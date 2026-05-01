//! `MoeRouter` — mixture-of-experts: top-k softmax gating over N
//! expert backends. Each request is dispatched to the top-k
//! experts; their replies are blended by gating weights.

use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

pub trait ExpertProtocol: Send + 'static {
    type Msg: Send + 'static;
    fn make_run(
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    ) -> Self::Msg;
}

pub trait GateFn: Send + Sync + 'static {
    /// Score every expert for the given input. Higher = more
    /// preferred. Output length must equal expert count.
    fn score(&self, input: &[f32]) -> Result<Vec<f32>, GpuError>;
}

impl<F> GateFn for F
where
    F: Fn(&[f32]) -> Result<Vec<f32>, GpuError> + Send + Sync + 'static,
{
    fn score(&self, input: &[f32]) -> Result<Vec<f32>, GpuError> {
        self(input)
    }
}

pub struct MoeConfig<P: ExpertProtocol> {
    pub experts: Vec<ActorRef<P::Msg>>,
    pub gate: Arc<dyn GateFn>,
    pub top_k: usize,
}

impl<P: ExpertProtocol> Clone for MoeConfig<P> {
    fn clone(&self) -> Self {
        Self {
            experts: self.experts.clone(),
            gate: self.gate.clone(),
            top_k: self.top_k,
        }
    }
}

pub enum MoeMsg<P: ExpertProtocol> {
    Run {
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<fn() -> P>),
}

pub struct MoeRouter<P: ExpertProtocol> {
    cfg: MoeConfig<P>,
}

impl<P: ExpertProtocol> MoeRouter<P> {
    pub fn props(cfg: MoeConfig<P>) -> Props<Self> {
        Props::create(move || MoeRouter { cfg: cfg.clone() })
    }
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![0.0; scores.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

#[async_trait]
impl<P: ExpertProtocol> Actor for MoeRouter<P> {
    type Msg = MoeMsg<P>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: MoeMsg<P>) {
        match msg {
            MoeMsg::_Phantom(_) => {}
            MoeMsg::Run { input, reply } => {
                let cfg = self.cfg.clone();
                tokio::spawn(async move {
                    let scores = match cfg.gate.score(&input) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            return;
                        }
                    };
                    if scores.len() != cfg.experts.len() {
                        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                            "MoE: gate produced {} scores, expected {}",
                            scores.len(),
                            cfg.experts.len()
                        ))));
                        return;
                    }
                    // Top-k selection.
                    let mut idx: Vec<usize> = (0..scores.len()).collect();
                    idx.sort_by(|&a, &b| {
                        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let k = cfg.top_k.min(idx.len()).max(1);
                    let chosen: Vec<usize> = idx[..k].to_vec();
                    // Softmax over the chosen subset's scores.
                    let chosen_scores: Vec<f32> = chosen.iter().map(|&i| scores[i]).collect();
                    let weights = softmax(&chosen_scores);

                    // Dispatch in parallel.
                    let mut rxs = Vec::with_capacity(k);
                    for &i in &chosen {
                        let (tx, rx) = oneshot::channel();
                        cfg.experts[i].tell(P::make_run(input.clone(), tx));
                        rxs.push(rx);
                    }
                    // Collect.
                    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(k);
                    for rx in rxs {
                        match rx.await {
                            Ok(Ok(o)) => outputs.push(o),
                            Ok(Err(e)) => {
                                let _ = reply.send(Err(e));
                                return;
                            }
                            Err(_) => {
                                let _ = reply.send(Err(GpuError::Unrecoverable(
                                    "MoE: expert dropped reply".into(),
                                )));
                                return;
                            }
                        }
                    }
                    // Weighted blend (assume all outputs same length).
                    let n = outputs.iter().map(|v| v.len()).max().unwrap_or(0);
                    let mut blended = vec![0.0f32; n];
                    for (out, w) in outputs.iter().zip(&weights) {
                        for (i, v) in out.iter().enumerate().take(n) {
                            blended[i] += v * w;
                        }
                    }
                    let _ = reply.send(Ok(blended));
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    enum ExpertMsg {
        Run { input: Vec<f32>, reply: oneshot::Sender<Result<Vec<f32>, GpuError>> },
    }
    struct Constant(f32);
    #[async_trait]
    impl Actor for Constant {
        type Msg = ExpertMsg;
        async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ExpertMsg) {
            match msg {
                ExpertMsg::Run { input, reply } => {
                    let v: Vec<f32> = input.iter().map(|x| x + self.0).collect();
                    let _ = reply.send(Ok(v));
                }
            }
        }
    }
    struct ExpertProto;
    impl ExpertProtocol for ExpertProto {
        type Msg = ExpertMsg;
        fn make_run(
            input: Vec<f32>,
            reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
        ) -> ExpertMsg {
            ExpertMsg::Run { input, reply }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn top1_routes_to_top_expert() {
        let sys = ActorSystem::create("moe-test", Config::empty()).await.unwrap();
        let e0 = sys.actor_of(rakka_core::actor::Props::create(|| Constant(10.0)), "e0").unwrap();
        let e1 = sys.actor_of(rakka_core::actor::Props::create(|| Constant(100.0)), "e1").unwrap();

        // Gate prefers e1 always.
        let gate: Arc<dyn GateFn> = Arc::new(|_input: &[f32]| Ok(vec![0.0, 10.0]));
        let cfg = MoeConfig::<ExpertProto> {
            experts: vec![e0, e1],
            gate,
            top_k: 1,
        };
        let router = sys.actor_of(MoeRouter::<ExpertProto>::props(cfg), "router").unwrap();
        let (tx, rx) = oneshot::channel();
        router.tell(MoeMsg::Run { input: vec![1.0, 2.0], reply: tx });
        let v = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        // Only e1 routed; output = input + 100 = [101, 102] weighted by softmax(top-1)=1.0.
        assert!((v[0] - 101.0).abs() < 1e-3);
        assert!((v[1] - 102.0).abs() < 1e-3);

        sys.terminate().await;
    }
}
