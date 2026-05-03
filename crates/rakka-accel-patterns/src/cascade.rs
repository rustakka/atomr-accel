//! `InferenceCascade<Req, Resp>` — chain of models with early-exit
//! routing. Cheap models run first; expensive models only run if the
//! cheap model's confidence is below a threshold.
//!
//! Generic over a user-supplied per-stage [`CascadeStage`] that takes
//! a `Req` and returns a `(Resp, confidence)` pair. The cascade
//! returns the first stage's response whose confidence ≥ threshold,
//! or the final stage's response if none clear the bar.

use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

/// User-supplied per-stage handler. Returns `(Resp, confidence)`.
pub trait CascadeStage<Req, Resp>: Send + Sync + 'static {
    fn run(&self, req: &Req) -> Result<(Resp, f32), GpuError>;
}

impl<Req, Resp, F> CascadeStage<Req, Resp> for F
where
    F: Fn(&Req) -> Result<(Resp, f32), GpuError> + Send + Sync + 'static,
{
    fn run(&self, req: &Req) -> Result<(Resp, f32), GpuError> {
        self(req)
    }
}

pub struct CascadeStageEntry<Req, Resp> {
    pub stage: Arc<dyn CascadeStage<Req, Resp>>,
    /// If the stage's confidence is ≥ this threshold, return early.
    pub confidence_threshold: f32,
}

pub struct CascadeConfig<Req, Resp> {
    pub stages: Vec<CascadeStageEntry<Req, Resp>>,
}

impl<Req, Resp> Clone for CascadeConfig<Req, Resp> {
    fn clone(&self) -> Self {
        Self {
            stages: self
                .stages
                .iter()
                .map(|e| CascadeStageEntry {
                    stage: e.stage.clone(),
                    confidence_threshold: e.confidence_threshold,
                })
                .collect(),
        }
    }
}

pub enum CascadeMsg<Req: Send + 'static, Resp: Send + 'static> {
    Predict {
        req: Req,
        reply: oneshot::Sender<Result<CascadeReply<Resp>, GpuError>>,
    },
}

#[derive(Debug)]
pub struct CascadeReply<Resp> {
    pub response: Resp,
    /// Index of the stage that produced the response.
    pub stage_index: usize,
    /// Confidence reported by that stage.
    pub confidence: f32,
}

pub struct InferenceCascade<Req: Send + 'static, Resp: Send + 'static> {
    config: CascadeConfig<Req, Resp>,
}

impl<Req: Send + 'static, Resp: Send + 'static> InferenceCascade<Req, Resp> {
    pub fn props(config: CascadeConfig<Req, Resp>) -> Props<Self> {
        Props::create(move || InferenceCascade { config: config.clone() })
    }
}

#[async_trait]
impl<Req: Send + 'static, Resp: Send + 'static> Actor for InferenceCascade<Req, Resp> {
    type Msg = CascadeMsg<Req, Resp>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: CascadeMsg<Req, Resp>) {
        match msg {
            CascadeMsg::Predict { req, reply } => {
                let n = self.config.stages.len();
                if n == 0 {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "InferenceCascade: no stages configured".into(),
                    )));
                    return;
                }
                let mut last: Option<(Resp, f32, usize)> = None;
                for (idx, entry) in self.config.stages.iter().enumerate() {
                    let result = entry.stage.run(&req);
                    match result {
                        Ok((resp, conf)) => {
                            // If we've cleared the threshold, return early.
                            if conf >= entry.confidence_threshold {
                                let _ = reply.send(Ok(CascadeReply {
                                    response: resp,
                                    stage_index: idx,
                                    confidence: conf,
                                }));
                                return;
                            }
                            last = Some((resp, conf, idx));
                        }
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            return;
                        }
                    }
                }
                if let Some((resp, conf, idx)) = last {
                    let _ = reply.send(Ok(CascadeReply {
                        response: resp,
                        stage_index: idx,
                        confidence: conf,
                    }));
                } else {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "InferenceCascade: no stage produced a response".into(),
                    )));
                }
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cascade_returns_first_confident_stage() {
        // Stage 0: low confidence (0.3). Stage 1: high (0.9).
        let s0: Arc<dyn CascadeStage<u32, u32>> =
            Arc::new(|x: &u32| Ok((*x + 1, 0.3)));
        let s1: Arc<dyn CascadeStage<u32, u32>> =
            Arc::new(|x: &u32| Ok((*x + 100, 0.9)));
        let cfg = CascadeConfig {
            stages: vec![
                CascadeStageEntry { stage: s0, confidence_threshold: 0.5 },
                CascadeStageEntry { stage: s1, confidence_threshold: 0.5 },
            ],
        };

        let sys = ActorSystem::create("cascade-test", Config::empty()).await.unwrap();
        let actor = sys
            .actor_of(InferenceCascade::<u32, u32>::props(cfg), "cascade")
            .unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(CascadeMsg::Predict { req: 5, reply: tx });
        let r = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        // Stage 1's threshold met first.
        assert_eq!(r.response, 105);
        assert_eq!(r.stage_index, 1);

        sys.terminate().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cascade_falls_through_to_last_stage() {
        // Both stages below threshold; cascade returns last.
        let s0: Arc<dyn CascadeStage<u32, u32>> =
            Arc::new(|x: &u32| Ok((*x + 1, 0.1)));
        let s1: Arc<dyn CascadeStage<u32, u32>> =
            Arc::new(|x: &u32| Ok((*x + 2, 0.2)));
        let cfg = CascadeConfig {
            stages: vec![
                CascadeStageEntry { stage: s0, confidence_threshold: 0.99 },
                CascadeStageEntry { stage: s1, confidence_threshold: 0.99 },
            ],
        };

        let sys = ActorSystem::create("cascade-fall", Config::empty()).await.unwrap();
        let actor = sys
            .actor_of(InferenceCascade::<u32, u32>::props(cfg), "cascade")
            .unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(CascadeMsg::Predict { req: 0, reply: tx });
        let r = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(r.response, 2);
        assert_eq!(r.stage_index, 1);

        sys.terminate().await;
    }
}
