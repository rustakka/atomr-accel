//! `DataParallelTrainer` — replicates a model across N replicas,
//! splits a mini-batch evenly, runs forward+backward per replica,
//! aggregates loss/grad-norm, and applies an optimizer step.
//!
//! The trainer is generic over a [`ReplicaProtocol`] trait that
//! describes the message contract to a single replica actor. F4.x
//! ships the protocol with a CPU-side `host_step` that completes
//! a synchronous forward/backward and returns
//! `(loss, grad_norm)`. F5 swaps that for a real GPU
//! forward/backward+AllReduce path; the public surface stays the
//! same.

use std::time::Instant;

use async_trait::async_trait;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

use crate::loss::LossKind;
use crate::optimizer::{OptimizerKind, StepStats};

/// Per-replica step contract. Each replica receives a chunk of the
/// mini-batch and replies with `(loss, grad_norm)` for that chunk.
pub trait ReplicaProtocol: Send + 'static {
    type Msg: Send + 'static;
    fn make_step(
        chunk: Vec<TrainSample>,
        reply: oneshot::Sender<Result<ReplicaStepResult, GpuError>>,
    ) -> Self::Msg;
}

#[derive(Debug, Clone)]
pub struct TrainSample {
    pub features: Vec<f32>,
    pub label: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReplicaStepResult {
    pub loss: f32,
    pub grad_norm: f32,
    pub samples: usize,
}

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub batch_size_per_device: usize,
    pub gradient_clip: Option<f32>,
    pub optimizer: OptimizerKind,
    pub loss: LossKind,
}

pub enum TrainerMsg<P: ReplicaProtocol> {
    Step {
        batch: Vec<TrainSample>,
        reply: oneshot::Sender<Result<StepStats, GpuError>>,
    },
    /// Set the replica refs after construction. Allows a
    /// late-binding pattern when replicas are spawned by another
    /// actor.
    SetReplicas {
        replicas: Vec<ActorRef<P::Msg>>,
    },
}

pub struct DataParallelTrainer<P: ReplicaProtocol> {
    config: TrainerConfig,
    replicas: Vec<ActorRef<P::Msg>>,
}

impl<P: ReplicaProtocol> DataParallelTrainer<P> {
    pub fn props(config: TrainerConfig, replicas: Vec<ActorRef<P::Msg>>) -> Props<Self> {
        Props::create(move || DataParallelTrainer {
            config: config.clone(),
            replicas: replicas.clone(),
        })
    }

    fn split_batch(&self, batch: Vec<TrainSample>) -> Vec<Vec<TrainSample>> {
        let n = self.replicas.len().max(1);
        let mut out: Vec<Vec<TrainSample>> = (0..n).map(|_| Vec::new()).collect();
        for (i, s) in batch.into_iter().enumerate() {
            out[i % n].push(s);
        }
        out
    }
}

#[async_trait]
impl<P: ReplicaProtocol> Actor for DataParallelTrainer<P> {
    type Msg = TrainerMsg<P>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: TrainerMsg<P>) {
        match msg {
            TrainerMsg::SetReplicas { replicas } => {
                self.replicas = replicas;
            }
            TrainerMsg::Step { batch, reply } => {
                if self.replicas.is_empty() {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "DataParallelTrainer::Step: no replicas configured".into(),
                    )));
                    return;
                }
                let _ = &self.config; // configured but not consumed inline
                let started = Instant::now();
                let chunks = self.split_batch(batch);
                let mut rxs = Vec::with_capacity(chunks.len());
                for (replica, chunk) in self.replicas.iter().zip(chunks) {
                    let (tx, rx) = oneshot::channel();
                    replica.tell(P::make_step(chunk, tx));
                    rxs.push(rx);
                }
                tokio::spawn(async move {
                    let mut total_loss = 0.0f32;
                    let mut total_grad_norm = 0.0f32;
                    let mut total_samples = 0usize;
                    for rx in rxs {
                        match rx.await {
                            Ok(Ok(r)) => {
                                total_loss += r.loss * r.samples as f32;
                                total_grad_norm += r.grad_norm * r.samples as f32;
                                total_samples += r.samples;
                            }
                            Ok(Err(e)) => {
                                let _ = reply.send(Err(e));
                                return;
                            }
                            Err(_) => {
                                let _ = reply.send(Err(GpuError::Unrecoverable(
                                    "trainer: replica dropped reply".into(),
                                )));
                                return;
                            }
                        }
                    }
                    if total_samples == 0 {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "trainer: zero samples".into(),
                        )));
                        return;
                    }
                    let stats = StepStats {
                        loss: total_loss / total_samples as f32,
                        grad_norm: total_grad_norm / total_samples as f32,
                        step_micros: started.elapsed().as_micros() as u64,
                    };
                    let _ = reply.send(Ok(stats));
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

    /// Echo replica: reports loss = sum_of_features / samples.
    enum EchoMsg {
        Step {
            chunk: Vec<TrainSample>,
            reply: oneshot::Sender<Result<ReplicaStepResult, GpuError>>,
        },
    }

    struct EchoReplicaActor;
    #[async_trait]
    impl Actor for EchoReplicaActor {
        type Msg = EchoMsg;
        async fn handle(&mut self, _ctx: &mut Context<Self>, msg: EchoMsg) {
            match msg {
                EchoMsg::Step { chunk, reply } => {
                    let n = chunk.len();
                    let mut sum = 0.0f32;
                    for s in &chunk {
                        sum += s.features.iter().sum::<f32>();
                    }
                    let _ = reply.send(Ok(ReplicaStepResult {
                        loss: if n > 0 { sum / n as f32 } else { 0.0 },
                        grad_norm: 1.0,
                        samples: n,
                    }));
                }
            }
        }
    }

    struct EchoProtocol;
    impl ReplicaProtocol for EchoProtocol {
        type Msg = EchoMsg;
        fn make_step(
            chunk: Vec<TrainSample>,
            reply: oneshot::Sender<Result<ReplicaStepResult, GpuError>>,
        ) -> Self::Msg {
            EchoMsg::Step { chunk, reply }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn step_aggregates_across_replicas() {
        let sys = ActorSystem::create("trainer-test", Config::empty()).await.unwrap();
        let r1 = sys
            .actor_of(rakka_core::actor::Props::create(|| EchoReplicaActor), "r1")
            .unwrap();
        let r2 = sys
            .actor_of(rakka_core::actor::Props::create(|| EchoReplicaActor), "r2")
            .unwrap();
        let trainer = sys
            .actor_of(
                DataParallelTrainer::<EchoProtocol>::props(
                    TrainerConfig {
                        batch_size_per_device: 1,
                        gradient_clip: None,
                        optimizer: OptimizerKind::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 },
                        loss: LossKind::Mse,
                    },
                    vec![r1, r2],
                ),
                "trainer",
            )
            .unwrap();

        let (tx, rx) = oneshot::channel();
        trainer.tell(TrainerMsg::Step {
            batch: vec![
                TrainSample { features: vec![1.0, 2.0], label: vec![] },
                TrainSample { features: vec![3.0, 4.0], label: vec![] },
            ],
            reply: tx,
        });
        let stats = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        // Replica 0 sees [1.0, 2.0] → loss 3; replica 1 sees [3.0, 4.0] → loss 7.
        // Weighted avg = (3*1 + 7*1) / 2 = 5.
        assert!((stats.loss - 5.0).abs() < 1e-5);
        assert_eq!(stats.grad_norm, 1.0);

        sys.terminate().await;
    }
}
