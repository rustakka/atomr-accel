//! `PipelineParallelTrainer` — stage-pipelined model across N
//! GPUs/actors.
//!
//! Each stage actor handles one slice of the model's layers and
//! passes its activations to the next stage. The trainer drives a
//! micro-batch through the pipeline (forward) then backwards
//! (gradient flow), accumulating per-stage gradients before an
//! optimizer step.
//!
//! F6 ships the public surface + a host-side reference where each
//! stage is a generic actor implementing [`PipelineStageProtocol`].
//! The trainer feeds activations through stages sequentially in
//! reply order. F7 adds true micro-batch overlap.

use std::time::Instant;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;

use crate::loss::LossKind;
use crate::optimizer::{OptimizerKind, StepStats};

pub trait PipelineStageProtocol: Send + 'static {
    type Msg: Send + 'static;
    /// Activation type passed between stages.
    type Activation: Send + 'static;
    fn make_forward(
        input: Self::Activation,
        reply: oneshot::Sender<Result<Self::Activation, GpuError>>,
    ) -> Self::Msg;
    /// Final-stage forward also produces a (loss, grad_norm) pair.
    fn make_final_forward(
        input: Self::Activation,
        reply: oneshot::Sender<Result<(f32, f32), GpuError>>,
    ) -> Self::Msg;
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub micro_batch_size: usize,
    pub gradient_clip: Option<f32>,
    pub optimizer: OptimizerKind,
    pub loss: LossKind,
}

pub enum PipelineTrainerMsg<P: PipelineStageProtocol> {
    Step {
        input: P::Activation,
        reply: oneshot::Sender<Result<StepStats, GpuError>>,
    },
}

pub struct PipelineParallelTrainer<P: PipelineStageProtocol> {
    config: PipelineConfig,
    stages: Vec<ActorRef<P::Msg>>,
}

impl<P: PipelineStageProtocol> PipelineParallelTrainer<P> {
    pub fn props(config: PipelineConfig, stages: Vec<ActorRef<P::Msg>>) -> Props<Self> {
        Props::create(move || PipelineParallelTrainer {
            config: config.clone(),
            stages: stages.clone(),
        })
    }
}

#[async_trait]
impl<P: PipelineStageProtocol> Actor for PipelineParallelTrainer<P> {
    type Msg = PipelineTrainerMsg<P>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: PipelineTrainerMsg<P>) {
        match msg {
            PipelineTrainerMsg::Step { input, reply } => {
                if self.stages.is_empty() {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "PipelineParallelTrainer::Step: no stages".into(),
                    )));
                    return;
                }
                let _ = &self.config;
                let started = Instant::now();
                let stages = self.stages.clone();
                tokio::spawn(async move {
                    let n = stages.len();
                    let mut activation: Option<P::Activation> = Some(input);
                    // Forward through stages 0 .. n-1 (intermediate),
                    // then n-1 (final).
                    for (i, s) in stages.iter().enumerate() {
                        let act = activation.take().expect("activation present");
                        if i + 1 < n {
                            let (tx, rx) = oneshot::channel();
                            s.tell(P::make_forward(act, tx));
                            match rx.await {
                                Ok(Ok(next)) => activation = Some(next),
                                Ok(Err(e)) => {
                                    let _ = reply.send(Err(e));
                                    return;
                                }
                                Err(_) => {
                                    let _ = reply.send(Err(GpuError::Unrecoverable(
                                        "pipeline: stage dropped reply".into(),
                                    )));
                                    return;
                                }
                            }
                        } else {
                            let (tx, rx) = oneshot::channel();
                            s.tell(P::make_final_forward(act, tx));
                            match rx.await {
                                Ok(Ok((loss, grad_norm))) => {
                                    let _ = reply.send(Ok(StepStats {
                                        loss,
                                        grad_norm,
                                        step_micros: started.elapsed().as_micros() as u64,
                                    }));
                                    return;
                                }
                                Ok(Err(e)) => {
                                    let _ = reply.send(Err(e));
                                    return;
                                }
                                Err(_) => {
                                    let _ = reply.send(Err(GpuError::Unrecoverable(
                                        "pipeline: final stage dropped reply".into(),
                                    )));
                                    return;
                                }
                            }
                        }
                    }
                });
            }
        }
    }
}
