//! `TensorParallelTrainer` — weight-sharded matmul: each replica
//! owns a slice of the weight matrix; activations are split, each
//! shard runs a partial matmul, then results are summed via
//! `AllReduce`.
//!
//! F6 ships the public surface + a host-side reference. Each shard
//! implements [`ShardProtocol`] which receives a partial input slice
//! and returns its partial output. The trainer collects all
//! partials and sums them.

use std::time::Instant;

use async_trait::async_trait;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

use crate::optimizer::{OptimizerKind, StepStats};

pub trait ShardProtocol: Send + 'static {
    type Msg: Send + 'static;
    /// Per-shard step: takes a slice of input + the local weight
    /// shard, returns `(partial_output, partial_loss, partial_grad_norm)`.
    fn make_step(
        input_slice: Vec<f32>,
        reply: oneshot::Sender<Result<ShardStepResult, GpuError>>,
    ) -> Self::Msg;
}

#[derive(Debug, Clone, Default)]
pub struct ShardStepResult {
    pub partial_output: Vec<f32>,
    pub loss: f32,
    pub grad_norm: f32,
    pub samples: usize,
}

#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    pub shard_count: usize,
    pub optimizer: OptimizerKind,
}

pub enum TensorParallelMsg<P: ShardProtocol> {
    Step {
        input: Vec<f32>,
        reply: oneshot::Sender<Result<(Vec<f32>, StepStats), GpuError>>,
    },
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<fn() -> P>),
}

pub struct TensorParallelTrainer<P: ShardProtocol> {
    config: TensorParallelConfig,
    shards: Vec<ActorRef<P::Msg>>,
}

impl<P: ShardProtocol> TensorParallelTrainer<P> {
    pub fn props(config: TensorParallelConfig, shards: Vec<ActorRef<P::Msg>>) -> Props<Self> {
        Props::create(move || TensorParallelTrainer {
            config: config.clone(),
            shards: shards.clone(),
        })
    }
}

#[async_trait]
impl<P: ShardProtocol> Actor for TensorParallelTrainer<P> {
    type Msg = TensorParallelMsg<P>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: TensorParallelMsg<P>) {
        match msg {
            TensorParallelMsg::_Phantom(_) => {}
            TensorParallelMsg::Step { input, reply } => {
                if self.shards.is_empty() {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "TensorParallelTrainer::Step: no shards".into(),
                    )));
                    return;
                }
                let _ = &self.config;
                let started = Instant::now();
                let n = self.shards.len();
                // Split input by chunks across shards (round-robin
                // along the row dimension).
                let chunk_size = (input.len() + n - 1) / n;
                let mut chunks: Vec<Vec<f32>> = Vec::with_capacity(n);
                for i in 0..n {
                    let lo = (i * chunk_size).min(input.len());
                    let hi = ((i + 1) * chunk_size).min(input.len());
                    chunks.push(input[lo..hi].to_vec());
                }
                let mut rxs = Vec::with_capacity(n);
                for (s, chunk) in self.shards.iter().zip(chunks) {
                    let (tx, rx) = oneshot::channel();
                    s.tell(P::make_step(chunk, tx));
                    rxs.push(rx);
                }
                tokio::spawn(async move {
                    let mut summed: Option<Vec<f32>> = None;
                    let mut total_loss = 0.0f32;
                    let mut total_grad = 0.0f32;
                    let mut total_samples = 0usize;
                    for rx in rxs {
                        match rx.await {
                            Ok(Ok(r)) => {
                                match summed.as_mut() {
                                    None => summed = Some(r.partial_output),
                                    Some(acc) => {
                                        if acc.len() != r.partial_output.len() {
                                            // Pad to longest.
                                            let m = acc.len().max(r.partial_output.len());
                                            acc.resize(m, 0.0);
                                            for (i, v) in r.partial_output.iter().enumerate() {
                                                acc[i] += *v;
                                            }
                                        } else {
                                            for (i, v) in r.partial_output.iter().enumerate() {
                                                acc[i] += *v;
                                            }
                                        }
                                    }
                                }
                                total_loss += r.loss * r.samples as f32;
                                total_grad += r.grad_norm * r.samples as f32;
                                total_samples += r.samples;
                            }
                            Ok(Err(e)) => {
                                let _ = reply.send(Err(e));
                                return;
                            }
                            Err(_) => {
                                let _ = reply.send(Err(GpuError::Unrecoverable(
                                    "tensor-parallel: shard dropped reply".into(),
                                )));
                                return;
                            }
                        }
                    }
                    let out = summed.unwrap_or_default();
                    let stats = StepStats {
                        loss: if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 },
                        grad_norm: if total_samples > 0 {
                            total_grad / total_samples as f32
                        } else {
                            0.0
                        },
                        step_micros: started.elapsed().as_micros() as u64,
                    };
                    let _ = reply.send(Ok((out, stats)));
                });
            }
        }
    }
}
