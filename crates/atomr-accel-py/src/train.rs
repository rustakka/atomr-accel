//! `atomr_accel_train` Python wrappers.
//!
//! Phase 2 ships:
//! - `AsyncParameterServer` — non-generic; spawn + `push_gradient` /
//!   `pull_weights` representative methods to exercise the dispatch
//!   path end-to-end.
//! - `DataParallelTrainer`, `PipelineParallelTrainer`,
//!   `TensorParallelTrainer` — generic over a user-supplied
//!   `ReplicaProtocol` / `PipelineStageProtocol` / `ShardProtocol`,
//!   so we ship them as **structural anchors** (no `__new__`, just a
//!   `__repr__`). Wiring those up means projecting the protocol
//!   trait into a Python typed-bytes contract; tracked as Phase 2.5.

use std::time::Duration;

use atomr_core::prelude::async_trait;
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;
use atomr_accel_train::data_parallel::{
    DataParallelTrainer, ReplicaProtocol, ReplicaStepResult, TrainSample, TrainerConfig, TrainerMsg,
};
use atomr_accel_train::loss::LossKind;
use atomr_accel_train::optimizer::OptimizerKind;
use atomr_accel_train::parameter_server::{
    AsyncParameterServer, ParameterServerMsg, ParameterServerStats, WorkerId,
};
use atomr_accel_train::pipeline_parallel::{
    PipelineConfig, PipelineParallelTrainer, PipelineStageProtocol, PipelineTrainerMsg,
};
use atomr_accel_train::tensor_parallel::{
    ShardProtocol, ShardStepResult, TensorParallelConfig, TensorParallelMsg, TensorParallelTrainer,
};
use atomr_core::actor::{Actor, ActorRef, Context, Props};

use crate::errors;
use crate::runtime::runtime;
use crate::system::PySystem;

/// `AsyncParameterServer` handle — central parameter store with
/// async gradient pushes and async weight pulls.
#[pyclass(name = "AsyncParameterServer", module = "atomr_accel._native")]
pub struct PyAsyncParameterServer {
    actor_ref: ActorRef<ParameterServerMsg>,
}

fn parse_optimizer(kind: &str, lr: f32) -> PyResult<OptimizerKind> {
    match kind.to_ascii_lowercase().as_str() {
        "sgd" => Ok(OptimizerKind::Sgd {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
        }),
        "adamw" => Ok(OptimizerKind::AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }),
        _ => Err(errors::map_str(format!(
            "optimizer must be 'sgd' or 'adamw' (got {kind:?})"
        ))),
    }
}

#[pymethods]
impl PyAsyncParameterServer {
    /// Spawn an `AsyncParameterServer` actor under `system`. Returns
    /// a handle wrapping its `ActorRef<ParameterServerMsg>`.
    ///
    /// `optimizer` is a string tag — `"sgd"` or `"adamw"` — with
    /// `lr` controlling the learning rate. Other hyperparameters
    /// default to standard values (Phase 2.5 will widen the surface).
    #[staticmethod]
    #[pyo3(signature = (
        system, initial_weights, optimizer="sgd", lr=0.01,
        max_staleness=4, name=None,
    ))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        initial_weights: Vec<f32>,
        optimizer: &str,
        lr: f32,
        max_staleness: u64,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let opt = parse_optimizer(optimizer, lr)?;
        let actor_name = name.unwrap_or_else(|| "parameter-server".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(
                    AsyncParameterServer::props(initial_weights, opt, max_staleness),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyAsyncParameterServer { actor_ref })
    }

    /// Push a gradient vector. Returns the new server-side version on
    /// success; raises `Unrecoverable` if the push exceeds
    /// `max_staleness`.
    #[pyo3(signature = (worker_id, worker_version, gradient, timeout_secs=10.0))]
    fn push_gradient(
        &self,
        py: Python<'_>,
        worker_id: u32,
        worker_version: u64,
        gradient: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<u64> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ParameterServerMsg::PushGradient {
                    worker: WorkerId(worker_id),
                    worker_version,
                    gradient,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("parameter server dropped reply")),
                    Err(_) => Err(errors::map_str("push_gradient timed out")),
                }
            })
        })
    }

    /// Pull the latest weights + version.
    #[pyo3(signature = (worker_id, timeout_secs=10.0))]
    fn pull_weights(
        &self,
        py: Python<'_>,
        worker_id: u32,
        timeout_secs: f64,
    ) -> PyResult<(Vec<f32>, u64)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ParameterServerMsg::PullWeights {
                    worker: WorkerId(worker_id),
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok((w, v))) => Ok((w, v)),
                    Ok(Err(_)) => Err(errors::map_str("parameter server dropped reply")),
                    Err(_) => Err(errors::map_str("pull_weights timed out")),
                }
            })
        })
    }

    /// Pull server-side stats (version, gradients applied, average
    /// staleness).
    #[pyo3(signature = (timeout_secs=2.0))]
    fn stats(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<(u64, u64, u64, f32)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ParameterServerMsg::Stats { reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(s)) => Ok(unpack_stats(s)),
                    Ok(Err(_)) => Err(errors::map_str("parameter server dropped reply")),
                    Err(_) => Err(errors::map_str("stats timed out")),
                }
            })
        })
    }

    // ─── Async (asyncio) variants ────────────────────────────────

    #[pyo3(signature = (worker_id, worker_version, gradient, timeout_secs=10.0))]
    fn push_gradient_async<'py>(
        &self,
        py: Python<'py>,
        worker_id: u32,
        worker_version: u64,
        gradient: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(ParameterServerMsg::PushGradient {
                worker: WorkerId(worker_id),
                worker_version,
                gradient,
                reply: tx,
            });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("parameter server dropped reply")),
                Err(_) => Err(errors::map_str("push_gradient timed out")),
            }
        })
    }

    #[pyo3(signature = (worker_id, timeout_secs=10.0))]
    fn pull_weights_async<'py>(
        &self,
        py: Python<'py>,
        worker_id: u32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(ParameterServerMsg::PullWeights {
                worker: WorkerId(worker_id),
                reply: tx,
            });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok((w, v))) => Ok((w, v)),
                Ok(Err(_)) => Err(errors::map_str("parameter server dropped reply")),
                Err(_) => Err(errors::map_str("pull_weights timed out")),
            }
        })
    }

    #[pyo3(signature = (timeout_secs=2.0))]
    fn stats_async<'py>(&self, py: Python<'py>, timeout_secs: f64) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(ParameterServerMsg::Stats { reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(s)) => Ok(unpack_stats(s)),
                Ok(Err(_)) => Err(errors::map_str("parameter server dropped reply")),
                Err(_) => Err(errors::map_str("stats timed out")),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "AsyncParameterServer(handle)"
    }
}

fn unpack_stats(s: ParameterServerStats) -> (u64, u64, u64, f32) {
    (
        s.version,
        s.gradients_applied,
        s.weights_pulled,
        s.avg_staleness,
    )
}

// ─── Phase 2.5 generic trainers — internal echo replicas/stages/shards
//
// Each generic trainer is monomorphized with a mock-mode replica /
// stage / shard actor that does the simplest reasonable work (sum
// of features for replica step, identity for pipeline forward,
// passthrough for shard step). This exercises the dispatch path
// end-to-end so Python callers can `step(...)` without wiring real
// GPU kernels yet. Phase 2.6 will accept Python-supplied actor refs.

// ─── DataParallelTrainer ────────────────────────────────────────

/// Internal echo replica for `DataParallelTrainer`. Reports
/// `loss = sum_of_features / samples`; `grad_norm = 1.0`.
pub(crate) enum EchoTrainReplicaMsg {
    Step {
        chunk: Vec<TrainSample>,
        reply: oneshot::Sender<Result<ReplicaStepResult, GpuError>>,
    },
}

pub(crate) struct EchoTrainReplicaActor;

#[async_trait]
impl Actor for EchoTrainReplicaActor {
    type Msg = EchoTrainReplicaMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: EchoTrainReplicaMsg) {
        match msg {
            EchoTrainReplicaMsg::Step { chunk, reply } => {
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

pub(crate) struct EchoTrainReplicaProto;
impl ReplicaProtocol for EchoTrainReplicaProto {
    type Msg = EchoTrainReplicaMsg;
    fn make_step(
        chunk: Vec<TrainSample>,
        reply: oneshot::Sender<Result<ReplicaStepResult, GpuError>>,
    ) -> Self::Msg {
        EchoTrainReplicaMsg::Step { chunk, reply }
    }
}

/// `DataParallelTrainer` handle (Phase 2.5: internal echo replicas
/// produce `loss = mean(features)` so callers can prove the dispatch
/// path is wired).
#[pyclass(name = "DataParallelTrainer", module = "atomr_accel._native")]
pub struct PyDataParallelTrainer {
    actor_ref: ActorRef<TrainerMsg<EchoTrainReplicaProto>>,
}

#[pymethods]
impl PyDataParallelTrainer {
    /// Spawn a `DataParallelTrainer` over `n_replicas` internal
    /// echo replicas. Each call to `step(...)` round-robins samples
    /// across replicas and aggregates `(loss, grad_norm)`.
    #[staticmethod]
    #[pyo3(signature = (
        system, n_replicas=2, batch_size_per_device=4, lr=0.01, name=None,
    ))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        n_replicas: usize,
        batch_size_per_device: usize,
        lr: f32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if n_replicas == 0 {
            return Err(errors::map_str("n_replicas must be ≥ 1"));
        }
        let actor_name = name.unwrap_or_else(|| "data-parallel".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            // Phase 2.5: typed payloads — internal echo replicas.
            let mut replicas = Vec::with_capacity(n_replicas);
            for i in 0..n_replicas {
                let r = system
                    .inner
                    .actor_of(
                        Props::create(|| EchoTrainReplicaActor),
                        &format!("{actor_name}-replica-{i}"),
                    )
                    .map_err(errors::map_str)?;
                replicas.push(r);
            }
            system
                .inner
                .actor_of(
                    DataParallelTrainer::<EchoTrainReplicaProto>::props(
                        TrainerConfig {
                            batch_size_per_device,
                            gradient_clip: None,
                            optimizer: OptimizerKind::Sgd {
                                lr,
                                momentum: 0.0,
                                weight_decay: 0.0,
                            },
                            loss: LossKind::Mse,
                        },
                        replicas,
                    ),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyDataParallelTrainer { actor_ref })
    }

    /// Run a single training step over `batch`. Each entry is a list
    /// of feature floats; labels are unused by the echo replicas.
    /// Returns `(loss, grad_norm, step_micros)`.
    #[pyo3(signature = (batch, timeout_secs=10.0))]
    fn step(
        &self,
        py: Python<'_>,
        batch: Vec<Vec<f32>>,
        timeout_secs: f64,
    ) -> PyResult<(f32, f32, u64)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let samples: Vec<TrainSample> = batch
                    .into_iter()
                    .map(|features| TrainSample {
                        features,
                        label: vec![],
                    })
                    .collect();
                let (tx, rx) = oneshot::channel();
                actor.tell(TrainerMsg::Step {
                    batch: samples,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(s))) => Ok((s.loss, s.grad_norm, s.step_micros)),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("trainer dropped reply")),
                    Err(_) => Err(errors::map_str("step timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "DataParallelTrainer(handle)"
    }
}

// ─── PipelineParallelTrainer ────────────────────────────────────

/// Internal pipeline stage actor. Intermediate stages forward the
/// activation unchanged; the final stage produces a synthetic
/// `(loss, grad_norm)`.
pub(crate) enum EchoPipelineStageMsg {
    Forward {
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
    FinalForward {
        input: Vec<f32>,
        reply: oneshot::Sender<Result<(f32, f32), GpuError>>,
    },
}

pub(crate) struct EchoPipelineStageActor;

#[async_trait]
impl Actor for EchoPipelineStageActor {
    type Msg = EchoPipelineStageMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: EchoPipelineStageMsg) {
        match msg {
            EchoPipelineStageMsg::Forward { input, reply } => {
                // Identity activation passthrough.
                let _ = reply.send(Ok(input));
            }
            EchoPipelineStageMsg::FinalForward { input, reply } => {
                let n = input.len().max(1) as f32;
                let sum: f32 = input.iter().sum();
                let loss = sum / n;
                let grad_norm = 1.0;
                let _ = reply.send(Ok((loss, grad_norm)));
            }
        }
    }
}

pub(crate) struct EchoPipelineProto;
impl PipelineStageProtocol for EchoPipelineProto {
    type Msg = EchoPipelineStageMsg;
    type Activation = Vec<f32>;
    fn make_forward(
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    ) -> Self::Msg {
        EchoPipelineStageMsg::Forward { input, reply }
    }
    fn make_final_forward(
        input: Vec<f32>,
        reply: oneshot::Sender<Result<(f32, f32), GpuError>>,
    ) -> Self::Msg {
        EchoPipelineStageMsg::FinalForward { input, reply }
    }
}

/// `PipelineParallelTrainer` handle (Phase 2.5: internal echo stages
/// — intermediate stages identity-forward; final stage reports
/// `loss = mean(microbatch)`).
#[pyclass(name = "PipelineParallelTrainer", module = "atomr_accel._native")]
pub struct PyPipelineParallelTrainer {
    actor_ref: ActorRef<PipelineTrainerMsg<EchoPipelineProto>>,
}

#[pymethods]
impl PyPipelineParallelTrainer {
    /// Spawn a `PipelineParallelTrainer` with `n_stages` internal
    /// echo stages.
    #[staticmethod]
    #[pyo3(signature = (
        system, n_stages=2, micro_batch_size=4, lr=0.01, name=None,
    ))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        n_stages: usize,
        micro_batch_size: usize,
        lr: f32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if n_stages == 0 {
            return Err(errors::map_str("n_stages must be ≥ 1"));
        }
        let actor_name = name.unwrap_or_else(|| "pipeline-parallel".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            // Phase 2.5: typed payloads — internal echo stages.
            let mut stages = Vec::with_capacity(n_stages);
            for i in 0..n_stages {
                let s = system
                    .inner
                    .actor_of(
                        Props::create(|| EchoPipelineStageActor),
                        &format!("{actor_name}-stage-{i}"),
                    )
                    .map_err(errors::map_str)?;
                stages.push(s);
            }
            system
                .inner
                .actor_of(
                    PipelineParallelTrainer::<EchoPipelineProto>::props(
                        PipelineConfig {
                            micro_batch_size,
                            gradient_clip: None,
                            optimizer: OptimizerKind::Sgd {
                                lr,
                                momentum: 0.0,
                                weight_decay: 0.0,
                            },
                            loss: LossKind::Mse,
                        },
                        stages,
                    ),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyPipelineParallelTrainer { actor_ref })
    }

    /// Drive `microbatch` through the pipeline; returns
    /// `(loss, grad_norm, step_micros)`.
    #[pyo3(signature = (microbatch, timeout_secs=10.0))]
    fn step(
        &self,
        py: Python<'_>,
        microbatch: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<(f32, f32, u64)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(PipelineTrainerMsg::Step {
                    input: microbatch,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(s))) => Ok((s.loss, s.grad_norm, s.step_micros)),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("pipeline trainer dropped reply")),
                    Err(_) => Err(errors::map_str("step timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "PipelineParallelTrainer(handle)"
    }
}

// ─── TensorParallelTrainer ──────────────────────────────────────

/// Internal tensor-parallel shard. Returns the input slice as the
/// partial output (so the trainer's all-sum produces the original
/// tensor under shard-by-row partitioning).
pub(crate) enum EchoShardMsg {
    Step {
        input_slice: Vec<f32>,
        reply: oneshot::Sender<Result<ShardStepResult, GpuError>>,
    },
}

pub(crate) struct EchoShardActor;

#[async_trait]
impl Actor for EchoShardActor {
    type Msg = EchoShardMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: EchoShardMsg) {
        match msg {
            EchoShardMsg::Step { input_slice, reply } => {
                let samples = input_slice.len();
                let sum: f32 = input_slice.iter().sum();
                let loss = if samples > 0 {
                    sum / samples as f32
                } else {
                    0.0
                };
                let _ = reply.send(Ok(ShardStepResult {
                    partial_output: input_slice,
                    loss,
                    grad_norm: 1.0,
                    samples,
                }));
            }
        }
    }
}

pub(crate) struct EchoShardProto;
impl ShardProtocol for EchoShardProto {
    type Msg = EchoShardMsg;
    fn make_step(
        input_slice: Vec<f32>,
        reply: oneshot::Sender<Result<ShardStepResult, GpuError>>,
    ) -> Self::Msg {
        EchoShardMsg::Step { input_slice, reply }
    }
}

/// `TensorParallelTrainer` handle (Phase 2.5: internal echo shards
/// where `partial_output = input_slice`; the trainer sums shard
/// outputs into a reconstructed tensor).
#[pyclass(name = "TensorParallelTrainer", module = "atomr_accel._native")]
pub struct PyTensorParallelTrainer {
    actor_ref: ActorRef<TensorParallelMsg<EchoShardProto>>,
}

#[pymethods]
impl PyTensorParallelTrainer {
    /// Spawn a `TensorParallelTrainer` with `n_shards` internal echo
    /// shards.
    #[staticmethod]
    #[pyo3(signature = (system, n_shards=2, lr=0.01, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        n_shards: usize,
        lr: f32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if n_shards == 0 {
            return Err(errors::map_str("n_shards must be ≥ 1"));
        }
        let actor_name = name.unwrap_or_else(|| "tensor-parallel".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            // Phase 2.5: typed payloads — internal echo shards.
            let mut shards = Vec::with_capacity(n_shards);
            for i in 0..n_shards {
                let s = system
                    .inner
                    .actor_of(
                        Props::create(|| EchoShardActor),
                        &format!("{actor_name}-shard-{i}"),
                    )
                    .map_err(errors::map_str)?;
                shards.push(s);
            }
            system
                .inner
                .actor_of(
                    TensorParallelTrainer::<EchoShardProto>::props(
                        TensorParallelConfig {
                            shard_count: n_shards,
                            optimizer: OptimizerKind::Sgd {
                                lr,
                                momentum: 0.0,
                                weight_decay: 0.0,
                            },
                        },
                        shards,
                    ),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyTensorParallelTrainer { actor_ref })
    }

    /// Forward pass: split `x` across shards, sum partial outputs,
    /// return the reconstructed tensor along with a `loss`.
    #[pyo3(signature = (x, timeout_secs=10.0))]
    fn forward(&self, py: Python<'_>, x: Vec<f32>, timeout_secs: f64) -> PyResult<(Vec<f32>, f32)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(TensorParallelMsg::Step {
                    input: x,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok((out, stats)))) => Ok((out, stats.loss)),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("tensor-parallel trainer dropped reply")),
                    Err(_) => Err(errors::map_str("forward timed out")),
                }
            })
        })
    }

    /// Backward pass — Phase 2.5 mock. The underlying actor's
    /// `Step` message handles forward+backward atomically; this
    /// method runs another `Step` against the gradient as input
    /// so callers see a working method, but the semantics are
    /// shard-echo, not real autograd.
    #[pyo3(signature = (grad, timeout_secs=10.0))]
    fn backward(&self, py: Python<'_>, grad: Vec<f32>, timeout_secs: f64) -> PyResult<f32> {
        // Phase 2.5: typed payloads — backward semantics are mocked
        // since the underlying `TensorParallelMsg` doesn't yet split
        // forward and backward into separate variants. Return the
        // grad_norm reported by the shards.
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(TensorParallelMsg::Step {
                    input: grad,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok((_out, stats)))) => Ok(stats.grad_norm),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("tensor-parallel trainer dropped reply")),
                    Err(_) => Err(errors::map_str("backward timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "TensorParallelTrainer(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsyncParameterServer>()?;
    m.add_class::<PyDataParallelTrainer>()?;
    m.add_class::<PyPipelineParallelTrainer>()?;
    m.add_class::<PyTensorParallelTrainer>()?;
    Ok(())
}
