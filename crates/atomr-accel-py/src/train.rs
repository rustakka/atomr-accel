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

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_train::optimizer::OptimizerKind;
use atomr_accel_train::parameter_server::{
    AsyncParameterServer, ParameterServerMsg, ParameterServerStats, WorkerId,
};
use atomr_core::actor::ActorRef;

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

// ─── Structural-anchor classes (generic actors) ──────────────────

/// `DataParallelTrainer` — N-replica trainer.
///
/// TODO Phase 2.5: bridge `ReplicaProtocol` to a Python-side typed
/// step contract so Python callers can wire replica actors and drive
/// `Step { batch }` messages.
#[pyclass(name = "DataParallelTrainer", module = "atomr_accel._native")]
pub struct PyDataParallelTrainer {}

#[pymethods]
impl PyDataParallelTrainer {
    fn __repr__(&self) -> &'static str {
        "DataParallelTrainer(handle, structural-anchor)"
    }
}

/// `PipelineParallelTrainer` — staged forward/backward.
///
/// TODO Phase 2.5: bridge `PipelineStageProtocol` for Python callers.
#[pyclass(name = "PipelineParallelTrainer", module = "atomr_accel._native")]
pub struct PyPipelineParallelTrainer {}

#[pymethods]
impl PyPipelineParallelTrainer {
    fn __repr__(&self) -> &'static str {
        "PipelineParallelTrainer(handle, structural-anchor)"
    }
}

/// `TensorParallelTrainer` — sharded matmul coordinator.
///
/// TODO Phase 2.5: bridge `ShardProtocol` for Python callers.
#[pyclass(name = "TensorParallelTrainer", module = "atomr_accel._native")]
pub struct PyTensorParallelTrainer {}

#[pymethods]
impl PyTensorParallelTrainer {
    fn __repr__(&self) -> &'static str {
        "TensorParallelTrainer(handle, structural-anchor)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsyncParameterServer>()?;
    m.add_class::<PyDataParallelTrainer>()?;
    m.add_class::<PyPipelineParallelTrainer>()?;
    m.add_class::<PyTensorParallelTrainer>()?;
    Ok(())
}
