//! `RngGenerator` — Python handle wrapping `ActorRef<RngMsg>`.
//!
//! Obtained via `Device.rng()` (only when the `curand` feature is
//! compiled in *and* the device's `EnabledLibraries::CURAND` flag is
//! set). Phase 1 ships:
//!
//! - `set_seed(seed)`
//! - `uniform_f32(buf, lo=0.0, hi=1.0)`
//! - `normal_f32(buf, mean=0.0, std=1.0)`
//!
//! Quasi-random generators, log-normal / Poisson / Exponential /
//! Beta / Cauchy / Gamma / Discrete distributions, and per-dtype
//! variants (`uniform_f64`, `uniform_u32`, …) follow in the Phase 1.5
//! cuRAND tracking issue.

#![cfg(feature = "curand")]

use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{Distribution, FillRequest, RngMsg};
use atomr_core::actor::ActorRef;

use crate::buffer::PyGpuBufferF32;
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "RngGenerator", module = "atomr_accel._native")]
pub struct PyRngGenerator {
    actor_ref: ActorRef<RngMsg>,
}

impl PyRngGenerator {
    pub fn new(actor_ref: ActorRef<RngMsg>) -> Self {
        Self { actor_ref }
    }
}

#[pymethods]
impl PyRngGenerator {
    /// Re-seed the active generator. No-op for quasi generators.
    #[pyo3(signature = (seed, timeout_secs=10.0))]
    fn set_seed(&self, py: Python<'_>, seed: u64, timeout_secs: f64) -> PyResult<()> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RngMsg::SetSeed { seed, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("set_seed timed out")),
                }
            })
        })
    }

    /// Fill an f32 buffer with samples from `Uniform(lo, hi)`. Note:
    /// cuRAND's native uniform is `(0, 1]`; callers needing other
    /// bounds get an affine transform applied internally.
    #[pyo3(signature = (buf, lo=0.0, hi=1.0, timeout_secs=10.0))]
    fn uniform_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        lo: f32,
        hi: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let g = buf
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                    buf: g,
                    dist: Distribution::Uniform { lo, hi },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("uniform_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f32 buffer with samples from `Normal(mean, std)`.
    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn normal_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        mean: f32,
        std: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let g = buf
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                    buf: g,
                    dist: Distribution::Normal { mean, std },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("normal_f32 timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "RngGenerator(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRngGenerator>()?;
    Ok(())
}
