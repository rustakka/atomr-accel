//! `Cub` — Python handle wrapping `ActorRef<CubMsg>`.
//!
//! Obtained via `Device.cub()` (only when the `cub` feature is compiled
//! in *and* the device's `KernelChildren` extras slot has the
//! `CubChildRef` registered). Phase 4 ships a single representative
//! method, `reduce_sum_f32`, that exercises the typed `ReduceRequest::<f32>`
//! dispatch end-to-end (mock-mode replies as `Unrecoverable`, and the
//! real path currently surfaces "kernel compile lands in Phase 5.1"
//! per the upstream actor).
//!
//! The remaining CUB family — scan, sort, histogram, select / partition,
//! segmented reduce — and the f64 / i32 / u32 / i64 / u64 / i8 / u8
//! axes follow in the Phase 4.5 CUB-coverage tracking issue.
//!
//! Mock-mode parity: the wrapping `ActorRef` is constructed by the
//! merge-time wiring agent; calls on a mock CubActor reply with
//! `GpuError::Unrecoverable("CubActor in mock mode (no GPU available)")`.

#![cfg(feature = "cub")]

use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cub::{CubMsg, ReduceRequest, ReductionOp};
use atomr_core::actor::ActorRef;

use crate::buffer::PyGpuBufferF32;
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Cub", module = "atomr_accel._native")]
pub struct PyCub {
    actor_ref: ActorRef<CubMsg>,
}

impl PyCub {
    pub fn new(actor_ref: ActorRef<CubMsg>) -> Self {
        Self { actor_ref }
    }
}

#[pymethods]
impl PyCub {
    /// Sum-reduction over an `f32` device buffer. `output` must point
    /// to a 1-element `GpuBufferF32`; on success the device-side scalar
    /// `Σ input[i]` lives at `output[0]`.
    ///
    /// Phase 4 wires the typed `ReduceRequest::<f32>` dispatch path;
    /// the actor's NVRTC compile-and-launch implementation lands in
    /// Phase 5.1 (until then, the actor replies with a structured
    /// `Unrecoverable` so callers can observe the path is plumbed).
    #[pyo3(signature = (input, output, timeout_secs=10.0))]
    fn reduce_sum_f32(
        &self,
        py: Python<'_>,
        input: Py<PyGpuBufferF32>,
        output: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let input = input
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("input consumed"))?;
        let output = output
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("output consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ReduceRequest::<f32>::new(ReductionOp::Sum, input, output, tx);
                actor.tell(CubMsg::Reduce(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cub dropped reply")),
                    Err(_) => Err(errors::map_str("reduce_sum_f32 timed out")),
                }
            })
        })
    }

    /// Async counterpart of `reduce_sum_f32`. Returns a Python
    /// awaitable that resolves to `None` on success or raises a
    /// `GpuRuntimeError` subclass on failure.
    #[pyo3(signature = (input, output, timeout_secs=10.0))]
    fn reduce_sum_f32_async<'py>(
        &self,
        py: Python<'py>,
        input: Py<PyGpuBufferF32>,
        output: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = input
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("input consumed"))?;
        let output = output
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("output consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            let req = ReduceRequest::<f32>::new(ReductionOp::Sum, input, output, tx);
            actor.tell(CubMsg::Reduce(Box::new(req)));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("cub dropped reply")),
                Err(_) => Err(errors::map_str("reduce_sum_f32 timed out")),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "Cub(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCub>()?;
    Ok(())
}
