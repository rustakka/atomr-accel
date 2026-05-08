//! `Solver` — Python handle wrapping `ActorRef<SolverMsg>`.
//!
//! Obtained via `Device.solver()` (only when the `cusolver` feature is
//! compiled in *and* the device's `EnabledLibraries::CUSOLVER` flag
//! is set). Phase 1 ships the handle as a structural anchor; LU /
//! QR / Cholesky / SVD / eigendecomposition follow in the Phase 1.5
//! cuSOLVER tracking issue.

#![cfg(feature = "cusolver")]

use pyo3::prelude::*;

use atomr_accel_cuda::kernel::SolverMsg;
use atomr_core::actor::ActorRef;

#[pyclass(name = "Solver", module = "atomr_accel._native")]
pub struct PySolver {
    #[allow(dead_code)]
    actor_ref: ActorRef<SolverMsg>,
}

impl PySolver {
    pub fn new(actor_ref: ActorRef<SolverMsg>) -> Self {
        Self { actor_ref }
    }
}

#[pymethods]
impl PySolver {
    fn __repr__(&self) -> &'static str {
        "Solver(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolver>()?;
    Ok(())
}
