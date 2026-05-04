//! `RngGenerator` Python wrapper.
//!
//! This module is gated on the `curand` feature. The `RngGenerator`
//! is constructed by `Device.rng()` once we have a way to expose the
//! pre-spawned `RngActor` ref — that requires `KernelChildren`
//! plumbing through Python, deferred to the next iteration. For now
//! we ship a typed Python class so downstream code can already
//! reference `atomr_accel.RngGenerator` via the facade.

use pyo3::prelude::*;
use atomr_accel_cuda::kernel::RngMsg;
use atomr_core::actor::ActorRef;

#[pyclass(name = "RngGenerator", module = "atomr_accel._native")]
pub struct PyRngGenerator {
    #[allow(dead_code)]
    actor_ref: ActorRef<RngMsg>,
}

impl PyRngGenerator {
    pub fn new(actor_ref: ActorRef<RngMsg>) -> Self {
        Self { actor_ref }
    }
}

#[pymethods]
impl PyRngGenerator {
    fn __repr__(&self) -> &'static str {
        "RngGenerator(...)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRngGenerator>()?;
    Ok(())
}
