//! `Collective` — Python handle wrapping `ActorRef<CollectiveMsg>`.
//!
//! Obtained via `Device.collective()` (only when the `nccl` feature is
//! compiled in *and* the device's `EnabledLibraries::NCCL` flag is
//! set). Phase 1 ships the handle as a structural anchor; all-reduce /
//! broadcast / all-gather / reduce-scatter / all-to-all / send-recv
//! across NcclReduceSupported dtypes follow in the Phase 1.5 NCCL
//! tracking issue (these require comm-group bootstrap which spans
//! multiple devices and is most natural to drive from Rust).

#![cfg(feature = "nccl")]

use pyo3::prelude::*;

use atomr_accel_cuda::kernel::CollectiveMsg;
use atomr_core::actor::ActorRef;

#[pyclass(name = "Collective", module = "atomr_accel._native")]
pub struct PyCollective {
    #[allow(dead_code)]
    actor_ref: ActorRef<CollectiveMsg>,
}

impl PyCollective {
    pub fn new(actor_ref: ActorRef<CollectiveMsg>) -> Self {
        Self { actor_ref }
    }
}

#[pymethods]
impl PyCollective {
    fn __repr__(&self) -> &'static str {
        "Collective(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCollective>()?;
    Ok(())
}
