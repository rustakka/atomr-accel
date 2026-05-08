//! `Fft` — Python handle wrapping `ActorRef<FftMsg>`.
//!
//! Obtained via `Device.fft()` (only when the `cufft` feature is
//! compiled in *and* the device's `EnabledLibraries::CUFFT` flag is
//! set). Phase 1 ships the handle as a structural anchor; typed FFT
//! plans (R2C/C2R/C2C across f32/f64/f16/bf16, 1-D / 2-D / 3-D, plan-
//! many, callback support) follow in the Phase 1.5 cuFFT tracking
//! issue — they require numpy↔complex marshalling that's out of scope
//! for this PR.

#![cfg(feature = "cufft")]

use pyo3::prelude::*;

use atomr_accel_cuda::kernel::FftMsg;
use atomr_core::actor::ActorRef;

#[pyclass(name = "Fft", module = "atomr_accel._native")]
pub struct PyFft {
    #[allow(dead_code)]
    actor_ref: ActorRef<FftMsg>,
}

impl PyFft {
    pub fn new(actor_ref: ActorRef<FftMsg>) -> Self {
        Self { actor_ref }
    }
}

#[pymethods]
impl PyFft {
    fn __repr__(&self) -> &'static str {
        "Fft(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFft>()?;
    Ok(())
}
