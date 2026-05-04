//! Python bindings for `atomr-accel-cuda`. Native extension module
//! `atomr_accel._native`; the user-facing API lives in
//! `python/atomr_accel/__init__.py`.

#![allow(
    clippy::useless_conversion,
    clippy::too_many_arguments,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::type_complexity,
    dead_code,
    unexpected_cfgs
)]

use pyo3::prelude::*;

mod buffer;
mod device;
mod errors;
mod runtime;
mod system;

#[cfg(feature = "nvrtc")]
mod nvrtc;
#[cfg(feature = "curand")]
mod rng;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Native bindings for atomr-accel-cuda. Import `atomr_accel` (the pure-Python facade) instead.",
    )?;

    errors::register(py, m)?;
    system::register(py, m)?;
    device::register(py, m)?;
    buffer::register(py, m)?;
    #[cfg(feature = "curand")]
    rng::register(py, m)?;
    #[cfg(feature = "nvrtc")]
    nvrtc::register(py, m)?;

    Ok(())
}
