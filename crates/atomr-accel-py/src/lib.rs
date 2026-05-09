//! Python bindings for `atomr-accel-cuda`. Native extension module
//! `atomr_accel._native`; the user-facing API lives in
//! `python/atomr_accel/__init__.py` (with per-domain submodules under
//! `python/atomr_accel/{system,device,blas,cudnn,fft,rng,solver,
//! collective,nvrtc,patterns,train,agents,realtime,telemetry,cub,
//! cutlass,flashattn,tensorrt,errors}.py`).

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

mod agents;
mod blas;
mod buffer;
mod device;
mod errors;
mod graph;
mod memory;
mod patterns;
mod realtime;
mod runtime;
mod system;
mod train;

#[cfg(feature = "nccl")]
mod collective;
#[cfg(feature = "cudnn")]
mod cudnn;
#[cfg(feature = "cufft")]
mod fft;
#[cfg(feature = "nvrtc")]
mod nvrtc;
#[cfg(feature = "curand")]
mod rng;
#[cfg(feature = "cusolver")]
mod solver;

#[cfg(feature = "telemetry")]
mod telemetry;

#[cfg(feature = "cub")]
mod cub;
#[cfg(feature = "cutlass")]
mod cutlass;
#[cfg(feature = "flashattn")]
mod flashattn;
#[cfg(feature = "tensorrt")]
mod tensorrt;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Native bindings for atomr-accel. Import `atomr_accel` (the pure-Python facade) instead.",
    )?;

    errors::register(py, m)?;
    system::register(py, m)?;
    device::register(py, m)?;
    buffer::register(py, m)?;
    blas::register(py, m)?;
    graph::register(py, m)?;
    memory::register(py, m)?;
    patterns::register(py, m)?;
    train::register(py, m)?;
    agents::register(py, m)?;
    realtime::register(py, m)?;
    #[cfg(feature = "cudnn")]
    cudnn::register(py, m)?;
    #[cfg(feature = "cufft")]
    fft::register(py, m)?;
    #[cfg(feature = "curand")]
    rng::register(py, m)?;
    #[cfg(feature = "cusolver")]
    solver::register(py, m)?;
    #[cfg(feature = "nccl")]
    collective::register(py, m)?;
    #[cfg(feature = "nvrtc")]
    nvrtc::register(py, m)?;

    #[cfg(feature = "telemetry")]
    telemetry::register(py, m)?;

    #[cfg(feature = "cub")]
    cub::register(py, m)?;
    #[cfg(feature = "cutlass")]
    cutlass::register(py, m)?;
    #[cfg(feature = "flashattn")]
    flashattn::register(py, m)?;
    #[cfg(feature = "tensorrt")]
    tensorrt::register(py, m)?;

    Ok(())
}
