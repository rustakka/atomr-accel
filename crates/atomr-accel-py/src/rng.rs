//! `RngGenerator` — Python handle wrapping `ActorRef<RngMsg>`.
//!
//! Obtained via `Device.rng()` (only when the `curand` feature is
//! compiled in *and* the device's `EnabledLibraries::CURAND` flag is
//! set). Phase 1.5 ships the full distribution × dtype matrix that
//! `FillRequest<T>` exposes:
//!
//! - `set_seed(seed)`
//! - `set_generator(kind)` — switch generator family at runtime
//!   (XorWow, Philox4_32_10, Mrg32K3A, Mtgp32, plus all four Sobol
//!   variants). Pseudo↔quasi switches are supported.
//! - `uniform_f32 / uniform_f64`
//! - `normal_f32 / normal_f64`
//! - `log_normal_f32 / log_normal_f64`
//! - `poisson_f32 / poisson_f64`
//! - `exponential_f32 / exponential_f64`
//! - `beta_f32 / beta_f64`
//! - `cauchy_f32 / cauchy_f64`
//! - `gamma_f32 / gamma_f64`
//! - `discrete_f32 / discrete_f64`
//! - `uniform_u32` — raw bit fill via the legacy
//!   `RngMsg::FillUniformU32` path. `FillRequest<u32>` is *not*
//!   `RngDispatch` (`u32: RngFloatSupported` is intentionally not
//!   provided), so this is the only u32 surface.
//!
//! Distribution variants whose kernel is not yet wired (Beta, Cauchy,
//! Gamma, Exponential, Discrete, Poisson into a float buffer, Uniform
//! with non-`(0,1]` bounds) surface a tagged `LibraryError` from the
//! actor — Python callers see the same `atomr_accel.LibraryError`
//! they would for any other unwired path. Mock-mode `RngActor` drops
//! the boxed `Fill` request without replying (surfaces as
//! `"rng dropped reply"`); legacy and control-plane variants reply
//! with `Unrecoverable`.
//!
//! Method bodies are repetitive on purpose: a `macro_rules!` shim
//! cannot generate `#[pyo3(signature = ...)]` arms because the
//! `#[pymethods]` proc-macro parses the impl block before any
//! declarative macro inside it expands. Each method is therefore
//! spelled out directly — the only differences are the buffer dtype,
//! the `FillRequest<T>` parameter, the `Distribution` variant, and
//! the timeout error label.

#![cfg(feature = "curand")]

use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{Distribution, FillRequest, RngGeneratorKind, RngMsg};
use atomr_core::actor::ActorRef;

use crate::buffer::{PyGpuBufferF32, PyGpuBufferF64, PyGpuBufferU32};
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

/// Map a user-facing string (case-insensitive) to a [`RngGeneratorKind`].
///
/// Accepted aliases mirror the cuRAND `curandRngType_t` names:
///
/// | input                                  | enum                |
/// |---|---|
/// | `"default"` / `"pseudo_default"`       | `PseudoDefault`     |
/// | `"philox"` / `"philox4_32_10"`         | `Philox4_32_10`     |
/// | `"xorwow"`                             | `XorWow`            |
/// | `"mrg32k3a"`                           | `Mrg32K3A`          |
/// | `"mtgp32"`                             | `Mtgp32`            |
/// | `"sobol32"`                            | `Sobol32`           |
/// | `"scrambled_sobol32"`                  | `ScrambledSobol32`  |
/// | `"sobol64"`                            | `Sobol64`           |
/// | `"scrambled_sobol64"`                  | `ScrambledSobol64`  |
fn rng_kind_from_str(s: &str) -> PyResult<RngGeneratorKind> {
    match s.to_ascii_lowercase().as_str() {
        "default" | "pseudo_default" => Ok(RngGeneratorKind::PseudoDefault),
        "philox" | "philox4_32_10" | "philox_4_32_10" => Ok(RngGeneratorKind::Philox4_32_10),
        "xorwow" => Ok(RngGeneratorKind::XorWow),
        "mrg32k3a" => Ok(RngGeneratorKind::Mrg32K3A),
        "mtgp32" => Ok(RngGeneratorKind::Mtgp32),
        "sobol32" => Ok(RngGeneratorKind::Sobol32),
        "scrambled_sobol32" | "scrambledsobol32" => Ok(RngGeneratorKind::ScrambledSobol32),
        "sobol64" => Ok(RngGeneratorKind::Sobol64),
        "scrambled_sobol64" | "scrambledsobol64" => Ok(RngGeneratorKind::ScrambledSobol64),
        other => Err(errors::map_str(format!("unknown RNG kind: {other:?}"))),
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

    /// Tear down the current generator and reconstruct it as `kind`.
    /// `kind` is a case-insensitive string; see [`rng_kind_from_str`]
    /// for the accepted aliases. Pseudo↔quasi switches are supported;
    /// quasi generators ignore subsequent `set_seed` calls.
    #[pyo3(signature = (kind, timeout_secs=10.0))]
    fn set_generator(&self, py: Python<'_>, kind: &str, timeout_secs: f64) -> PyResult<()> {
        let kind_enum = rng_kind_from_str(kind)?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RngMsg::SetGenerator {
                    kind: kind_enum,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("set_generator timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Uniform
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `Uniform(lo, hi)`. Note:
    /// cuRAND's native uniform is `(0, 1]`; callers needing other
    /// bounds get an affine transform applied internally (Phase 1:
    /// returns `LibraryError` for non-trivial bounds).
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

    /// Fill an f64 buffer with samples from `Uniform(lo, hi)`.
    #[pyo3(signature = (buf, lo=0.0, hi=1.0, timeout_secs=10.0))]
    fn uniform_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        lo: f64,
        hi: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Uniform { lo, hi },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("uniform_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Normal
    // ----------------------------------------------------------------

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

    /// Fill an f64 buffer with samples from `Normal(mean, std)`.
    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn normal_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        mean: f64,
        std: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Normal { mean, std },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("normal_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // LogNormal
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `LogNormal(mean, std)`.
    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn log_normal_f32(
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
                    dist: Distribution::LogNormal { mean, std },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("log_normal_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with samples from `LogNormal(mean, std)`.
    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn log_normal_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        mean: f64,
        std: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::LogNormal { mean, std },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("log_normal_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Poisson — `lambda` is always f64 regardless of the output dtype
    // (matches `curandGeneratePoisson`'s ABI). Phase 1 surfaces a
    // tagged `LibraryError` because Poisson into floats requires a
    // host-side widen step that lands in Phase 2.
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `Poisson(lambda)`.
    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn poisson_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        lambda: f64,
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
                    dist: Distribution::Poisson { lambda },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("poisson_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with samples from `Poisson(lambda)`.
    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn poisson_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        lambda: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Poisson { lambda },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("poisson_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Exponential — Phase 1 returns `LibraryError` (needs a custom
    // kernel; the type-level surface is wired so callers can write
    // code today that auto-grows when the kernel lands).
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `Exponential(lambda)`.
    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn exponential_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        lambda: f32,
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
                    dist: Distribution::Exponential { lambda },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("exponential_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with samples from `Exponential(lambda)`.
    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn exponential_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        lambda: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Exponential { lambda },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("exponential_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Beta — Python keyword `beta` is permitted and maps to the
    // `Distribution::Beta::beta` field of the same name.
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `Beta(alpha, beta)`.
    /// (Phase 1: returns `LibraryError` — needs custom kernel.)
    #[pyo3(signature = (buf, alpha, beta, timeout_secs=10.0))]
    fn beta_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        alpha: f32,
        beta: f32,
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
                    dist: Distribution::Beta { alpha, beta },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("beta_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with samples from `Beta(alpha, beta)`.
    #[pyo3(signature = (buf, alpha, beta, timeout_secs=10.0))]
    fn beta_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        alpha: f64,
        beta: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Beta { alpha, beta },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("beta_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Cauchy
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `Cauchy(loc, scale)`.
    /// (Phase 1: returns `LibraryError` — needs custom kernel.)
    #[pyo3(signature = (buf, loc, scale, timeout_secs=10.0))]
    fn cauchy_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        loc: f32,
        scale: f32,
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
                    dist: Distribution::Cauchy { loc, scale },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("cauchy_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with samples from `Cauchy(loc, scale)`.
    #[pyo3(signature = (buf, loc, scale, timeout_secs=10.0))]
    fn cauchy_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        loc: f64,
        scale: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Cauchy { loc, scale },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("cauchy_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Gamma
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with samples from `Gamma(shape, scale)`.
    /// (Phase 1: returns `LibraryError` — needs custom kernel.)
    #[pyo3(signature = (buf, shape, scale, timeout_secs=10.0))]
    fn gamma_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        shape: f32,
        scale: f32,
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
                    dist: Distribution::Gamma { shape, scale },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("gamma_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with samples from `Gamma(shape, scale)`.
    #[pyo3(signature = (buf, shape, scale, timeout_secs=10.0))]
    fn gamma_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        shape: f64,
        scale: f64,
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
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Gamma { shape, scale },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("gamma_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Discrete — categorical sampling. `weights` is *always* an f32
    // GpuBuffer regardless of the output dtype (that's the shape of
    // `Distribution::Discrete { weights: GpuRef<f32> }`).
    // ----------------------------------------------------------------

    /// Fill an f32 buffer with categorical samples drawn against
    /// `weights` (un-normalised f32 probabilities). (Phase 1: returns
    /// `LibraryError` — needs `curandCreatePoissonDistribution` plus a
    /// custom kernel.)
    #[pyo3(signature = (buf, weights, timeout_secs=10.0))]
    fn discrete_f32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF32>,
        weights: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let g = buf
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("buf consumed"))?;
        let w = weights
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("weights consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                    buf: g,
                    dist: Distribution::Discrete { weights: w },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("discrete_f32 timed out")),
                }
            })
        })
    }

    /// Fill an f64 buffer with categorical samples drawn against
    /// `weights` (un-normalised f32 probabilities). The weights stay
    /// f32 even when the output is f64 — that's the shape of
    /// [`Distribution::Discrete`].
    #[pyo3(signature = (buf, weights, timeout_secs=10.0))]
    fn discrete_f64(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferF64>,
        weights: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let g = buf
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("buf consumed"))?;
        let w = weights
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("weights consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                    buf: g,
                    dist: Distribution::Discrete { weights: w },
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("discrete_f64 timed out")),
                }
            })
        })
    }

    // ----------------------------------------------------------------
    // Integer dtypes
    //
    // `FillRequest<u32>` is *not* `RngDispatch` — `u32:
    // RngFloatSupported` is intentionally not provided, so the only
    // u32 surface is the legacy `RngMsg::FillUniformU32` (raw bits via
    // `curandGenerate`). The `#[allow(deprecated)]` is intentional:
    // it's still the only u32 path supported by the actor.
    //
    // u64 raw-bit fill (`curandGenerateLongLong`) is *not* surfaced
    // because the actor doesn't expose a `FillUniformU64` legacy
    // variant — adding one is a follow-up.
    // ----------------------------------------------------------------

    /// Fill a u32 buffer with raw 32-bit uniform bits via
    /// `curandGenerate`. No affine transform — equivalent to
    /// numpy's `np.random.bit_generator.random_raw(size)`.
    #[pyo3(signature = (buf, timeout_secs=10.0))]
    fn uniform_u32(
        &self,
        py: Python<'_>,
        buf: Py<PyGpuBufferU32>,
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
                #[allow(deprecated)]
                actor.tell(RngMsg::FillUniformU32 { dst: g, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                    Err(_) => Err(errors::map_str("uniform_u32 timed out")),
                }
            })
        })
    }

    // ─────────────────────── Async (asyncio) variants ─────────────

    #[pyo3(signature = (seed, timeout_secs=10.0))]
    fn set_seed_async<'py>(
        &self,
        py: Python<'py>,
        seed: u64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::SetSeed { seed, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("set_seed timed out")),
            }
        })
    }

    #[pyo3(signature = (kind, timeout_secs=10.0))]
    fn set_generator_async<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kind_enum = rng_kind_from_str(kind)?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::SetGenerator { kind: kind_enum, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("set_generator timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, lo=0.0, hi=1.0, timeout_secs=10.0))]
    fn uniform_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        lo: f32,
        hi: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Uniform { lo, hi }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("uniform_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, lo=0.0, hi=1.0, timeout_secs=10.0))]
    fn uniform_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        lo: f64,
        hi: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Uniform { lo, hi }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("uniform_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn normal_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        mean: f32,
        std: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Normal { mean, std }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("normal_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn normal_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        mean: f64,
        std: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Normal { mean, std }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("normal_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn log_normal_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        mean: f32,
        std: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::LogNormal { mean, std }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("log_normal_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, mean=0.0, std=1.0, timeout_secs=10.0))]
    fn log_normal_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        mean: f64,
        std: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::LogNormal { mean, std }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("log_normal_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn poisson_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        lambda: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Poisson { lambda }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("poisson_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn poisson_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        lambda: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Poisson { lambda }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("poisson_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn exponential_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        lambda: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Exponential { lambda }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("exponential_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, lambda, timeout_secs=10.0))]
    fn exponential_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        lambda: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Exponential { lambda }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("exponential_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, alpha, beta, timeout_secs=10.0))]
    fn beta_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Beta { alpha, beta }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("beta_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, alpha, beta, timeout_secs=10.0))]
    fn beta_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        alpha: f64,
        beta: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Beta { alpha, beta }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("beta_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, loc, scale, timeout_secs=10.0))]
    fn cauchy_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        loc: f32,
        scale: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Cauchy { loc, scale }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("cauchy_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, loc, scale, timeout_secs=10.0))]
    fn cauchy_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        loc: f64,
        scale: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Cauchy { loc, scale }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("cauchy_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, shape, scale, timeout_secs=10.0))]
    fn gamma_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        shape: f32,
        scale: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Gamma { shape, scale }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("gamma_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, shape, scale, timeout_secs=10.0))]
    fn gamma_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        shape: f64,
        scale: f64,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Gamma { shape, scale }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("gamma_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, weights, timeout_secs=10.0))]
    fn discrete_f32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF32>,
        weights: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let w = weights.borrow(py).clone_ref().ok_or_else(|| errors::map_str("weights consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f32> {
                buf: g, dist: Distribution::Discrete { weights: w }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("discrete_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, weights, timeout_secs=10.0))]
    fn discrete_f64_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferF64>,
        weights: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let w = weights.borrow(py).clone_ref().ok_or_else(|| errors::map_str("weights consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(RngMsg::Fill(Box::new(FillRequest::<f64> {
                buf: g, dist: Distribution::Discrete { weights: w }, reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("discrete_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (buf, timeout_secs=10.0))]
    fn uniform_u32_async<'py>(
        &self,
        py: Python<'py>,
        buf: Py<PyGpuBufferU32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = buf.borrow(py).clone_ref().ok_or_else(|| errors::map_str("buf consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            #[allow(deprecated)]
            actor.tell(RngMsg::FillUniformU32 { dst: g, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("rng dropped reply")),
                Err(_) => Err(errors::map_str("uniform_u32 timed out")),
            }
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
