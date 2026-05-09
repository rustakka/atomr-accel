//! `Cutlass` — Python handle wrapping `CutlassActor`.
//!
//! `CutlassActor` is a host-only synchronous actor (it does not flow
//! through `atomr_core::actor::ActorRef<CutlassMsg>` because the upstream
//! crate intentionally avoids depending on `atomr-accel-cuda` to break
//! a cargo cycle — see the comment in `atomr-accel-cutlass/Cargo.toml`).
//! Each call to `actor.handle()` is `&self` (interior mutable plan
//! cache), so we wrap an `Arc<CutlassActor>` and dispatch directly with
//! the GIL released.
//!
//! Phase 4 ships `gemm_f32_plan`, which constructs a `GemmRequest::<f32>`,
//! hands it to the actor, and returns the running dispatched-message
//! count. This exercises the typed Gemm plan-cache path end-to-end (the
//! NVRTC compile step is wired through the optional `compile_sink`,
//! which the merge-time wiring agent populates with the cuda crate's
//! `NvrtcActor::Compile` shim).
//!
//! Grouped GEMM, conv (fwd / dgrad / wgrad), refit, EVT, and the
//! fp16 / bf16 / fp8 / fp4 dtype axes follow in the Phase 4.5 CUTLASS
//! tracking issue. The `gemm_f64_plan` variant is added as a second
//! representative so callers can observe the dtype generic crossing
//! the PyO3 boundary.

#![cfg(feature = "cutlass")]

use std::sync::Arc;

use pyo3::prelude::*;

use atomr_accel_cutlass::{CutlassActor, CutlassMsg, GemmRequest, GemmShape, SmArch};

use crate::errors;

#[pyclass(name = "Cutlass", module = "atomr_accel._native")]
pub struct PyCutlass {
    actor: Arc<CutlassActor>,
}

impl PyCutlass {
    pub fn new(actor: Arc<CutlassActor>) -> Self {
        Self { actor }
    }
}

fn arch_from_str(s: &str) -> PyResult<SmArch> {
    match s.to_ascii_lowercase().as_str() {
        "sm_80" | "sm80" => Ok(SmArch::Sm80),
        "sm_86" | "sm86" => Ok(SmArch::Sm86),
        "sm_89" | "sm89" => Ok(SmArch::Sm89),
        "sm_90" | "sm90" => Ok(SmArch::Sm90),
        "sm_90a" | "sm90a" => Ok(SmArch::Sm90a),
        "sm_100" | "sm100" => Ok(SmArch::Sm100),
        "sm_120" | "sm120" => Ok(SmArch::Sm120),
        _ => Err(errors::map_str(format!(
            "arch must be one of sm_80/sm_86/sm_89/sm_90/sm_90a/sm_100/sm_120 (got {s:?})"
        ))),
    }
}

#[pymethods]
impl PyCutlass {
    /// Construct an `f32` `GemmRequest` for `(m × k) · (k × n) → (m × n)`
    /// targeting `arch` (default `sm_80`), dispatch it through the
    /// actor, and return the running dispatched-message counter. The
    /// actor inserts the rendered `.cu` source into its plan cache; if
    /// a `compile_sink` is wired in, the source is forwarded to the
    /// downstream NVRTC actor for compilation.
    #[pyo3(signature = (m, n, k, arch="sm_80"))]
    fn gemm_f32_plan(&self, py: Python<'_>, m: u32, n: u32, k: u32, arch: &str) -> PyResult<u64> {
        let arch = arch_from_str(arch)?;
        let actor = self.actor.clone();
        py.allow_threads(|| {
            let req = GemmRequest::<f32>::new(GemmShape::new(m, n, k), arch);
            actor.handle(CutlassMsg::Gemm(Box::new(req)));
            Ok(actor.inner().dispatched())
        })
    }

    /// f64 variant of `gemm_f32_plan`. Same semantics; exercises the
    /// dtype generic.
    #[pyo3(signature = (m, n, k, arch="sm_80"))]
    fn gemm_f64_plan(&self, py: Python<'_>, m: u32, n: u32, k: u32, arch: &str) -> PyResult<u64> {
        let arch = arch_from_str(arch)?;
        let actor = self.actor.clone();
        py.allow_threads(|| {
            let req = GemmRequest::<f64>::new(GemmShape::new(m, n, k), arch);
            actor.handle(CutlassMsg::Gemm(Box::new(req)));
            Ok(actor.inner().dispatched())
        })
    }

    /// Number of messages this actor has processed since construction.
    #[getter]
    fn dispatched(&self) -> u64 {
        self.actor.inner().dispatched()
    }

    /// Number of distinct plans cached (one per `(template_id, shape,
    /// dtype, arch)` tuple).
    #[getter]
    fn plan_cache_len(&self) -> usize {
        self.actor.inner().plan_cache.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Cutlass(dispatched={}, plans={})",
            self.actor.inner().dispatched(),
            self.actor.inner().plan_cache.len(),
        )
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCutlass>()?;
    Ok(())
}
