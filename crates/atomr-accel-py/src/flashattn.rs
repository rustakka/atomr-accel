//! `FlashAttn` — Python handle wrapping `ActorRef<FlashAttnMsg>`.
//!
//! Obtained via `Device.flashattn()` (only when the `flashattn` feature
//! is compiled in *and* the device's `KernelChildren` extras slot has
//! the actor ref registered). Phase 4 ships a single representative
//! method, `forward_f16`, that constructs an [`Fa2FwdRequest::<F16>`]
//! and dispatches it through the actor; the constructor validates the
//! dispatch cell up front so callers get a typed error if the
//! `(arch, head_dim, mask, …)` tuple isn't in the FA dispatch table.
//!
//! Mock-mode parity: the upstream actor's mock handler intentionally
//! drops the message rather than replying, so the receiver fires
//! `Err(_)` which we surface as `Unrecoverable("flashattn dropped reply")`.
//!
//! Backward (`Fa2BwdRequest`), FA3 forward (`Fa3FwdRequest`), varlen,
//! paged KV-cache, chunked prefill, and the bf16 / fp8 axes follow in
//! the Phase 4.5 FlashAttention tracking issue. The constructor
//! signature on each request type is sufficiently uniform (arch,
//! head_dim, gqa_ratio, mask, bias, sink_tokens, softmax_scale) that
//! the follow-up wrappers will reuse the same string-arg dispatch
//! pattern used here.

#![cfg(feature = "flashattn")]

use std::time::Duration;

use pyo3::prelude::*;

use atomr_accel_flashattn::{Fa2FwdRequest, FlashAttnMsg, MaskKind, PositionBias, SmArch, F16};
use atomr_core::actor::ActorRef;

use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "FlashAttn", module = "atomr_accel._native")]
pub struct PyFlashAttn {
    actor_ref: ActorRef<FlashAttnMsg>,
}

impl PyFlashAttn {
    pub fn new(actor_ref: ActorRef<FlashAttnMsg>) -> Self {
        Self { actor_ref }
    }
}

fn arch_from_str(s: &str) -> PyResult<SmArch> {
    match s.to_ascii_lowercase().as_str() {
        "sm_80" | "sm80" => Ok(SmArch::Sm80),
        "sm_89" | "sm89" => Ok(SmArch::Sm89),
        "sm_90a" | "sm90a" => Ok(SmArch::Sm90a),
        "sm_100" | "sm100" => Ok(SmArch::Sm100),
        _ => Err(errors::map_str(format!(
            "arch must be one of sm_80/sm_89/sm_90a/sm_100 (got {s:?})"
        ))),
    }
}

fn mask_from_str(s: &str, window: Option<u32>) -> PyResult<MaskKind> {
    match s.to_ascii_lowercase().as_str() {
        "full" => Ok(MaskKind::Full),
        "causal" => Ok(MaskKind::Causal),
        "sliding_causal" => Ok(MaskKind::SlidingCausal {
            window: window.ok_or_else(|| {
                errors::map_str("sliding_causal mask requires the `window` argument")
            })?,
        }),
        "sliding_full" => Ok(MaskKind::SlidingFull {
            window: window.ok_or_else(|| {
                errors::map_str("sliding_full mask requires the `window` argument")
            })?,
        }),
        _ => Err(errors::map_str(format!(
            "mask must be 'full', 'causal', 'sliding_causal', or 'sliding_full' (got {s:?})"
        ))),
    }
}

#[pymethods]
impl PyFlashAttn {
    /// FA2 forward pass with `f16` Q/K/V. Validates the dispatch cell
    /// at request-construction time (errors surface as `GpuRuntimeError`
    /// with the underlying [`atomr_accel_flashattn::FlashAttnError`]
    /// `Display` text). On success the actor processes the request on
    /// its background stream; mock-mode replies drop the message and
    /// the helper surfaces `Unrecoverable("flashattn dropped reply")`.
    ///
    /// `mask` accepts `"full"`, `"causal"`, `"sliding_causal"`, or
    /// `"sliding_full"`; the latter two require a non-zero `window`.
    /// `softmax_scale` defaults to `1.0 / sqrt(head_dim)` when not
    /// supplied.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        arch="sm_80",
        head_dim=64,
        gqa_ratio=1,
        mask="causal",
        window=None,
        alibi=false,
        sink_tokens=0,
        softmax_scale=None,
        timeout_secs=10.0,
    ))]
    fn forward_f16(
        &self,
        py: Python<'_>,
        arch: &str,
        head_dim: u32,
        gqa_ratio: u32,
        mask: &str,
        window: Option<u32>,
        alibi: bool,
        sink_tokens: u32,
        softmax_scale: Option<f32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let arch = arch_from_str(arch)?;
        let mask = mask_from_str(mask, window)?;
        let bias = if alibi {
            PositionBias::Alibi
        } else {
            PositionBias::None
        };
        let scale = softmax_scale.unwrap_or_else(|| 1.0_f32 / (head_dim as f32).sqrt());
        let (req, rx) =
            Fa2FwdRequest::<F16>::new(arch, head_dim, gqa_ratio, mask, bias, sink_tokens, scale)
                .map_err(|e| errors::map_str(e.to_string()))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                actor.tell(FlashAttnMsg::forward(req));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_str(e.to_string())),
                    Ok(Err(_)) => Err(errors::map_str(
                        "flashattn dropped reply (mock mode or actor crashed)",
                    )),
                    Err(_) => Err(errors::map_str("forward_f16 timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "FlashAttn(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlashAttn>()?;
    Ok(())
}
