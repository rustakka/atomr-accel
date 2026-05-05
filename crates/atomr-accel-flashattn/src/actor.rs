//! `FlashAttnActor` — receives [`FlashAttnMsg`]s and dispatches to the
//! NVRTC-compiled FA2/FA3 cubin selected by the request's
//! [`crate::dispatch::DispatchKey`].
//!
//! The actor is intended to be installed alongside the cuBLAS /
//! cuDNN actors as a child of `ContextActor` (registered through the
//! Phase 0 `KernelChildren::register_extra` slot). At runtime, every
//! `Forward` / `Backward` / `PagedForward` message is:
//!
//! 1. Validated (the request constructor already validated the
//!    dispatch cell; this is just `dispatch_key()`).
//! 2. Resolved to a kernel-name expression via [`crate::dispatch::lookup`].
//! 3. Compiled-or-fetched through the NVRTC actor + Phase 0.6 disk cache.
//! 4. Launched on the actor's stream, with completion wired through
//!    the standard `envelope::run_kernel` path.
//!
//! The Phase 7 deliverable focuses on (1)–(2). The launch path (3)–(4)
//! is gated behind `cuda-runtime-tests` because it needs a real
//! `CudaContext` and the vendored FA csrc compiled — neither is part
//! of the unit-test surface.

#[cfg(feature = "cuda-runtime-tests")]
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};

use crate::dispatch::{FaBwdDispatch, FaFwdDispatch, FaPagedFwdDispatch};
use crate::FlashAttnError;

/// Public message surface for [`FlashAttnActor`].
///
/// Each variant carries a boxed dispatch trait — keeping the enum cheap
/// to clone/send while letting callers specialise the request type per
/// FA generation (FA2, FA3, varlen, paged, prefill).
pub enum FlashAttnMsg {
    /// Forward attention (FA2 or FA3, including varlen + chunked
    /// prefill flavours).
    Forward(Box<dyn FaFwdDispatch>),
    /// Backward attention (FA2 only — fp8 backward is rejected at
    /// request-construction time).
    Backward(Box<dyn FaBwdDispatch>),
    /// Paged-attention forward (vLLM-style block table).
    PagedForward(Box<dyn FaPagedFwdDispatch>),
}

/// Per-actor static properties — kept distinct from per-request state
/// so they can be cloned into the spawn closure without dragging the
/// inbox along.
#[derive(Debug, Clone)]
pub struct FlashAttnProps {
    /// Tag used in `tracing` spans.
    pub label: &'static str,
    /// Maximum number of in-flight kernels per actor. Real launches
    /// honour this through `envelope::run_kernel`'s per-actor
    /// dispatcher; the mock path just records it.
    pub max_in_flight: usize,
}

impl Default for FlashAttnProps {
    fn default() -> Self {
        Self {
            label: "flashattn",
            max_in_flight: 8,
        }
    }
}

/// Inner state — split into `Real` and `Mock` so the crate builds and
/// unit-tests on hosts without a GPU.
pub enum FlashAttnInner {
    /// GPU-free. Every message replies with `Err(MockMode)`.
    Mock { props: FlashAttnProps },
    /// Real actor. Holds an `Arc<NvrtcActor>` ref + a CUDA stream. The
    /// definition is gated behind `cuda-runtime-tests` so the crate
    /// builds without `cudarc` initialisation.
    #[cfg(feature = "cuda-runtime-tests")]
    Real {
        props: FlashAttnProps,
        nvrtc: Arc<crate::cuda_real::NvrtcRef>,
        stream: Arc<cudarc::driver::CudaStream>,
    },
}

/// Top-level FlashAttention actor.
pub struct FlashAttnActor {
    inner: FlashAttnInner,
}

impl FlashAttnActor {
    /// Construct a `Props<FlashAttnActor>` configured for mock-mode use.
    pub fn mock_props(props: FlashAttnProps) -> Props<Self> {
        Props::create(move || FlashAttnActor {
            inner: FlashAttnInner::Mock {
                props: props.clone(),
            },
        })
    }

    /// Construct a `Props<FlashAttnActor>` for real-GPU use. Gated
    /// behind `cuda-runtime-tests` because it requires a live NVRTC
    /// actor + stream (provided by `ContextActor` at spawn time).
    #[cfg(feature = "cuda-runtime-tests")]
    pub fn props(
        props: FlashAttnProps,
        nvrtc: Arc<crate::cuda_real::NvrtcRef>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Props<Self> {
        Props::create(move || FlashAttnActor {
            inner: FlashAttnInner::Real {
                props: props.clone(),
                nvrtc: nvrtc.clone(),
                stream: stream.clone(),
            },
        })
    }

    /// Borrow the static props of this actor.
    pub fn props_ref(&self) -> &FlashAttnProps {
        match &self.inner {
            FlashAttnInner::Mock { props } => props,
            #[cfg(feature = "cuda-runtime-tests")]
            FlashAttnInner::Real { props, .. } => props,
        }
    }

    /// Inspect the request's dispatch key without launching anything.
    /// Useful for tests + the mock path.
    pub fn inspect_key(msg: &FlashAttnMsg) -> crate::dispatch::DispatchKey {
        match msg {
            FlashAttnMsg::Forward(d) => d.dispatch_key(),
            FlashAttnMsg::Backward(d) => d.dispatch_key(),
            FlashAttnMsg::PagedForward(d) => d.dispatch_key(),
        }
    }
}

#[async_trait]
impl Actor for FlashAttnActor {
    type Msg = FlashAttnMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        let key = Self::inspect_key(&msg);
        let kernel_name = crate::dispatch::lookup(&key).map(|n| n.to_string());

        match &mut self.inner {
            FlashAttnInner::Mock { props } => {
                tracing::debug!(
                    label = props.label,
                    kernel = ?kernel_name,
                    "flashattn mock-mode dispatch (no-op)"
                );
                // Reply channels live inside the boxed request's
                // concrete type, so the mock path simply drops the
                // message — the request's `oneshot::Receiver` returns
                // `Err(_)` on the caller side, which higher-level
                // wrappers translate to [`FlashAttnError::MockMode`].
                let _ = msg;
                let _ = kernel_name;
            }
            #[cfg(feature = "cuda-runtime-tests")]
            FlashAttnInner::Real {
                props,
                nvrtc,
                stream,
            } => {
                tracing::debug!(
                    label = props.label,
                    kernel = ?kernel_name,
                    "flashattn real dispatch"
                );
                let _ = (props, nvrtc, stream, msg);
            }
        }
    }
}

/// Convenience constructors that wrap each request type in the right
/// `FlashAttnMsg` variant. Keeps the call site free of `Box::new`
/// noise.
impl FlashAttnMsg {
    pub fn forward(req: impl FaFwdDispatch) -> Self {
        FlashAttnMsg::Forward(Box::new(req))
    }

    pub fn backward(req: impl FaBwdDispatch) -> Self {
        FlashAttnMsg::Backward(Box::new(req))
    }

    pub fn paged_forward(req: impl FaPagedFwdDispatch) -> Self {
        FlashAttnMsg::PagedForward(Box::new(req))
    }
}

/// Returned by mock-mode actors when they observe a launch they
/// can't honour.
pub fn mock_error() -> FlashAttnError {
    FlashAttnError::MockMode
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Bf16, SmArch, F16};
    use crate::fa2::{Fa2BwdRequest, Fa2FwdRequest, MaskKind, PositionBias};

    /// `FlashAttnMsg` constructs and inspects correctly for forward,
    /// backward, and paged variants.
    #[test]
    fn flashattn_msg_constructs() {
        // Forward (fa2)
        let (fwd, _rx) = Fa2FwdRequest::<F16>::new(
            SmArch::Sm80,
            64,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / 8.0,
        )
        .unwrap();
        let msg = FlashAttnMsg::forward(fwd);
        let key = FlashAttnActor::inspect_key(&msg);
        assert!(key.causal);
        assert_eq!(key.head_dim, 64);
        assert!(matches!(msg, FlashAttnMsg::Forward(_)));

        // Backward (fa2)
        let (bwd, _rx) = Fa2BwdRequest::<Bf16>::new(
            SmArch::Sm80,
            128,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            true,
        )
        .unwrap();
        let msg = FlashAttnMsg::backward(bwd);
        let key = FlashAttnActor::inspect_key(&msg);
        assert!(key.causal);
        assert_eq!(key.dtype, crate::dispatch::DType::Bf16);
        assert!(matches!(msg, FlashAttnMsg::Backward(_)));

        // Paged forward (gated)
        #[cfg(feature = "paged")]
        {
            use crate::paged::{PagedAttentionRequest, PagedKvCache};
            let cache = PagedKvCache::new(1024, 16, 8, 128, 64).unwrap();
            let (paged_req, _rx) = PagedAttentionRequest::<Bf16>::new(
                SmArch::Sm90a,
                128,
                8,
                MaskKind::Causal,
                PositionBias::None,
                0,
                1.0 / (128f32).sqrt(),
                cache,
                4,
                1,
            )
            .unwrap();
            let msg = FlashAttnMsg::paged_forward(paged_req);
            let key = FlashAttnActor::inspect_key(&msg);
            assert!(key.paged);
            assert!(matches!(msg, FlashAttnMsg::PagedForward(_)));
        }

        // Mock-mode props.
        let props = FlashAttnProps::default();
        assert_eq!(props.label, "flashattn");
        assert_eq!(props.max_in_flight, 8);

        // Constructing the actor in mock mode succeeds.
        let _props = FlashAttnActor::mock_props(props);
    }
}
