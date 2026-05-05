//! `BlasActor` — full cuBLAS surface (Phase 1 cuBLAS slice).
//!
//! Wraps a [`cudarc::cublas::CudaBlas`] handle, performs cuBLAS L1/L2/
//! L3 ops on its assigned stream, and returns completion via the
//! configured [`CompletionStrategy`] (§3.2 stateless-handle archetype +
//! §5.10 callback wiring).
//!
//! Sub-modules:
//! - [`gemm`] — typed `Gemm<T>` for f32/f64/f16/bf16 (cudarc safe layer)
//!   and the legacy `SgemmRequest` adapter that routes through
//!   `Gemm<f32>` for back-compat.
//! - [`gemm_strided_batched`] — strided-batched gemm via cudarc's safe
//!   layer for f32/f64/f16/bf16; can drop to
//!   [`crate::sys::cublas::gemm_strided_batched_ex`] if more dtypes are
//!   needed in a follow-up.
//! - [`l1`] — axpy / dot / nrm2 / scal / asum / iamax / iamin / copy /
//!   swap / rot via the cuBLAS ex-suffix entry points.
//! - [`l2`] — gemv / ger via cudarc's `Gemv<T>` and the local
//!   `cublasGemv_v2` / `cublasGer_v2` wrappers.
//! - [`l3`] — geam / syrk / trsm via the local `cublasSgeam` /
//!   `cublasSsyrk_v2` / `cublasStrsm_v2` wrappers (and dgeam/dsyrk/
//!   dtrsm).
//! - [`scaling`] — fp8 scaling-factor helpers (per-tensor / per-row),
//!   stubbed under the `cublas-fp8` feature for use by `cublasGemmEx`
//!   on Hopper+.
//!
//! The mailbox is freed immediately after the kernel is enqueued — the
//! actor never blocks on the GPU (§5.2). Reply delivery happens on the
//! Tokio task spawned by [`crate::kernel::envelope::run_kernel`].

use std::sync::Arc;

use atomr_core::actor::{Context, Props};
use atomr_macros::Actor;
use cudarc::cublas::CudaBlas;

use crate::completion::{CompletionStrategy, HostFnCompletion};
use crate::device::{DeviceState, SgemmRequest};
use crate::error::GpuError;
use crate::kernel::dispatch::{
    BlasDispatchCtx, BlasL1Dispatch, BlasL2Dispatch, BlasL3Dispatch, GemmDispatch,
    GemmStridedBatchedDispatch,
};
use crate::stream::{ActorHints, StreamAllocator};

pub mod gemm;
pub mod gemm_strided_batched;
pub mod l1;
pub mod l2;
pub mod l3;
pub mod scaling;

pub use gemm::GemmRequest;
pub use gemm_strided_batched::GemmStridedBatchedRequest;
pub use l1::{
    AsumRequest, AxpyRequest, CopyRequest, DotRequest, IamaxRequest, IaminRequest, Nrm2Request,
    RotRequest, ScalRequest, SwapRequest,
};
pub use l2::{GemvRequest, GerRequest};
pub use l3::{GeamRequest, SyrkRequest, TrsmRequest};

/// Public messages for `BlasActor`. Each variant boxes a typed
/// dispatcher trait object so the dtype dimension travels through the
/// box without forcing an N-fold mailbox explosion.
pub enum BlasMsg {
    /// Generic typed gemm (canonical form). Construct via
    /// [`BlasMsg::gemm::<T>`] or
    /// [`gemm::GemmRequest::<T>::into_msg`].
    Gemm(Box<dyn GemmDispatch>),
    /// L1 ops boxed in [`BlasL1Dispatch`].
    L1(Box<dyn BlasL1Dispatch>),
    /// L2 ops boxed in [`BlasL2Dispatch`].
    L2(Box<dyn BlasL2Dispatch>),
    /// L3 ops other than gemm (geam / syrk / trsm).
    L3(Box<dyn BlasL3Dispatch>),
    /// Strided-batched gemm.
    GemmStridedBatched(Box<dyn GemmStridedBatchedDispatch>),
    /// Legacy alias kept for back-compat — routes through `Gemm<f32>`
    /// internally.
    #[deprecated(note = "use BlasMsg::gemm::<f32>(GemmRequest::<f32> { ... })")]
    Sgemm(Box<crate::device::SgemmRequest>),
}

impl BlasMsg {
    /// Construct a `BlasMsg::Gemm` from a typed [`GemmRequest<T>`].
    /// Convenience wrapper so callers don't have to box manually.
    pub fn gemm<T: crate::dtype::GemmSupported>(req: GemmRequest<T>) -> Self
    where
        GemmRequest<T>: GemmDispatch,
    {
        Self::Gemm(Box::new(req))
    }

    /// Construct a `BlasMsg::GemmStridedBatched` from a typed request.
    pub fn gemm_strided_batched<T: crate::dtype::GemmSupported>(
        req: GemmStridedBatchedRequest<T>,
    ) -> Self
    where
        GemmStridedBatchedRequest<T>: GemmStridedBatchedDispatch,
    {
        Self::GemmStridedBatched(Box::new(req))
    }
}

/// Two-track construction: a real cuBLAS-backed actor (`props`), and a
/// mock variant used by `examples/echo_no_gpu` and unit tests where no
/// GPU is present.
#[derive(Actor)]
#[msg(BlasMsg)]
pub struct BlasActor {
    inner: BlasInner,
}

pub(crate) enum BlasInner {
    Real {
        cublas: Arc<CudaBlas>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    },
    Mock,
}

impl BlasActor {
    /// Build a [`Props<BlasActor>`] from a stream+allocator+completion
    /// triple. Panics from inside the factory closure with
    /// `"ContextPoisoned: CudaBlas::new failed: …"` so the supervisor
    /// can restart the actor on handle-creation failure.
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        let actor_stream = allocator.acquire(ActorHints::default());
        debug_assert!(Arc::ptr_eq(&actor_stream, &stream));
        Props::create(move || {
            let cublas = match CudaBlas::new(stream.clone()) {
                Ok(b) => b,
                Err(e) => panic!("ContextPoisoned: CudaBlas::new failed: {e}"),
            };
            BlasActor {
                inner: BlasInner::Real {
                    cublas: Arc::new(cublas),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                },
            }
        })
    }

    /// Back-compat shim for callers using the F1 constructor signature.
    /// Wraps the legacy `(stream, PerActorAllocator, HostFnCompletion)`
    /// into the F2 form. New code should call [`BlasActor::props`].
    pub fn props_legacy(
        stream: Arc<cudarc::driver::CudaStream>,
        allocator: crate::stream::PerActorAllocator,
        completion: HostFnCompletion,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        let alloc: Arc<dyn StreamAllocator> = Arc::new(allocator);
        let comp: Arc<dyn CompletionStrategy> = Arc::new(completion);
        Self::props(stream, alloc, comp, state)
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| BlasActor {
            inner: BlasInner::Mock,
        })
    }
}

impl BlasActor {
    async fn handle_msg(&mut self, _ctx: &mut Context<Self>, msg: BlasMsg) {
        // Decompose the message into a known shape, then run the
        // dispatcher with a borrowed `BlasDispatchCtx` so each variant
        // shares the same enqueue/completion plumbing.
        match &self.inner {
            BlasInner::Mock => match msg {
                BlasMsg::Gemm(d) => mock_reply(d.op_name()),
                BlasMsg::L1(d) => mock_reply(d.op_name()),
                BlasMsg::L2(d) => mock_reply(d.op_name()),
                BlasMsg::L3(d) => mock_reply(d.op_name()),
                BlasMsg::GemmStridedBatched(d) => mock_reply(d.op_name()),
                #[allow(deprecated)]
                BlasMsg::Sgemm(req) => {
                    let _ = req.reply.send(Err(GpuError::Unrecoverable(
                        "Sgemm not supported in mock mode".into(),
                    )));
                }
            },
            BlasInner::Real {
                cublas,
                stream,
                completion,
                state,
            } => {
                let ctx = BlasDispatchCtx {
                    cublas,
                    stream,
                    completion,
                    state,
                };
                match msg {
                    BlasMsg::Gemm(d) => d.dispatch(&ctx),
                    BlasMsg::L1(d) => d.dispatch(&ctx),
                    BlasMsg::L2(d) => d.dispatch(&ctx),
                    BlasMsg::L3(d) => d.dispatch(&ctx),
                    BlasMsg::GemmStridedBatched(d) => d.dispatch(&ctx),
                    #[allow(deprecated)]
                    BlasMsg::Sgemm(req) => {
                        // Route through Gemm<f32> internally so all
                        // back-compat callers benefit from the same
                        // dispatch path.
                        let SgemmRequest {
                            a,
                            b,
                            c,
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            reply,
                        } = *req;
                        let typed = GemmRequest::<f32> {
                            a,
                            b,
                            c,
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            trans_a: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                            trans_b: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                            lda: m,
                            ldb: k,
                            ldc: m,
                            reply,
                        };
                        let boxed: Box<dyn GemmDispatch> = Box::new(typed);
                        boxed.dispatch(&ctx);
                    }
                }
            }
        }
    }
}

/// Mock-mode reply helper. We don't have access to the request's
/// `oneshot::Sender` (it lives inside the boxed dispatcher), so the
/// only thing we can do is drop the dispatcher — the receiver
/// observes `Err(RecvError)` which surfaces as a typed error at the
/// caller. Tracing logs the dropped op name so tests can spot it.
fn mock_reply(op: &'static str) {
    tracing::debug!(op, "BlasActor (mock): dropping op without reply");
}
