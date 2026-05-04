//! `BlasActor` — wraps a [`cudarc::cublas::CudaBlas`] handle, performs
//! SGEMM on its assigned stream, and returns completion via the
//! configured [`CompletionStrategy`] (§3.2 stateless-handle archetype +
//! §5.10 callback wiring).
//!
//! The mailbox is freed immediately after the kernel is enqueued — the
//! actor never blocks on the GPU (§5.2). Reply delivery happens on the
//! Tokio task spawned by [`envelope::run_kernel`].

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use atomr_core::actor::{Context, Props};
use atomr_macros::Actor;

use crate::completion::{CompletionStrategy, HostFnCompletion};
use crate::device::{DeviceState, SgemmRequest};
use crate::error::GpuError;
use crate::kernel::envelope;
use crate::stream::{ActorHints, StreamAllocator};

/// Public messages for `BlasActor`.
pub enum BlasMsg {
    Sgemm(Box<SgemmRequest>),
}

/// Two-track construction: a real cuBLAS-backed actor (`props`), and a
/// mock variant used by `examples/echo_no_gpu` and unit tests where no
/// GPU is present.
#[derive(Actor)]
#[msg(BlasMsg)]
pub struct BlasActor {
    inner: BlasInner,
}

enum BlasInner {
    Real {
        blas: CudaBlas,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        #[allow(dead_code)]
        state: Arc<DeviceState>,
    },
    Mock,
}

const LIB: &str = "cublas";

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
            let blas = match CudaBlas::new(stream.clone()) {
                Ok(b) => b,
                Err(e) => panic!("ContextPoisoned: CudaBlas::new failed: {e}"),
            };
            BlasActor {
                inner: BlasInner::Real {
                    blas,
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
        match msg {
            BlasMsg::Sgemm(req) => match &self.inner {
                BlasInner::Mock => {
                    let _ = req.reply.send(Err(GpuError::Unrecoverable(
                        "Sgemm not supported in mock mode".into(),
                    )));
                }
                BlasInner::Real {
                    blas,
                    stream,
                    completion,
                    ..
                } => {
                    enqueue_sgemm(blas, stream, completion, *req);
                }
            },
        }
    }
}

/// Validate operands, enqueue the cuBLAS call, and hand off to
/// [`envelope::run_kernel`] for completion wiring.
fn enqueue_sgemm(
    blas: &CudaBlas,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    req: SgemmRequest,
) {
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
    } = req;

    // Validate generation tokens (§5.8). Pre-launch failures are
    // reported synchronously through the reply channel so callers
    // observe them as `Err(...)` instead of timing out.
    let (a_slice, b_slice, c_slice) = match envelope::access_all_3(&a, &b, &c) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    // F1 only supports the no-transpose case. cuBLAS uses column-major
    // strides; see cublas docs for the convention.
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m,
        n,
        k,
        alpha,
        lda: m,
        ldb: k,
        beta,
        ldc: m,
    };

    // cudarc's `gemm` requires `&mut C: DevicePtrMut<f32>`. An
    // `Arc<CudaSlice<f32>>` doesn't satisfy that: we have to unwrap
    // the Arc. The caller must hold the unique `GpuRef` to the output
    // buffer or the unwrap fails — single-writer enforcement.
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_arc) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SGEMM target buffer C has more than one live reference; \
                 caller must hold the unique GpuRef to write to it"
                    .into(),
            )));
            return;
        }
    };

    // Track that this stream wrote to `c` so downstream consumers
    // (P2P, pipeline stages) can inject a wait without round-tripping
    // to the host.
    c.record_write(stream);

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        // SAFETY: cudarc's `gemm` is unsafe because invalid
        // m/n/k/lda/ldb/ldc can read out of bounds. The caller is
        // responsible for valid dims.
        let res = unsafe { blas.gemm(cfg, &*a_slice, &*b_slice, &mut c_owned) };
        match res {
            Ok(()) => {
                // keep-alive carries everything cudarc needs to
                // remain valid until the kernel completes.
                Ok((a_slice, b_slice, c_owned))
            }
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("sgemm enqueue: {e}"),
            }),
        }
    });
}
