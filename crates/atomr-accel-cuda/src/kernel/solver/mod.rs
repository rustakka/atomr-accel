//! `SolverActor` — wraps a [`cudarc::cusolver::DnHandle`] for dense
//! linear algebra and a [`cudarc::cusolver::SpHandle`] for sparse
//! solves (gated `cusolver-sp`).
//!
//! Phase 1 cuSOLVER scope:
//! - Dense: `Qr`, `Lu` (factorize / solve), `Cholesky`, `Svd`,
//!   `Syevd` for f32 and f64 (see [`dense`]).
//! - Batched: `getrfBatched` (cuBLAS-side LU, lifted into this
//!   actor for symmetry), `potrfBatched`, `gesvdjBatched` (see
//!   [`batched`]).
//! - Generalized symmetric eigenvalue: `Sygvd` / `Hegvd` (real
//!   variants today; complex Hermitian deferred — see
//!   [`generalized`]).
//! - Sparse: `cusolverSp` `Cholesky` / `QR` solves over CSR matrices
//!   (gated `cusolver-sp`, see [`sparse`]).
//!
//! Implementation notes:
//! - cudarc 0.19's safe layer exposes only handle management; per-op
//!   entry points live in `cusolver::sys` and are wired through
//!   [`crate::sys::cusolver::SolverScalar`] into a dtype-generic
//!   surface (see `crate/src/sys/cusolver.rs`).
//! - Each op queries the cuSOLVER workspace size, grows our on-demand
//!   `CudaSlice<u8>` workspace, then dispatches the factorisation. The
//!   1-element `info` buffer is read back to detect failures (singular
//!   matrix, illegal arg, etc.).
//! - `SolverMsg::Op(Box<dyn SolverDispatch>)` is the canonical
//!   surface; the typed `Qr` / `Lu` / `Cholesky` / `Svd` / `Syevd`
//!   variants are kept as `#[deprecated]` aliases for backward
//!   compatibility with the f32-only Phase 0 layout.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::cusolver::DnHandle;
use cudarc::driver::CudaSlice;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::stream::StreamAllocator;

pub mod batched;
pub mod dense;
pub mod generalized;
#[cfg(feature = "cusolver-sp")]
pub mod sparse;
mod workspace;

pub use batched::{GesvdjBatchedRequest, GetrfBatchedRequest, PotrfBatchedRequest};
pub use dense::{CholeskyRequest, LuRequest, LuSolveRequest, QrRequest, SvdRequest, SyevdRequest};
pub use generalized::{HegvdRequest, SygvdRequest};
#[cfg(feature = "cusolver-sp")]
pub use sparse::{SparseCholeskyRequest, SparseLuRequest, SparseQrRequest};

/// Storage triangle for symmetric / Hermitian / triangular factorisations.
#[derive(Debug, Clone, Copy)]
pub enum Uplo {
    Upper,
    Lower,
}

impl Uplo {
    pub(crate) fn as_cusolver_fill(self) -> cudarc::cusolver::sys::cublasFillMode_t {
        use cudarc::cusolver::sys::cublasFillMode_t;
        match self {
            Uplo::Upper => cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            Uplo::Lower => cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        }
    }
}

/// Crate-private cells the dispatch traits operate against. Passed
/// to [`SolverDispatch::dispatch`] as a single bundle so each op
/// implementation only depends on what it actually uses.
///
/// The struct itself is publicly visible because [`SolverDispatch`]
/// is a public trait whose method takes a `SolverCells<'_>`, but
/// every field is `pub(crate)` since the `SendDn` / `SendSp`
/// newtypes leak FFI handles that have no stable external
/// representation. External code wires custom solver ops by
/// implementing `SolverDispatch` only when also living inside this
/// crate.
pub struct SolverCells<'a> {
    pub(crate) handle: &'a Mutex<SendDn>,
    pub(crate) stream: &'a Arc<cudarc::driver::CudaStream>,
    pub(crate) completion: &'a Arc<dyn CompletionStrategy>,
    pub(crate) workspace: &'a Mutex<Option<CudaSlice<u8>>>,
    pub(crate) info: &'a Mutex<CudaSlice<i32>>,
    #[cfg(feature = "cusolver-sp")]
    pub(crate) sp_handle: &'a Mutex<Option<SendSp>>,
}

/// Trait implemented by every solver request. The actor turns the
/// boxed trait object back into a typed launcher and forwards the
/// runtime cells.
pub trait SolverDispatch: Send + 'static {
    /// Execute the request against a real cuSOLVER handle.
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>);

    /// Reply with a "mock mode" error without touching the GPU.
    /// Default impl drops `self` so the caller's oneshot closes;
    /// per-request impls override to send a typed `Err`.
    fn dispatch_mock(self: Box<Self>) {
        drop(self);
    }
}

pub enum SolverMsg {
    /// Canonical, dtype-generic surface. New code should prefer this
    /// over the legacy enum variants.
    Op(Box<dyn SolverDispatch>),

    /// Legacy QR factorize. Use [`QrRequest`] via [`SolverMsg::Op`]
    /// instead.
    #[deprecated(note = "use SolverMsg::Op(Box::new(QrRequest { .. }))")]
    QrFactorize {
        a: GpuRef<f32>,
        m: i32,
        n: i32,
        tau: GpuRef<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Legacy LU factorize. Use [`LuRequest`] via [`SolverMsg::Op`].
    #[deprecated(note = "use SolverMsg::Op(Box::new(LuRequest { .. }))")]
    LuFactorize {
        a: GpuRef<f32>,
        m: i32,
        n: i32,
        ipiv: GpuRef<i32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Legacy LU solve. Use [`LuSolveRequest`] via [`SolverMsg::Op`].
    #[deprecated(note = "use SolverMsg::Op(Box::new(LuSolveRequest { .. }))")]
    LuSolve {
        lu: GpuRef<f32>,
        ipiv: GpuRef<i32>,
        b: GpuRef<f32>,
        n: i32,
        nrhs: i32,
        trans: bool,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Legacy Cholesky. Use [`CholeskyRequest`] via [`SolverMsg::Op`].
    #[deprecated(note = "use SolverMsg::Op(Box::new(CholeskyRequest { .. }))")]
    Cholesky {
        a: GpuRef<f32>,
        n: i32,
        uplo: Uplo,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Legacy SVD. Use [`SvdRequest`] via [`SolverMsg::Op`].
    #[deprecated(note = "use SolverMsg::Op(Box::new(SvdRequest { .. }))")]
    Svd {
        a: GpuRef<f32>,
        m: i32,
        n: i32,
        s: GpuRef<f32>,
        u: Option<GpuRef<f32>>,
        vt: Option<GpuRef<f32>>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Legacy symmetric eigendecomposition. Use [`SyevdRequest`] via
    /// [`SolverMsg::Op`].
    #[deprecated(note = "use SolverMsg::Op(Box::new(SyevdRequest { .. }))")]
    Syevd {
        a: GpuRef<f32>,
        n: i32,
        uplo: Uplo,
        w: GpuRef<f32>,
        compute_vectors: bool,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct SolverActor {
    inner: SolverInner,
}

pub(crate) struct SendDn(pub(crate) DnHandle);
unsafe impl Send for SendDn {}
unsafe impl Sync for SendDn {}

#[cfg(feature = "cusolver-sp")]
pub(crate) struct SendSp(pub(crate) cudarc::cusolver::SpHandle);
#[cfg(feature = "cusolver-sp")]
unsafe impl Send for SendSp {}
#[cfg(feature = "cusolver-sp")]
unsafe impl Sync for SendSp {}

#[allow(dead_code)]
enum SolverInner {
    Real {
        handle: Mutex<SendDn>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        /// On-demand-grown scratch buffer (in bytes; we widen from
        /// the per-op f32/f64 workspaces by multiplying out
        /// `lwork * size_of::<T>()`). Never shrunk; rebuilt fresh on
        /// context restart.
        workspace: Mutex<Option<CudaSlice<u8>>>,
        /// 1-element `i32` info buffer reused across calls.
        info: Mutex<CudaSlice<i32>>,
        /// Lazy `cusolverSp` handle; created on first sparse op.
        #[cfg(feature = "cusolver-sp")]
        sp_handle: Mutex<Option<SendSp>>,
    },
    Mock,
}

impl SolverActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            let handle = match DnHandle::new(stream.clone()) {
                Ok(h) => h,
                Err(e) => panic!("ContextPoisoned: DnHandle::new failed: {e}"),
            };
            let info = stream
                .alloc_zeros::<i32>(1)
                .unwrap_or_else(|e| panic!("ContextPoisoned: alloc info: {e}"));
            SolverActor {
                inner: SolverInner::Real {
                    handle: Mutex::new(SendDn(handle)),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                    workspace: Mutex::new(None),
                    info: Mutex::new(info),
                    #[cfg(feature = "cusolver-sp")]
                    sp_handle: Mutex::new(None),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| SolverActor {
            inner: SolverInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for SolverActor {
    type Msg = SolverMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SolverMsg) {
        match &self.inner {
            SolverInner::Mock => mock_reply(msg),
            SolverInner::Real {
                handle,
                stream,
                completion,
                workspace,
                info,
                #[cfg(feature = "cusolver-sp")]
                sp_handle,
                ..
            } => {
                let cells = SolverCells {
                    handle,
                    stream,
                    completion,
                    workspace,
                    info,
                    #[cfg(feature = "cusolver-sp")]
                    sp_handle,
                };
                dispatch_msg(msg, cells);
            }
        }
    }
}

#[allow(deprecated)]
fn dispatch_msg(msg: SolverMsg, cells: SolverCells<'_>) {
    match msg {
        SolverMsg::Op(op) => op.dispatch(cells),
        SolverMsg::QrFactorize {
            a,
            m,
            n,
            tau,
            reply,
        } => Box::new(QrRequest::<f32> {
            a,
            m,
            n,
            tau,
            reply,
        })
        .dispatch(cells),
        SolverMsg::LuFactorize {
            a,
            m,
            n,
            ipiv,
            reply,
        } => Box::new(LuRequest::<f32> {
            a,
            m,
            n,
            ipiv,
            reply,
        })
        .dispatch(cells),
        SolverMsg::LuSolve {
            lu,
            ipiv,
            b,
            n,
            nrhs,
            trans,
            reply,
        } => Box::new(LuSolveRequest::<f32> {
            lu,
            ipiv,
            b,
            n,
            nrhs,
            trans,
            reply,
        })
        .dispatch(cells),
        SolverMsg::Cholesky { a, n, uplo, reply } => {
            Box::new(CholeskyRequest::<f32> { a, n, uplo, reply }).dispatch(cells)
        }
        SolverMsg::Svd {
            a,
            m,
            n,
            s,
            u,
            vt,
            reply,
        } => Box::new(SvdRequest::<f32> {
            a,
            m,
            n,
            s,
            u,
            vt,
            reply,
        })
        .dispatch(cells),
        SolverMsg::Syevd {
            a,
            n,
            uplo,
            w,
            compute_vectors,
            reply,
        } => Box::new(SyevdRequest::<f32> {
            a,
            n,
            uplo,
            w,
            compute_vectors,
            reply,
        })
        .dispatch(cells),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    /// The deprecated `SolverMsg::QrFactorize` alias must still
    /// construct (and the actor must still route it through the
    /// dispatch path) so existing applications can compile during
    /// the Phase 1 transition window.
    #[test]
    #[allow(deprecated)]
    fn deprecated_qr_alias_still_constructs() {
        // We can't run the actor without a GPU; just ensure the
        // variant is constructible and matches the documented shape.
        let (tx, _rx) = oneshot::channel::<Result<(), GpuError>>();
        // Use placeholder values; we never `tell` it to a live actor.
        // Construction is enough to assert the deprecated surface
        // hasn't been removed.
        let make = move |reply: oneshot::Sender<Result<(), GpuError>>| -> &'static str {
            // Compile-time only: build the deprecated variant.
            // We avoid constructing a real GpuRef by deferring to a
            // closure that's never called.
            #[allow(dead_code)]
            #[allow(deprecated)]
            fn _check(
                a: GpuRef<f32>,
                tau: GpuRef<f32>,
                reply: oneshot::Sender<Result<(), GpuError>>,
            ) -> SolverMsg {
                SolverMsg::QrFactorize {
                    a,
                    m: 0,
                    n: 0,
                    tau,
                    reply,
                }
            }
            drop(reply);
            "ok"
        };
        assert_eq!(make(tx), "ok");
    }
}

#[allow(deprecated)]
fn mock_reply(msg: SolverMsg) {
    let err = || GpuError::Unrecoverable("SolverActor in mock mode".into());
    match msg {
        SolverMsg::Op(op) => op.dispatch_mock(),
        SolverMsg::QrFactorize { reply, .. }
        | SolverMsg::LuFactorize { reply, .. }
        | SolverMsg::LuSolve { reply, .. }
        | SolverMsg::Cholesky { reply, .. }
        | SolverMsg::Svd { reply, .. }
        | SolverMsg::Syevd { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
    }
}
