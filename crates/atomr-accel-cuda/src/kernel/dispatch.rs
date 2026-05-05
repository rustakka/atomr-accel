//! Per-actor dispatch traits — Phase 0.2 (cuSPARSE + cuSPARSELt slice
//! only, scoped for Phase 4).
//!
//! Phase 0's full cut adds `GemmDispatch`, `BlasLtDispatch`,
//! `CudnnDispatch`, `FftDispatch`, `RngDispatch`, `SolverDispatch`,
//! `TensorDispatch`, `CollectiveDispatch`, and `AllocDispatch`. Phase 4
//! ships only the SparseDispatch + SparseLtDispatch slot — every other
//! actor still uses its concrete typed message.
//!
//! The pattern is intentionally minimal: a `Box<dyn SparseDispatch>` is
//! the canonical Phase-4 message payload. The dispatcher reaches into
//! `SparseDispatchCtx` for the cuSPARSE handle, the actor stream, and
//! the completion strategy, then calls
//! [`SparseDispatch::dispatch`] which owns the closure body that builds
//! descriptors and submits the kernel.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::dtype::DType;
use crate::error::GpuError;

#[cfg(feature = "cusparse")]
use cudarc::cusparse::sys as cs;

/// Send-wrapper for the raw cuSPARSE handle. The handle is `!Send` by
/// default because it's a raw pointer; cuSPARSE itself is thread-safe so
/// long as a given handle is only touched by one stream at a time, which
/// the actor-per-handle invariant guarantees.
#[cfg(feature = "cusparse")]
pub struct SendSparseHandle(pub cs::cusparseHandle_t);
#[cfg(feature = "cusparse")]
unsafe impl Send for SendSparseHandle {}
#[cfg(feature = "cusparse")]
unsafe impl Sync for SendSparseHandle {}

/// Per-message context the [`crate::kernel::sparse::SparseActor`] hands
/// to a `SparseDispatch::dispatch` call. Carries the live cuSPARSE
/// handle, the actor stream, the completion strategy, and the shared
/// workspace mutex (an on-demand-grown `CudaSlice<u8>`).
#[cfg(feature = "cusparse")]
pub struct SparseDispatchCtx<'a> {
    pub handle: &'a Mutex<SendSparseHandle>,
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub workspace: &'a Mutex<Option<cudarc::driver::CudaSlice<u8>>>,
}

/// The op-kind tag a `SparseDispatch` exposes — used for tracing,
/// graph-capture replay, and `EnqueueDispatch::op_name` introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseOp {
    SpMv,
    SpMm,
    SpGemm,
    SpSv,
    Sddmm,
    DenseToSparse,
    SparseToDense,
    Convert,
}

impl SparseOp {
    pub fn as_str(self) -> &'static str {
        match self {
            SparseOp::SpMv => "spmv",
            SparseOp::SpMm => "spmm",
            SparseOp::SpGemm => "spgemm",
            SparseOp::SpSv => "spsv",
            SparseOp::Sddmm => "sddmm",
            SparseOp::DenseToSparse => "dense_to_sparse",
            SparseOp::SparseToDense => "sparse_to_dense",
            SparseOp::Convert => "convert",
        }
    }
}

/// Box-erased cuSPARSE op. A request struct
/// (`SpMvRequest`, `SpMmRequest`, …) implements this trait so the
/// `SparseActor` mailbox can carry a single canonical `Op(Box<dyn …>)`
/// variant in addition to the deprecated typed ones.
///
/// `dispatch` consumes the box and is responsible for calling
/// [`crate::kernel::envelope::run_kernel`] with the right keep-alive
/// tuple. `op_name` and `dtype` are introspection-only — they MUST NOT
/// have side effects and MUST be cheap.
#[cfg(feature = "cusparse")]
pub trait SparseDispatch: Send + 'static {
    fn op_name(&self) -> SparseOp;
    fn dtype(&self) -> DType;
    fn dispatch(self: Box<Self>, ctx: &SparseDispatchCtx<'_>);
}

// ---------------------------------------------------------------------
// cuSPARSELt
// ---------------------------------------------------------------------

/// Dispatch context for cuSPARSELt ops. Phase 4 keeps it minimal — the
/// handle pointer + stream + completion + reply oneshot.
#[cfg(feature = "cusparse-lt")]
pub struct SparseLtDispatchCtx<'a> {
    pub handle: &'a Mutex<crate::sys::cusparse_lt::SendCuSparseLtHandle>,
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseLtOp {
    Prune,
    Compress,
    Matmul,
}

impl SparseLtOp {
    pub fn as_str(self) -> &'static str {
        match self {
            SparseLtOp::Prune => "prune",
            SparseLtOp::Compress => "compress",
            SparseLtOp::Matmul => "matmul",
        }
    }
}

#[cfg(feature = "cusparse-lt")]
pub trait SparseLtDispatch: Send + 'static {
    fn op_name(&self) -> SparseLtOp;
    fn dtype(&self) -> DType;
    fn dispatch(self: Box<Self>, ctx: &SparseLtDispatchCtx<'_>);
}

/// Helper used by the no-handle-yet test path: surface a typed error on
/// the reply oneshot.
#[allow(dead_code)]
#[inline]
pub(crate) fn send_unrecoverable<T: Send + 'static>(
    reply: oneshot::Sender<Result<T, GpuError>>,
    msg: &'static str,
) {
    let _ = reply.send(Err(GpuError::Unrecoverable(msg.into())));
}
