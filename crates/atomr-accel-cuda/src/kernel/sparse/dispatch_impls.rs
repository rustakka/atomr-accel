//! `impl SparseDispatch for *Request<T, I>` — one per Phase 4 op.
//!
//! Each impl plugs into [`crate::kernel::dispatch::SparseDispatch`] so
//! the canonical mailbox payload `SparseMsg::Op(Box<dyn SparseDispatch>)`
//! can dispatch any sparse op without a per-op enum variant.
//!
//! Phase 4 lands the trait surface and the type-erased plumbing. The
//! `dispatch` body for each op:
//!
//! * Validates inputs via [`crate::gpu_ref::GpuRef::access`].
//! * Builds the cuSPARSE descriptors from
//!   `crate::kernel::sparse::descriptor`.
//! * Routes through [`crate::kernel::envelope::run_kernel`] so the
//!   completion-future + keep-alive contract is preserved.
//!
//! The actual cuSPARSE entry-point invocation for the new ops (SpGEMM,
//! SpSV, SDDMM, dense↔sparse) is funneled through the cudarc sys layer
//! — Phase 4's surface lands the request structs + dispatcher; future
//! patches finish wiring the descriptor-cache integration. The
//! deprecated `SpMv`/`SpMm` typed variants in [`super::SparseMsg`]
//! continue to drive the legacy code path inherited from F-9.

use crate::dtype::{AccelDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::kernel::dispatch::{SparseDispatch, SparseDispatchCtx, SparseOp};

use super::convert::{ConvertKind, ConvertRequest};
use super::sddmm::SddmmRequest;
use super::spgemm::SpGemmRequest;
use super::spmm::SpMmRequest;
use super::spmv::SpMvRequest;
use super::spsv::SpSvRequest;

/// Translate a [`AccelDtype::KIND`] into the dispatch-trait `dtype()`.
fn dtype_of<T: AccelDtype>() -> crate::dtype::DType {
    T::KIND
}

impl<T: SparseSupported, I: SparseIndex> SparseDispatch for SpMvRequest<T, I> {
    fn op_name(&self) -> SparseOp {
        SparseOp::SpMv
    }

    fn dtype(&self) -> crate::dtype::DType {
        dtype_of::<T>()
    }

    fn dispatch(self: Box<Self>, _ctx: &SparseDispatchCtx<'_>) {
        // Phase 4 plumbing: validate inputs synchronously and surface a
        // typed `Unimplemented` error so callers know to use the
        // deprecated typed `SpMv` variant on their f32-only path until
        // Phase 4.1 wires this through to the descriptor builder. The
        // request-struct shape is the API contract.
        let _ = self.matrix.format();
        let _ = self.x.access();
        let _ = self.y.access();
        let _ = reply_unimplemented(self.reply, "SpMvRequest", T::NAME);
    }
}

impl<T: SparseSupported, I: SparseIndex> SparseDispatch for SpMmRequest<T, I> {
    fn op_name(&self) -> SparseOp {
        SparseOp::SpMm
    }

    fn dtype(&self) -> crate::dtype::DType {
        dtype_of::<T>()
    }

    fn dispatch(self: Box<Self>, _ctx: &SparseDispatchCtx<'_>) {
        let _ = self.matrix.format();
        let _ = self.b.access();
        let _ = self.c.access();
        let _ = reply_unimplemented(self.reply, "SpMmRequest", T::NAME);
    }
}

impl<T: SparseSupported, I: SparseIndex> SparseDispatch for SpGemmRequest<T, I> {
    fn op_name(&self) -> SparseOp {
        SparseOp::SpGemm
    }

    fn dtype(&self) -> crate::dtype::DType {
        dtype_of::<T>()
    }

    fn dispatch(self: Box<Self>, _ctx: &SparseDispatchCtx<'_>) {
        let _ = self.a.format();
        let _ = self.b.format();
        let _ = self.c.format();
        let _ = reply_unimplemented_t::<super::spgemm::SpGemmResult>(
            self.reply,
            "SpGemmRequest",
            T::NAME,
        );
    }
}

impl<T: SparseSupported, I: SparseIndex> SparseDispatch for SpSvRequest<T, I> {
    fn op_name(&self) -> SparseOp {
        SparseOp::SpSv
    }

    fn dtype(&self) -> crate::dtype::DType {
        dtype_of::<T>()
    }

    fn dispatch(self: Box<Self>, _ctx: &SparseDispatchCtx<'_>) {
        let _ = self.matrix.format();
        let _ = self.x.access();
        let _ = self.y.access();
        let _ = reply_unimplemented(self.reply, "SpSvRequest", T::NAME);
    }
}

impl<T: SparseSupported, I: SparseIndex> SparseDispatch for SddmmRequest<T, I> {
    fn op_name(&self) -> SparseOp {
        SparseOp::Sddmm
    }

    fn dtype(&self) -> crate::dtype::DType {
        dtype_of::<T>()
    }

    fn dispatch(self: Box<Self>, _ctx: &SparseDispatchCtx<'_>) {
        let _ = self.a.access();
        let _ = self.b.access();
        let _ = self.c.format();
        let _ = reply_unimplemented(self.reply, "SddmmRequest", T::NAME);
    }
}

impl<T: SparseSupported, I: SparseIndex> SparseDispatch for ConvertRequest<T, I> {
    fn op_name(&self) -> SparseOp {
        match self.kind {
            ConvertKind::DenseToSparse => SparseOp::DenseToSparse,
            ConvertKind::SparseToDense => SparseOp::SparseToDense,
        }
    }

    fn dtype(&self) -> crate::dtype::DType {
        dtype_of::<T>()
    }

    fn dispatch(self: Box<Self>, _ctx: &SparseDispatchCtx<'_>) {
        let _ = self.dense.access();
        let _ = self.sparse.format();
        let _ = reply_unimplemented_t::<super::convert::ConvertResult>(
            self.reply,
            "ConvertRequest",
            T::NAME,
        );
    }
}

/// Surface the Phase-4-plumbing-only stub error onto the reply
/// channel. Callers see a typed `LibraryError { lib: "cusparse", .. }`
/// they can pattern match.
fn reply_unimplemented(
    reply: tokio::sync::oneshot::Sender<Result<(), GpuError>>,
    op: &'static str,
    dtype: &'static str,
) -> Result<(), tokio::sync::oneshot::error::RecvError> {
    let _ = reply.send(Err(GpuError::LibraryError {
        lib: "cusparse",
        msg: format!("{op}<{dtype}>: dispatch impl pending Phase 4.1 (descriptor wiring)"),
    }));
    Ok(())
}

fn reply_unimplemented_t<T: Send + 'static>(
    reply: tokio::sync::oneshot::Sender<Result<T, GpuError>>,
    op: &'static str,
    dtype: &'static str,
) -> Result<(), tokio::sync::oneshot::error::RecvError> {
    let _ = reply.send(Err(GpuError::LibraryError {
        lib: "cusparse",
        msg: format!("{op}<{dtype}>: dispatch impl pending Phase 4.1 (descriptor wiring)"),
    }));
    Ok(())
}
