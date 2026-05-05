//! Format / dtype / index-type conversion entry points.
//!
//! Wraps `cusparseDenseToSparse_*` (dense → CSR/COO) and
//! `cusparseSparseToDense` (CSR/COO/Blocked-ELL → dense) in
//! request-style structs. Index-type and dtype mismatches are
//! statically rejected via the [`SparseIndex`] / [`SparseSupported`]
//! markers.

use tokio::sync::oneshot;

use crate::dtype::{SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

use super::format::SparseMatrix;
use super::spmm::DenseOrder;

/// Direction of a [`ConvertRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertKind {
    DenseToSparse,
    SparseToDense,
}

/// Convert dense ↔ sparse for a given fixed sparse format. The sparse
/// side carries pre-allocated index/value buffers sized to the target
/// nnz; the actor only fills them.
pub struct ConvertRequest<T: SparseSupported, I: SparseIndex> {
    pub kind: ConvertKind,
    pub dense: GpuRef<T>,
    pub dense_rows: i64,
    pub dense_cols: i64,
    pub dense_ld: i64,
    pub dense_order: DenseOrder,
    pub sparse: SparseMatrix<T, I>,
    pub reply: oneshot::Sender<Result<ConvertResult, GpuError>>,
}

/// Result of a conversion.
#[derive(Debug, Clone, Copy)]
pub struct ConvertResult {
    /// Final nnz of the sparse side (only populated for
    /// `DenseToSparse`).
    pub nnz: i64,
}

impl<T: SparseSupported, I: SparseIndex> ConvertRequest<T, I> {
    #[allow(clippy::too_many_arguments)]
    pub fn dense_to_sparse(
        dense: GpuRef<T>,
        rows: i64,
        cols: i64,
        ld: i64,
        order: DenseOrder,
        sparse: SparseMatrix<T, I>,
        reply: oneshot::Sender<Result<ConvertResult, GpuError>>,
    ) -> Self {
        Self {
            kind: ConvertKind::DenseToSparse,
            dense,
            dense_rows: rows,
            dense_cols: cols,
            dense_ld: ld,
            dense_order: order,
            sparse,
            reply,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_to_dense(
        sparse: SparseMatrix<T, I>,
        dense: GpuRef<T>,
        rows: i64,
        cols: i64,
        ld: i64,
        order: DenseOrder,
        reply: oneshot::Sender<Result<ConvertResult, GpuError>>,
    ) -> Self {
        Self {
            kind: ConvertKind::SparseToDense,
            dense,
            dense_rows: rows,
            dense_cols: cols,
            dense_ld: ld,
            dense_order: order,
            sparse,
            reply,
        }
    }
}
