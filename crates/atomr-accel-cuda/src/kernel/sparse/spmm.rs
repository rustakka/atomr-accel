//! `SpMmRequest<T, I>` — `C = alpha * op(A) * op(B) + beta * C`.
//!
//! Backed by `cusparseSpMM` via the generic API. Supported sparse
//! formats: CSR, COO, Blocked-ELL.

use tokio::sync::oneshot;

use crate::dtype::{AccelDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::sys::cusparse::SpMmAlg;

use super::format::SparseMatrix;
use super::spmv::SpMvOp;

/// Memory order of dense `B`/`C`. cuSPARSE supports both column- and
/// row-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseOrder {
    ColMajor,
    RowMajor,
}

impl DenseOrder {
    pub fn raw(self) -> cudarc::cusparse::sys::cusparseOrder_t {
        match self {
            DenseOrder::ColMajor => cudarc::cusparse::sys::cusparseOrder_t::CUSPARSE_ORDER_COL,
            DenseOrder::RowMajor => cudarc::cusparse::sys::cusparseOrder_t::CUSPARSE_ORDER_ROW,
        }
    }
}

/// Generic SpMm request.
pub struct SpMmRequest<T: SparseSupported, I: SparseIndex> {
    pub matrix: SparseMatrix<T, I>,
    pub b: GpuRef<T>,
    pub c: GpuRef<T>,
    pub b_cols: i64,
    pub ldb: i64,
    pub ldc: i64,
    pub alpha: <T as AccelDtype>::Scalar,
    pub beta: <T as AccelDtype>::Scalar,
    pub op_a: SpMvOp,
    pub op_b: SpMvOp,
    pub order: DenseOrder,
    pub alg: SpMmAlg,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: SparseSupported, I: SparseIndex> SpMmRequest<T, I> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        matrix: SparseMatrix<T, I>,
        b: GpuRef<T>,
        c: GpuRef<T>,
        b_cols: i64,
        ldb: i64,
        ldc: i64,
        alpha: <T as AccelDtype>::Scalar,
        beta: <T as AccelDtype>::Scalar,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            matrix,
            b,
            c,
            b_cols,
            ldb,
            ldc,
            alpha,
            beta,
            op_a: SpMvOp::NonTranspose,
            op_b: SpMvOp::NonTranspose,
            order: DenseOrder::ColMajor,
            alg: SpMmAlg::Default,
            reply,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spmm_request_round_trip() {
        fn _ct<T: SparseSupported, I: SparseIndex>() {}
        _ct::<f32, i32>();
        _ct::<f64, i32>();
        _ct::<f32, i64>();
        _ct::<f64, i64>();
        #[cfg(feature = "f16")]
        {
            _ct::<half::f16, i32>();
            _ct::<half::bf16, i64>();
        }
        // Order tags round trip.
        let _ = DenseOrder::ColMajor.raw();
        let _ = DenseOrder::RowMajor.raw();
    }
}
