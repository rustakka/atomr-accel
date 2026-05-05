//! `SddmmRequest<T, I>` — sampled dense-dense matrix multiply.
//!
//! Computes `C = alpha * (A @ B) * sample(C_sparsity_pattern) + beta * C`
//! where `A` and `B` are dense and `C` is sparse — only the non-zero
//! positions of `C` are written, leaving its sparsity pattern intact.
//! Backed by `cusparseSDDMM`.

use tokio::sync::oneshot;

use crate::dtype::{AccelDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::sys::cusparse::SddmmAlg;

use super::format::SparseMatrix;
use super::spmm::DenseOrder;
use super::spmv::SpMvOp;

pub struct SddmmRequest<T: SparseSupported, I: SparseIndex> {
    /// Dense `A`, layout in `order`. Shape `m × k`.
    pub a: GpuRef<T>,
    pub a_rows: i64,
    pub a_cols: i64,
    pub lda: i64,
    /// Dense `B`. Shape `k × n`.
    pub b: GpuRef<T>,
    pub b_rows: i64,
    pub b_cols: i64,
    pub ldb: i64,
    /// Sparse `C`. Pattern preserved; values written at non-zero
    /// positions only.
    pub c: SparseMatrix<T, I>,
    pub alpha: <T as AccelDtype>::Scalar,
    pub beta: <T as AccelDtype>::Scalar,
    pub op_a: SpMvOp,
    pub op_b: SpMvOp,
    pub order: DenseOrder,
    pub alg: SddmmAlg,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: SparseSupported, I: SparseIndex> SddmmRequest<T, I> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: GpuRef<T>,
        a_rows: i64,
        a_cols: i64,
        lda: i64,
        b: GpuRef<T>,
        b_rows: i64,
        b_cols: i64,
        ldb: i64,
        c: SparseMatrix<T, I>,
        alpha: <T as AccelDtype>::Scalar,
        beta: <T as AccelDtype>::Scalar,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            a,
            a_rows,
            a_cols,
            lda,
            b,
            b_rows,
            b_cols,
            ldb,
            c,
            alpha,
            beta,
            op_a: SpMvOp::NonTranspose,
            op_b: SpMvOp::NonTranspose,
            order: DenseOrder::ColMajor,
            alg: SddmmAlg::Default,
            reply,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sddmm_request_round_trip() {
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
    }
}
