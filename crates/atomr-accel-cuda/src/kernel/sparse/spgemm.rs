//! `SpGemmRequest<T, I>` — sparse-sparse matrix multiply
//! `C = alpha * op(A) * op(B) + beta * C` where A, B, C are all sparse.
//!
//! Backed by the three-stage `cusparseSpGEMM_workEstimation`,
//! `cusparseSpGEMM_compute`, `cusparseSpGEMM_copy` flow.

use tokio::sync::oneshot;

use crate::dtype::{AccelDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::sys::cusparse::SpGemmAlg;

use super::format::SparseMatrix;
use super::spmv::SpMvOp;

/// Generic SpGEMM request — A, B, C are all sparse. C is **output-only**:
/// the actor allocates the result indices/values once `cusparseSpGEMM_copy`
/// has reported the final nnz, and surfaces them via `result`.
pub struct SpGemmRequest<T: SparseSupported, I: SparseIndex> {
    pub a: SparseMatrix<T, I>,
    pub b: SparseMatrix<T, I>,
    /// Pre-allocated C placeholder — actor fills in the descriptor then
    /// `cusparseSpGEMM_copy`s the result into it. The `nnz` field is
    /// initially `0`; the actor updates it post-compute.
    pub c: SparseMatrix<T, I>,
    pub alpha: <T as AccelDtype>::Scalar,
    pub beta: <T as AccelDtype>::Scalar,
    pub op_a: SpMvOp,
    pub op_b: SpMvOp,
    pub alg: SpGemmAlg,
    pub reply: oneshot::Sender<Result<SpGemmResult, GpuError>>,
}

/// Result of a SpGEMM — the final nnz of `C` (the indices/values are
/// written in-place into the placeholder buffers carried by
/// [`SpGemmRequest::c`]).
#[derive(Debug, Clone, Copy)]
pub struct SpGemmResult {
    pub nnz: i64,
}

impl<T: SparseSupported, I: SparseIndex> SpGemmRequest<T, I> {
    pub fn new(
        a: SparseMatrix<T, I>,
        b: SparseMatrix<T, I>,
        c: SparseMatrix<T, I>,
        alpha: <T as AccelDtype>::Scalar,
        beta: <T as AccelDtype>::Scalar,
        reply: oneshot::Sender<Result<SpGemmResult, GpuError>>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            alpha,
            beta,
            op_a: SpMvOp::NonTranspose,
            op_b: SpMvOp::NonTranspose,
            alg: SpGemmAlg::Default,
            reply,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spgemm_request_round_trip() {
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
        let r = SpGemmResult { nnz: 42 };
        assert_eq!(r.nnz, 42);
    }
}
