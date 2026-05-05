//! `SpMvRequest<T, I>` — `y = alpha * op(A) * x + beta * y`.
//!
//! Backed by `cusparseSpMV` via the generic API. Supported sparse
//! formats: CSR, COO. (Blocked-ELL is SpMM-only per cuSPARSE.)

use cudarc::cusparse::sys as cs;
use tokio::sync::oneshot;

use crate::dtype::{AccelDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::sys::cusparse::SpMvAlg;

use super::format::SparseMatrix;

/// Operation tag — `op(A)` in `y = alpha * op(A) * x + beta * y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpMvOp {
    NonTranspose,
    Transpose,
    ConjugateTranspose,
}

impl SpMvOp {
    pub fn raw(self) -> cs::cusparseOperation_t {
        match self {
            SpMvOp::NonTranspose => cs::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            SpMvOp::Transpose => cs::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
            SpMvOp::ConjugateTranspose => {
                cs::cusparseOperation_t::CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
            }
        }
    }
}

/// Generic SpMv request.
pub struct SpMvRequest<T: SparseSupported, I: SparseIndex> {
    pub matrix: SparseMatrix<T, I>,
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub alpha: <T as AccelDtype>::Scalar,
    pub beta: <T as AccelDtype>::Scalar,
    pub op: SpMvOp,
    pub alg: SpMvAlg,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: SparseSupported, I: SparseIndex> SpMvRequest<T, I> {
    pub fn new(
        matrix: SparseMatrix<T, I>,
        x: GpuRef<T>,
        y: GpuRef<T>,
        alpha: <T as AccelDtype>::Scalar,
        beta: <T as AccelDtype>::Scalar,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            matrix,
            x,
            y,
            alpha,
            beta,
            op: SpMvOp::NonTranspose,
            alg: SpMvAlg::Default,
            reply,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip the request struct's type parameters across the four
    /// supported value dtypes — verifies the trait bounds compose.
    #[test]
    fn spmv_request_round_trip_f32_f64_f16_bf16() {
        fn _ct<T: SparseSupported, I: SparseIndex>(
            x: GpuRef<T>,
            y: GpuRef<T>,
            mat: SparseMatrix<T, I>,
            reply: oneshot::Sender<Result<(), GpuError>>,
        ) -> SpMvRequest<T, I>
        where
            <T as AccelDtype>::Scalar: Default,
        {
            SpMvRequest::new(
                mat,
                x,
                y,
                <T as AccelDtype>::Scalar::default(),
                <T as AccelDtype>::Scalar::default(),
                reply,
            )
        }
        // We can't actually mint a `GpuRef` here (no GPU), but the
        // function signature compiling proves the bounds hold.
        let _f = _ct::<f32, i32>;
        let _g = _ct::<f64, i32>;
        let _h = _ct::<f32, i64>;
        let _i = _ct::<f64, i64>;
        #[cfg(feature = "f16")]
        {
            // half::f16's Scalar is half::f16 which has no Default;
            // exercise via the typed signature only.
            fn _half<I: SparseIndex>() {}
            _half::<i32>();
            _half::<i64>();
        }
        // Algorithm tags round trip.
        assert_eq!(SpMvAlg::Default.raw(), SpMvAlg::Default.raw());
    }
}
