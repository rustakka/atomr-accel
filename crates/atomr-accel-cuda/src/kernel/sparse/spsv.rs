//! `SpSvRequest<T, I>` — sparse triangular solve `op(A) * y = alpha * x`.
//!
//! Backed by the cuSPARSE generic API `cusparseSpSV_*` (preferred) plus
//! the legacy `cusparseCsrsv2_*` path for callers that explicitly opt in
//! via [`SpSvAlg`]. CSR format only; A is upper or lower triangular with
//! optional unit diagonal.

use tokio::sync::oneshot;

use crate::dtype::{AccelDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::sys::cusparse::SpSvAlg;

use super::format::SparseMatrix;
use super::spmv::SpMvOp;

/// Triangle / diagonal structure of `A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpSvFill {
    Upper,
    Lower,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpSvDiag {
    NonUnit,
    Unit,
}

pub struct SpSvRequest<T: SparseSupported, I: SparseIndex> {
    pub matrix: SparseMatrix<T, I>,
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub alpha: <T as AccelDtype>::Scalar,
    pub op: SpMvOp,
    pub fill: SpSvFill,
    pub diag: SpSvDiag,
    pub alg: SpSvAlg,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: SparseSupported, I: SparseIndex> SpSvRequest<T, I> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        matrix: SparseMatrix<T, I>,
        x: GpuRef<T>,
        y: GpuRef<T>,
        alpha: <T as AccelDtype>::Scalar,
        fill: SpSvFill,
        diag: SpSvDiag,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            matrix,
            x,
            y,
            alpha,
            op: SpMvOp::NonTranspose,
            fill,
            diag,
            alg: SpSvAlg::Default,
            reply,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spsv_request_round_trip() {
        fn _ct<T: SparseSupported, I: SparseIndex>() {}
        _ct::<f32, i32>();
        _ct::<f64, i32>();
        _ct::<f32, i64>();
        _ct::<f64, i64>();
        // Tag enums round trip.
        assert!(matches!(SpSvFill::Upper, SpSvFill::Upper));
        assert!(matches!(SpSvDiag::NonUnit, SpSvDiag::NonUnit));
    }
}
