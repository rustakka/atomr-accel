//! Dtype-generic plumbing for atomr-accel-cuda.
//!
//! Phase 0.2 ships a minimal local `CudaDtype` trait plus capability
//! markers (`GemmSupported`, …). The trait is intentionally cudarc /
//! CUDA-specific — a backend-agnostic `AccelDtype` trait can be lifted
//! into `crates/atomr-accel/src/dtype.rs` later without changing this
//! module, because every `impl CudaDtype` here is concrete on a
//! primitive numeric type (or `half::{f16,bf16}` under the existing
//! `f16` cargo feature).
//!
//! Runtime maps:
//! - [`CudaDtype::cuda_data_type`] returns the cuDNN/cuBLAS
//!   `cudaDataType_t` enum value.
//! - [`CudaDtype::cublas_compute_type`] returns the matching
//!   `cublasComputeType_t` for that operand type. Used by the
//!   ex-suffix entry points (`cublasGemmEx`, `cublasGemmStridedBatchedEx`,
//!   `cublasAxpyEx`, `cublasScalEx`, `cublasNrm2Ex`, `cublasDotEx`,
//!   `cublasIamaxEx`, `cublasIaminEx`, `cublasAsumEx`, `cublasCopyEx`,
//!   `cublasSwapEx`, `cublasRotEx`).
//!
//! Capability markers gate ops at compile time:
//! - [`GemmSupported`] — every dtype cuBLAS gemm accepts (f32, f64,
//!   f16, bf16; fp8 lights up with the `cublas-fp8` feature).
//! - [`AxpyDotNrm2Supported`] — ex-suffix L1 ops accept the same
//!   dtypes as gemm minus integer types.
//! - [`TrsmSupported`] — cuBLAS `trsm` is f32/f64-only (f16/bf16 not
//!   supported by hardware).
//! - [`SyrkSupported`], [`GeamSupported`], [`GemvSupported`],
//!   [`GerSupported`] — the rest of the L2/L3 surface, wider than
//!   trsm (f32/f64), narrower than gemm (no fp8 path).

use cudarc::cublas::sys::{cublasComputeType_t, cudaDataType_t};

/// Local equivalent of the future backend-agnostic `AccelDtype`. Every
/// type that flows through atomr-accel-cuda's typed message API
/// implements this — primitive numeric types, plus `half::f16` /
/// `half::bf16` under the `f16` cargo feature.
pub trait CudaDtype: Copy + Send + Sync + 'static + std::fmt::Debug {
    /// The accumulator scalar used by ex-suffix cuBLAS entry points.
    /// For f32/f64 it equals `Self`; for f16/bf16/fp8 it widens to
    /// `f32` so callers can pass alpha/beta in the higher precision
    /// cuBLAS expects.
    type Scalar: Copy + Send + Sync + 'static + std::fmt::Debug;

    /// `cudaDataType_t` enum value matching `Self`.
    fn cuda_data_type() -> cudaDataType_t;

    /// `cublasComputeType_t` matching the natural compute precision
    /// for this operand type. Used in `cublasGemmEx` /
    /// `cublasGemmStridedBatchedEx` calls.
    fn cublas_compute_type() -> cublasComputeType_t;

    /// Stable string name (e.g. `"f32"`, `"f16"`, `"bf16"`,
    /// `"f8e4m3"`). Used in error messages and trace events.
    fn name() -> &'static str;
}

/// Marker: cuBLAS gemm accepts this dtype. Implemented for every
/// `CudaDtype` we ship.
pub trait GemmSupported: CudaDtype {}

/// Marker: cuBLAS ex-suffix L1 ops (axpy/dot/nrm2/scal/asum/iamax/iamin/
/// copy/swap/rot) accept this dtype.
pub trait AxpyDotNrm2Supported: CudaDtype {}

/// Marker: cuBLAS `trsm` accepts this dtype. f32/f64 only — f16/bf16
/// are not hardware-supported here.
pub trait TrsmSupported: CudaDtype {}

/// Marker: cuBLAS `syrk` accepts this dtype.
pub trait SyrkSupported: CudaDtype {}

/// Marker: cuBLAS `geam` accepts this dtype. f32/f64 in the cudarc
/// 0.19 sys layer.
pub trait GeamSupported: CudaDtype {}

/// Marker: cuBLAS `gemv` accepts this dtype. f32/f64 via cudarc's safe
/// `Gemv<T>`; f16/bf16 callers should use a gemm with `m=1`.
pub trait GemvSupported: CudaDtype {}

/// Marker: cuBLAS `ger` accepts this dtype. f32/f64.
pub trait GerSupported: CudaDtype {}

// ───────────────────────── primitive impls ─────────────────────────

impl CudaDtype for f32 {
    type Scalar = f32;
    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_32F
    }
    fn cublas_compute_type() -> cublasComputeType_t {
        cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
    fn name() -> &'static str {
        "f32"
    }
}
impl GemmSupported for f32 {}
impl AxpyDotNrm2Supported for f32 {}
impl TrsmSupported for f32 {}
impl SyrkSupported for f32 {}
impl GeamSupported for f32 {}
impl GemvSupported for f32 {}
impl GerSupported for f32 {}

impl CudaDtype for f64 {
    type Scalar = f64;
    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_64F
    }
    fn cublas_compute_type() -> cublasComputeType_t {
        cublasComputeType_t::CUBLAS_COMPUTE_64F
    }
    fn name() -> &'static str {
        "f64"
    }
}
impl GemmSupported for f64 {}
impl AxpyDotNrm2Supported for f64 {}
impl TrsmSupported for f64 {}
impl SyrkSupported for f64 {}
impl GeamSupported for f64 {}
impl GemvSupported for f64 {}
impl GerSupported for f64 {}

#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    type Scalar = f32;
    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_16F
    }
    fn cublas_compute_type() -> cublasComputeType_t {
        cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
    fn name() -> &'static str {
        "f16"
    }
}
#[cfg(feature = "f16")]
impl GemmSupported for half::f16 {}
#[cfg(feature = "f16")]
impl AxpyDotNrm2Supported for half::f16 {}

#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    type Scalar = f32;
    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_16BF
    }
    fn cublas_compute_type() -> cublasComputeType_t {
        cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
    fn name() -> &'static str {
        "bf16"
    }
}
#[cfg(feature = "f16")]
impl GemmSupported for half::bf16 {}
#[cfg(feature = "f16")]
impl AxpyDotNrm2Supported for half::bf16 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_dtypes_have_distinct_runtime_codes() {
        assert_ne!(f32::cuda_data_type(), f64::cuda_data_type());
        assert_eq!(f32::name(), "f32");
        assert_eq!(f64::name(), "f64");
    }

    #[test]
    fn primitive_compute_types_match_natural_precision() {
        assert_eq!(
            f32::cublas_compute_type(),
            cublasComputeType_t::CUBLAS_COMPUTE_32F
        );
        assert_eq!(
            f64::cublas_compute_type(),
            cublasComputeType_t::CUBLAS_COMPUTE_64F
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_and_bf16_widen_alpha_to_f32() {
        // The Scalar associated type is the alpha/beta type the user
        // hands to cuBLAS-Ex entry points; for half-precision operand
        // types it is f32.
        fn assert_scalar_is_f32<T: CudaDtype<Scalar = f32>>() {}
        assert_scalar_is_f32::<half::f16>();
        assert_scalar_is_f32::<half::bf16>();
    }
}
