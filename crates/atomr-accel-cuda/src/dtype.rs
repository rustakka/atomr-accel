//! `CudaDtype` + capability marker traits (Phase 0.2 minimal slice).
//!
//! This is the dtype-generic backbone the kernel-level `*Dispatch`
//! traits require. The full `AccelDtype` trait lives in the
//! backend-agnostic `atomr-accel` core crate; this module provides the
//! CUDA-side mappings and the per-op capability markers
//! (`GemmSupported`, `CudnnSupported`, …) that gate which dtypes can
//! flow into which library actor at compile time.
//!
//! The design intentionally keeps the trait surface small: every op
//! site writes `T: GemmSupported` (or whichever capability marker it
//! needs) and gets `cuda_data_type()` + `cublas_compute_type()` for
//! free. New dtypes (fp8 e4m3/e5m2, fp4 e2m1) add a single `impl`
//! block here without touching message enums.

use cudarc::cublaslt::sys as cublaslt_sys;
// cudarc 0.19.4 does not re-export `cudaDataType_t` from
// `cudarc::driver::sys`. The cuBLAS / cuBLASLt sys modules each
// re-export their own copy. We canonicalize on `cublaslt_sys` since
// every dtype-aware op site in this crate already imports from it.
use cudarc::cublaslt::sys::cudaDataType_t;

/// Stable kind tag for a CUDA-supported scalar dtype. Used as the
/// dtype-axis of cache keys (heuristic cache, plan cache, …) without
/// pulling cudarc's `cudaDataType_t` (which is a bindgen enum that
/// changes between CUDA versions) into the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum DTypeKind {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    Bf16 = 3,
    I8 = 4,
    I32 = 5,
    U8 = 6,
    F8E4m3 = 7,
    F8E5m2 = 8,
}

/// CUDA-side dtype trait. Every numeric scalar that can flow through a
/// kernel actor implements this.
pub trait CudaDtype: Copy + Send + Sync + 'static + std::fmt::Debug {
    /// Scalar type used for `alpha`/`beta`/scale arguments. For most
    /// dtypes this is `Self`; for fp8/fp4 wrappers it's `f32` because
    /// cuBLASLt expects f32 scaling factors.
    type Scalar: Copy + Send + Sync + 'static;

    /// Stable cache-key tag (independent of cudarc/CUDA version).
    const KIND: DTypeKind;

    /// Element size in bytes.
    const SIZE: usize;

    /// Human-readable name (used in error messages, telemetry).
    const NAME: &'static str;

    /// cudarc-bindgen `cudaDataType_t` for this dtype.
    fn cuda_data_type() -> cudaDataType_t;

    /// Default cuBLAS compute type for matmul/gemm with this input
    /// dtype. Callers can override per-op via the request struct.
    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t;
}

/// Marker: this dtype is accepted by cuBLAS / cuBLASLt gemm.
pub trait GemmSupported: CudaDtype {}

// ---- f32 ----------------------------------------------------------

impl CudaDtype for f32 {
    type Scalar = f32;
    const KIND: DTypeKind = DTypeKind::F32;
    const SIZE: usize = 4;
    const NAME: &'static str = "f32";

    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_32F
    }

    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t {
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    }
}
impl GemmSupported for f32 {}

// ---- f64 ----------------------------------------------------------

impl CudaDtype for f64 {
    type Scalar = f64;
    const KIND: DTypeKind = DTypeKind::F64;
    const SIZE: usize = 8;
    const NAME: &'static str = "f64";

    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_64F
    }

    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t {
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_64F
    }
}
impl GemmSupported for f64 {}

// ---- half / bf16 (feature `f16`) ---------------------------------

#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    type Scalar = f32;
    const KIND: DTypeKind = DTypeKind::F16;
    const SIZE: usize = 2;
    const NAME: &'static str = "f16";

    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_16F
    }

    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t {
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}
#[cfg(feature = "f16")]
impl GemmSupported for half::f16 {}

#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    type Scalar = f32;
    const KIND: DTypeKind = DTypeKind::Bf16;
    const SIZE: usize = 2;
    const NAME: &'static str = "bf16";

    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_16BF
    }

    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t {
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}
#[cfg(feature = "f16")]
impl GemmSupported for half::bf16 {}

// ---- fp8 (feature `cublas-fp8`) ----------------------------------
//
// Newtype wrappers around `u8` so we never accidentally arithmetic on
// raw fp8 bits. The actual bit-level conversion to/from `f32` happens
// on the GPU via cuBLASLt's per-tensor scale pointers; on the host
// these types are pure storage tags.

/// Storage type for `__nv_fp8_e4m3` (range-optimized fp8).
#[cfg(feature = "cublas-fp8")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F8E4m3(pub u8);

#[cfg(feature = "cublas-fp8")]
impl CudaDtype for F8E4m3 {
    type Scalar = f32;
    const KIND: DTypeKind = DTypeKind::F8E4m3;
    const SIZE: usize = 1;
    const NAME: &'static str = "f8e4m3";

    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_8F_E4M3
    }

    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t {
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}
#[cfg(feature = "cublas-fp8")]
impl GemmSupported for F8E4m3 {}

/// Storage type for `__nv_fp8_e5m2` (precision-optimized fp8).
#[cfg(feature = "cublas-fp8")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F8E5m2(pub u8);

#[cfg(feature = "cublas-fp8")]
impl CudaDtype for F8E5m2 {
    type Scalar = f32;
    const KIND: DTypeKind = DTypeKind::F8E5m2;
    const SIZE: usize = 1;
    const NAME: &'static str = "f8e5m2";

    fn cuda_data_type() -> cudaDataType_t {
        cudaDataType_t::CUDA_R_8F_E5M2
    }

    fn cublas_compute_type() -> cublaslt_sys::cublasComputeType_t {
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}
#[cfg(feature = "cublas-fp8")]
impl GemmSupported for F8E5m2 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_dtype_kind_and_size() {
        assert_eq!(<f32 as CudaDtype>::KIND, DTypeKind::F32);
        assert_eq!(<f32 as CudaDtype>::SIZE, 4);
        assert_eq!(<f32 as CudaDtype>::NAME, "f32");
    }

    #[test]
    fn f64_dtype_kind_and_size() {
        assert_eq!(<f64 as CudaDtype>::KIND, DTypeKind::F64);
        assert_eq!(<f64 as CudaDtype>::SIZE, 8);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_bf16_dtype_kinds() {
        assert_eq!(<half::f16 as CudaDtype>::KIND, DTypeKind::F16);
        assert_eq!(<half::bf16 as CudaDtype>::KIND, DTypeKind::Bf16);
    }

    #[cfg(feature = "cublas-fp8")]
    #[test]
    fn fp8_dtype_kinds() {
        assert_eq!(<F8E4m3 as CudaDtype>::KIND, DTypeKind::F8E4m3);
        assert_eq!(<F8E5m2 as CudaDtype>::KIND, DTypeKind::F8E5m2);
        assert_eq!(<F8E4m3 as CudaDtype>::SIZE, 1);
    }
}
