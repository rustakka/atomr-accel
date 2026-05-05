//! Compile-time dtype + capability traits for CUDA library actors
//! (Phase 0.1 of the CUDA-coverage roadmap).
//!
//! This module exposes:
//!
//! * [`CudaDtype`] — minimal mapping from a Rust scalar type to its
//!   cudarc / cuDNN data-type enum and its scaling-parameter type
//!   (e.g. `f16` accumulates against `f32` for cuDNN scaling).
//! * Capability marker traits gating per-library ops at compile time:
//!   * [`CudnnSupported`] — dtypes cuDNN can run on.
//!
//! Other parallel Phase 2 actors (NCCL, cuTENSOR) own their own marker
//! traits in their own files. The capability traits in this file are
//! the ones touched by the cuDNN slice.

use std::fmt::Debug;

use cudarc::cublas::sys::cudaDataType_t;

/// Common dtype trait: every type usable as a tensor element on CUDA
/// implements this.
pub trait CudaDtype: Copy + Send + Sync + Debug + 'static {
    /// Element type used for `alpha` / `beta` scaling parameters in
    /// most cuBLAS / cuDNN APIs. For `f32`/`f64` this is `Self`. For
    /// half-precision (`f16`/`bf16`) it is `f32` per cuDNN's scaling
    /// rules.
    type Scalar: Copy + Send + Sync + Debug + 'static;

    /// Cudarc / CUDA driver data-type enum for this scalar (used by
    /// cuBLAS/cuTENSOR/cuFFT generic APIs).
    const CUDA_DATA_TYPE: cudaDataType_t;

    /// Static name (for logs, errors, debug).
    const NAME: &'static str;

    /// Size in bytes.
    const SIZE: usize;

    /// cuDNN data-type enum for this scalar. Always present (cuDNN
    /// supports every scalar this trait is implemented for) but gated
    /// behind the `cudnn` cudarc feature so the symbol is resolvable.
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudarc::cudnn::sys::cudnnDataType_t;
}

/// Capability marker: this dtype is acceptable to cuDNN tensor ops
/// (conv, pool, norm, attention, RNN, activation, …).
///
/// Implemented for `f32`, `f64`, `f16`, `bf16`, `i8`. Implementation
/// of [`CudaDtype`] is required.
pub trait CudnnSupported: CudaDtype {}

// ----- Implementations --------------------------------------------------

impl CudaDtype for f32 {
    type Scalar = f32;
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_32F;
    const NAME: &'static str = "f32";
    const SIZE: usize = 4;
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudarc::cudnn::sys::cudnnDataType_t {
        cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_FLOAT
    }
}
impl CudnnSupported for f32 {}

impl CudaDtype for f64 {
    type Scalar = f64;
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_64F;
    const NAME: &'static str = "f64";
    const SIZE: usize = 8;
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudarc::cudnn::sys::cudnnDataType_t {
        cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_DOUBLE
    }
}
impl CudnnSupported for f64 {}

impl CudaDtype for i8 {
    type Scalar = f32;
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_8I;
    const NAME: &'static str = "i8";
    const SIZE: usize = 1;
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudarc::cudnn::sys::cudnnDataType_t {
        cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_INT8
    }
}
impl CudnnSupported for i8 {}

#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    type Scalar = f32;
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_16F;
    const NAME: &'static str = "f16";
    const SIZE: usize = 2;
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudarc::cudnn::sys::cudnnDataType_t {
        cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_HALF
    }
}
#[cfg(feature = "f16")]
impl CudnnSupported for half::f16 {}

#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    type Scalar = f32;
    const CUDA_DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_16BF;
    const NAME: &'static str = "bf16";
    const SIZE: usize = 2;
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudarc::cudnn::sys::cudnnDataType_t {
        cudarc::cudnn::sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16
    }
}
#[cfg(feature = "f16")]
impl CudnnSupported for half::bf16 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_dtype_basic_facts() {
        assert_eq!(<f32 as CudaDtype>::NAME, "f32");
        assert_eq!(<f64 as CudaDtype>::SIZE, 8);
        assert_eq!(<i8 as CudaDtype>::SIZE, 1);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn half_dtypes_have_f32_scalar() {
        fn requires_f32_scalar<T: CudaDtype<Scalar = f32>>() {}
        requires_f32_scalar::<half::f16>();
        requires_f32_scalar::<half::bf16>();
    }

    #[cfg(feature = "cudnn")]
    #[test]
    fn cudnn_data_types_resolve() {
        let _ = <f32 as CudaDtype>::cudnn_data_type();
        let _ = <f64 as CudaDtype>::cudnn_data_type();
        let _ = <i8 as CudaDtype>::cudnn_data_type();
    }
}
