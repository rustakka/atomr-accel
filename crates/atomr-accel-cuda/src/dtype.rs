//! Phase 0.1 / 0.2 dtype foundation — minimal surface needed by Phase 4.
//!
//! The full `AccelDtype` story (cudnn/fft/rng/nccl marker traits, fp8/fp4
//! newtypes, `cublas_compute_type`, …) ships with the canonical Phase 0
//! land. This file ships **just enough** for Phase 4 — cuSPARSE +
//! cuSPARSELt — to compile and dispatch correctly across f32/f64/f16/bf16
//! on the value side and i32/i64 on the index side.
//!
//! Conventions:
//! * [`AccelDtype`] lives here (not in `atomr-accel`) until Phase 0 lifts
//!   it to the backend-agnostic core crate. Phase 4 sees only the trait
//!   bound, not its location, so the migration is a re-export-and-rename.
//! * [`CudaDtype::cuda_data_type`] surfaces the `cudaDataType_t` mapping
//!   every cuSPARSE generic-API entry point needs — gated behind
//!   `feature = "cusparse"` because that is where the type lives.
//! * Capability marker traits ([`SparseSupported`], [`SparseIndex`]) are
//!   blanket-extending sub-traits that gate ops at compile time. A type
//!   that doesn't implement `SparseIndex` cannot be used as the
//!   `row_offsets` / `col_indices` element type of a
//!   [`crate::kernel::sparse::SparseMatrix`].

use std::fmt::Debug;

/// Backend-agnostic dtype kinds. The `non_exhaustive` lets us add fp8/fp4
/// in Phase 0 without breaking match patterns in callers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// Backend-agnostic numeric-type capability. Phase 0's full version adds
/// `Scalar`, `nan()`, etc.; Phase 4 only needs `KIND` + `NAME` + `zero` +
/// `one` for plumbing tests.
pub trait AccelDtype: Copy + Send + Sync + Debug + 'static {
    /// Host-side scalar used in `alpha`/`beta` slots. For real numeric
    /// types this is `Self`; in Phase 0 the fp8 newtypes will widen to
    /// `f32`.
    type Scalar: Copy + Send + Sync + Debug + 'static;

    const KIND: DType;
    const SIZE: usize;
    const NAME: &'static str;

    fn zero() -> Self;
    fn one() -> Self;
}

/// CUDA-specific dtype mapping. Phase 0 expands this with cublas/cudnn
/// helpers. Phase 4 only needs `cuda_data_type` (cuSPARSE generic API).
///
/// `cuda_data_type` is gated on `feature = "cusparse"` because cudarc
/// 0.19.4 only exposes `cudaDataType_t` from inside the cusparse module
/// tree. The Phase 0 land will lift this to a feature-agnostic location.
pub trait CudaDtype: AccelDtype {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t;
}

/// Capability marker — `T` is supported as the **value** dtype of a
/// cuSPARSE sparse matrix. Implemented for `f32`, `f64`, and (under
/// `feature = "f16"`) `half::f16` / `half::bf16`.
pub trait SparseSupported: CudaDtype {}

/// Capability marker — `I` is supported as a cuSPARSE **index** dtype.
/// Implemented for `i32` and `i64` only. Any attempt to construct a
/// `SparseMatrix<_, u32>` (or any other type) fails to compile because
/// `u32: SparseIndex` is not provided.
pub trait SparseIndex: AccelDtype {
    #[cfg(feature = "cusparse")]
    fn cusparse_index_type() -> cudarc::cusparse::sys::cusparseIndexType_t;
}

// -- f32 ----------------------------------------------------------------

impl AccelDtype for f32 {
    type Scalar = f32;
    const KIND: DType = DType::F32;
    const SIZE: usize = 4;
    const NAME: &'static str = "f32";
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}
impl CudaDtype for f32 {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t {
        cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F
    }
}
impl SparseSupported for f32 {}

// -- f64 ----------------------------------------------------------------

impl AccelDtype for f64 {
    type Scalar = f64;
    const KIND: DType = DType::F64;
    const SIZE: usize = 8;
    const NAME: &'static str = "f64";
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}
impl CudaDtype for f64 {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t {
        cudarc::cusparse::sys::cudaDataType_t::CUDA_R_64F
    }
}
impl SparseSupported for f64 {}

// -- f16 / bf16 (gated) -------------------------------------------------

#[cfg(feature = "f16")]
impl AccelDtype for half::f16 {
    type Scalar = half::f16;
    const KIND: DType = DType::F16;
    const SIZE: usize = 2;
    const NAME: &'static str = "f16";
    fn zero() -> Self {
        half::f16::from_f32(0.0)
    }
    fn one() -> Self {
        half::f16::from_f32(1.0)
    }
}
#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t {
        cudarc::cusparse::sys::cudaDataType_t::CUDA_R_16F
    }
}
#[cfg(feature = "f16")]
impl SparseSupported for half::f16 {}

#[cfg(feature = "f16")]
impl AccelDtype for half::bf16 {
    type Scalar = half::bf16;
    const KIND: DType = DType::Bf16;
    const SIZE: usize = 2;
    const NAME: &'static str = "bf16";
    fn zero() -> Self {
        half::bf16::from_f32(0.0)
    }
    fn one() -> Self {
        half::bf16::from_f32(1.0)
    }
}
#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t {
        cudarc::cusparse::sys::cudaDataType_t::CUDA_R_16BF
    }
}
#[cfg(feature = "f16")]
impl SparseSupported for half::bf16 {}

// -- i32 / i64 ----------------------------------------------------------

impl AccelDtype for i32 {
    type Scalar = i32;
    const KIND: DType = DType::I32;
    const SIZE: usize = 4;
    const NAME: &'static str = "i32";
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl CudaDtype for i32 {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t {
        cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32I
    }
}
impl SparseIndex for i32 {
    #[cfg(feature = "cusparse")]
    fn cusparse_index_type() -> cudarc::cusparse::sys::cusparseIndexType_t {
        cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I
    }
}

impl AccelDtype for i64 {
    type Scalar = i64;
    const KIND: DType = DType::I64;
    const SIZE: usize = 8;
    const NAME: &'static str = "i64";
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl CudaDtype for i64 {
    #[cfg(feature = "cusparse")]
    fn cuda_data_type() -> cudarc::cusparse::sys::cudaDataType_t {
        cudarc::cusparse::sys::cudaDataType_t::CUDA_R_64I
    }
}
impl SparseIndex for i64 {
    #[cfg(feature = "cusparse")]
    fn cusparse_index_type() -> cudarc::cusparse::sys::cusparseIndexType_t {
        cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_64I
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_supported_value_dtypes() {
        assert_eq!(<f32 as AccelDtype>::KIND, DType::F32);
        assert_eq!(<f64 as AccelDtype>::KIND, DType::F64);
        assert_eq!(<f32 as AccelDtype>::SIZE, 4);
        assert_eq!(<f64 as AccelDtype>::SIZE, 8);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn sparse_supported_half_dtypes() {
        assert_eq!(<half::f16 as AccelDtype>::KIND, DType::F16);
        assert_eq!(<half::bf16 as AccelDtype>::KIND, DType::Bf16);
    }

    #[test]
    fn sparse_index_admits_only_i32_i64() {
        // Compile-time: u8/u32 do NOT impl SparseIndex; the only way to
        // observe that here is by exercising the marker on the admitted
        // types and trusting `compile_fail` doctests for negative cases.
        fn assert_index<I: SparseIndex>() {}
        assert_index::<i32>();
        assert_index::<i64>();
    }

    /// The whole point of [`SparseIndex`] is to reject non-{i32,i64} index
    /// types at compile time. This doctest fails to compile, which the
    /// rustdoc test harness verifies.
    ///
    /// ```compile_fail
    /// use atomr_accel_cuda::dtype::SparseIndex;
    /// fn assert_index<I: SparseIndex>() {}
    /// assert_index::<u8>();
    /// ```
    #[allow(dead_code)]
    fn _doc_compile_fail_anchor() {}
}
