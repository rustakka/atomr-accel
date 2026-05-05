//! Dtype traits used by dtype-generic actor messages.
//!
//! This is a *minimal* slice of the Phase 0.1 `AccelDtype` design that
//! lives on the backend-agnostic core crate. The cuFFT actor only
//! needs to identify a dtype kind at runtime (for the plan-cache key)
//! and to gate ops at compile time via marker traits — the full
//! cuda/cublas/cudnn data-type plumbing is deliberately deferred to
//! the broader Phase 0 PR train.
//!
//! Once the umbrella `atomr-accel/src/dtype.rs` lands, this module
//! becomes a thin re-export of the core trait plus CUDA-specific
//! capability markers; consumers are insulated by the typed marker
//! traits.
//!
//! # Capability markers
//!
//! - [`FftSupported`] — implemented for every dtype that can flow
//!   through `cufftExec*` calls: `f32`, `f64`, and (under the `f16`
//!   feature) `half::f16`. Per the cuFFT documentation the f16 path
//!   only supports power-of-two sizes through `cufftXtMakePlanMany`,
//!   but the marker itself is dtype-only.

use std::fmt::Debug;

/// Concrete dtype kinds used by plan-cache keys and runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    I32,
    I64,
    U8,
    U32,
    U64,
}

impl DType {
    /// Size in bytes of a single scalar element of this dtype.
    pub const fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F16 | DType::Bf16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }

    /// Static name for tracing / error messages.
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::U64 => "u64",
        }
    }
}

/// Minimal CUDA-side dtype trait. The full Phase 0.1 surface adds
/// `cuda_data_type()`, `cublas_compute_type()`, etc.; for the cuFFT
/// slice we only need a runtime [`DType`] tag.
pub trait CudaDtype: Copy + Send + Sync + 'static + Debug {
    const KIND: DType;
    const SIZE: usize = Self::KIND.size();
    const NAME: &'static str = Self::KIND.name();
}

impl CudaDtype for f32 {
    const KIND: DType = DType::F32;
}
impl CudaDtype for f64 {
    const KIND: DType = DType::F64;
}
impl CudaDtype for i8 {
    const KIND: DType = DType::I8;
}
impl CudaDtype for i32 {
    const KIND: DType = DType::I32;
}
impl CudaDtype for i64 {
    const KIND: DType = DType::I64;
}
impl CudaDtype for u8 {
    const KIND: DType = DType::U8;
}
impl CudaDtype for u32 {
    const KIND: DType = DType::U32;
}
impl CudaDtype for u64 {
    const KIND: DType = DType::U64;
}

#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    const KIND: DType = DType::F16;
}
#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    const KIND: DType = DType::Bf16;
}

/// Marker trait: dtypes that cuFFT can transform.
///
/// cuFFT supports `f32` (R2C/C2R/C2C), `f64` (D2Z/Z2D/Z2Z), and `f16`
/// through the `cufftXtMakePlanMany` half-precision path (sizes must
/// be powers of two; this restriction is enforced at plan-creation
/// time, not by the marker).
pub trait FftSupported: CudaDtype {}

impl FftSupported for f32 {}
impl FftSupported for f64 {}
#[cfg(feature = "f16")]
impl FftSupported for half::f16 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_kind_and_size() {
        assert_eq!(<f32 as CudaDtype>::KIND, DType::F32);
        assert_eq!(<f32 as CudaDtype>::SIZE, 4);
        assert_eq!(<f64 as CudaDtype>::KIND, DType::F64);
        assert_eq!(<f64 as CudaDtype>::SIZE, 8);
    }

    #[test]
    fn dtype_name_round_trip() {
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F64.name(), "f64");
        assert_eq!(DType::F16.name(), "f16");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_dtype_kind() {
        assert_eq!(<half::f16 as CudaDtype>::KIND, DType::F16);
        assert_eq!(<half::bf16 as CudaDtype>::KIND, DType::Bf16);
    }
}
