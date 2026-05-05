//! Capability markers used by per-library dispatch traits to
//! constrain the dtype set each cuRAND/cuBLAS/cuDNN/etc. entry point
//! actually supports.
//!
//! The traits are deliberately *empty* — they exist purely as
//! type-level proofs, so a generic `RngDispatch<T: RngFloatSupported>`
//! can be parameterised over `f32` / `f64` only and reject `i32`,
//! `u8`, etc. at compile time.

/// Common parent: any element type that has a stable CUDA on-device
/// representation. Implemented for the numeric primitives the rest of
/// `atomr-accel-cuda` allocates via `DeviceMsg::Allocate*`.
pub trait CudaDtype: Copy + Send + Sync + 'static {}

impl CudaDtype for f32 {}
impl CudaDtype for f64 {}
impl CudaDtype for u32 {}
impl CudaDtype for u64 {}
impl CudaDtype for i8 {}
impl CudaDtype for i32 {}
impl CudaDtype for i64 {}
impl CudaDtype for u8 {}
#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {}
#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {}

/// Marker: cuRAND host-API `fill_with_uniform / fill_with_normal /
/// fill_with_log_normal` accept this float type.
pub trait RngFloatSupported: CudaDtype {
    /// Scalar form used for distribution parameters (mean, std, lo,
    /// hi, …). For `f32` and `f64` it is the type itself.
    type Scalar: Copy + Send + Sync + 'static + std::fmt::Debug;
}

impl RngFloatSupported for f32 {
    type Scalar = f32;
}
impl RngFloatSupported for f64 {
    type Scalar = f64;
}

/// Marker: cuRAND host-API supports an integer fill of this width
/// (`curandGenerate` for u32, `curandGenerateLongLong` for u64). Used
/// by the `Discrete` and raw-bit paths.
pub trait RngIntSupported: CudaDtype {}

impl RngIntSupported for u32 {}
impl RngIntSupported for u64 {}
