//! `CudaDtype` — CUDA-side dtype mappings and capability markers.
//!
//! The backend-agnostic [`AccelDtype`] trait (in `atomr-accel`) names
//! the dtype and gives identity values. `CudaDtype` adds the cudarc-
//! enum mappings every kernel actor needs:
//!
//! - [`cuda_data_type`](CudaDtype::cuda_data_type) — `cudaDataType_t`
//!   (consumed by cuBLAS, cuBLASLt, cuSPARSE, cuSOLVER, cuTENSOR).
//! - [`cublas_compute_type`](CudaDtype::cublas_compute_type) — the
//!   natural `cublasComputeType_t` for matmul accumulation.
//! - [`cudnn_data_type`](CudaDtype::cudnn_data_type) — `cudnnDataType_t`
//!   (cuDNN tensor descriptor element type), gated on `cudnn`.
//! - [`nccl_data_type`](CudaDtype::nccl_data_type) — `ncclDataType_t`
//!   (collective-op element type), gated on `nccl`.
//! - [`cuda_type_name`](CudaDtype::cuda_type_name) — CUDA C++ type
//!   name (`"float"`, `"__half"`, `"__nv_bfloat16"`, …) for NVRTC
//!   kernel source generation.
//!
//! Capability markers ([`GemmSupported`], [`CudnnSupported`], …) are
//! the compile-time gate keeping operations from being dispatched
//! against unsupported dtypes — `BlasMsg::gemm::<i64>(...)` does not
//! compile because `i64: GemmSupported` has no impl.

use cudarc::cublas::sys as cublas_sys;
#[cfg(feature = "cudnn")]
use cudarc::cudnn::sys as cudnn_sys;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
#[cfg(feature = "nccl")]
use cudarc::nccl::sys as nccl_sys;

/// Re-export `atomr_accel::DType` so existing `crate::dtype::DType`
/// imports inside `atomr-accel-cuda` (added by Phase 0.4) keep working
/// without changing every call site.
pub use atomr_accel::DType;

/// Re-export so `crate::dtype::AccelDtype` resolves for actor modules
/// that prefer the unified import path.
pub use atomr_accel::AccelDtype;

/// Alias used by `BlasLtDispatch::dtype_kind` and other Phase 1 dispatchers.
pub use atomr_accel::DType as DTypeKind;

/// Local fp8 / fp4 wrappers (`#[repr(transparent)]` over `u8`) that
/// satisfy cudarc's orphan-rule constraint for `unsafe impl DeviceRepr`.
/// Convertible from/to the backend-agnostic `atomr_accel::dtype::*`
/// equivalents.
/// 64-bit interleaved complex (`{re, im}` of f32). Layout matches
/// `cufft_sys::float2` and `numpy.complex64`. Phase 1.5++.
///
/// `#[repr(transparent)]` over `[f32; 2]` so `unsafe impl DeviceRepr`
/// is sound (cudarc's orphan rule blocks blanket impls on tuple-struct
/// shapes from foreign crates) and so transmutes from `Vec<C32>` to /
/// from `Vec<num_complex::Complex<f32>>` (also `#[repr(C)]` with two
/// `f32` fields) are layout-safe.
///
/// Maps to `cudaDataType_t::CUDA_C_32F` and `cuComplex` in CUDA C++.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct C32(pub [f32; 2]);

impl C32 {
    /// Construct from real / imaginary components.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self([re, im])
    }
    #[inline]
    pub fn re(self) -> f32 {
        self.0[0]
    }
    #[inline]
    pub fn im(self) -> f32 {
        self.0[1]
    }
}

/// 128-bit interleaved complex (`{re, im}` of f64). Layout matches
/// `cufft_sys::double2` and `numpy.complex128`. Phase 1.5++.
///
/// Maps to `cudaDataType_t::CUDA_C_64F` and `cuDoubleComplex`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct C64(pub [f64; 2]);

impl C64 {
    /// Construct from real / imaginary components.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self([re, im])
    }
    #[inline]
    pub fn re(self) -> f64 {
        self.0[0]
    }
    #[inline]
    pub fn im(self) -> f64 {
        self.0[1]
    }
}

// Layout bridges to the cudarc cuFFT FFI structs. `cufft_sys::float2`
// is `#[repr(C)] #[repr(align(8))] { x: f32, y: f32 }` — slightly
// stricter alignment than `[f32; 2]` (align 4). The `From` impl is a
// field-by-field copy, not a transmute, so the alignment mismatch is
// harmless on the value side. (Transmuting `*const float2` → `*const
// C32` is _not_ sound because the alignment shrinks; callers needing
// a pointer-level bridge should keep `cufft_sys::float2` typed.)
//
// Gated on the `cufft` cargo feature because the cudarc::cufft module
// is itself feature-gated.
#[cfg(feature = "cufft")]
impl From<cudarc::cufft::sys::float2> for C32 {
    #[inline]
    fn from(v: cudarc::cufft::sys::float2) -> Self {
        C32([v.x, v.y])
    }
}

#[cfg(feature = "cufft")]
impl From<C32> for cudarc::cufft::sys::float2 {
    #[inline]
    fn from(v: C32) -> Self {
        cudarc::cufft::sys::float2 {
            x: v.0[0],
            y: v.0[1],
        }
    }
}

#[cfg(feature = "cufft")]
impl From<cudarc::cufft::sys::double2> for C64 {
    #[inline]
    fn from(v: cudarc::cufft::sys::double2) -> Self {
        C64([v.x, v.y])
    }
}

#[cfg(feature = "cufft")]
impl From<C64> for cudarc::cufft::sys::double2 {
    #[inline]
    fn from(v: C64) -> Self {
        cudarc::cufft::sys::double2 {
            x: v.0[0],
            y: v.0[1],
        }
    }
}

// SAFETY: `C32` / `C64` are `#[repr(transparent)]` over `[f32; 2]` /
// `[f64; 2]`, both of which are POD; cudarc allows arbitrary `Copy +
// 'static` over device-mappable bit patterns to be `DeviceRepr`. All
// bit patterns of `f32` / `f64` (including NaN, inf, signaling NaN)
// are valid floats, so `ValidAsZeroBits` is sound — the all-zeros
// pattern represents `+0.0 + 0.0i`.
unsafe impl DeviceRepr for C32 {}
unsafe impl ValidAsZeroBits for C32 {}
unsafe impl DeviceRepr for C64 {}
unsafe impl ValidAsZeroBits for C64 {}

// `AccelDtype` requires a `DType` discriminant. The base atomr-accel
// `DType` enum has no Complex variant; reuse the matching scalar lane
// (`F32` for C32, `F64` for C64) — the same convention `FftKind`'s
// `scalar_dtype()` already uses for cuFFT plan keys. Callers that
// need to distinguish complex from real branch on `T` directly, not
// on `KIND`.
impl atomr_accel::AccelDtype for C32 {
    type Scalar = f32;
    const KIND: DType = DType::F32;
    const SIZE: usize = 8;
    const NAME: &'static str = "complex64";
    #[inline]
    fn zero() -> Self {
        C32([0.0, 0.0])
    }
    #[inline]
    fn one() -> Self {
        C32([1.0, 0.0])
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(C32([f32::NAN, f32::NAN]))
    }
}

impl atomr_accel::AccelDtype for C64 {
    type Scalar = f64;
    const KIND: DType = DType::F64;
    const SIZE: usize = 16;
    const NAME: &'static str = "complex128";
    #[inline]
    fn zero() -> Self {
        C64([0.0, 0.0])
    }
    #[inline]
    fn one() -> Self {
        C64([1.0, 0.0])
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(C64([f64::NAN, f64::NAN]))
    }
}

#[cfg(feature = "f8")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F8E4m3(pub u8);

#[cfg(feature = "f8")]
impl F8E4m3 {
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self(atomr_accel::dtype::F8E4m3::from_f32(x).0)
    }
    #[inline]
    pub fn to_f32(self) -> f32 {
        atomr_accel::dtype::F8E4m3(self.0).to_f32()
    }
}

#[cfg(feature = "f8")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F8E5m2(pub u8);

#[cfg(feature = "f8")]
impl F8E5m2 {
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self(atomr_accel::dtype::F8E5m2::from_f32(x).0)
    }
    #[inline]
    pub fn to_f32(self) -> f32 {
        atomr_accel::dtype::F8E5m2(self.0).to_f32()
    }
}

#[cfg(feature = "f8")]
impl atomr_accel::AccelDtype for F8E4m3 {
    type Scalar = f32;
    const KIND: DType = DType::F8E4m3;
    const SIZE: usize = 1;
    const NAME: &'static str = "f8_e4m3";
    #[inline]
    fn zero() -> Self {
        F8E4m3(0x00)
    }
    #[inline]
    fn one() -> Self {
        F8E4m3(0x38)
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(F8E4m3(0x7f))
    }
}

#[cfg(feature = "f8")]
impl atomr_accel::AccelDtype for F8E5m2 {
    type Scalar = f32;
    const KIND: DType = DType::F8E5m2;
    const SIZE: usize = 1;
    const NAME: &'static str = "f8_e5m2";
    #[inline]
    fn zero() -> Self {
        F8E5m2(0x00)
    }
    #[inline]
    fn one() -> Self {
        F8E5m2(0x3c)
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(F8E5m2(0x7e))
    }
}

/// CUDA-specific layer over [`AccelDtype`].
///
/// The cudarc bounds (`DeviceRepr`, `ValidAsZeroBits`) are part of the
/// supertrait list so dispatch payloads can call `stream.alloc_zeros::<T>`
/// behind a single `T: CudaDtype` bound.
pub trait CudaDtype: AccelDtype + DeviceRepr + ValidAsZeroBits {
    fn cuda_data_type() -> cublas_sys::cudaDataType_t;
    fn cublas_compute_type() -> cublas_sys::cublasComputeType_t;
    /// CUDA C++ type name for NVRTC source generation.
    fn cuda_type_name() -> &'static str;
    #[cfg(feature = "cudnn")]
    fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t;
    #[cfg(feature = "nccl")]
    fn nccl_data_type() -> nccl_sys::ncclDataType_t;
}

/// Capability marker — type may be a cuBLAS GEMM operand.
pub trait GemmSupported: CudaDtype {}

/// Capability marker — type may be a cuDNN tensor element.
pub trait CudnnSupported: CudaDtype {}

/// Capability marker — type may be a cuFFT element.
pub trait FftSupported: CudaDtype {}

/// Capability marker — type may be a cuRAND distribution element
/// (`Self` is one of the float dtypes accepted by `curandGenerate*`).
pub trait RngFloatSupported: CudaDtype {}

/// Capability marker — type may be an NCCL collective-op element.
pub trait NcclReduceSupported: CudaDtype {}

/// Capability marker — type may be a cuSOLVER dense factorization
/// element (real or complex float).
pub trait SolverSupported: CudaDtype {}

/// Capability marker — type may be a cuSPARSE SpMV/SpMM/SpGEMM
/// element.
pub trait SparseSupported: CudaDtype {}

/// Capability marker — type may be a cuTENSOR contraction operand.
pub trait TensorSupported: CudaDtype {}

/// Capability marker — cuRAND integer-fill operand. `curandGenerate` produces u32,
/// `curandGenerateLongLong` produces u64. Used by `Discrete` and raw-bit paths.
pub trait RngIntSupported: CudaDtype {}

/// Phase 4 cuSPARSE index-type marker. Only `i32` and `i64` are
/// representable cuSPARSE row/col index dtypes.
pub trait SparseIndex: AccelDtype {
    #[cfg(feature = "cusparse")]
    fn cusparse_index_type() -> cudarc::cusparse::sys::cusparseIndexType_t;
}

#[cfg(feature = "cusparse")]
impl SparseIndex for i32 {
    fn cusparse_index_type() -> cudarc::cusparse::sys::cusparseIndexType_t {
        cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I
    }
}
#[cfg(feature = "cusparse")]
impl SparseIndex for i64 {
    fn cusparse_index_type() -> cudarc::cusparse::sys::cusparseIndexType_t {
        cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_64I
    }
}
#[cfg(not(feature = "cusparse"))]
impl SparseIndex for i32 {}
#[cfg(not(feature = "cusparse"))]
impl SparseIndex for i64 {}

// Phase 1 cuBLAS sub-marker traits (per-op dtype subsets).

pub trait AxpyDotNrm2Supported: CudaDtype {}
pub trait GemvSupported: CudaDtype {}
pub trait GerSupported: CudaDtype {}
pub trait GeamSupported: CudaDtype {}
pub trait SyrkSupported: CudaDtype {}
pub trait TrsmSupported: CudaDtype {}

macro_rules! impl_cuda_dtype {
    (
        $rust:ty,
        $cuda:ident,
        $compute:ident,
        $name:literal,
        cudnn: $cudnn:ident,
        nccl: $($nccl:ident)?
    ) => {
        impl CudaDtype for $rust {
            #[inline]
            fn cuda_data_type() -> cublas_sys::cudaDataType_t {
                cublas_sys::cudaDataType_t::$cuda
            }
            #[inline]
            fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
                cublas_sys::cublasComputeType_t::$compute
            }
            #[inline]
            fn cuda_type_name() -> &'static str { $name }
            #[cfg(feature = "cudnn")]
            #[inline]
            fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
                cudnn_sys::cudnnDataType_t::$cudnn
            }
            #[cfg(feature = "nccl")]
            #[inline]
            fn nccl_data_type() -> nccl_sys::ncclDataType_t {
                impl_cuda_dtype!(@nccl_arm $($nccl)?)
            }
        }
    };
    (@nccl_arm $variant:ident) => { nccl_sys::ncclDataType_t::$variant };
    (@nccl_arm) => { panic!("dtype not supported by NCCL") };
}

impl_cuda_dtype!(f32, CUDA_R_32F, CUBLAS_COMPUTE_32F, "float",
    cudnn: CUDNN_DATA_FLOAT, nccl: ncclFloat32);
impl_cuda_dtype!(f64, CUDA_R_64F, CUBLAS_COMPUTE_64F, "double",
    cudnn: CUDNN_DATA_DOUBLE, nccl: ncclFloat64);
impl_cuda_dtype!(i8, CUDA_R_8I, CUBLAS_COMPUTE_32I, "char",
    cudnn: CUDNN_DATA_INT8, nccl: ncclInt8);
impl_cuda_dtype!(u8, CUDA_R_8U, CUBLAS_COMPUTE_32I, "unsigned char",
    cudnn: CUDNN_DATA_UINT8, nccl: ncclUint8);
impl_cuda_dtype!(i32, CUDA_R_32I, CUBLAS_COMPUTE_32I, "int",
    cudnn: CUDNN_DATA_INT32, nccl: ncclInt32);
impl_cuda_dtype!(u32, CUDA_R_32U, CUBLAS_COMPUTE_32I, "unsigned int",
    cudnn: CUDNN_DATA_INT32, nccl: ncclUint32);
impl_cuda_dtype!(i64, CUDA_R_64I, CUBLAS_COMPUTE_32I, "long long",
    cudnn: CUDNN_DATA_INT64, nccl: ncclInt64);
impl_cuda_dtype!(u64, CUDA_R_64U, CUBLAS_COMPUTE_32I, "unsigned long long",
    cudnn: CUDNN_DATA_INT64, nccl: ncclUint64);

#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    #[inline]
    fn cuda_data_type() -> cublas_sys::cudaDataType_t {
        cublas_sys::cudaDataType_t::CUDA_R_16F
    }
    #[inline]
    fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
    #[inline]
    fn cuda_type_name() -> &'static str {
        "__half"
    }
    #[cfg(feature = "cudnn")]
    #[inline]
    fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
        cudnn_sys::cudnnDataType_t::CUDNN_DATA_HALF
    }
    #[cfg(feature = "nccl")]
    #[inline]
    fn nccl_data_type() -> nccl_sys::ncclDataType_t {
        nccl_sys::ncclDataType_t::ncclFloat16
    }
}

#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    #[inline]
    fn cuda_data_type() -> cublas_sys::cudaDataType_t {
        cublas_sys::cudaDataType_t::CUDA_R_16BF
    }
    #[inline]
    fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
    #[inline]
    fn cuda_type_name() -> &'static str {
        "__nv_bfloat16"
    }
    #[cfg(feature = "cudnn")]
    #[inline]
    fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
        cudnn_sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16
    }
    #[cfg(feature = "nccl")]
    #[inline]
    fn nccl_data_type() -> nccl_sys::ncclDataType_t {
        nccl_sys::ncclDataType_t::ncclBfloat16
    }
}

impl GemmSupported for f32 {}
impl GemmSupported for f64 {}
impl GemmSupported for i8 {}
impl GemmSupported for i32 {}
#[cfg(feature = "f16")]
impl GemmSupported for half::f16 {}
#[cfg(feature = "f16")]
impl GemmSupported for half::bf16 {}

#[cfg(feature = "f8")]
mod fp8_impls {
    use super::*;
    use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

    unsafe impl DeviceRepr for F8E4m3 {}
    unsafe impl ValidAsZeroBits for F8E4m3 {}
    unsafe impl DeviceRepr for F8E5m2 {}
    unsafe impl ValidAsZeroBits for F8E5m2 {}

    impl CudaDtype for F8E4m3 {
        #[inline]
        fn cuda_data_type() -> cublas_sys::cudaDataType_t {
            cublas_sys::cudaDataType_t::CUDA_R_8F_E4M3
        }
        #[inline]
        fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        }
        #[inline]
        fn cuda_type_name() -> &'static str {
            "__nv_fp8_e4m3"
        }
        #[cfg(feature = "cudnn")]
        #[inline]
        fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FP8_E4M3
        }
        #[cfg(feature = "nccl")]
        #[inline]
        fn nccl_data_type() -> nccl_sys::ncclDataType_t {
            nccl_sys::ncclDataType_t::ncclFloat8e4m3
        }
    }

    impl CudaDtype for F8E5m2 {
        #[inline]
        fn cuda_data_type() -> cublas_sys::cudaDataType_t {
            cublas_sys::cudaDataType_t::CUDA_R_8F_E5M2
        }
        #[inline]
        fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        }
        #[inline]
        fn cuda_type_name() -> &'static str {
            "__nv_fp8_e5m2"
        }
        #[cfg(feature = "cudnn")]
        #[inline]
        fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FP8_E5M2
        }
        #[cfg(feature = "nccl")]
        #[inline]
        fn nccl_data_type() -> nccl_sys::ncclDataType_t {
            nccl_sys::ncclDataType_t::ncclFloat8e5m2
        }
    }

    impl GemmSupported for F8E4m3 {}
    impl GemmSupported for F8E5m2 {}
    impl CudnnSupported for F8E4m3 {}
    impl CudnnSupported for F8E5m2 {}
    impl NcclReduceSupported for F8E4m3 {}
    impl NcclReduceSupported for F8E5m2 {}
}

impl CudaDtype for C32 {
    #[inline]
    fn cuda_data_type() -> cublas_sys::cudaDataType_t {
        cublas_sys::cudaDataType_t::CUDA_C_32F
    }
    #[inline]
    fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
    #[inline]
    fn cuda_type_name() -> &'static str {
        "cuComplex"
    }
    #[cfg(feature = "cudnn")]
    #[inline]
    fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
        // cuDNN has no native complex tensor element. Phase 1.5++ does
        // not surface a CudnnSupported impl for complex dtypes; calling
        // this method is a programmer error.
        panic!("C32 is not a cuDNN tensor element type");
    }
    #[cfg(feature = "nccl")]
    #[inline]
    fn nccl_data_type() -> nccl_sys::ncclDataType_t {
        // NCCL has no native complex reduce element. Same gating as
        // above — Phase 1.5++ does not impl `NcclReduceSupported` for
        // complex dtypes.
        panic!("C32 is not an NCCL reduce element type");
    }
}

impl CudaDtype for C64 {
    #[inline]
    fn cuda_data_type() -> cublas_sys::cudaDataType_t {
        cublas_sys::cudaDataType_t::CUDA_C_64F
    }
    #[inline]
    fn cublas_compute_type() -> cublas_sys::cublasComputeType_t {
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_64F
    }
    #[inline]
    fn cuda_type_name() -> &'static str {
        "cuDoubleComplex"
    }
    #[cfg(feature = "cudnn")]
    #[inline]
    fn cudnn_data_type() -> cudnn_sys::cudnnDataType_t {
        panic!("C64 is not a cuDNN tensor element type");
    }
    #[cfg(feature = "nccl")]
    #[inline]
    fn nccl_data_type() -> nccl_sys::ncclDataType_t {
        panic!("C64 is not an NCCL reduce element type");
    }
}

impl CudnnSupported for f32 {}
impl CudnnSupported for f64 {}
impl CudnnSupported for i8 {}
impl CudnnSupported for u8 {}
impl CudnnSupported for i32 {}
impl CudnnSupported for i64 {}
#[cfg(feature = "f16")]
impl CudnnSupported for half::f16 {}
#[cfg(feature = "f16")]
impl CudnnSupported for half::bf16 {}

impl FftSupported for f32 {}
impl FftSupported for f64 {}
impl FftSupported for C32 {}
impl FftSupported for C64 {}
#[cfg(feature = "f16")]
impl FftSupported for half::f16 {}

impl RngFloatSupported for f32 {}
impl RngFloatSupported for f64 {}

impl RngIntSupported for u32 {}
impl RngIntSupported for u64 {}

impl AxpyDotNrm2Supported for f32 {}
impl AxpyDotNrm2Supported for f64 {}
#[cfg(feature = "f16")]
impl AxpyDotNrm2Supported for half::f16 {}
#[cfg(feature = "f16")]
impl AxpyDotNrm2Supported for half::bf16 {}

impl GemvSupported for f32 {}
impl GemvSupported for f64 {}

impl GerSupported for f32 {}
impl GerSupported for f64 {}

impl GeamSupported for f32 {}
impl GeamSupported for f64 {}

impl SyrkSupported for f32 {}
impl SyrkSupported for f64 {}

impl TrsmSupported for f32 {}
impl TrsmSupported for f64 {}

impl NcclReduceSupported for f32 {}
impl NcclReduceSupported for f64 {}
impl NcclReduceSupported for i8 {}
impl NcclReduceSupported for u8 {}
impl NcclReduceSupported for i32 {}
impl NcclReduceSupported for u32 {}
impl NcclReduceSupported for i64 {}
impl NcclReduceSupported for u64 {}
#[cfg(feature = "f16")]
impl NcclReduceSupported for half::f16 {}
#[cfg(feature = "f16")]
impl NcclReduceSupported for half::bf16 {}

impl SolverSupported for f32 {}
impl SolverSupported for f64 {}

impl SparseSupported for f32 {}
impl SparseSupported for f64 {}
#[cfg(feature = "f16")]
impl SparseSupported for half::f16 {}
#[cfg(feature = "f16")]
impl SparseSupported for half::bf16 {}

impl TensorSupported for f32 {}
impl TensorSupported for f64 {}
#[cfg(feature = "f16")]
impl TensorSupported for half::f16 {}
#[cfg(feature = "f16")]
impl TensorSupported for half::bf16 {}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_accel::DType;

    #[test]
    fn cuda_data_type_round_trip() {
        assert_eq!(<f32 as AccelDtype>::KIND, DType::F32);
        assert_eq!(
            <f32 as CudaDtype>::cuda_data_type(),
            cublas_sys::cudaDataType_t::CUDA_R_32F
        );
        assert_eq!(
            <f64 as CudaDtype>::cuda_data_type(),
            cublas_sys::cudaDataType_t::CUDA_R_64F
        );
        assert_eq!(<f32 as CudaDtype>::cuda_type_name(), "float");
        assert_eq!(<f64 as CudaDtype>::cuda_type_name(), "double");
    }

    #[test]
    fn integer_compute_types() {
        assert_eq!(
            <i32 as CudaDtype>::cublas_compute_type(),
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32I
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_mappings() {
        assert_eq!(
            <half::f16 as CudaDtype>::cuda_data_type(),
            cublas_sys::cudaDataType_t::CUDA_R_16F
        );
        assert_eq!(
            <half::bf16 as CudaDtype>::cuda_data_type(),
            cublas_sys::cudaDataType_t::CUDA_R_16BF
        );
        assert_eq!(<half::f16 as CudaDtype>::cuda_type_name(), "__half");
        assert_eq!(<half::bf16 as CudaDtype>::cuda_type_name(), "__nv_bfloat16");
    }

    fn _assert_capability_bounds<G, C, F, N>()
    where
        G: GemmSupported,
        C: CudnnSupported,
        F: FftSupported,
        N: NcclReduceSupported,
    {
    }

    #[test]
    fn capability_compile_time_check() {
        _assert_capability_bounds::<f32, f32, f32, f32>();
        _assert_capability_bounds::<f64, f64, f64, f64>();
    }

    #[test]
    fn complex_dtype_size_and_layout() {
        assert_eq!(<C32 as AccelDtype>::SIZE, 8);
        assert_eq!(<C64 as AccelDtype>::SIZE, 16);
        assert_eq!(<C32 as AccelDtype>::KIND, DType::F32);
        assert_eq!(<C64 as AccelDtype>::KIND, DType::F64);
        assert_eq!(<C32 as AccelDtype>::NAME, "complex64");
        assert_eq!(<C64 as AccelDtype>::NAME, "complex128");

        // Layout matches `[T; 2]` exactly (transparent).
        assert_eq!(std::mem::size_of::<C32>(), 8);
        assert_eq!(std::mem::size_of::<C64>(), 16);
        assert_eq!(std::mem::align_of::<C32>(), std::mem::align_of::<f32>());
        assert_eq!(std::mem::align_of::<C64>(), std::mem::align_of::<f64>());
    }

    #[test]
    fn complex_cuda_data_type_mapping() {
        assert_eq!(
            <C32 as CudaDtype>::cuda_data_type(),
            cublas_sys::cudaDataType_t::CUDA_C_32F
        );
        assert_eq!(
            <C64 as CudaDtype>::cuda_data_type(),
            cublas_sys::cudaDataType_t::CUDA_C_64F
        );
        assert_eq!(<C32 as CudaDtype>::cuda_type_name(), "cuComplex");
        assert_eq!(<C64 as CudaDtype>::cuda_type_name(), "cuDoubleComplex");
        assert_eq!(
            <C32 as CudaDtype>::cublas_compute_type(),
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        );
        assert_eq!(
            <C64 as CudaDtype>::cublas_compute_type(),
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_64F
        );
    }

    #[test]
    fn complex_fft_supported_compile_time_check() {
        fn _check<T: FftSupported>() {}
        _check::<C32>();
        _check::<C64>();
    }

    #[test]
    fn complex_zero_one_nan_identities() {
        assert_eq!(<C32 as AccelDtype>::zero(), C32([0.0, 0.0]));
        assert_eq!(<C32 as AccelDtype>::one(), C32([1.0, 0.0]));
        assert!(<C32 as AccelDtype>::nan()
            .map(|n| n.0[0].is_nan() && n.0[1].is_nan())
            .unwrap_or(false));
        assert_eq!(<C64 as AccelDtype>::zero(), C64([0.0, 0.0]));
        assert_eq!(<C64 as AccelDtype>::one(), C64([1.0, 0.0]));
    }

    #[cfg(feature = "cufft")]
    #[test]
    fn complex_round_trips_cufft_sys() {
        use cudarc::cufft::sys as s;
        let f = s::float2 { x: 1.5, y: -2.5 };
        let c: C32 = f.into();
        assert_eq!(c, C32([1.5, -2.5]));
        let f2: s::float2 = c.into();
        assert_eq!(f2.x, 1.5);
        assert_eq!(f2.y, -2.5);

        let d = s::double2 { x: 7.0, y: 8.0 };
        let c64: C64 = d.into();
        assert_eq!(c64, C64([7.0, 8.0]));
        let d2: s::double2 = c64.into();
        assert_eq!(d2.x, 7.0);
        assert_eq!(d2.y, 8.0);
    }
}
