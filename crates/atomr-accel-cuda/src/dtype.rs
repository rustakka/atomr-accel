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

use atomr_accel::AccelDtype;

use cudarc::cublas::sys as cublas_sys;
#[cfg(feature = "cudnn")]
use cudarc::cudnn::sys as cudnn_sys;
#[cfg(feature = "nccl")]
use cudarc::nccl::sys as nccl_sys;

/// CUDA-specific layer over [`AccelDtype`].
pub trait CudaDtype: AccelDtype {
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
#[cfg(feature = "f16")]
impl FftSupported for half::f16 {}

impl RngFloatSupported for f32 {}
impl RngFloatSupported for f64 {}

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
}
