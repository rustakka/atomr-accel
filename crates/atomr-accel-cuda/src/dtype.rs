//! Dtype marker traits used by the dtype-generic actor messages.
//!
//! cuTENSOR (Phase 2) is the first call site that consumes the
//! [`TensorSupported`] marker — it gates `ContractRequest<T>`,
//! `ReductionRequest<T>` etc. so that only known-good scalar types
//! (`f32`, `f64`, `half::f16`, `half::bf16`) instantiate.
//!
//! The full Phase-0 [`AccelDtype`] trait described in the architecture
//! plan is **not** materialised in this slice: cuTENSOR only needs the
//! pieces required to (a) pick a `cudaDataType_t` for a tensor
//! descriptor and (b) gate generic op requests at compile time.
//! Subsequent phases (cuDNN, NCCL, cuBLAS) extend this module
//! independently. We keep the surface intentionally small so parallel
//! agents touching cuDNN / NCCL can grow it without trampling our
//! commits.

use cudarc::cutensor::sys as ct_sys;

/// Compile-time gate for `cuTENSOR` operations.
///
/// Implementors are scalar types that cuTENSOR understands as the
/// element type of an operand tensor. The trait carries the
/// `cudaDataType_t` enum and a stable string name used in plan-cache
/// keys and error messages.
///
/// `Send + Sync + 'static` so a `Box<dyn TensorDispatch>` can carry a
/// generic `Request<T>` across actor mailboxes.
pub trait TensorSupported: Copy + Send + Sync + 'static {
    /// `cudaDataType_t` value passed to `cutensorCreateTensorDescriptor`.
    fn cuda_data_type() -> ct_sys::cudaDataType_t;

    /// Stable human-readable tag used in cache keys and panic messages.
    fn dtype_tag() -> &'static str;
}

impl TensorSupported for f32 {
    fn cuda_data_type() -> ct_sys::cudaDataType_t {
        ct_sys::cudaDataType_t::CUDA_R_32F
    }
    fn dtype_tag() -> &'static str {
        "f32"
    }
}

impl TensorSupported for f64 {
    fn cuda_data_type() -> ct_sys::cudaDataType_t {
        ct_sys::cudaDataType_t::CUDA_R_64F
    }
    fn dtype_tag() -> &'static str {
        "f64"
    }
}

#[cfg(feature = "f16")]
impl TensorSupported for half::f16 {
    fn cuda_data_type() -> ct_sys::cudaDataType_t {
        ct_sys::cudaDataType_t::CUDA_R_16F
    }
    fn dtype_tag() -> &'static str {
        "f16"
    }
}

#[cfg(feature = "f16")]
impl TensorSupported for half::bf16 {
    fn cuda_data_type() -> ct_sys::cudaDataType_t {
        ct_sys::cudaDataType_t::CUDA_R_16BF
    }
    fn dtype_tag() -> &'static str {
        "bf16"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_tags_are_unique() {
        assert_eq!(<f32 as TensorSupported>::dtype_tag(), "f32");
        assert_eq!(<f64 as TensorSupported>::dtype_tag(), "f64");
    }

    #[test]
    fn cuda_data_types_match_expected() {
        assert_eq!(
            <f32 as TensorSupported>::cuda_data_type(),
            ct_sys::cudaDataType_t::CUDA_R_32F
        );
        assert_eq!(
            <f64 as TensorSupported>::cuda_data_type(),
            ct_sys::cudaDataType_t::CUDA_R_64F
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn half_dtype_tags() {
        assert_eq!(<half::f16 as TensorSupported>::dtype_tag(), "f16");
        assert_eq!(<half::bf16 as TensorSupported>::dtype_tag(), "bf16");
    }
}
