//! Thin local wrappers over `cudarc::cublaslt::sys` for the entry
//! points cudarc 0.19.4's safe layer doesn't expose. Each helper is
//! `unsafe fn` if it dereferences raw handles; the safe shape is just
//! "create / set-attribute / destroy" RAII over the opaque handle.
//!
//! Used by:
//! - [`crate::kernel::blas_lt::heuristic`] for
//!   `cublasLtMatmulPreferenceCreate` + `cublasLtMatmulAlgoGetHeuristic`,
//! - [`crate::kernel::blas_lt::scaling`] for the per-tensor fp8 scale
//!   pointer descriptor attributes,
//! - [`crate::kernel::blas_lt::epilogue`] for setting the
//!   `CUBLASLT_MATMUL_DESC_EPILOGUE` attribute.

use std::ffi::c_void;
use std::ptr;

use cudarc::cublaslt::sys::{
    cublasLtMatmulDescAttributes_t, cublasLtMatmulDesc_t, cublasLtMatmulPreferenceAttributes_t,
    cublasLtMatmulPreferenceOpaque_t, cublasLtMatmulPreference_t, cublasStatus_t,
};
#[cfg(test)]
use cudarc::cublaslt::sys::cublasLtMatmulDescOpaque_t;

/// Map a `cublasStatus_t` into an `Err(String)` for the cuBLASLt
/// status codes we care about. We don't go through `CublasError` here
/// because cudarc's `result` module doesn't expose every variant we
/// touch — the string form is sufficient since these errors funnel
/// straight into `GpuError::LibraryError`.
pub fn check(status: cublasStatus_t, op: &str) -> Result<(), String> {
    match status {
        cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
        other => Err(format!("{op}: {other:?}")),
    }
}

/// RAII handle for a `cublasLtMatmulPreference_t`. Created via
/// [`Preference::new`], destroyed on Drop.
pub struct Preference {
    pub raw: cublasLtMatmulPreference_t,
}

// SAFETY: the underlying preference object is plain CPU-side state
// that cuBLASLt synchronizes internally. We only ever touch it through
// the cuBLASLt API.
unsafe impl Send for Preference {}
unsafe impl Sync for Preference {}

impl Preference {
    pub fn new() -> Result<Self, String> {
        let mut raw: cublasLtMatmulPreference_t =
            ptr::null_mut::<cublasLtMatmulPreferenceOpaque_t>();
        let status = unsafe { cudarc::cublaslt::sys::cublasLtMatmulPreferenceCreate(&mut raw) };
        check(status, "cublasLtMatmulPreferenceCreate")?;
        Ok(Self { raw })
    }

    /// Set a u64-valued preference attribute (the most common case —
    /// `MAX_WORKSPACE_BYTES`, `IMPL_MASK`, `REDUCTION_SCHEME_MASK` all
    /// take a u64).
    pub fn set_u64(
        &self,
        attr: cublasLtMatmulPreferenceAttributes_t,
        value: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            cudarc::cublaslt::sys::cublasLtMatmulPreferenceSetAttribute(
                self.raw,
                attr,
                &value as *const u64 as *const c_void,
                std::mem::size_of::<u64>(),
            )
        };
        check(status, "cublasLtMatmulPreferenceSetAttribute")
    }
}

impl Drop for Preference {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = cudarc::cublaslt::sys::cublasLtMatmulPreferenceDestroy(self.raw);
            }
            self.raw = ptr::null_mut::<cublasLtMatmulPreferenceOpaque_t>();
        }
    }
}

/// Set a pointer-valued attribute on a matmul descriptor (used for
/// `BIAS_POINTER`, `EPILOGUE_AUX_POINTER`, fp8 scale pointers).
///
/// # Safety
///
/// `desc` must be a live `cublasLtMatmulDesc_t`; `ptr` must remain
/// valid for the entire lifetime of any matmul call that uses `desc`.
pub unsafe fn set_desc_pointer_attr(
    desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    ptr: *const c_void,
) -> Result<(), String> {
    let status = unsafe {
        cudarc::cublaslt::sys::cublasLtMatmulDescSetAttribute(
            desc,
            attr,
            &ptr as *const *const c_void as *const c_void,
            std::mem::size_of::<*const c_void>(),
        )
    };
    check(status, "cublasLtMatmulDescSetAttribute(pointer)")
}

/// Set an i32-valued attribute on a matmul descriptor (used for
/// `EPILOGUE`, the cudarc-bindgen `cublasLtEpilogue_t` is repr(u32)
/// but the cuBLASLt API expects `int`).
///
/// # Safety
///
/// `desc` must be a live `cublasLtMatmulDesc_t`.
pub unsafe fn set_desc_i32_attr(
    desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    value: i32,
) -> Result<(), String> {
    let status = unsafe {
        cudarc::cublaslt::sys::cublasLtMatmulDescSetAttribute(
            desc,
            attr,
            &value as *const i32 as *const c_void,
            std::mem::size_of::<i32>(),
        )
    };
    check(status, "cublasLtMatmulDescSetAttribute(i32)")
}

/// Stand-in opaque-descriptor type used by tests that mock attribute
/// writes without touching the real CUDA driver.
#[cfg(test)]
pub fn mock_desc_handle() -> cublasLtMatmulDesc_t {
    let leaked: Box<cublasLtMatmulDescOpaque_t> = Box::new(unsafe { std::mem::zeroed() });
    Box::into_raw(leaked)
}

/// Drop a mock descriptor allocated via [`mock_desc_handle`].
#[cfg(test)]
pub unsafe fn drop_mock_desc(desc: cublasLtMatmulDesc_t) {
    if !desc.is_null() {
        let _ = unsafe { Box::from_raw(desc) };
    }
}
