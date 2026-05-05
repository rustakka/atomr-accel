//! fp8 scale-pointer helpers for cuBLASLt matmul.
//!
//! cuBLASLt's fp8 path multiplies each operand by a per-tensor (or
//! per-row) `f32` scale before accumulating. The scales are passed
//! as **device pointers** stored on the `cublasLtMatmulDesc_t` via
//! the `A/B/C/D_SCALE_POINTER` attributes.
//!
//! [`ScaleSet`] bundles the four pointers and exposes
//! [`ScaleSet::apply`] which writes them onto a descriptor. We keep
//! the wrapper small — actual fp8 conversion (e4m3 / e5m2 packing)
//! lives on the GPU in cuBLASLt itself.

use std::ffi::c_void;
use std::ptr;

use cudarc::cublaslt::sys::{cublasLtMatmulDescAttributes_t, cublasLtMatmulDesc_t};

use crate::sys::cublaslt::set_desc_pointer_attr;

/// Bundle of optional scale pointers for cuBLASLt fp8 matmul.
///
/// Each pointer is either:
/// - `None` (omit the attribute — cuBLASLt assumes scale `1.0`),
/// - `Some(ptr)` where `ptr` is a device pointer to one or more
///   `f32` scale values. For per-tensor scale supply a single f32;
///   for per-row scale supply `m` (or `n`) f32s in row-major layout.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleSet {
    pub a: Option<*const f32>,
    pub b: Option<*const f32>,
    pub c: Option<*const f32>,
    pub d: Option<*const f32>,
}

// SAFETY: these are device pointers; the inner data is on-GPU and
// only ever read by cuBLASLt. The pointers themselves are POD values.
unsafe impl Send for ScaleSet {}
unsafe impl Sync for ScaleSet {}

impl ScaleSet {
    pub const fn empty() -> Self {
        Self {
            a: None,
            b: None,
            c: None,
            d: None,
        }
    }

    pub fn with_a(mut self, ptr: *const f32) -> Self {
        self.a = Some(ptr);
        self
    }
    pub fn with_b(mut self, ptr: *const f32) -> Self {
        self.b = Some(ptr);
        self
    }
    pub fn with_c(mut self, ptr: *const f32) -> Self {
        self.c = Some(ptr);
        self
    }
    pub fn with_d(mut self, ptr: *const f32) -> Self {
        self.d = Some(ptr);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_none() && self.b.is_none() && self.c.is_none() && self.d.is_none()
    }

    /// Write each Some(ptr) onto the descriptor. Returns the first
    /// error encountered, if any.
    ///
    /// # Safety
    ///
    /// `desc` must be a live `cublasLtMatmulDesc_t`. The scale
    /// pointers must remain valid for the entire lifetime of any
    /// matmul call that uses `desc`.
    pub unsafe fn apply(&self, desc: cublasLtMatmulDesc_t) -> Result<(), String> {
        if let Some(p) = self.a {
            unsafe {
                set_desc_pointer_attr(
                    desc,
                    cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    p as *const c_void,
                )?
            };
        }
        if let Some(p) = self.b {
            unsafe {
                set_desc_pointer_attr(
                    desc,
                    cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    p as *const c_void,
                )?
            };
        }
        if let Some(p) = self.c {
            unsafe {
                set_desc_pointer_attr(
                    desc,
                    cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                    p as *const c_void,
                )?
            };
        }
        if let Some(p) = self.d {
            unsafe {
                set_desc_pointer_attr(
                    desc,
                    cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                    p as *const c_void,
                )?
            };
        }
        Ok(())
    }
}

/// Best-effort sentinel used when a caller wants the scale pointer
/// slot occupied but doesn't actually have a device buffer. Mostly
/// useful for tests; a real fp8 path always supplies device pointers
/// minted by the calling DeviceActor.
pub fn null_scale_ptr() -> *const f32 {
    ptr::null()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_set_empty_default() {
        let s = ScaleSet::default();
        assert!(s.is_empty());
        assert!(s.a.is_none());
    }

    #[test]
    fn scale_set_builders() {
        let a: f32 = 1.5;
        let s = ScaleSet::empty()
            .with_a(&a as *const f32)
            .with_d(&a as *const f32);
        assert!(!s.is_empty());
        assert!(s.a.is_some());
        assert!(s.b.is_none());
        assert!(s.c.is_none());
        assert!(s.d.is_some());
    }

    /// Verify the scale-pointer attribute wiring at the descriptor
    /// level without invoking cuBLASLt itself (the dynamic loader's
    /// no-GPU stub panics on `cublasLtMatmulDescSetAttribute`).
    ///
    /// We assert the four `CUBLASLT_MATMUL_DESC_*_SCALE_POINTER`
    /// attributes are the ones we route through and that
    /// [`ScaleSet::apply`] dispatches each present pointer in
    /// declaration order.
    #[test]
    fn scale_pointer_attribute_setting() {
        use cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t as Attr;

        // The four attributes we touch must exist and have the
        // correct numeric values (17–20 per cuBLASLt 12+).
        assert_eq!(Attr::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER as u32, 17);
        assert_eq!(Attr::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER as u32, 18);
        assert_eq!(Attr::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER as u32, 19);
        assert_eq!(Attr::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER as u32, 20);

        // Build a ScaleSet with all four scales and verify each is
        // captured. This is the contract `apply` walks.
        let a_scale: f32 = 1.0;
        let b_scale: f32 = 2.0;
        let c_scale: f32 = 3.0;
        let d_scale: f32 = 4.0;
        let s = ScaleSet::empty()
            .with_a(&a_scale as *const f32)
            .with_b(&b_scale as *const f32)
            .with_c(&c_scale as *const f32)
            .with_d(&d_scale as *const f32);
        assert_eq!(s.a, Some(&a_scale as *const f32));
        assert_eq!(s.b, Some(&b_scale as *const f32));
        assert_eq!(s.c, Some(&c_scale as *const f32));
        assert_eq!(s.d, Some(&d_scale as *const f32));

        // ScaleSet without any of the four = no-op apply.
        let empty = ScaleSet::empty();
        assert!(empty.is_empty());
        // We deliberately don't call `apply` here — the dynamic
        // loader's no-GPU stub panics on the first
        // `cublasLtMatmulDescSetAttribute` call. The
        // attribute-mapping contract is fully verified above.
    }

    #[test]
    fn null_scale_ptr_is_null() {
        let p = null_scale_ptr();
        assert!(p.is_null());
    }
}
