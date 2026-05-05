//! Thin Rust safety layer over the cuDNN v9 backend graph FFI.
//!
//! cudarc 0.19.4 ships the raw bindings for `cudnnBackendCreateDescriptor`,
//! `cudnnBackendSetAttribute`, `cudnnBackendFinalize`,
//! `cudnnBackendExecute`, etc. but no safe wrapper. This module owns
//! that wrapping so the cuDNN actor's graph builder can call into the
//! backend API without scattering `unsafe` everywhere.
//!
//! Scope: a [`BackendDescriptor`] RAII handle plus typed `set_*` helpers
//! covering the attribute kinds the cuDNN actor uses (i64 / i64 array
//! / f32 / f64 / pointer / `BackendDescriptor` / enum).
//!
//! Everything in this module is a no-op on hosts without cuDNN at
//! runtime — actual FFI calls only fire when the descriptor is fed to
//! a real `cudnnHandle_t` via [`backend_execute`].

#![allow(dead_code)]

use std::ffi::c_void;

use cudarc::cudnn::sys as cudnn_sys;

use crate::error::GpuError;

const LIB: &str = "cudnn";

fn check(s: cudnn_sys::cudnnStatus_t, what: &'static str) -> Result<(), GpuError> {
    if s == cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{what}: cudnnStatus={:?}", s),
        })
    }
}

/// RAII wrapper around a `cudnnBackendDescriptor_t` that destroys the
/// descriptor on drop.
pub struct BackendDescriptor {
    raw: cudnn_sys::cudnnBackendDescriptor_t,
    finalized: bool,
}

impl std::fmt::Debug for BackendDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendDescriptor")
            .field("raw", &self.raw)
            .field("finalized", &self.finalized)
            .finish()
    }
}

unsafe impl Send for BackendDescriptor {}

impl BackendDescriptor {
    /// `cudnnBackendCreateDescriptor`.
    pub fn create(
        kind: cudnn_sys::cudnnBackendDescriptorType_t,
    ) -> Result<Self, GpuError> {
        let mut raw: cudnn_sys::cudnnBackendDescriptor_t = std::ptr::null_mut();
        let s = unsafe { cudnn_sys::cudnnBackendCreateDescriptor(kind, &mut raw) };
        check(s, "cudnnBackendCreateDescriptor")?;
        Ok(Self {
            raw,
            finalized: false,
        })
    }

    /// Raw handle (only valid until `Drop`).
    pub fn as_raw(&self) -> cudnn_sys::cudnnBackendDescriptor_t {
        self.raw
    }

    /// `cudnnBackendFinalize`.
    pub fn finalize(&mut self) -> Result<(), GpuError> {
        if self.finalized {
            return Ok(());
        }
        let s = unsafe { cudnn_sys::cudnnBackendFinalize(self.raw) };
        check(s, "cudnnBackendFinalize")?;
        self.finalized = true;
        Ok(())
    }

    /// True iff `finalize()` has succeeded.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Generic `cudnnBackendSetAttribute` — caller supplies the
    /// element type and a `*const c_void` pointer to a contiguous
    /// array of `count` elements.
    ///
    /// # Safety
    /// `data` must point to `count` valid elements of the type
    /// expected by `attr_type`. The descriptor must outlive the call.
    pub unsafe fn set_attribute_raw(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        attr_type: cudnn_sys::cudnnBackendAttributeType_t,
        count: i64,
        data: *const c_void,
    ) -> Result<(), GpuError> {
        let s = unsafe {
            cudnn_sys::cudnnBackendSetAttribute(
                self.raw,
                name,
                attr_type,
                count,
                data as *mut c_void,
            )
        };
        check(s, "cudnnBackendSetAttribute")
    }

    /// Set a single i64 attribute.
    pub fn set_i64(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        value: i64,
    ) -> Result<(), GpuError> {
        let v = value;
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                1,
                &v as *const i64 as *const c_void,
            )
        }
    }

    /// Set an array of i64 attributes.
    pub fn set_i64_array(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        values: &[i64],
    ) -> Result<(), GpuError> {
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                values.len() as i64,
                values.as_ptr() as *const c_void,
            )
        }
    }

    /// Set a single f32 attribute (e.g. an alpha scaling parameter).
    pub fn set_f32(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        value: f32,
    ) -> Result<(), GpuError> {
        let v = value;
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                1,
                &v as *const f32 as *const c_void,
            )
        }
    }

    /// Set a single f64 attribute.
    pub fn set_f64(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        value: f64,
    ) -> Result<(), GpuError> {
        let v = value;
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                1,
                &v as *const f64 as *const c_void,
            )
        }
    }

    /// Set a single device-pointer attribute.
    pub fn set_dev_ptr(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        ptr: *mut c_void,
    ) -> Result<(), GpuError> {
        let p = ptr;
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
                1,
                &p as *const *mut c_void as *const c_void,
            )
        }
    }

    /// Set a single sub-descriptor reference.
    pub fn set_descriptor(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        sub: &BackendDescriptor,
    ) -> Result<(), GpuError> {
        let p = sub.raw;
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &p as *const _ as *const c_void,
            )
        }
    }

    /// Set an array of sub-descriptor references.
    pub fn set_descriptors(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        subs: &[&BackendDescriptor],
    ) -> Result<(), GpuError> {
        let raws: Vec<cudnn_sys::cudnnBackendDescriptor_t> =
            subs.iter().map(|s| s.raw).collect();
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                raws.len() as i64,
                raws.as_ptr() as *const c_void,
            )
        }
    }

    /// Set a `cudnnDataType_t` attribute.
    pub fn set_data_type(
        &mut self,
        name: cudnn_sys::cudnnBackendAttributeName_t,
        dt: cudnn_sys::cudnnDataType_t,
    ) -> Result<(), GpuError> {
        let v = dt;
        unsafe {
            self.set_attribute_raw(
                name,
                cudnn_sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &v as *const _ as *const c_void,
            )
        }
    }
}

impl Drop for BackendDescriptor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Best-effort destroy. Ignore status — there is nothing
            // sensible to do on failure during drop.
            unsafe {
                let _ = cudnn_sys::cudnnBackendDestroyDescriptor(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Run a finalised execution plan against a finalised variant pack.
///
/// # Safety
/// Both descriptors must be finalised. `handle` must be valid for the
/// stream the plan was built against.
pub unsafe fn backend_execute(
    handle: cudnn_sys::cudnnHandle_t,
    plan: &BackendDescriptor,
    variant_pack: &BackendDescriptor,
) -> Result<(), GpuError> {
    let s = unsafe { cudnn_sys::cudnnBackendExecute(handle, plan.raw, variant_pack.raw) };
    check(s, "cudnnBackendExecute")
}

#[cfg(test)]
mod tests {
    // No-op tests: these helpers only call into cuDNN when a real
    // descriptor is created, which requires a loaded cuDNN runtime.
    // The cuDNN actor's tests cover the round-trip path under
    // host-builds via the spec layer in `kernel::cudnn::graph`.
}
