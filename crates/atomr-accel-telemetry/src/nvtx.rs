//! NVTX-backed [`crate::KernelTrace`] implementation.
//!
//! `before_enqueue` pushes a domain range named `info.op_name`. The
//! returned cookie owns an `nvtxRangeId_t` plus a `start: Instant`.
//! `after_complete` pops the range, ending it for the profiler.
//!
//! cudarc 0.19.4 exposes the safe NVTX wrappers via
//! `cudarc::nvtx::result::*`. The `Domain` type is not part of
//! cudarc's safe layer, so this module wraps the raw
//! `nvtxDomainHandle_t` in a small `Domain` newtype that owns the
//! handle and destroys it on drop.

use std::any::Any;
use std::ffi::CString;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::nvtx::result as nvtx_result;
use cudarc::nvtx::sys as nvtx_sys;

use crate::trace::{KernelInfo, KernelTrace};

/// Owning wrapper around an `nvtxDomainHandle_t`. The handle is
/// created lazily on `Domain::new` and destroyed on drop. Ranges are
/// pushed/popped via the unsafe `cudarc::nvtx::result::domain_*`
/// wrappers.
#[derive(Debug)]
pub struct Domain {
    handle: nvtx_sys::nvtxDomainHandle_t,
}

// nvtxDomainHandle_t is an opaque pointer; NVTX is thread-safe per
// NVIDIA's documentation, and we never expose mutable state through
// the handle.
unsafe impl Send for Domain {}
unsafe impl Sync for Domain {}

impl Domain {
    /// Create a new NVTX domain. Domains are how NVIDIA Nsight
    /// groups ranges, so each actor that wants its own swim lane
    /// should create a distinct domain.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self {
            handle: nvtx_result::domain_create(name),
        }
    }

    /// Underlying raw handle. Exposed for advanced users that need
    /// to call additional NVTX entry points directly.
    pub fn raw(&self) -> nvtx_sys::nvtxDomainHandle_t {
        self.handle
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        // Safety: handle was returned by nvtxDomainCreateA above and
        // has not been destroyed elsewhere.
        unsafe { nvtx_result::domain_destroy(self.handle) }
    }
}

/// `KernelTrace` that pushes / pops NVTX domain ranges around every
/// kernel launch. Construct via `NvtxKernelTrace::new()` (uses a
/// default domain name) or `NvtxKernelTrace::with_domain_name(name)`.
pub struct NvtxKernelTrace {
    domain: Domain,
}

impl Default for NvtxKernelTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl NvtxKernelTrace {
    /// Construct with the default `"atomr-accel"` domain. Use
    /// `with_domain_name` to give each device / actor its own swim
    /// lane in Nsight.
    pub fn new() -> Self {
        Self::with_domain_name("atomr-accel")
    }

    /// Construct with a caller-supplied domain name.
    pub fn with_domain_name(name: impl AsRef<str>) -> Self {
        Self {
            domain: Domain::new(name),
        }
    }

    /// Wrap into an `Arc<dyn KernelTrace>` for installation on a
    /// kernel envelope.
    pub fn shared() -> Arc<dyn KernelTrace> {
        Arc::new(Self::new())
    }
}

/// Cookie returned from `before_enqueue`. The envelope hands it back
/// to `after_complete`.
struct NvtxCookie {
    /// NVTX range id returned by `nvtxDomainRangeStartEx`.
    range_id: nvtx_sys::nvtxRangeId_t,
    /// Wall-clock at push time, used to backfill `duration` if the
    /// envelope didn't measure it.
    start: Instant,
}

fn build_event(message: &CString) -> nvtx_sys::nvtxEventAttributes_t {
    nvtx_sys::nvtxEventAttributes_t {
        version: 3,
        size: std::mem::size_of::<nvtx_sys::nvtxEventAttributes_t>() as u16,
        category: 0,
        colorType: nvtx_sys::nvtxColorType_t::NVTX_COLOR_UNKNOWN as u32 as i32,
        color: 0,
        payloadType: nvtx_sys::nvtxPayloadType_t::NVTX_PAYLOAD_UNKNOWN as u32 as i32,
        reserved0: 0,
        payload: nvtx_sys::nvtxEventAttributes_v2_payload_t { iValue: 0 },
        messageType: nvtx_sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as u32 as i32,
        message: nvtx_sys::nvtxMessageValue_t {
            ascii: message.as_ptr(),
        },
    }
}

impl KernelTrace for NvtxKernelTrace {
    fn before_enqueue(&self, info: &KernelInfo) -> Box<dyn Any + Send> {
        // op_name is &'static str so this allocation is tight; we
        // can't avoid a CString because NVTX's ASCII path needs a
        // null terminator.
        let msg = match CString::new(info.op_name) {
            Ok(c) => c,
            // Operation name contained a NUL — fall back to lib_tag.
            Err(_) => {
                CString::new(info.lib_tag).unwrap_or_else(|_| CString::new("kernel").unwrap())
            }
        };
        let attr = build_event(&msg);
        // Safety: `self.domain.handle` is alive for the lifetime of
        // self; `attr` lives for the duration of this call.
        let range_id = unsafe { nvtx_result::domain_range_start(self.domain.handle, &attr) };
        Box::new(NvtxCookie {
            range_id,
            start: Instant::now(),
        })
    }

    fn after_complete(&self, _info: &KernelInfo, cookie: Box<dyn Any + Send>, _duration: Duration) {
        if let Ok(cookie) = cookie.downcast::<NvtxCookie>() {
            // `start` is observed only via the event-pair semantics
            // NVTX captures, but we keep it on the cookie so future
            // backends (CUPTI co-trace) can read it without an extra
            // map.
            let _ = cookie.start;
            // Safety: domain handle is alive; range_id was returned
            // by `domain_range_start` above.
            unsafe { nvtx_result::domain_range_end(self.domain.handle, cookie.range_id) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time + zero-arg-constructor smoke test. Confirms
    /// `NvtxKernelTrace` implements `KernelTrace`, the constructors
    /// type-check, and the public surface can be wrapped in an
    /// `Arc<dyn KernelTrace>` (the install shape used by the kernel
    /// envelope).
    ///
    /// Calling [`NvtxKernelTrace::new`] would invoke
    /// `nvtxDomainCreateA` through cudarc's libloading wrapper,
    /// which panics on hosts where `libnvToolsExt.so` is missing.
    /// CI runs without NVTX installed, so the test exercises only
    /// the trait-bound machinery.
    #[test]
    fn nvtx_kernel_trace_implements_kernel_trace() {
        // Static assertion: the trait bound is satisfied.
        fn assert_trace<T: KernelTrace>() {}
        assert_trace::<NvtxKernelTrace>();

        // Static assertion: zero-arg constructor exists and returns
        // the right shape. We reference it by name to force
        // type-checking without actually calling it.
        let _ctor: fn() -> NvtxKernelTrace = NvtxKernelTrace::new;
        let _ctor_default: fn() -> NvtxKernelTrace = <NvtxKernelTrace as Default>::default;
        let _shared_ctor: fn() -> Arc<dyn KernelTrace> = NvtxKernelTrace::shared;
    }
}
