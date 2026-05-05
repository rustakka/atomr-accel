//! [`KernelTrace`] — observability hook installed on
//! `atomr_accel_cuda::kernel::envelope::run_kernel`. The kernel
//! envelope owns the shared `Arc<dyn KernelTrace>` and calls the
//! pre-/post-launch hooks around every cudarc enqueue.
//!
//! The hook is split into `before_enqueue` / `after_complete` so a
//! pair of NVTX `Push` / `Pop` calls bracket the kernel's wall-clock
//! time. CUPTI listeners can mirror the same shape.
//!
//! The trait is intentionally minimal — a richer payload (kernel
//! arguments, dtype, byte counts) lives in the
//! [`KernelInfo`] struct that the envelope hands to each callback.
//!
//! ## Forward compatibility
//!
//! When Phase 0.7 of the roadmap lands the canonical home for this
//! trait will be `atomr_accel_cuda::kernel::envelope::KernelTrace`.
//! `atomr_accel_cuda` will then re-export this module's types so
//! existing callers keep compiling. Until then, anything using the
//! envelope may store a `SharedTrace` (`Arc<dyn KernelTrace>`)
//! whose default value is `Arc::new(NoopKernelTrace)`.

use std::any::Any;
use std::time::Duration;

/// Static metadata describing a kernel launch. Populated by the
/// kernel envelope before each `before_enqueue` call.
#[derive(Clone, Debug)]
pub struct KernelInfo {
    /// Library tag, matching the `lib_tag` argument of
    /// `envelope::run_kernel` (`"cublas"`, `"cudnn"`, `"cufft"`, …).
    pub lib_tag: &'static str,

    /// Operation name. For cuBLAS this is `"sgemm"` /
    /// `"sgemm_strided_batched"`; for cuDNN, `"conv_fwd"` /
    /// `"softmax"`; etc. NVTX ranges use this verbatim.
    pub op_name: &'static str,

    /// Optional device index (0-based) the kernel is targeting. The
    /// envelope passes `None` if the actor has no notion of a
    /// specific device.
    pub device_index: Option<u32>,

    /// Optional per-launch correlation token. CUPTI activity records
    /// already carry their own `correlationId`; this field is the
    /// caller's logical token (e.g. a request id) that telemetry can
    /// join across hooks.
    pub correlation_id: Option<u64>,
}

impl KernelInfo {
    /// Construct a minimal `KernelInfo` for tests / synthetic events.
    pub fn synthetic(lib_tag: &'static str, op_name: &'static str) -> Self {
        Self {
            lib_tag,
            op_name,
            device_index: None,
            correlation_id: None,
        }
    }
}

/// Observability hook installed on the kernel envelope. Both
/// callbacks are infallible — if a backend fails it must log and
/// swallow the error so the kernel path is never blocked.
///
/// `before_enqueue` returns an opaque cookie of type `Box<dyn Any +
/// Send>` that the envelope keeps alive until completion. NVTX
/// implementations stash a `Range` guard there so the range ends on
/// `drop` (i.e. exactly when `after_complete` returns).
pub trait KernelTrace: Send + Sync + 'static {
    /// Invoked synchronously on the actor task immediately before
    /// the cudarc enqueue. The returned cookie is held by the
    /// envelope and passed to [`KernelTrace::after_complete`] once
    /// stream completion fires.
    fn before_enqueue(&self, info: &KernelInfo) -> Box<dyn Any + Send>;

    /// Invoked from the completion task once the stream signal
    /// arrives. `cookie` is the value previously returned from
    /// `before_enqueue`. `duration` is the wall-clock time between
    /// the two calls; backends that don't care can ignore it.
    fn after_complete(&self, info: &KernelInfo, cookie: Box<dyn Any + Send>, duration: Duration);
}

/// Zero-overhead no-op implementation, used as the envelope's
/// default when no observability backend is installed.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoopKernelTrace;

impl KernelTrace for NoopKernelTrace {
    fn before_enqueue(&self, _info: &KernelInfo) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn after_complete(
        &self,
        _info: &KernelInfo,
        _cookie: Box<dyn Any + Send>,
        _duration: Duration,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn noop_kernel_trace_round_trip() {
        let t = NoopKernelTrace;
        let info = KernelInfo::synthetic("test", "noop");
        let cookie = t.before_enqueue(&info);
        t.after_complete(&info, cookie, Duration::from_millis(0));
    }

    #[test]
    fn kernel_info_synthetic_constructor() {
        let info = KernelInfo::synthetic("cublas", "sgemm");
        assert_eq!(info.lib_tag, "cublas");
        assert_eq!(info.op_name, "sgemm");
        assert!(info.device_index.is_none());
        assert!(info.correlation_id.is_none());
    }
}
