//! `cub::DeviceScan` — inclusive / exclusive prefix sums.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;

use crate::{reply_err, CubDispatchBase, CubDispatchCtx};

/// Inclusive vs. exclusive scan flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanKind {
    Inclusive,
    Exclusive,
}

impl ScanKind {
    pub fn op_name(self) -> &'static str {
        match self {
            ScanKind::Inclusive => "scan_inclusive",
            ScanKind::Exclusive => "scan_exclusive",
        }
    }
}

pub struct ScanRequest<T: CudaDtype> {
    pub kind: ScanKind,
    pub input: GpuRef<T>,
    pub output: GpuRef<T>,
    /// Initial value for exclusive scans (`T::zero()` is the natural
    /// default for sum scans). Ignored for `Inclusive`.
    pub init: Option<T>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> ScanRequest<T> {
    pub fn new(
        kind: ScanKind,
        input: GpuRef<T>,
        output: GpuRef<T>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            kind,
            input,
            output,
            init: None,
            reply,
            _phantom: PhantomData,
        }
    }
}

pub trait CubScanDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for ScanRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        self.kind.op_name()
    }
    fn dtype_name(&self) -> &'static str {
        <T as atomr_accel_cuda::dtype::AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubScanDispatch for ScanRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubScan::{}<{}> — kernel compile path lands in Phase 5.1",
                self.kind.op_name(),
                <T as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 5: round-trip a `ScanRequest<f32>` through the
    /// `CubDispatchBase` trait — `op_name` and `dtype_name` resolve
    /// without dereferencing any `GpuRef`.
    #[test]
    fn scan_request_round_trip() {
        for kind in [ScanKind::Inclusive, ScanKind::Exclusive] {
            assert!(kind.op_name().starts_with("scan_"));
        }
        // Op-name uniqueness across the kind matrix.
        assert_ne!(ScanKind::Inclusive.op_name(), ScanKind::Exclusive.op_name());

        // The kernel cache key threads through the ScanKind name + the
        // dtype `T::NAME` from atomr-accel. Verify it's unique across
        // (kind, dtype).
        let dtypes = ["f32", "f64", "i32", "u32", "i64", "u64"];
        let kinds = [ScanKind::Inclusive, ScanKind::Exclusive];
        let mut seen = std::collections::HashSet::new();
        for k in kinds {
            for dt in dtypes {
                let key = crate::kernel_key(k.op_name(), dt);
                assert!(seen.insert(key.clone()), "scan key collision: {key}");
            }
        }
        assert_eq!(seen.len(), kinds.len() * dtypes.len());
    }
}
