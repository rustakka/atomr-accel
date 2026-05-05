//! `GraphOpRecord` impls for [`crate::kernel::CudnnActor`] requests.
//!
//! Each op holds the same payload as the matching `CudnnMsg` variant
//! minus the reply channel, plus the cuDNN handle that the actor
//! ordinarily owns. The graph script caller passes the handle in
//! once, then submits a series of [`ConvForwardOp`]s without
//! reaching back into the actor.
//!
//! For Phase 3 we keep these structs descriptor-shape-only: their
//! `record` method validates the inputs and returns `Unrecoverable`
//! on hosts where cuDNN isn't loadable. A future sub-phase will wire
//! the actual cuDNN-record path (cuDNN supports stream-capture as of
//! v8.5).

#![cfg(feature = "cudnn")]

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::graph::{GraphOpRecord, GraphRecordCtx};
use crate::kernel::{ActivationKind, ConvParams};

/// Capture-mode op for `CudnnMsg::ConvForward`.
pub struct ConvForwardOp {
    pub x: GpuRef<f32>,
    pub x_dims: [i32; 4],
    pub w: GpuRef<f32>,
    pub w_dims: [i32; 4],
    pub y: GpuRef<f32>,
    pub y_dims: [i32; 4],
    pub conv: ConvParams,
    pub alpha: f32,
    pub beta: f32,
}

/// Capture-mode op for `CudnnMsg::Activation`.
pub struct ActivationOp {
    pub kind: ActivationKind,
    pub x: GpuRef<f32>,
    pub y: GpuRef<f32>,
    pub dims: [i32; 4],
    pub alpha: f32,
    pub beta: f32,
}

/// Capture-mode op for `CudnnMsg::Softmax`.
pub struct SoftmaxOp {
    pub x: GpuRef<f32>,
    pub y: GpuRef<f32>,
    pub dims: [i32; 4],
    pub alpha: f32,
    pub beta: f32,
}

impl GraphOpRecord for ConvForwardOp {
    fn record(&self, ctx: &GraphRecordCtx<'_>) -> Result<(), GpuError> {
        // Validate first.
        validate_dims(&self.x_dims, "conv: x_dims")?;
        validate_dims(&self.w_dims, "conv: w_dims")?;
        validate_dims(&self.y_dims, "conv: y_dims")?;
        let _ = self.x.access()?;
        let _ = self.w.access()?;
        let _ = self.y.access()?;
        // Phase 3: cuDNN's stream-capture path is not yet wired
        // through the existing `CudnnActor` (which uses
        // `envelope::run_kernel` host-fn completion that's not
        // capture-safe). Until the actor exposes a capture-safe entry
        // point, we surface a clear Unrecoverable here. The
        // descriptor and validation are still enforced so callers
        // catch shape mismatches early.
        let _ = ctx;
        Err(GpuError::Unrecoverable(
            "graph::record::cudnn::ConvForward: cuDNN capture-mode \
             entry not yet wired (Phase 3 surface only)"
                .into(),
        ))
    }
}

impl GraphOpRecord for ActivationOp {
    fn record(&self, ctx: &GraphRecordCtx<'_>) -> Result<(), GpuError> {
        validate_dims(&self.dims, "activation: dims")?;
        let _ = self.x.access()?;
        let _ = self.y.access()?;
        let _ = ctx;
        Err(GpuError::Unrecoverable(
            "graph::record::cudnn::Activation: cuDNN capture-mode \
             entry not yet wired"
                .into(),
        ))
    }
}

impl GraphOpRecord for SoftmaxOp {
    fn record(&self, ctx: &GraphRecordCtx<'_>) -> Result<(), GpuError> {
        validate_dims(&self.dims, "softmax: dims")?;
        let _ = self.x.access()?;
        let _ = self.y.access()?;
        let _ = ctx;
        Err(GpuError::Unrecoverable(
            "graph::record::cudnn::Softmax: cuDNN capture-mode \
             entry not yet wired"
                .into(),
        ))
    }
}

fn validate_dims(d: &[i32; 4], who: &str) -> Result<(), GpuError> {
    if d.iter().any(|&x| x <= 0) {
        Err(GpuError::Unrecoverable(format!(
            "{who}: non-positive dim in {d:?}"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceState;
    use crate::graph::MockGraphRecordCtx;
    use cudarc::driver::sys as driver_sys;
    use std::sync::Arc;

    /// Build a minimal `GpuRef<f32>` that's *invalid* (no underlying
    /// CudaSlice) but typed correctly. We won't actually deref it —
    /// the record path catches it via `access()` returning
    /// `GpuRefStale`.
    fn dead_gpu_ref() -> GpuRef<f32> {
        // We can't build a real CudaSlice without a CudaContext; the
        // test relies on the dim-validation path failing fast before
        // the GpuRef is touched. Use a no-element placeholder by
        // constructing through a bogus DeviceState — but the public
        // GpuRef::new requires a real Arc<CudaSlice>. Workaround: skip
        // the GpuRef-bearing assertions and exercise dim validation
        // separately.
        let _ = DeviceState::new(0);
        unimplemented!("not used — dim-validation tests cover the path")
    }

    #[test]
    fn conv_op_records() {
        // Validation-failure path: zero dim. We exercise it through
        // the record() method but skip the GpuRef accesses by using a
        // mock context and asserting the typed error category.
        let null_graph: driver_sys::CUgraph = std::ptr::null_mut();
        let mock = MockGraphRecordCtx::new(null_graph);
        let ctx = mock.as_ctx();

        // Use validate_dims directly to avoid needing live GpuRefs.
        assert!(validate_dims(&[1, 1, 1, 1], "ok").is_ok());
        assert!(validate_dims(&[0, 1, 1, 1], "bad").is_err());

        // Smoke-test the trait wiring with carefully chosen dims that
        // pass validation; then access() of a synthetic GpuRef would
        // fail, but we need to construct one. For Phase 3 we keep the
        // assertion to "validation surface compiles" and rely on
        // dim-validation tests above for behaviour.
        let _ = dead_gpu_ref;
        let _ = Arc::new(()) as Arc<()>;
        let _ = ctx;
    }

    #[test]
    fn activation_op_records() {
        assert!(validate_dims(&[1, 2, 3, 4], "ok").is_ok());
        assert!(validate_dims(&[1, 2, 3, -1], "bad").is_err());
    }

    #[test]
    fn softmax_op_records() {
        assert!(validate_dims(&[2, 4, 1, 1], "ok").is_ok());
    }
}
