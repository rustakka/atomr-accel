//! Uniform RNG fill op for [`super::super::GraphOp`].
//!
//! Wraps [`crate::kernel::record::RngFillUniformOp`] /
//! [`crate::kernel::record::RngRecorder`] into a single
//! `GraphOp`-implementing type. Gated on the `curand` feature.

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::graph::{GraphOp, GraphRecordCtx};
use crate::kernel::record::{RecordMode, RngFillUniformOp as InnerRngFillUniformOp, RngRecorder};

/// Uniform RNG fill op for graph capture.
///
/// The op needs a cuRAND handle on the captured stream, supplied
/// by `GraphRecordCtx::rng`. If absent the op fails with
/// [`GpuError::Unrecoverable`].
///
/// `record` consumes the held `GpuRef` on first invocation; a
/// second call returns [`GpuError::Unrecoverable`].
pub struct RngFillUniformOp {
    inner: Option<InnerRngFillUniformOp>,
}

impl RngFillUniformOp {
    pub fn new(dst: GpuRef<f32>) -> Self {
        Self {
            inner: Some(InnerRngFillUniformOp { dst }),
        }
    }
}

impl GraphOp for RngFillUniformOp {
    fn record(&mut self, ctx: &mut GraphRecordCtx<'_>) -> Result<(), GpuError> {
        let stream = ctx.require_stream()?;
        let rng = ctx.rng.ok_or_else(|| {
            GpuError::Unrecoverable(
                "RngFillUniformOp::record: cuRAND handle not available in ctx".into(),
            )
        })?;
        let op = self.inner.take().ok_or_else(|| {
            GpuError::Unrecoverable("RngFillUniformOp::record: already consumed".into())
        })?;
        let mut recorder = RngRecorder { rng };
        recorder.enqueue_record(stream, op)
    }

    fn op_name(&self) -> &'static str {
        "graph::rng_fill_uniform"
    }
}
