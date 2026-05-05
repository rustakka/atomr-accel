//! Device-to-device memcpy op for [`super::super::GraphOp`].
//!
//! Wraps [`crate::kernel::record::MemcpyOp`] /
//! [`crate::kernel::record::MemcpyRecorder`] into a single
//! `GraphOp`-implementing type.

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::graph::{GraphOp, GraphRecordCtx};
use crate::kernel::record::{MemcpyOp as InnerMemcpyOp, MemcpyRecorder, RecordMode};

/// Device-to-device memcpy op for the captured stream. Capture-safe.
///
/// `record` consumes the held `GpuRef`s on first invocation; a
/// second call returns [`GpuError::Unrecoverable`].
pub struct MemcpyOp {
    inner: Option<InnerMemcpyOp>,
}

impl MemcpyOp {
    pub fn new(src: GpuRef<f32>, dst: GpuRef<f32>) -> Self {
        Self {
            inner: Some(InnerMemcpyOp { src, dst }),
        }
    }
}

impl GraphOp for MemcpyOp {
    fn record(&mut self, ctx: &mut GraphRecordCtx<'_>) -> Result<(), GpuError> {
        let stream = ctx.require_stream()?;
        let op = self
            .inner
            .take()
            .ok_or_else(|| GpuError::Unrecoverable("MemcpyOp::record: already consumed".into()))?;
        let mut recorder = MemcpyRecorder;
        recorder.enqueue_record(stream, op)
    }

    fn op_name(&self) -> &'static str {
        "graph::memcpy"
    }
}
