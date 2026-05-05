//! 1-D R2C FFT op for [`super::super::GraphOp`].
//!
//! Wraps [`crate::kernel::record::FftR2COp`] /
//! [`crate::kernel::record::FftRecorder`] into a single
//! `GraphOp`-implementing type. Gated on the `cufft` feature.

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::graph::{GraphOp, GraphRecordCtx};
use crate::kernel::record::{FftR2COp as InnerFftR2COp, FftRecorder, RecordMode};

/// 1-D R2C FFT op for graph capture. The user installs a pre-built
/// `cudarc::cufft::CudaFft` plan via `GraphMsg::SetFftPlan` before
/// recording; the plan is borrowed through `GraphRecordCtx::fft`.
///
/// `record` consumes the held `GpuRef`s on first invocation; a
/// second call returns [`GpuError::Unrecoverable`].
pub struct FftR2COp {
    inner: Option<InnerFftR2COp>,
}

impl FftR2COp {
    pub fn new(src: GpuRef<f32>, dst: GpuRef<cudarc::cufft::sys::float2>) -> Self {
        Self {
            inner: Some(InnerFftR2COp { src, dst }),
        }
    }
}

impl GraphOp for FftR2COp {
    fn record(&mut self, ctx: &mut GraphRecordCtx<'_>) -> Result<(), GpuError> {
        let stream = ctx.require_stream()?;
        let plan = ctx.fft.ok_or_else(|| {
            GpuError::Unrecoverable(
                "FftR2COp::record: no cuFFT plan installed; call GraphMsg::SetFftPlan first".into(),
            )
        })?;
        let op = self
            .inner
            .take()
            .ok_or_else(|| GpuError::Unrecoverable("FftR2COp::record: already consumed".into()))?;
        let mut recorder = FftRecorder { plan };
        recorder.enqueue_record(stream, op)
    }

    fn op_name(&self) -> &'static str {
        "graph::fft_r2c"
    }
}
