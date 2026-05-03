//! [`PipelineStage`] trait.

use std::sync::Arc;

use cudarc::driver::{CudaEvent, CudaStream};

use crate::error::GpuError;

/// One stage in a multi-stream GPU pipeline.
///
/// Implementations enqueue their kernel onto `stream` synchronously
/// (no host wait) and return a `CudaEvent` marking the completion of
/// that stage's GPU work, plus the typed output. The executor
/// arranges that the next stage's `wait_for` is the previous stage's
/// returned event, so cross-stage synchronization is on-device only.
pub trait PipelineStage: Send + 'static {
    type In: Send + 'static;
    type Out: Send + 'static;

    fn enqueue(
        &mut self,
        stream: &Arc<CudaStream>,
        wait_for: Option<&CudaEvent>,
        input: Self::In,
    ) -> Result<(CudaEvent, Self::Out), GpuError>;
}
