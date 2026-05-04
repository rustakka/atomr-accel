//! `SyncCompletion` (§5.10 `BlockingCompletion`) — blocks a dedicated
//! thread in `cudaStreamSynchronize`. **Stub for F1.** Used by the
//! deterministic-replay harness once that ships in B1.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::FutureExt;

use crate::error::GpuError;

use super::CompletionStrategy;

#[derive(Clone, Default)]
pub struct SyncCompletion;

impl SyncCompletion {
    pub fn new() -> Self {
        Self
    }
}

impl CompletionStrategy for SyncCompletion {
    fn await_completion(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> BoxFuture<'static, Result<(), GpuError>> {
        let stream = stream.clone();
        async move {
            // tokio::task::spawn_blocking is the right tool here — we
            // genuinely block waiting for the GPU. F1 keeps this minimal;
            // the production path is HostFnCompletion.
            tokio::task::spawn_blocking(move || stream.synchronize())
                .await
                .map_err(|e| GpuError::Driver(format!("sync-completion task: {e}")))?
                .map_err(|e| GpuError::Driver(format!("cudaStreamSynchronize: {e}")))
        }
        .boxed()
    }
}
