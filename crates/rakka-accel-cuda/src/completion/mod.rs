//! Completion strategies (§5.10).
//!
//! `CompletionStrategy` decides how the runtime detects that a stream
//! has finished its outstanding work. The F1 default is
//! `HostFnCompletion`, which uses `cuLaunchHostFunc` to schedule a
//! callback that wakes a Tokio waker — sub-microsecond latency, no
//! polling overhead, scales to many concurrent operations.
//!
//! Two fallback strategies are present as stubs for F2: `PolledCompletion`
//! for environments that block host-functions, and `SyncCompletion` for
//! debugging / deterministic-replay testing.

mod host_fn;
mod poll;
mod sync;

pub use host_fn::HostFnCompletion;
pub use poll::PolledCompletion;
pub use sync::SyncCompletion;

use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::error::GpuError;

pub trait CompletionStrategy: Send + Sync {
    /// Return a future that resolves when all preceding work on `stream`
    /// has completed. Implementations differ in how completion is
    /// detected.
    fn await_completion(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> BoxFuture<'static, Result<(), GpuError>>;
}
