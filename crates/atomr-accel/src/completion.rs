//! `CompletionStrategy` — async wakeup contract for kernel
//! completion. Promoted to the core so every backend can plug into
//! the same family (host-fn callback, sync block, polled query).

use async_trait::async_trait;

use crate::backend::AccelBackend;
use crate::error::AccelError;

/// Strategy for awaiting kernel completion on a stream.
///
/// Three canonical implementations live in each backend:
/// - **HostFn** — `cuLaunchHostFunc` / `hipLaunchHostFunc` /
///   `MTLCommandBuffer.addCompletedHandler` callback. Sub-µs
///   wakeup, no host-side blocking.
/// - **Sync** — explicit `cudaStreamSynchronize`. Easy reasoning,
///   parks a host thread.
/// - **Polled** — periodic `cuEventQuery` with a timeout. Hard
///   upper bound at the cost of a polling loop.
#[async_trait]
pub trait CompletionStrategy<B: AccelBackend>: Send + Sync + 'static {
    /// Resolve once every kernel previously enqueued on `stream`
    /// has finished. `Ok(())` on completion; `Err` if the device
    /// poisoned mid-flight.
    async fn await_completion(&self, stream: &B::Stream) -> Result<(), AccelError>;
}
