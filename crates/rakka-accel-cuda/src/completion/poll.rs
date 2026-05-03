//! `PolledCompletion` (§5.10) — periodic `cuEventQuery`-style polling.
//!
//! Useful where `cuLaunchHostFunc` (used by [`super::HostFnCompletion`])
//! is forbidden by deployment policy. Trade-off: every outstanding
//! kernel costs one tokio task waking on a timer.
//!
//! Implementation: `await_completion` records a `CudaEvent` on the
//! supplied stream, then drives a tokio sleep loop calling
//! `event.is_complete()` at `interval` until it returns true.

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;
use futures_util::FutureExt;

use crate::error::GpuError;

use super::CompletionStrategy;

#[derive(Clone, Debug)]
pub struct PolledCompletion {
    pub interval: Duration,
    /// Hard cap on total wait time. `None` = unbounded. The bound
    /// is necessary because a stuck driver could otherwise spin
    /// forever; default 5 minutes.
    pub timeout: Option<Duration>,
}

impl PolledCompletion {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            timeout: Some(Duration::from_secs(300)),
        }
    }
}

impl Default for PolledCompletion {
    fn default() -> Self {
        Self::new(Duration::from_micros(50))
    }
}

impl CompletionStrategy for PolledCompletion {
    fn await_completion(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> BoxFuture<'static, Result<(), GpuError>> {
        let stream = stream.clone();
        let interval = self.interval;
        let timeout = self.timeout;
        async move {
            // Record an event after all currently-queued work on the
            // stream. Catch panics from the FFI loader on no-driver
            // hosts and surface them as a typed error.
            let event_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stream.record_event(None)
            }));
            let event = match event_res {
                Ok(Ok(e)) => e,
                Ok(Err(e)) => {
                    return Err(GpuError::LibraryError {
                        lib: "driver",
                        msg: format!("PolledCompletion: record_event: {e}"),
                    });
                }
                Err(_) => {
                    return Err(GpuError::Unrecoverable(
                        "PolledCompletion: CUDA driver not loadable".into(),
                    ));
                }
            };
            let started = std::time::Instant::now();
            loop {
                let complete = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    event.is_complete()
                }))
                .unwrap_or(false);
                if complete {
                    return Ok(());
                }
                if let Some(t) = timeout {
                    if started.elapsed() >= t {
                        return Err(GpuError::Timeout);
                    }
                }
                tokio::time::sleep(interval).await;
            }
        }
        .boxed()
    }
}
