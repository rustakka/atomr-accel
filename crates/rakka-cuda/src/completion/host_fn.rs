//! `HostFnCompletion` (§5.10) — callback-based completion detection.
//!
//! After enqueueing a kernel onto a `CudaStream`, register a
//! `cuLaunchHostFunc` callback that runs after all preceding work on the
//! stream completes. The callback fulfils a `oneshot::Sender<Result>`,
//! which the actor's `await_completion` future awaits.
//!
//! The CUDA-side callback context is severely restricted (no CUDA API
//! calls, no most locks, no blocking). The trampoline below does only:
//!
//! - reconstruct an `Arc<oneshot::Sender>` from a raw pointer,
//! - send `Ok(())` on it,
//!
//! all of which are permitted.

use std::ffi::c_void;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use tokio::sync::oneshot;

use crate::error::GpuError;

use super::CompletionStrategy;

#[derive(Clone, Default)]
pub struct HostFnCompletion;

impl HostFnCompletion {
    pub fn new() -> Self {
        Self
    }
}

/// Trampoline invoked by CUDA on the stream callback worker thread. It
/// reconstructs the boxed `oneshot::Sender` and signals completion.
///
/// SAFETY: `data` must have been produced by `Box::into_raw(Box::new(slot))`
/// in [`HostFnCompletion::await_completion`]; CUDA invokes this exactly
/// once per `cuLaunchHostFunc` registration.
unsafe extern "C" fn wake_trampoline(data: *mut c_void) {
    if data.is_null() {
        return;
    }
    // Reclaim the box and drop, which fulfils the oneshot.
    let slot: Box<oneshot::Sender<Result<(), GpuError>>> = Box::from_raw(data.cast());
    let _ = slot.send(Ok(()));
}

impl CompletionStrategy for HostFnCompletion {
    fn await_completion(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> BoxFuture<'static, Result<(), GpuError>> {
        let stream = stream.clone();
        let (tx, rx) = oneshot::channel::<Result<(), GpuError>>();
        let boxed = Box::new(tx);
        let arg = Box::into_raw(boxed) as *mut c_void;

        // Use the documented `result::launch_host_function` wrapper.
        // This is unsafe because the callback signature is unconstrained
        // — we satisfy the requirements by hand: the trampoline does no
        // CUDA work and does not block.
        let launch_res = unsafe {
            cudarc::driver::result::stream::launch_host_function(
                stream.cu_stream(),
                wake_trampoline,
                arg,
            )
        };

        if let Err(e) = launch_res {
            // Reclaim the box so we don't leak the sender; this also
            // drops the oneshot, so `rx.await` returns a closed-channel
            // error that we map to a typed failure.
            unsafe {
                drop(Box::from_raw(arg as *mut oneshot::Sender<Result<(), GpuError>>));
            }
            let msg = format!("cuLaunchHostFunc failed: {e}");
            return async move { Err(GpuError::Driver(msg)) }.boxed();
        }

        async move {
            match rx.await {
                Ok(r) => r,
                Err(_) => Err(GpuError::Driver(
                    "host-function callback dropped without firing".into(),
                )),
            }
        }
        .boxed()
    }
}
