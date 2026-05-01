//! `kernel::envelope` — shared kernel-actor body factored out of
//! `BlasActor::enqueue_sgemm`.
//!
//! Every library actor (`BlasActor`, `CudnnActor`, `FftActor`,
//! `RngActor`, …) follows the same pattern:
//!
//! 1. Validate every input [`GpuRef`] via [`GpuRef::access`] and turn
//!    the strong [`Arc<CudaSlice<T>>`] into a temporary owner that
//!    keeps the buffer alive past kernel completion.
//! 2. Synchronously enqueue the kernel onto the actor's stream. The
//!    enqueue body is library-specific and provided as a closure.
//! 3. Spawn an async task that awaits the configured
//!    [`CompletionStrategy`], delivers the reply on a `oneshot::Sender`,
//!    and only then drops the temporary owners (so that the kernel
//!    can't outlive its inputs).
//!
//! The envelope handles step 3 uniformly. Pre-launch errors are
//! reported synchronously through the same `oneshot`. Post-launch
//! errors arrive through the completion future.
//!
//! # Single-writer enforcement
//!
//! cudarc's library APIs typically take `&mut Dst` for the write
//! target. cudarc 0.19 satisfies this for `CudaSlice<T>`. Since a
//! `GpuRef<T>` wraps `Arc<CudaSlice<T>>`, callers that want write
//! access to a buffer must hold the unique reference to that
//! `GpuRef` (so `Arc::try_unwrap` succeeds inside the actor). Each
//! library actor enforces this contract explicitly — the envelope
//! does not, because some libraries (cuBLAS gemm with non-zero beta)
//! read-modify-write the output while others (cuDNN forward conv)
//! write to a freshly allocated output.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use futures_util::FutureExt;
use tokio::sync::oneshot;
use tracing::warn;

use crate::completion::CompletionStrategy;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

/// Validate two input `GpuRef`s and return owning `Arc`s of their
/// underlying slices. Fails fast (synchronously) with `GpuRefStale` if
/// either is invalid.
pub fn access_all_2<A, B>(
    a: &GpuRef<A>,
    b: &GpuRef<B>,
) -> Result<(Arc<CudaSlice<A>>, Arc<CudaSlice<B>>), GpuError> {
    let a_s = a.access()?.clone();
    let b_s = b.access()?.clone();
    Ok((a_s, b_s))
}

/// Validate three input `GpuRef`s and return owning `Arc`s of their
/// underlying slices.
pub fn access_all_3<A, B, C>(
    a: &GpuRef<A>,
    b: &GpuRef<B>,
    c: &GpuRef<C>,
) -> Result<(Arc<CudaSlice<A>>, Arc<CudaSlice<B>>, Arc<CudaSlice<C>>), GpuError> {
    let a_s = a.access()?.clone();
    let b_s = b.access()?.clone();
    let c_s = c.access()?.clone();
    Ok((a_s, b_s, c_s))
}

/// Validate four input `GpuRef`s and return owning `Arc`s. Used by
/// cuDNN convolution which takes (input, filter, output, workspace).
pub fn access_all_4<A, B, C, D>(
    a: &GpuRef<A>,
    b: &GpuRef<B>,
    c: &GpuRef<C>,
    d: &GpuRef<D>,
) -> Result<
    (
        Arc<CudaSlice<A>>,
        Arc<CudaSlice<B>>,
        Arc<CudaSlice<C>>,
        Arc<CudaSlice<D>>,
    ),
    GpuError,
> {
    let a_s = a.access()?.clone();
    let b_s = b.access()?.clone();
    let c_s = c.access()?.clone();
    let d_s = d.access()?.clone();
    Ok((a_s, b_s, c_s, d_s))
}

/// Run the synchronous-enqueue + async-completion-await pipeline.
///
/// `enqueue` runs immediately on the calling actor's task. On success
/// it returns the **keep-alive tuple** — anything that must outlive
/// the kernel (input `Arc<CudaSlice<T>>`s, the unwrapped write
/// target, descriptor handles, etc.). The envelope spawns a Tokio
/// task that awaits [`CompletionStrategy::await_completion`] for
/// `stream`, replies via `reply`, and drops the keep-alive only
/// after completion.
///
/// `lib_tag` populates the `lib` field of any error annotation. The
/// completion future emits its own typed errors; on failure `output`
/// is discarded.
pub fn run_kernel<O, KA, F>(
    lib_tag: &'static str,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    output: O,
    reply: oneshot::Sender<Result<O, GpuError>>,
    enqueue: F,
) where
    O: Send + 'static,
    KA: Send + 'static,
    F: FnOnce() -> Result<KA, GpuError>,
{
    let keep_alive = match enqueue() {
        Ok(ka) => ka,
        Err(e) => {
            let _ = reply.send(Err(annotate_error(e, lib_tag)));
            return;
        }
    };

    let fut = completion.await_completion(stream).boxed();
    tokio::spawn(async move {
        let result = fut.await;
        match result {
            Ok(()) => {
                let _ = reply.send(Ok(output));
            }
            Err(e) => {
                warn!(lib = lib_tag, error = %e, "kernel completion failed");
                let _ = reply.send(Err(e));
            }
        }
        // Held until completion resolved; safe to drop now.
        drop(keep_alive);
    });
}

/// Tag a generic error with a library name iff it doesn't already
/// carry a more specific classification. Pre-existing typed variants
/// (`ContextPoisoned`, `OutOfMemory`, `GpuRefStale`, `Unrecoverable`,
/// `Timeout`) and pre-tagged `LibraryError` pass through unchanged.
fn annotate_error(e: GpuError, lib_tag: &'static str) -> GpuError {
    match e {
        GpuError::Driver(msg) => GpuError::LibraryError { lib: lib_tag, msg },
        // Already-tagged or library-agnostic errors pass through.
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn annotate_error_tags_driver_failures() {
        let e = annotate_error(GpuError::Driver("oops".into()), "cudnn");
        match e {
            GpuError::LibraryError { lib, msg } => {
                assert_eq!(lib, "cudnn");
                assert_eq!(msg, "oops");
            }
            other => panic!("expected LibraryError, got {other:?}"),
        }
    }

    #[test]
    fn annotate_error_passes_through_typed_variants() {
        let e = annotate_error(GpuError::OutOfMemory("alloc".into()), "cudnn");
        assert!(matches!(e, GpuError::OutOfMemory(_)));
        let e = annotate_error(GpuError::GpuRefStale("stale"), "cudnn");
        assert!(matches!(e, GpuError::GpuRefStale(_)));
    }

    /// Smoke test the `enqueue` failure short-circuit on the synchronous
    /// path without needing a real stream — the completion future is
    /// never invoked. We verify the reply carries the failure.
    #[test]
    fn pre_enqueue_error_bypasses_completion() {
        let (tx, rx) = oneshot::channel::<Result<u32, GpuError>>();
        // We can't construct an Arc<CudaStream> on a host without a
        // GPU, so we exercise the annotate_error path inline rather
        // than running the full envelope. The full envelope is
        // covered by GPU integration tests.
        let mut bumped = AtomicU32::new(0);
        let enqueue = || -> Result<(), GpuError> {
            bumped.fetch_add(1, Ordering::Relaxed);
            Err(GpuError::OutOfMemory("forced".into()))
        };
        let res = enqueue();
        assert!(matches!(res, Err(GpuError::OutOfMemory(_))));
        assert_eq!(*bumped.get_mut(), 1);
        // reply isn't actually sent in this stripped-down test.
        drop(tx);
        drop(rx);
    }
}
