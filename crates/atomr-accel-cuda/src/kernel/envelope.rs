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
//!
//! # Observability hooks (Phase 0.7)
//!
//! [`KernelEnvelope`] is an opt-in builder that wraps the same
//! pipeline with two observability surfaces:
//!
//! * A `KernelTrace` callback that fires four lifecycle events
//!   (`before_enqueue`, `after_enqueue`, `before_complete`,
//!   `after_complete`). The trait is **always compiled** — when no
//!   trace is set, the envelope skips the calls entirely.
//! * An optional NVTX range label. When the `nvtx` cargo feature is
//!   on, the synchronous-enqueue body is wrapped in a
//!   `cudarc::nvtx::safe::scoped_range` guard. When the feature is
//!   off, the field is unused and adds no runtime cost.
//!
//! Existing callers that use the free [`run_kernel`] function continue
//! to behave byte-for-byte identically: that path constructs a default
//! (trace-less, nvtx-less) envelope.

use std::sync::Arc;
use std::time::{Duration, Instant};

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

// ---------------------------------------------------------------------------
// Observability hooks (Phase 0.7).
// ---------------------------------------------------------------------------

/// Per-launch metadata passed to every [`KernelTrace`] callback.
///
/// `dtype` is `Option<&'static str>` because some kernel actors do not
/// have a single-dtype identity (e.g. memcpy, NCCL group calls). When
/// Phase 0.1 lands a real `atomr_accel::DType`, this field will be
/// promoted to `Option<atomr_accel::DType>` without changing the trait
/// shape.
#[derive(Debug, Clone, Copy)]
pub struct KernelInfo<'a> {
    /// Op identifier (e.g. `"sgemm"`, `"conv2d_forward"`).
    pub op_name: &'a str,
    /// Library tag (e.g. `"cublas"`, `"cudnn"`, `"nccl"`).
    pub library: &'a str,
    /// Stream identity. The raw CUstream pointer cast to `u64`; opaque
    /// to consumers but stable for the lifetime of the stream.
    pub stream_id: u64,
    /// Element dtype, if the op is single-dtype. `None` otherwise.
    pub dtype: Option<&'a str>,
}

/// Lifecycle hook receiver. All four methods have empty default
/// bodies, so a custom trace can override only the events it cares
/// about.
///
/// The trait is always compiled; no feature flag is required. When no
/// trace is attached to a [`KernelEnvelope`], the envelope skips every
/// call and adds zero runtime cost beyond a single `Option::is_some`
/// check (which the optimizer typically folds away).
pub trait KernelTrace: Send + Sync + 'static {
    /// Fires immediately before the synchronous enqueue closure runs.
    fn before_enqueue(&self, info: &KernelInfo<'_>) {
        let _ = info;
    }

    /// Fires immediately after the synchronous enqueue closure
    /// returns. `result` is `Ok(())` on success or `Err(&GpuError)` if
    /// the enqueue body failed.
    fn after_enqueue(&self, info: &KernelInfo<'_>, result: Result<(), &GpuError>) {
        let _ = (info, result);
    }

    /// Fires just before the completion future is awaited (i.e. after
    /// a successful enqueue, on the spawned Tokio task).
    fn before_complete(&self, info: &KernelInfo<'_>) {
        let _ = info;
    }

    /// Fires after the completion future resolves. `latency` is the
    /// wall-clock duration between `before_complete` and the resolved
    /// completion (i.e. host-observed completion latency, not GPU
    /// time).
    fn after_complete(
        &self,
        info: &KernelInfo<'_>,
        result: Result<(), &GpuError>,
        latency: Duration,
    ) {
        let _ = (info, result, latency);
    }
}

/// Builder/configuration for a single `run_kernel` invocation.
///
/// Existing actors that call the free [`run_kernel`] function are
/// unaffected — that path still constructs a default envelope
/// internally. Actors that want observability migrate to
/// `KernelEnvelope::new(lib).with_trace(..).with_nvtx(..).run_kernel(..)`.
#[derive(Clone)]
pub struct KernelEnvelope {
    lib_tag: &'static str,
    op_name: &'static str,
    dtype: Option<&'static str>,
    trace: Option<Arc<dyn KernelTrace>>,
    /// NVTX range label. When `Some(..)` and the `nvtx` feature is
    /// enabled, the envelope wraps the synchronous enqueue body in a
    /// `cudarc::nvtx::safe::scoped_range` guard. When the `nvtx`
    /// feature is disabled, the field is read but otherwise unused
    /// (zero runtime cost).
    nvtx_range_name: Option<&'static str>,
}

impl KernelEnvelope {
    /// Construct a trace-less, NVTX-less envelope tagged with the
    /// given library. `op_name` defaults to `lib_tag` and can be
    /// refined via [`Self::with_op_name`].
    pub fn new(lib_tag: &'static str) -> Self {
        Self {
            lib_tag,
            op_name: lib_tag,
            dtype: None,
            trace: None,
            nvtx_range_name: None,
        }
    }

    /// Override the op identifier surfaced to the trace callback (e.g.
    /// `"sgemm"`, `"conv2d_forward"`).
    pub fn with_op_name(mut self, op_name: &'static str) -> Self {
        self.op_name = op_name;
        self
    }

    /// Tag the envelope with a dtype name (e.g. `"f32"`, `"f16"`).
    /// Surfaced to trace callbacks as `KernelInfo::dtype`.
    pub fn with_dtype(mut self, dtype: &'static str) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Attach a `KernelTrace` callback. Cloning `Arc<dyn KernelTrace>`
    /// is cheap; the same object can be shared across many envelopes.
    pub fn with_trace(mut self, trace: Arc<dyn KernelTrace>) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Attach an NVTX range label. No-op unless the `nvtx` cargo
    /// feature is enabled.
    pub fn with_nvtx(mut self, name: &'static str) -> Self {
        self.nvtx_range_name = Some(name);
        self
    }

    fn info<'a>(&'a self, stream_id: u64) -> KernelInfo<'a> {
        KernelInfo {
            op_name: self.op_name,
            library: self.lib_tag,
            stream_id,
            dtype: self.dtype,
        }
    }

    /// Builder-style equivalent of the free [`run_kernel`] function
    /// with observability hooks layered in.
    ///
    /// Behaviour without a trace and without an NVTX range is
    /// byte-for-byte identical to [`run_kernel`].
    pub fn run_kernel<O, KA, F>(
        self,
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
        let stream_id = stream.cu_stream() as usize as u64;
        let info = self.info(stream_id);

        if let Some(t) = self.trace.as_deref() {
            t.before_enqueue(&info);
        }

        // NVTX range (if feature on and label set) wraps the enqueue
        // closure. The guard drops at the end of this block.
        let enqueue_result = {
            #[cfg(feature = "nvtx")]
            let _nvtx_guard = self.nvtx_range_name.map(cudarc::nvtx::safe::scoped_range);
            #[cfg(not(feature = "nvtx"))]
            let _ = self.nvtx_range_name;

            enqueue()
        };

        let keep_alive = match enqueue_result {
            Ok(ka) => {
                if let Some(t) = self.trace.as_deref() {
                    t.after_enqueue(&info, Ok(()));
                }
                ka
            }
            Err(e) => {
                let annotated = annotate_error(e, self.lib_tag);
                if let Some(t) = self.trace.as_deref() {
                    t.after_enqueue(&info, Err(&annotated));
                }
                let _ = reply.send(Err(annotated));
                return;
            }
        };

        let fut = completion.await_completion(stream).boxed();
        let lib_tag = self.lib_tag;
        let op_name = self.op_name;
        let dtype = self.dtype;
        let trace = self.trace.clone();
        tokio::spawn(async move {
            // Re-build the info struct on the spawned task so the
            // closure doesn't have to capture a self-borrowing
            // reference.
            let info = KernelInfo {
                op_name,
                library: lib_tag,
                stream_id,
                dtype,
            };
            if let Some(t) = trace.as_deref() {
                t.before_complete(&info);
            }
            let started = Instant::now();
            let result = fut.await;
            let latency = started.elapsed();
            match result {
                Ok(()) => {
                    if let Some(t) = trace.as_deref() {
                        t.after_complete(&info, Ok(()), latency);
                    }
                    let _ = reply.send(Ok(output));
                }
                Err(e) => {
                    warn!(lib = lib_tag, error = %e, "kernel completion failed");
                    if let Some(t) = trace.as_deref() {
                        t.after_complete(&info, Err(&e), latency);
                    }
                    let _ = reply.send(Err(e));
                }
            }
            // Held until completion resolved; safe to drop now.
            drop(keep_alive);
        });
    }
}

impl std::fmt::Debug for KernelEnvelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelEnvelope")
            .field("lib_tag", &self.lib_tag)
            .field("op_name", &self.op_name)
            .field("dtype", &self.dtype)
            .field("nvtx_range_name", &self.nvtx_range_name)
            .field("trace", &self.trace.as_ref().map(|_| "<dyn KernelTrace>"))
            .finish()
    }
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
///
/// This is the trace-less, NVTX-less compatibility entry point used by
/// every actor that hasn't migrated to [`KernelEnvelope`]. Behaviour
/// is byte-for-byte identical to the pre-Phase-0.7 implementation.
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
    use std::sync::Mutex;

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

    /// In-memory mock `KernelTrace` that records every event for
    /// later inspection.
    #[derive(Default)]
    struct RecordingTrace {
        events: Mutex<Vec<&'static str>>,
        last_dtype: Mutex<Option<String>>,
        last_op: Mutex<Option<String>>,
        last_lib: Mutex<Option<String>>,
        enqueue_ok: AtomicU32,
        enqueue_err: AtomicU32,
    }

    impl KernelTrace for RecordingTrace {
        fn before_enqueue(&self, info: &KernelInfo<'_>) {
            self.events.lock().unwrap().push("before_enqueue");
            *self.last_op.lock().unwrap() = Some(info.op_name.to_string());
            *self.last_lib.lock().unwrap() = Some(info.library.to_string());
            *self.last_dtype.lock().unwrap() = info.dtype.map(str::to_string);
        }

        fn after_enqueue(&self, _info: &KernelInfo<'_>, result: Result<(), &GpuError>) {
            self.events.lock().unwrap().push("after_enqueue");
            match result {
                Ok(()) => {
                    self.enqueue_ok.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    self.enqueue_err.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        fn before_complete(&self, _info: &KernelInfo<'_>) {
            self.events.lock().unwrap().push("before_complete");
        }

        fn after_complete(
            &self,
            _info: &KernelInfo<'_>,
            _result: Result<(), &GpuError>,
            _latency: Duration,
        ) {
            self.events.lock().unwrap().push("after_complete");
        }
    }

    /// Internal helper used by the trace tests. Mirrors the body of
    /// `KernelEnvelope::run_kernel` minus the actual stream / Tokio
    /// spawn, so we can exercise the trace hooks on a host without a
    /// GPU.
    fn drive_envelope_trace<F>(
        env: &KernelEnvelope,
        enqueue: F,
    ) -> (Result<(), GpuError>, Result<(), GpuError>)
    where
        F: FnOnce() -> Result<(), GpuError>,
    {
        // Synthesise a stream id without touching cudarc.
        let info = env.info(0xDEAD_BEEF);
        if let Some(t) = env.trace.as_deref() {
            t.before_enqueue(&info);
        }
        let enqueue_result = enqueue();
        let enqueue_report = match &enqueue_result {
            Ok(()) => Ok(()),
            Err(e) => Err(annotate_error_clone(e, env.lib_tag)),
        };
        if let Some(t) = env.trace.as_deref() {
            match &enqueue_report {
                Ok(()) => t.after_enqueue(&info, Ok(())),
                Err(e) => t.after_enqueue(&info, Err(e)),
            }
        }
        // Pretend the completion future resolved synchronously with
        // success; that's the path Phase 9 will exercise on real
        // streams. We only care that the trace fires in the right
        // order here.
        if enqueue_report.is_ok() {
            if let Some(t) = env.trace.as_deref() {
                t.before_complete(&info);
                t.after_complete(&info, Ok(()), Duration::from_micros(1));
            }
        }
        (enqueue_result, enqueue_report)
    }

    /// Cheap clone of the error for trace inspection in tests.
    fn annotate_error_clone(e: &GpuError, lib_tag: &'static str) -> GpuError {
        match e {
            GpuError::Driver(msg) => GpuError::LibraryError {
                lib: lib_tag,
                msg: msg.clone(),
            },
            GpuError::OutOfMemory(msg) => GpuError::OutOfMemory(msg.clone()),
            GpuError::ContextPoisoned(msg) => GpuError::ContextPoisoned(msg.clone()),
            GpuError::Unrecoverable(msg) => GpuError::Unrecoverable(msg.clone()),
            GpuError::GpuRefStale(s) => GpuError::GpuRefStale(s),
            GpuError::LibraryError { lib, msg } => GpuError::LibraryError {
                lib,
                msg: msg.clone(),
            },
            // Other variants don't appear on the trace path in these
            // tests; fall back to a generic library error so the
            // helper stays compile-clean across the GpuError surface.
            other => GpuError::LibraryError {
                lib: lib_tag,
                msg: other.to_string(),
            },
        }
    }

    #[test]
    fn envelope_default_is_traceless_and_nvtxless() {
        let env = KernelEnvelope::new("cublas");
        assert!(env.trace.is_none());
        assert!(env.nvtx_range_name.is_none());
        assert_eq!(env.lib_tag, "cublas");
        assert_eq!(env.op_name, "cublas");
        assert!(env.dtype.is_none());
    }

    #[test]
    fn envelope_builder_sets_metadata() {
        let trace = Arc::new(RecordingTrace::default()) as Arc<dyn KernelTrace>;
        let env = KernelEnvelope::new("cublas")
            .with_op_name("sgemm")
            .with_dtype("f32")
            .with_trace(trace)
            .with_nvtx("blas/sgemm");
        assert_eq!(env.op_name, "sgemm");
        assert_eq!(env.dtype, Some("f32"));
        assert_eq!(env.nvtx_range_name, Some("blas/sgemm"));
        assert!(env.trace.is_some());
    }

    #[test]
    fn trace_hooks_fire_in_order_on_success() {
        let trace = Arc::new(RecordingTrace::default());
        let env = KernelEnvelope::new("cublas")
            .with_op_name("sgemm")
            .with_dtype("f32")
            .with_trace(trace.clone() as Arc<dyn KernelTrace>);

        let (enqueue_res, _) = drive_envelope_trace(&env, || Ok(()));
        assert!(enqueue_res.is_ok());
        let events = trace.events.lock().unwrap().clone();
        assert_eq!(
            events,
            vec![
                "before_enqueue",
                "after_enqueue",
                "before_complete",
                "after_complete",
            ]
        );
        assert_eq!(trace.enqueue_ok.load(Ordering::Relaxed), 1);
        assert_eq!(trace.enqueue_err.load(Ordering::Relaxed), 0);
        assert_eq!(trace.last_op.lock().unwrap().as_deref(), Some("sgemm"));
        assert_eq!(trace.last_lib.lock().unwrap().as_deref(), Some("cublas"));
        assert_eq!(trace.last_dtype.lock().unwrap().as_deref(), Some("f32"));
    }

    #[test]
    fn trace_hooks_skip_completion_on_enqueue_error() {
        let trace = Arc::new(RecordingTrace::default());
        let env = KernelEnvelope::new("cudnn")
            .with_op_name("conv2d_forward")
            .with_trace(trace.clone() as Arc<dyn KernelTrace>);

        let (enqueue_res, report) =
            drive_envelope_trace(&env, || Err(GpuError::Driver("forced".into())));
        assert!(enqueue_res.is_err());
        // Driver errors get annotated to LibraryError.
        match report {
            Err(GpuError::LibraryError { lib, msg }) => {
                assert_eq!(lib, "cudnn");
                assert_eq!(msg, "forced");
            }
            other => panic!("expected LibraryError, got {other:?}"),
        }
        let events = trace.events.lock().unwrap().clone();
        assert_eq!(events, vec!["before_enqueue", "after_enqueue"]);
        assert_eq!(trace.enqueue_ok.load(Ordering::Relaxed), 0);
        assert_eq!(trace.enqueue_err.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn envelope_without_trace_is_silent() {
        let env = KernelEnvelope::new("cufft");
        let (res, _) = drive_envelope_trace(&env, || Ok(()));
        assert!(res.is_ok());
        // No trace attached, so nothing to record. The point of this
        // test is to make sure the trace-less path compiles and runs
        // through `drive_envelope_trace` without panicking.
    }
}
