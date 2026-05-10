//! # atomr-accel-cub
//!
//! Device-wide CUB primitives surfaced as an atomr actor. CUB ships
//! template-heavy block / device-level reductions, scans, sorts,
//! histograms, and select / partition operators inside its
//! `<cub/device/...>` headers; we wrap them as per-(op, dtype) NVRTC
//! kernel sources so the actor compiles them lazily on first
//! invocation and replays the cubin on subsequent calls through the
//! Phase 0.6 disk cache shared with `atomr-accel-cuda`.
//!
//! ## Architecture
//!
//! [`CubActor`] is a child actor of an `atomr-accel-cuda::ContextActor`.
//! Construction goes through [`cub_props`], which stashes the actor
//! ref into [`atomr_accel_cuda::device::KernelChildren`] via
//! `register_extra::<ActorRef<CubMsg>>(...)`. The host side never owns
//! a CUB handle — every call boils down to:
//!
//! 1. Look up (or compile) the per-(op, dtype) NVRTC kernel source,
//! 2. Invoke it through `NvrtcActor` on the actor's stream,
//! 3. Reply via the `oneshot::Sender` carried by the dispatch payload.
//!
//! ## Mailbox surface
//!
//! [`CubMsg`] is a sum of seven boxed-dispatch variants — one per CUB
//! family. Each request is generic over `T: CudaDtype`; reductions are
//! parameterised by [`ReductionOp`] (Sum, Max, Min, ArgMax, ArgMin),
//! sorts by key/value dtype + ascending/descending, etc. The boxed
//! traits ([`CubReduceDispatch`], [`CubScanDispatch`], …) follow the
//! Phase 0.3 `*Dispatch` pattern from `atomr-accel-cuda`.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context, Props};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_accel_cuda::completion::CompletionStrategy;
use atomr_accel_cuda::device::DeviceState;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::kernel::nvrtc::SmArch;
use atomr_accel_cuda::kernel::{KernelHandle, NvrtcMsg};

pub mod dispatch;

pub mod histogram;
pub mod kernels;
pub mod reduce;
pub mod scan;
pub mod segmented;
pub mod select;
pub mod sort;

pub use histogram::{CubHistogramDispatch, HistogramRequest};
pub use reduce::{CubReduceDispatch, ReduceRequest, ReductionOp};
pub use scan::{CubScanDispatch, ScanKind, ScanRequest};
pub use segmented::{CubSegmentedReduceDispatch, SegmentedReduceRequest};
pub use select::{
    CubPartitionDispatch, CubSelectDispatch, PartitionRequest, SelectMode, SelectRequest,
};
pub use sort::{CubSortDispatch, SortDirection, SortRequest};

/// Public mailbox of [`CubActor`]. Each variant boxes a dispatch
/// trait whose concrete payload carries the typed `GpuRef<T>` inputs
/// and a `oneshot::Sender` for the reply.
pub enum CubMsg {
    Reduce(Box<dyn CubReduceDispatch>),
    Scan(Box<dyn CubScanDispatch>),
    Sort(Box<dyn CubSortDispatch>),
    Histogram(Box<dyn CubHistogramDispatch>),
    Select(Box<dyn CubSelectDispatch>),
    Partition(Box<dyn CubPartitionDispatch>),
    SegmentedReduce(Box<dyn CubSegmentedReduceDispatch>),
}

/// Per-call context bundle handed to every CUB dispatcher. Captures
/// the actor's stream, completion strategy, `DeviceState` snapshot,
/// and the (optional) `NvrtcActor` ref so dispatch impls can
/// JIT-compile the per-(op, dtype) kernel without knowing how the
/// parent supervisor wired things up.
///
/// All fields are passed by reference so the dispatcher can clone
/// just the bits it needs (the `Arc<ActorRef<NvrtcMsg>>`, the stream,
/// the kernel cache) into a spawned `tokio::spawn(...)` task that
/// drives the async compile + launch round-trip. The borrow does not
/// outlive the [`CubActor::handle`] call frame.
pub struct CubDispatchCtx<'a> {
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub state: &'a Arc<DeviceState>,
    pub ctx: &'a Arc<cudarc::driver::CudaContext>,
    /// In-actor `(op, dtype) → KernelHandle` cache. Wrapped in an
    /// `Arc<Mutex<…>>` so the dispatcher can clone it into a spawned
    /// task without a 'static-lifetime headache.
    pub kernel_cache: &'a Arc<Mutex<KernelSourceCache>>,
    /// Phase 5.1 — `NvrtcActor` ref used to compile + launch the
    /// per-(op, dtype) kernel. `None` when the actor was constructed
    /// without an NVRTC sibling (dispatchers in that mode reply with a
    /// structured `GpuError::Unrecoverable("CubActor: NvrtcActor not
    /// wired")`).
    pub nvrtc: Option<&'a Arc<ActorRef<NvrtcMsg>>>,
    /// Phase 5.1 — detected SM compute capability for the current
    /// device. Drives the `--gpu-architecture=…` flag passed into
    /// NVRTC. Defaults to [`SmArch::Sm80`] when detection fails (a
    /// reasonable Ampere-or-newer baseline).
    pub arch: SmArch,
}

/// In-actor cache mapping `(op, dtype)` to the NVRTC compile result.
/// The persistent disk cache (shared with `atomr-accel-cuda`'s
/// `NvrtcCache`) lives one level below; this map is the per-actor
/// hot-path lookup that avoids re-compiling and re-loading the
/// already-resolved [`KernelHandle`].
///
/// Phase 5.1 extends the cache to also hold the loaded
/// [`KernelHandle`]; the original PTX-bytes path stays for callers
/// (e.g. tests) that round-trip raw cubin without going through the
/// NvrtcActor.
#[derive(Default)]
pub struct KernelSourceCache {
    inner: std::collections::HashMap<(String, String), Arc<Vec<u8>>>,
    handles: std::collections::HashMap<(String, String), KernelHandle>,
}

impl KernelSourceCache {
    pub fn get(&self, op: &str, dtype: &str) -> Option<Arc<Vec<u8>>> {
        self.inner
            .get(&(op.to_string(), dtype.to_string()))
            .cloned()
    }
    pub fn insert(&mut self, op: &str, dtype: &str, ptx: Arc<Vec<u8>>) {
        self.inner.insert((op.to_string(), dtype.to_string()), ptx);
    }
    /// Phase 5.1 — fetch a previously compiled [`KernelHandle`] for
    /// `(op, dtype)`. Returns `None` on first invocation; the
    /// dispatcher then runs the NVRTC compile and inserts via
    /// [`Self::insert_handle`].
    pub fn get_handle(&self, op: &str, dtype: &str) -> Option<KernelHandle> {
        self.handles
            .get(&(op.to_string(), dtype.to_string()))
            .cloned()
    }
    /// Phase 5.1 — store a freshly compiled [`KernelHandle`] for
    /// future cache hits in the same actor lifetime.
    pub fn insert_handle(&mut self, op: &str, dtype: &str, handle: KernelHandle) {
        self.handles
            .insert((op.to_string(), dtype.to_string()), handle);
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Number of cached compiled kernel handles. Useful for tests that
    /// want to assert a second invocation hit the cache.
    pub fn handle_count(&self) -> usize {
        self.handles.len()
    }
}

/// `CubActor` — owns no CUDA handle of its own; every op routes
/// through NVRTC against the parent context.
pub struct CubActor {
    inner: CubInner,
}

enum CubInner {
    Real {
        ctx: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        kernel_cache: Arc<Mutex<KernelSourceCache>>,
        nvrtc: Option<Arc<ActorRef<NvrtcMsg>>>,
        arch: SmArch,
    },
    Mock,
}

impl CubActor {
    /// Build a [`Props`] for a CUB child of the given context.
    ///
    /// `nvrtc` is the parent context's `NvrtcActor` ref — passing
    /// `None` keeps the actor in a "no compile path" mode where every
    /// dispatch returns [`GpuError::NotWired`]. Pass `Some(...)` for a
    /// fully functional CUB actor.
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        ctx: Arc<cudarc::driver::CudaContext>,
        nvrtc: Option<Arc<ActorRef<NvrtcMsg>>>,
    ) -> Props<Self> {
        let arch = detect_sm_arch(&stream);
        Props::create(move || CubActor {
            inner: CubInner::Real {
                ctx: ctx.clone(),
                stream: stream.clone(),
                completion: completion.clone(),
                state: state.clone(),
                kernel_cache: Arc::new(Mutex::new(KernelSourceCache::default())),
                nvrtc: nvrtc.clone(),
                arch,
            },
        })
    }

    /// Mock-mode props: every request replies `Unrecoverable("CubActor in mock mode")`.
    pub fn mock_props() -> Props<Self> {
        Props::create(|| CubActor {
            inner: CubInner::Mock,
        })
    }
}

/// Convenience wrapper for `CubActor::props` that returns a `Props`
/// callers can register as a child of `atomr-accel-cuda::ContextActor`.
/// Stash the resulting `ActorRef<CubMsg>` into the context's
/// `KernelChildren::register_extra::<ActorRef<CubMsg>>(...)` so the
/// supervisor can fold it into the device's restart graph.
pub fn cub_props(
    stream: Arc<cudarc::driver::CudaStream>,
    completion: Arc<dyn CompletionStrategy>,
    state: Arc<DeviceState>,
    ctx: Arc<cudarc::driver::CudaContext>,
    nvrtc: Option<Arc<ActorRef<NvrtcMsg>>>,
) -> Props<CubActor> {
    CubActor::props(stream, completion, state, ctx, nvrtc)
}

/// Probe the CUDA stream's underlying device for a compute-capability
/// `SmArch`. Falls back to [`SmArch::Sm80`] if the cudarc query fails;
/// the kernel JIT will still succeed because PTX is forward-compatible
/// and the loader retargets at module-load time.
fn detect_sm_arch(stream: &Arc<cudarc::driver::CudaStream>) -> SmArch {
    use cudarc::driver::sys::CUdevice_attribute::*;
    let ctx = stream.context();
    let major = ctx
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(8) as u32;
    let minor = ctx
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0) as u32;
    match (major, minor) {
        (8, 0) => SmArch::Sm80,
        (8, 6) => SmArch::Sm86,
        (8, 9) => SmArch::Sm89,
        (9, 0) => SmArch::Sm90,
        (10, _) => SmArch::Sm100,
        (12, _) => SmArch::Sm120,
        // Fall back to the lowest Ampere-class arch we support; PTX
        // produced for sm_80 JITs cleanly on every newer device.
        _ => SmArch::Sm80,
    }
}

#[async_trait]
impl Actor for CubActor {
    type Msg = CubMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: CubMsg) {
        match &self.inner {
            CubInner::Mock => mock_reply(msg),
            CubInner::Real {
                ctx,
                stream,
                completion,
                state,
                kernel_cache,
                nvrtc,
                arch,
            } => {
                let dispatch_ctx = CubDispatchCtx {
                    stream,
                    completion,
                    state,
                    ctx,
                    kernel_cache,
                    nvrtc: nvrtc.as_ref(),
                    arch: *arch,
                };
                match msg {
                    CubMsg::Reduce(d) => d.dispatch(&dispatch_ctx),
                    CubMsg::Scan(d) => d.dispatch(&dispatch_ctx),
                    CubMsg::Sort(d) => d.dispatch(&dispatch_ctx),
                    CubMsg::Histogram(d) => d.dispatch(&dispatch_ctx),
                    CubMsg::Select(d) => d.dispatch(&dispatch_ctx),
                    CubMsg::Partition(d) => d.dispatch(&dispatch_ctx),
                    CubMsg::SegmentedReduce(d) => d.dispatch(&dispatch_ctx),
                }
            }
        }
    }
}

fn mock_reply(msg: CubMsg) {
    let err = || GpuError::Unrecoverable("CubActor in mock mode (no GPU available)".into());
    match msg {
        CubMsg::Reduce(d) => d.cancel(err()),
        CubMsg::Scan(d) => d.cancel(err()),
        CubMsg::Sort(d) => d.cancel(err()),
        CubMsg::Histogram(d) => d.cancel(err()),
        CubMsg::Select(d) => d.cancel(err()),
        CubMsg::Partition(d) => d.cancel(err()),
        CubMsg::SegmentedReduce(d) => d.cancel(err()),
    }
}

/// Shared marker required of every CUB dispatch trait.
///
/// `op_name` and `dtype_name` feed into the per-(op, dtype) NVRTC
/// kernel-source cache key; `cancel` lets the actor abort the request
/// in mock mode (or on context-restart) by delivering an error through
/// the request's own oneshot.
pub trait CubDispatchBase: Send + 'static {
    fn op_name(&self) -> &'static str;
    fn dtype_name(&self) -> &'static str;
    fn cancel(self: Box<Self>, err: GpuError);
}

/// Convenience helper used by every dispatch impl to send a typed
/// reply error through a `oneshot::Sender`.
pub(crate) fn reply_err<T>(reply: oneshot::Sender<Result<T, GpuError>>, err: GpuError) {
    let _ = reply.send(Err(err));
}

/// Convenience: helper constructing a string label of the form
/// `"{op}_{dtype}"` used for NVRTC cache-key disambiguation. Surfaced
/// as a public function so tests / benches can reproduce the same key
/// the actor would.
pub fn kernel_key(op: &str, dtype: &str) -> String {
    format!("cub_{op}_{dtype}")
}

/// Tag-only marker so callers' `register_extra::<CubChildRef>(actor_ref)`
/// can disambiguate. Wraps the `ActorRef<CubMsg>` so the supervisor
/// can stop / restart it with everything else.
#[derive(Clone)]
pub struct CubChildRef(pub ActorRef<CubMsg>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_key_format() {
        assert_eq!(kernel_key("reduce_sum", "f32"), "cub_reduce_sum_f32");
        assert_eq!(
            kernel_key("scan_inclusive", "i64"),
            "cub_scan_inclusive_i64"
        );
    }

    #[test]
    fn kernel_source_cache_round_trip() {
        let mut c = KernelSourceCache::default();
        assert!(c.is_empty());
        let bytes = Arc::new(vec![0xAA; 32]);
        c.insert("reduce_sum", "f32", bytes.clone());
        assert_eq!(c.len(), 1);
        let got = c.get("reduce_sum", "f32").expect("hit");
        assert_eq!(got.as_slice(), bytes.as_slice());
        assert!(c.get("reduce_sum", "f64").is_none());
    }
}
