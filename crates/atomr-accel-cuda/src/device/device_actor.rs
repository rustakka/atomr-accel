//! `DeviceActor` — the outer tier of the §5.11 two-tier supervision tree.
//!
//! Responsibilities:
//! - Stable address: `ActorRef<DeviceMsg>` survives unlimited
//!   `ContextActor` restarts.
//! - Spawns the `ContextActor` child (which owns the `Arc<CudaContext>`)
//!   and queues `WorkRequest`s while the context is being (re)built.
//! - Holds the shared `Arc<DeviceState>` that outlives any single
//!   `ContextActor` incarnation.

use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context, Props};
use bitflags::bitflags;
use parking_lot::RwLock;
use tokio::sync::oneshot;
use tracing::{debug, warn};

use crate::dtype::CudaDtype;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::BlasMsg;

use super::alloc_dispatch::{
    AllocDispatch, AllocReq, CopyFromHostDispatch, CopyFromHostReq, CopyToHostDispatch,
    CopyToHostReq,
};
use super::alloc_msg::{DeviceLoad, HostBuf};
use super::context_actor::{ContextActor, ContextMsg};
use super::state::DeviceState;

bitflags! {
    /// Per-device opt-in flags for which library actors to spawn.
    /// Compile-time `feature = "..."` gates still apply — a flag for
    /// a library that wasn't compiled in is silently ignored.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EnabledLibraries: u32 {
        const BLAS     = 1 << 0;
        const CUDNN    = 1 << 1;
        const CUFFT    = 1 << 2;
        const CURAND   = 1 << 3;
        const CUSOLVER = 1 << 4;
        const CUBLASLT = 1 << 5;
        const NVRTC    = 1 << 6;
        // Phase 0.8 — additional library + extension actor opt-ins.
        const CUTENSOR   = 1 << 7;
        const CUSPARSE   = 1 << 8;
        const NCCL       = 1 << 9;
        const CUTLASS    = 1 << 10;
        const TENSORRT   = 1 << 11;
        const FLASHATTN  = 1 << 12;
        const CUB_THRUST = 1 << 13;
        const TELEMETRY  = 1 << 14;

        const ALL = Self::BLAS.bits()
            | Self::CUDNN.bits()
            | Self::CUFFT.bits()
            | Self::CURAND.bits()
            | Self::CUSOLVER.bits()
            | Self::CUBLASLT.bits()
            | Self::NVRTC.bits()
            | Self::CUTENSOR.bits()
            | Self::CUSPARSE.bits()
            | Self::NCCL.bits()
            | Self::CUTLASS.bits()
            | Self::TENSORRT.bits()
            | Self::FLASHATTN.bits()
            | Self::CUB_THRUST.bits()
            | Self::TELEMETRY.bits();
    }
}

impl Default for EnabledLibraries {
    /// Sensible default: BLAS only (matches F1 semantics). Enable
    /// other libraries explicitly via [`DeviceConfig::with_libraries`].
    fn default() -> Self {
        Self::BLAS
    }
}

/// Public configuration for a `DeviceActor`.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub device_id: u32,
    /// When true, the `ContextActor` skips real cudarc calls and just
    /// drives the supervision plumbing. Used by `examples/echo_no_gpu`
    /// and unit tests on hosts without a GPU.
    pub mock_mode: bool,
    /// Internal queue cap for work received before the context is ready
    /// (or while it is being rebuilt). Bounds the §5.4 backpressure
    /// surface.
    pub pending_queue_capacity: usize,
    /// Which library actors `ContextActor` should spawn under this
    /// device. Defaults to `BLAS` only (F1 behaviour). Compile-time
    /// `feature = "..."` gates still apply.
    pub enabled_libraries: EnabledLibraries,
}

impl DeviceConfig {
    pub fn new(device_id: u32) -> Self {
        Self {
            device_id,
            mock_mode: false,
            pending_queue_capacity: 1024,
            enabled_libraries: EnabledLibraries::default(),
        }
    }

    pub fn mock(device_id: u32) -> Self {
        Self {
            device_id,
            mock_mode: true,
            pending_queue_capacity: 1024,
            enabled_libraries: EnabledLibraries::default(),
        }
    }

    /// Builder: select which libraries' kernel actors to spawn.
    pub fn with_libraries(mut self, libs: EnabledLibraries) -> Self {
        self.enabled_libraries = libs;
        self
    }
}

/// Public messages sent to a `DeviceActor`.
///
/// **Phase 0.4** — the formerly-21 dtype-enumerated `Allocate*` /
/// `CopyToHost*` / `CopyFromHost*` variants collapse into 3 boxed
/// dispatchers:
///
/// - [`DeviceMsg::Alloc`] — typed allocation
/// - [`DeviceMsg::CopyToHost`] — D2H async copy
/// - [`DeviceMsg::CopyFromHost`] — H2D async copy
///
/// Each carries a `Box<dyn …Dispatch>` whose concrete payload is an
/// `AllocReq<T>` / `CopyToHostReq<T>` / `CopyFromHostReq<T>` for some
/// `T: CudaDtype`. `GpuRef<T>` keeps its static dtype on both ends —
/// the box is purely a uniform mailbox surface.
///
/// The legacy `Allocate*` / `CopyToHost*` / `CopyFromHost*` variants
/// remain as `#[deprecated]` aliases. Existing call sites compile and
/// run unchanged; the handler arm constructs the equivalent
/// `Box<dyn …Dispatch>` and forwards through the new path.
pub enum DeviceMsg {
    /// Phase 0.4 generic alloc. Construct via
    /// [`DeviceMsg::alloc::<T>`](Self::alloc) or
    /// `Box::new(AllocReq::<T> { … })` directly.
    Alloc(Box<dyn AllocDispatch>),
    /// Phase 0.4 generic D2H copy.
    CopyToHost(Box<dyn CopyToHostDispatch>),
    /// Phase 0.4 generic H2D copy.
    CopyFromHost(Box<dyn CopyFromHostDispatch>),

    /// **Deprecated alias** for [`DeviceMsg::AllocateF32`]. F1
    /// callers wrote `Allocate { len, reply }` — kept for back-compat.
    #[deprecated(note = "use DeviceMsg::alloc::<f32>(len, reply)")]
    Allocate {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<f32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<f32>(len, reply)")]
    AllocateF32 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<f32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<f64>(len, reply)")]
    AllocateF64 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<f64>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<i8>(len, reply)")]
    AllocateI8 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<i8>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<i32>(len, reply)")]
    AllocateI32 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<i32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<i64>(len, reply)")]
    AllocateI64 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<i64>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<u8>(len, reply)")]
    AllocateU8 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<u8>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<u32>(len, reply)")]
    AllocateU32 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<u32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::alloc::<u64>(len, reply)")]
    AllocateU64 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<u64>, GpuError>>,
    },
    #[cfg(feature = "f16")]
    #[deprecated(note = "use DeviceMsg::alloc::<half::f16>(len, reply)")]
    AllocateF16 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<half::f16>, GpuError>>,
    },
    #[cfg(feature = "f16")]
    #[deprecated(note = "use DeviceMsg::alloc::<half::bf16>(len, reply)")]
    AllocateBf16 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<half::bf16>, GpuError>>,
    },

    /// D2H async copy — buffer round-trips back via the reply so a
    /// pinned buffer can return to its pool.
    #[deprecated(note = "use DeviceMsg::copy_to_host::<f32>(src, dst, reply)")]
    CopyToHostF32 {
        src: GpuRef<f32>,
        dst: HostBuf<f32>,
        reply: oneshot::Sender<Result<HostBuf<f32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_from_host::<f32>(src, dst, reply)")]
    CopyFromHostF32 {
        src: HostBuf<f32>,
        dst: GpuRef<f32>,
        reply: oneshot::Sender<Result<HostBuf<f32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_to_host::<f64>(src, dst, reply)")]
    CopyToHostF64 {
        src: GpuRef<f64>,
        dst: HostBuf<f64>,
        reply: oneshot::Sender<Result<HostBuf<f64>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_from_host::<f64>(src, dst, reply)")]
    CopyFromHostF64 {
        src: HostBuf<f64>,
        dst: GpuRef<f64>,
        reply: oneshot::Sender<Result<HostBuf<f64>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_to_host::<i32>(src, dst, reply)")]
    CopyToHostI32 {
        src: GpuRef<i32>,
        dst: HostBuf<i32>,
        reply: oneshot::Sender<Result<HostBuf<i32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_from_host::<i32>(src, dst, reply)")]
    CopyFromHostI32 {
        src: HostBuf<i32>,
        dst: GpuRef<i32>,
        reply: oneshot::Sender<Result<HostBuf<i32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_to_host::<u32>(src, dst, reply)")]
    CopyToHostU32 {
        src: GpuRef<u32>,
        dst: HostBuf<u32>,
        reply: oneshot::Sender<Result<HostBuf<u32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_from_host::<u32>(src, dst, reply)")]
    CopyFromHostU32 {
        src: HostBuf<u32>,
        dst: GpuRef<u32>,
        reply: oneshot::Sender<Result<HostBuf<u32>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_to_host::<u8>(src, dst, reply)")]
    CopyToHostU8 {
        src: GpuRef<u8>,
        dst: HostBuf<u8>,
        reply: oneshot::Sender<Result<HostBuf<u8>, GpuError>>,
    },
    #[deprecated(note = "use DeviceMsg::copy_from_host::<u8>(src, dst, reply)")]
    CopyFromHostU8 {
        src: HostBuf<u8>,
        dst: GpuRef<u8>,
        reply: oneshot::Sender<Result<HostBuf<u8>, GpuError>>,
    },

    /// Fire an SGEMM through the context's BlasActor.
    Sgemm(Box<SgemmRequest>),

    /// F4: Snapshot the underlying `Arc<CudaContext>` so a top-level
    /// observer (P2pTopology, NcclWorldActor) can build cross-device
    /// machinery. Replies `None` if the context isn't ready.
    SnapshotContext {
        reply: oneshot::Sender<Option<Arc<cudarc::driver::CudaContext>>>,
    },

    /// Phase 4.5++ — Snapshot the device's primary `Arc<CudaStream>`
    /// (the stream owned by `ContextActor`). Returned to downstream
    /// raw-pointer FFI users (TensorRT `enqueueV3`, custom kernel
    /// launchers) that need to share a single CUDA execution timeline
    /// with the rest of the device's library actors.
    ///
    /// Replies `None` if the context isn't ready (e.g. mock mode, or
    /// before `ContextReady`). On real hardware the returned stream
    /// is the same one that BLAS / cuDNN / cuFFT child actors were
    /// minted off.
    SnapshotStream {
        reply: oneshot::Sender<Option<Arc<cudarc::driver::CudaStream>>>,
    },

    /// F7: Snapshot the current `KernelChildren` so application code
    /// can talk to library actors directly (e.g. `RngActor`,
    /// `CudnnActor`). Replies `None` until `ContextActor::Init`
    /// completes.
    SnapshotChildren {
        reply: oneshot::Sender<Option<KernelChildren>>,
    },

    /// F9: Subscribe to the device's `DeviceState::generation_watch`.
    /// The receiver fires every time the underlying `CudaContext`
    /// rebuilds. Used by `NcclWorldActor` and `P2pTopology` to
    /// react to context loss.
    WatchGeneration {
        reply: oneshot::Sender<tokio::sync::watch::Receiver<u64>>,
    },

    /// F5: Per-device load snapshot for placement scheduling.
    Stats { reply: oneshot::Sender<DeviceLoad> },

    /// Internal: `ContextActor` has finished initialising and the
    /// kernel actors are live.
    ContextReady { children: KernelChildren },
    /// Internal: `ContextActor` notifies that the context was torn
    /// down (e.g. on poisoning); pending work should be re-stashed
    /// until a new `ContextReady` arrives.
    ContextLost,
}

/// Set of kernel-actor refs spawned by a `ContextActor`. Each is
/// `Some` only when both the cargo feature is compiled in and the
/// `DeviceConfig::enabled_libraries` flag is set.
///
/// **Phase 0.8 extension.** In addition to the typed fields below
/// (which keep existing call sites compiling), `KernelChildren`
/// carries an open `extras` map keyed by [`TypeId`]. Future actor
/// crates (`atomr-accel-cutlass`, `-tensorrt`, `-flashattn`,
/// `-telemetry`, `-cub`) stash their `ActorRef` here so the device
/// supervisor can hand them out via [`KernelChildren::extra`]
/// without the core having to know their concrete message type.
#[derive(Clone)]
pub struct KernelChildren {
    pub blas: ActorRef<BlasMsg>,
    #[cfg(feature = "cudnn")]
    pub cudnn: Option<ActorRef<crate::kernel::CudnnMsg>>,
    #[cfg(feature = "cufft")]
    pub fft: Option<ActorRef<crate::kernel::FftMsg>>,
    #[cfg(feature = "curand")]
    pub rng: Option<ActorRef<crate::kernel::RngMsg>>,
    #[cfg(feature = "cusolver")]
    pub solver: Option<ActorRef<crate::kernel::SolverMsg>>,
    #[cfg(feature = "nvrtc")]
    pub nvrtc: Option<ActorRef<crate::kernel::NvrtcMsg>>,
    /// TypeId-keyed registry for child actors not represented by a
    /// typed field above. The `Arc<RwLock<…>>` keeps `KernelChildren`
    /// `Clone` while letting later library crates register / look up
    /// their own refs.
    extras: Arc<RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
}

impl KernelChildren {
    /// Construct a `KernelChildren` with the given `BlasActor` ref
    /// and no library children or extras. Mirrors the `..Default::default()`
    /// pattern used by callers but keeps `blas` mandatory.
    pub fn new(blas: ActorRef<BlasMsg>) -> Self {
        Self {
            blas,
            #[cfg(feature = "cudnn")]
            cudnn: None,
            #[cfg(feature = "cufft")]
            fft: None,
            #[cfg(feature = "curand")]
            rng: None,
            #[cfg(feature = "cusolver")]
            solver: None,
            #[cfg(feature = "nvrtc")]
            nvrtc: None,
            extras: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an extra child actor (or any `Send + Sync` handle) by
    /// type. Future actor crates (`atomr-accel-cutlass`, `-tensorrt`,
    /// `-flashattn`, `-telemetry`, `-cub`) stash their `ActorRef`
    /// here so the device supervisor can route stop/restart messages.
    ///
    /// If a value of the same type is already registered, it is
    /// overwritten — typical use is one-shot registration during
    /// `ContextActor::run_init`.
    pub fn register_extra<T: Any + Send + Sync>(&self, value: T) {
        let mut g = self.extras.write();
        g.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// Look up a previously registered extra by type. Returns a clone
    /// of the stored `T` if and only if a value of that exact type
    /// was registered.
    pub fn extra<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        let g = self.extras.read();
        g.get(&TypeId::of::<T>())
            .and_then(|v| v.clone().downcast::<T>().ok())
            .map(|arc| (*arc).clone())
    }

    /// Number of registered extras (for stats / observability).
    pub fn extras_len(&self) -> usize {
        self.extras.read().len()
    }
}

impl DeviceMsg {
    /// Phase 0.4: typed-allocation constructor. Boxes an
    /// [`AllocReq<T>`] into the generic [`DeviceMsg::Alloc`] variant.
    pub fn alloc<T: CudaDtype>(
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<T>, GpuError>>,
    ) -> Self {
        DeviceMsg::Alloc(Box::new(AllocReq::<T> { len, reply }))
    }

    /// Phase 0.4: typed D2H copy constructor.
    pub fn copy_to_host<T: CudaDtype>(
        src: GpuRef<T>,
        dst: HostBuf<T>,
        reply: oneshot::Sender<Result<HostBuf<T>, GpuError>>,
    ) -> Self {
        DeviceMsg::CopyToHost(Box::new(CopyToHostReq::<T> { src, dst, reply }))
    }

    /// Phase 0.4: typed H2D copy constructor.
    pub fn copy_from_host<T: CudaDtype>(
        src: HostBuf<T>,
        dst: GpuRef<T>,
        reply: oneshot::Sender<Result<HostBuf<T>, GpuError>>,
    ) -> Self {
        DeviceMsg::CopyFromHost(Box::new(CopyFromHostReq::<T> { src, dst, reply }))
    }
}

/// Body of a `DeviceMsg::Sgemm` request. Boxed because it's larger than
/// the surrounding enum's other variants and we want the enum cheap to
/// clone/send. `reply` is `oneshot`, so each request must be unique.
pub struct SgemmRequest {
    pub a: GpuRef<f32>,
    pub b: GpuRef<f32>,
    pub c: GpuRef<f32>,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: f32,
    pub beta: f32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// Pending work item — anything DeviceActor stashes while the context
/// is not ready. Mirrors the user-facing variants of `DeviceMsg` minus
/// internal messages.
pub enum WorkRequest {
    /// Forward to ContextActor as a typed allocate. We collapse all
    /// per-dtype variants into a single `Pending` that re-issues the
    /// original message verbatim — `Box<dyn FnOnce>` keeps each
    /// pending op type-safe without needing N enum variants.
    Boxed(Box<dyn FnOnce(&ActorRef<ContextMsg>, &ActorRef<BlasMsg>) + Send>),
    Sgemm(Box<SgemmRequest>),
    /// Reply slot for callers who can't be re-driven (e.g. a
    /// SnapshotContext while context isn't ready).
    SnapshotContext {
        reply: oneshot::Sender<Option<Arc<cudarc::driver::CudaContext>>>,
    },
}

pub struct DeviceActor {
    config: DeviceConfig,
    state: Arc<DeviceState>,
    context_ref: Option<ActorRef<ContextMsg>>,
    children: Option<KernelChildren>,
    pending: VecDeque<WorkRequest>,
}

impl DeviceActor {
    pub fn new(config: DeviceConfig) -> Self {
        let state = Arc::new(DeviceState::new(config.device_id));
        Self {
            config,
            state,
            context_ref: None,
            children: None,
            pending: VecDeque::new(),
        }
    }

    /// Construct a `Props<DeviceActor>` with the given configuration.
    pub fn props(config: DeviceConfig) -> Props<Self> {
        let cfg = config.clone();
        Props::create(move || DeviceActor::new(cfg.clone()))
    }

    /// Shared device state — exposed for tests and for `KernelActor`s
    /// that need to mint `GpuRef`s.
    pub fn state(&self) -> &Arc<DeviceState> {
        &self.state
    }

    fn enqueue_pending(&mut self, work: WorkRequest) {
        if self.pending.len() >= self.config.pending_queue_capacity {
            warn!(
                device_id = self.config.device_id,
                cap = self.config.pending_queue_capacity,
                "dropping work — pending queue full"
            );
            // Drop on the floor with a typed error. The Boxed variant
            // owns its reply channel internally so we just drop it
            // and the caller observes oneshot::Receiver::Err.
            match work {
                WorkRequest::Sgemm(req) => {
                    let _ = req.reply.send(Err(GpuError::Unrecoverable(
                        "device pending queue full".into(),
                    )));
                }
                WorkRequest::SnapshotContext { reply } => {
                    let _ = reply.send(None);
                }
                WorkRequest::Boxed(_) => { /* reply drops with closure */ }
            }
            return;
        }
        self.pending.push_back(work);
    }

    fn drain_pending(&mut self) {
        let Some(children) = self.children.clone() else {
            return;
        };
        let Some(ctx) = self.context_ref.clone() else {
            return;
        };
        while let Some(work) = self.pending.pop_front() {
            match work {
                WorkRequest::Boxed(f) => f(&ctx, &children.blas),
                WorkRequest::Sgemm(req) => {
                    children.blas.tell(BlasMsg::Sgemm(req));
                }
                WorkRequest::SnapshotContext { reply } => {
                    // No way to ask a non-mailbox method to fetch the
                    // current context here; the user can re-issue.
                    let _ = reply.send(self.state.current_context());
                }
            }
        }
    }
}

#[async_trait]
impl Actor for DeviceActor {
    type Msg = DeviceMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        debug!(device_id = self.config.device_id, "DeviceActor pre_start");
        let parent_ref = ctx.self_ref().clone();
        let props = ContextActor::props(self.state.clone(), self.config.clone(), parent_ref);
        match ctx.spawn::<ContextActor>(props, "ctx") {
            Ok(r) => {
                self.context_ref = Some(r);
            }
            Err(e) => {
                // Spawn failure here is structural; surface via panic so
                // a user-installed root supervisor sees it.
                panic!("Unrecoverable: failed to spawn ContextActor: {e}");
            }
        }
    }

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: DeviceMsg) {
        // Phase 0.4: the alloc/copy fan-out collapses into 3 generic
        // arms. Legacy `Allocate*` / `CopyToHost*` / `CopyFromHost*`
        // variants are translated into the new boxed dispatchers
        // before forwarding, so the rest of the pipeline (stash /
        // drain / context handler) sees a single shape.
        #[allow(deprecated)]
        let msg = match msg {
            // -- legacy alloc → AllocReq<T> -------------------
            DeviceMsg::Allocate { len, reply } | DeviceMsg::AllocateF32 { len, reply } => {
                DeviceMsg::alloc::<f32>(len, reply)
            }
            DeviceMsg::AllocateF64 { len, reply } => DeviceMsg::alloc::<f64>(len, reply),
            DeviceMsg::AllocateI8 { len, reply } => DeviceMsg::alloc::<i8>(len, reply),
            DeviceMsg::AllocateI32 { len, reply } => DeviceMsg::alloc::<i32>(len, reply),
            DeviceMsg::AllocateI64 { len, reply } => DeviceMsg::alloc::<i64>(len, reply),
            DeviceMsg::AllocateU8 { len, reply } => DeviceMsg::alloc::<u8>(len, reply),
            DeviceMsg::AllocateU32 { len, reply } => DeviceMsg::alloc::<u32>(len, reply),
            DeviceMsg::AllocateU64 { len, reply } => DeviceMsg::alloc::<u64>(len, reply),
            #[cfg(feature = "f16")]
            DeviceMsg::AllocateF16 { len, reply } => DeviceMsg::alloc::<half::f16>(len, reply),
            #[cfg(feature = "f16")]
            DeviceMsg::AllocateBf16 { len, reply } => DeviceMsg::alloc::<half::bf16>(len, reply),
            // -- legacy copy_to_host → CopyToHostReq<T> -------
            DeviceMsg::CopyToHostF32 { src, dst, reply } => {
                DeviceMsg::copy_to_host::<f32>(src, dst, reply)
            }
            DeviceMsg::CopyToHostF64 { src, dst, reply } => {
                DeviceMsg::copy_to_host::<f64>(src, dst, reply)
            }
            DeviceMsg::CopyToHostI32 { src, dst, reply } => {
                DeviceMsg::copy_to_host::<i32>(src, dst, reply)
            }
            DeviceMsg::CopyToHostU32 { src, dst, reply } => {
                DeviceMsg::copy_to_host::<u32>(src, dst, reply)
            }
            DeviceMsg::CopyToHostU8 { src, dst, reply } => {
                DeviceMsg::copy_to_host::<u8>(src, dst, reply)
            }
            // -- legacy copy_from_host → CopyFromHostReq<T> ---
            DeviceMsg::CopyFromHostF32 { src, dst, reply } => {
                DeviceMsg::copy_from_host::<f32>(src, dst, reply)
            }
            DeviceMsg::CopyFromHostF64 { src, dst, reply } => {
                DeviceMsg::copy_from_host::<f64>(src, dst, reply)
            }
            DeviceMsg::CopyFromHostI32 { src, dst, reply } => {
                DeviceMsg::copy_from_host::<i32>(src, dst, reply)
            }
            DeviceMsg::CopyFromHostU32 { src, dst, reply } => {
                DeviceMsg::copy_from_host::<u32>(src, dst, reply)
            }
            DeviceMsg::CopyFromHostU8 { src, dst, reply } => {
                DeviceMsg::copy_from_host::<u8>(src, dst, reply)
            }
            // already-collapsed / non-alloc variants pass through
            other => other,
        };

        let ready = self.context_ref.is_some() && self.children.is_some();

        match msg {
            // Phase 0.4: 3 arms for the generic forms.
            DeviceMsg::Alloc(boxed) => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::Alloc(boxed));
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::Alloc(boxed))
                    })));
                }
            }
            DeviceMsg::CopyToHost(boxed) => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyToHost(boxed));
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::CopyToHost(boxed))
                    })));
                }
            }
            DeviceMsg::CopyFromHost(boxed) => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyFromHost(boxed));
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::CopyFromHost(boxed))
                    })));
                }
            }

            // Legacy variants are unreachable here: the upstream
            // translation stage (above) rewrote every one into the
            // generic form.
            #[allow(deprecated)]
            DeviceMsg::Allocate { .. }
            | DeviceMsg::AllocateF32 { .. }
            | DeviceMsg::AllocateF64 { .. }
            | DeviceMsg::AllocateI8 { .. }
            | DeviceMsg::AllocateI32 { .. }
            | DeviceMsg::AllocateI64 { .. }
            | DeviceMsg::AllocateU8 { .. }
            | DeviceMsg::AllocateU32 { .. }
            | DeviceMsg::AllocateU64 { .. }
            | DeviceMsg::CopyToHostF32 { .. }
            | DeviceMsg::CopyFromHostF32 { .. }
            | DeviceMsg::CopyToHostF64 { .. }
            | DeviceMsg::CopyFromHostF64 { .. }
            | DeviceMsg::CopyToHostI32 { .. }
            | DeviceMsg::CopyFromHostI32 { .. }
            | DeviceMsg::CopyToHostU32 { .. }
            | DeviceMsg::CopyFromHostU32 { .. }
            | DeviceMsg::CopyToHostU8 { .. }
            | DeviceMsg::CopyFromHostU8 { .. } => unreachable!(
                "Phase 0.4 translation collapses all legacy alloc/copy variants \
                 into DeviceMsg::Alloc / CopyToHost / CopyFromHost"
            ),
            #[cfg(feature = "f16")]
            #[allow(deprecated)]
            DeviceMsg::AllocateF16 { .. } | DeviceMsg::AllocateBf16 { .. } => {
                unreachable!(
                    "Phase 0.4 translation collapses all legacy alloc/copy variants \
                     into DeviceMsg::Alloc"
                )
            }

            DeviceMsg::Sgemm(req) => match &self.children {
                Some(c) => c.blas.tell(BlasMsg::Sgemm(req)),
                None => self.enqueue_pending(WorkRequest::Sgemm(req)),
            },

            DeviceMsg::SnapshotContext { reply } => {
                let _ = reply.send(self.state.current_context());
            }
            DeviceMsg::SnapshotStream { reply } => {
                // Forward to ContextActor — it owns the primary
                // `Arc<CudaStream>`. If the context isn't ready (mock
                // mode pre-ready or while a rebuild is in flight) the
                // ContextActor handler replies `None` to the same
                // oneshot. We don't stash this as a `WorkRequest`
                // because a `None` reply is correct in that case —
                // callers re-issue if they need the stream.
                if let Some(ctx) = self.context_ref.as_ref() {
                    ctx.tell(ContextMsg::SnapshotStream { reply });
                } else {
                    let _ = reply.send(None);
                }
            }
            DeviceMsg::SnapshotChildren { reply } => {
                let _ = reply.send(self.children.clone());
            }
            DeviceMsg::WatchGeneration { reply } => {
                let _ = reply.send(self.state.generation_watch());
            }
            DeviceMsg::Stats { reply } => {
                let _ = reply.send(self.snapshot_load());
            }

            DeviceMsg::ContextReady { children } => {
                debug!(device_id = self.config.device_id, "context ready");
                self.children = Some(children);
                self.drain_pending();
            }
            DeviceMsg::ContextLost => {
                debug!(device_id = self.config.device_id, "context lost");
                self.children = None;
            }
        }
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        debug!(device_id = self.config.device_id, "DeviceActor post_stop");
        self.state.begin_shutdown();
        // Drain pending replies with stale errors so callers don't hang.
        while let Some(work) = self.pending.pop_front() {
            match work {
                WorkRequest::Boxed(_) => { /* reply drops with closure */ }
                WorkRequest::Sgemm(req) => {
                    let _ = req
                        .reply
                        .send(Err(GpuError::GpuRefStale("device shutting down")));
                }
                WorkRequest::SnapshotContext { reply } => {
                    let _ = reply.send(None);
                }
            }
        }
    }
}

impl DeviceActor {
    fn snapshot_load(&self) -> DeviceLoad {
        DeviceLoad {
            free_bytes: 0,
            total_bytes: 0,
            active_streams: 0,
            queue_depth: self.pending.len() as u32,
            compute_cap: (0, 0),
        }
    }
}

#[cfg(test)]
#[allow(deprecated)] // exercised on purpose: legacy variants must keep routing.
mod tests {
    use super::*;
    use crate::dtype::DType;
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;
    use std::time::Duration;

    /// Phase 0.8 — bit values are part of the on-the-wire surface
    /// (config files / persisted device specs). Lock them down so a
    /// future re-ordering of the bitflag declaration is caught.
    #[test]
    fn enabled_libraries_bit_values_are_stable() {
        assert_eq!(EnabledLibraries::BLAS.bits(), 1 << 0);
        assert_eq!(EnabledLibraries::CUDNN.bits(), 1 << 1);
        assert_eq!(EnabledLibraries::CUFFT.bits(), 1 << 2);
        assert_eq!(EnabledLibraries::CURAND.bits(), 1 << 3);
        assert_eq!(EnabledLibraries::CUSOLVER.bits(), 1 << 4);
        assert_eq!(EnabledLibraries::CUBLASLT.bits(), 1 << 5);
        assert_eq!(EnabledLibraries::NVRTC.bits(), 1 << 6);
        // Phase 0.8 additions.
        assert_eq!(EnabledLibraries::CUTENSOR.bits(), 1 << 7);
        assert_eq!(EnabledLibraries::CUSPARSE.bits(), 1 << 8);
        assert_eq!(EnabledLibraries::NCCL.bits(), 1 << 9);
        assert_eq!(EnabledLibraries::CUTLASS.bits(), 1 << 10);
        assert_eq!(EnabledLibraries::TENSORRT.bits(), 1 << 11);
        assert_eq!(EnabledLibraries::FLASHATTN.bits(), 1 << 12);
        assert_eq!(EnabledLibraries::CUB_THRUST.bits(), 1 << 13);
        assert_eq!(EnabledLibraries::TELEMETRY.bits(), 1 << 14);
    }

    #[test]
    fn enabled_libraries_round_trip_via_bits() {
        let original = EnabledLibraries::BLAS
            | EnabledLibraries::CUTENSOR
            | EnabledLibraries::FLASHATTN
            | EnabledLibraries::TELEMETRY;
        let bits = original.bits();
        let restored =
            EnabledLibraries::from_bits(bits).expect("known bits round-trip through from_bits");
        assert_eq!(original, restored);
        assert!(restored.contains(EnabledLibraries::FLASHATTN));
        assert!(!restored.contains(EnabledLibraries::CUDNN));
    }

    #[test]
    fn enabled_libraries_all_contains_every_phase_0_8_bit() {
        let all = EnabledLibraries::ALL;
        for bit in [
            EnabledLibraries::BLAS,
            EnabledLibraries::CUDNN,
            EnabledLibraries::CUFFT,
            EnabledLibraries::CURAND,
            EnabledLibraries::CUSOLVER,
            EnabledLibraries::CUBLASLT,
            EnabledLibraries::NVRTC,
            EnabledLibraries::CUTENSOR,
            EnabledLibraries::CUSPARSE,
            EnabledLibraries::NCCL,
            EnabledLibraries::CUTLASS,
            EnabledLibraries::TENSORRT,
            EnabledLibraries::FLASHATTN,
            EnabledLibraries::CUB_THRUST,
            EnabledLibraries::TELEMETRY,
        ] {
            assert!(all.contains(bit), "ALL missing {bit:?}");
        }
    }

    /// Phase 0.8 — `KernelChildren::register_extra` / `extra` round-trip.
    /// Uses a dummy non-actor type because spawning a real actor here
    /// would pull in the full ActorSystem and is unnecessary for the
    /// API contract under test.
    #[test]
    fn kernel_children_extras_register_and_retrieve_by_type() {
        // Build a KernelChildren manually using a dummy BlasActor ref.
        // We can't easily construct an ActorRef<BlasMsg> outside an
        // ActorSystem, so this test only touches the extras map by
        // building it directly.
        let extras: Arc<RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Stand-in helper that mirrors KernelChildren::register_extra /
        // extra / extras_len semantics but operates on the bare extras
        // map. Keeping the test free of an ActorSystem dep.
        fn register<T: Any + Send + Sync>(
            map: &Arc<RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
            v: T,
        ) {
            map.write().insert(TypeId::of::<T>(), Arc::new(v));
        }
        fn lookup<T: Any + Send + Sync + Clone>(
            map: &Arc<RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
        ) -> Option<T> {
            map.read()
                .get(&TypeId::of::<T>())
                .and_then(|v| v.clone().downcast::<T>().ok())
                .map(|arc| (*arc).clone())
        }

        #[derive(Clone, PartialEq, Eq, Debug)]
        struct CutlassRef(u32);
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct TensorRtRef(&'static str);

        register(&extras, CutlassRef(7));
        register(&extras, TensorRtRef("trt"));

        assert_eq!(lookup::<CutlassRef>(&extras), Some(CutlassRef(7)));
        assert_eq!(lookup::<TensorRtRef>(&extras), Some(TensorRtRef("trt")));
        // Unregistered type returns None.
        #[derive(Clone)]
        struct Unknown;
        assert!(lookup::<Unknown>(&extras).is_none());
        assert_eq!(extras.read().len(), 2);

        // Re-registering the same type overwrites.
        register(&extras, CutlassRef(99));
        assert_eq!(lookup::<CutlassRef>(&extras), Some(CutlassRef(99)));
        assert_eq!(extras.read().len(), 2);
    }

    /// End-to-end exercise of the actual `KernelChildren` API by going
    /// through a mock `DeviceActor` so we have a real `BlasActor` ref
    /// to seed the struct.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kernel_children_extras_via_snapshot() {
        let sys = ActorSystem::create("kc_extras", Config::empty())
            .await
            .unwrap();
        let dev = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev0")
            .unwrap();

        // Wait for ContextReady by repeatedly probing SnapshotChildren.
        let mut snap: Option<KernelChildren> = None;
        for _ in 0..50 {
            let (tx, rx) = oneshot::channel();
            dev.tell(DeviceMsg::SnapshotChildren { reply: tx });
            if let Ok(Some(c)) = rx.await {
                snap = Some(c);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let children = snap.expect("KernelChildren snapshot should arrive in mock mode");
        assert_eq!(children.extras_len(), 0);

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct FakeCutlassRef(u64);
        children.register_extra(FakeCutlassRef(42));
        assert_eq!(children.extras_len(), 1);
        assert_eq!(children.extra::<FakeCutlassRef>(), Some(FakeCutlassRef(42)));
        // Clones share the same extras map (Arc<RwLock<…>> inside).
        let cloned = children.clone();
        assert_eq!(cloned.extras_len(), 1);
        assert_eq!(cloned.extra::<FakeCutlassRef>(), Some(FakeCutlassRef(42)));

        sys.terminate().await;
    }

    /// Smoke test — DeviceActor in mock mode should accept Allocate
    /// requests and reply (with an error from mock BlasActor or with a
    /// fabricated success). This exercises the whole spawn / ContextReady
    /// / drain_pending plumbing without touching cudarc at runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_work_drains_on_context_ready() {
        let sys = ActorSystem::create("test", Config::empty()).await.unwrap();
        let dev = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev0")
            .unwrap();

        // Send Allocate before ContextReady can possibly have arrived. In
        // mock mode the ContextActor responds with success quickly; we
        // give it a generous timeout.
        let (tx, rx) = oneshot::channel();
        dev.tell(DeviceMsg::Allocate { len: 16, reply: tx });
        let res = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .expect("alloc reply should arrive within timeout")
            .expect("oneshot dropped");
        // In mock mode the allocation returns the synthetic error
        // documented in ContextActor::handle. We just verify a reply
        // arrived — the plumbing is what's under test here.
        assert!(matches!(res, Err(GpuError::Unrecoverable(_))));

        sys.terminate().await;
    }

    /// Phase 0.4: the typed `DeviceMsg::alloc::<T>` constructor should
    /// build an `AllocReq<T>`-shaped boxed dispatcher and round-trip
    /// through the actor pipeline, replying with the same kind of
    /// error the legacy variant produces in mock mode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn alloc_dispatch_via_typed_constructor() {
        let sys = ActorSystem::create("test", Config::empty()).await.unwrap();
        let dev = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev1")
            .unwrap();

        let (tx, rx) = oneshot::channel::<Result<GpuRef<f32>, GpuError>>();
        dev.tell(DeviceMsg::alloc::<f32>(64, tx));
        let res = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .expect("alloc reply within timeout")
            .expect("oneshot dropped");
        assert!(matches!(res, Err(GpuError::Unrecoverable(_))));

        sys.terminate().await;
    }

    /// Phase 0.4: every `*Dispatch` trait carries a runtime dtype tag
    /// reflecting the concrete `T: CudaDtype`. We never go through the
    /// actor system here — boxing is enough to verify dispatch.
    #[test]
    fn alloc_dispatch_dtype_kind_correct() {
        // f32
        let (tx, _rx) = oneshot::channel::<Result<GpuRef<f32>, GpuError>>();
        let boxed: Box<dyn AllocDispatch> = Box::new(AllocReq::<f32> { len: 4, reply: tx });
        assert_eq!(boxed.dtype(), DType::F32);
        assert_eq!(boxed.len(), 4);

        // i32
        let (tx, _rx) = oneshot::channel::<Result<GpuRef<i32>, GpuError>>();
        let boxed: Box<dyn AllocDispatch> = Box::new(AllocReq::<i32> { len: 7, reply: tx });
        assert_eq!(boxed.dtype(), DType::I32);

        // u8
        let (tx, _rx) = oneshot::channel::<Result<GpuRef<u8>, GpuError>>();
        let boxed: Box<dyn AllocDispatch> = Box::new(AllocReq::<u8> { len: 1, reply: tx });
        assert_eq!(boxed.dtype(), DType::U8);
    }

    /// Phase 0.4: legacy `DeviceMsg::AllocateF32` constructor is
    /// `#[deprecated]` but still compiles and routes correctly into
    /// the new generic pipeline.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deprecated_allocate_f32_still_works() {
        let sys = ActorSystem::create("test", Config::empty()).await.unwrap();
        let dev = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev2")
            .unwrap();

        let (tx, rx) = oneshot::channel::<Result<GpuRef<f32>, GpuError>>();
        // NOTE: explicitly using the deprecated variant. The
        // `#[allow(deprecated)]` on the mod silences the warning.
        dev.tell(DeviceMsg::AllocateF32 { len: 8, reply: tx });
        let res = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .expect("alloc reply within timeout")
            .expect("oneshot dropped");
        assert!(matches!(res, Err(GpuError::Unrecoverable(_))));

        sys.terminate().await;
    }

    /// Phase 0.4: the `CopyToHostDispatch` trait carries a runtime
    /// dtype tag. We exercise this with a stub dispatcher (no real
    /// `GpuRef<T>` involved — that would require a live CudaContext)
    /// and confirm the boxed dtype reports `T::KIND`.
    #[test]
    fn copy_to_host_typed() {
        struct Stub<T: CudaDtype>(std::marker::PhantomData<T>);
        impl<T: CudaDtype> CopyToHostDispatch for Stub<T> {
            fn dtype(&self) -> DType {
                T::KIND
            }
            fn run(
                self: Box<Self>,
                _stream: Arc<cudarc::driver::CudaStream>,
                _completion: Arc<dyn crate::completion::CompletionStrategy>,
            ) {
                // never invoked in unit tests
            }
        }

        let boxed: Box<dyn CopyToHostDispatch> = Box::new(Stub::<f32>(std::marker::PhantomData));
        assert_eq!(boxed.dtype(), DType::F32);
        let boxed: Box<dyn CopyToHostDispatch> = Box::new(Stub::<i32>(std::marker::PhantomData));
        assert_eq!(boxed.dtype(), DType::I32);

        // Smoke: the typed constructor builds the matching variant.
        // We can wrap the stub in DeviceMsg::CopyToHost manually and
        // assert the variant tag.
        let msg = DeviceMsg::CopyToHost(Box::new(Stub::<u32>(std::marker::PhantomData)));
        match msg {
            DeviceMsg::CopyToHost(b) => assert_eq!(b.dtype(), DType::U32),
            _ => panic!("expected CopyToHost variant"),
        }
    }

    /// Phase 0.4: H2D mirror of `copy_to_host_typed`.
    #[test]
    fn copy_from_host_typed() {
        struct Stub<T: CudaDtype>(std::marker::PhantomData<T>);
        impl<T: CudaDtype> CopyFromHostDispatch for Stub<T> {
            fn dtype(&self) -> DType {
                T::KIND
            }
            fn run(
                self: Box<Self>,
                _stream: Arc<cudarc::driver::CudaStream>,
                _completion: Arc<dyn crate::completion::CompletionStrategy>,
            ) {
            }
        }

        let boxed: Box<dyn CopyFromHostDispatch> = Box::new(Stub::<u8>(std::marker::PhantomData));
        assert_eq!(boxed.dtype(), DType::U8);

        let msg = DeviceMsg::CopyFromHost(Box::new(Stub::<f64>(std::marker::PhantomData)));
        match msg {
            DeviceMsg::CopyFromHost(b) => assert_eq!(b.dtype(), DType::F64),
            _ => panic!("expected CopyFromHost variant"),
        }
    }
}
