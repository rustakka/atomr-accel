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

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::BlasMsg;

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
/// Allocation variants are typed per dtype so `GpuRef<T>` keeps its
/// static type on the receive side. Memcpy variants accept either
/// owned `Vec<T>` or pinned [`PinnedBuf<T>`] via [`HostBuf<T>`].
pub enum DeviceMsg {
    /// **Deprecated alias** for [`DeviceMsg::AllocateF32`]. F1
    /// callers wrote `Allocate { len, reply }` — kept for back-compat.
    Allocate {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<f32>, GpuError>>,
    },
    AllocateF32 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<f32>, GpuError>>,
    },
    AllocateF64 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<f64>, GpuError>>,
    },
    AllocateI8 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<i8>, GpuError>>,
    },
    AllocateI32 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<i32>, GpuError>>,
    },
    AllocateI64 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<i64>, GpuError>>,
    },
    AllocateU8 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<u8>, GpuError>>,
    },
    AllocateU32 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<u32>, GpuError>>,
    },
    AllocateU64 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<u64>, GpuError>>,
    },
    #[cfg(feature = "f16")]
    AllocateF16 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<half::f16>, GpuError>>,
    },
    #[cfg(feature = "f16")]
    AllocateBf16 {
        len: usize,
        reply: oneshot::Sender<Result<GpuRef<half::bf16>, GpuError>>,
    },

    /// D2H async copy — buffer round-trips back via the reply so a
    /// pinned buffer can return to its pool.
    CopyToHostF32 {
        src: GpuRef<f32>,
        dst: HostBuf<f32>,
        reply: oneshot::Sender<Result<HostBuf<f32>, GpuError>>,
    },
    CopyFromHostF32 {
        src: HostBuf<f32>,
        dst: GpuRef<f32>,
        reply: oneshot::Sender<Result<HostBuf<f32>, GpuError>>,
    },
    CopyToHostF64 {
        src: GpuRef<f64>,
        dst: HostBuf<f64>,
        reply: oneshot::Sender<Result<HostBuf<f64>, GpuError>>,
    },
    CopyFromHostF64 {
        src: HostBuf<f64>,
        dst: GpuRef<f64>,
        reply: oneshot::Sender<Result<HostBuf<f64>, GpuError>>,
    },
    CopyToHostI32 {
        src: GpuRef<i32>,
        dst: HostBuf<i32>,
        reply: oneshot::Sender<Result<HostBuf<i32>, GpuError>>,
    },
    CopyFromHostI32 {
        src: HostBuf<i32>,
        dst: GpuRef<i32>,
        reply: oneshot::Sender<Result<HostBuf<i32>, GpuError>>,
    },
    CopyToHostU32 {
        src: GpuRef<u32>,
        dst: HostBuf<u32>,
        reply: oneshot::Sender<Result<HostBuf<u32>, GpuError>>,
    },
    CopyFromHostU32 {
        src: HostBuf<u32>,
        dst: GpuRef<u32>,
        reply: oneshot::Sender<Result<HostBuf<u32>, GpuError>>,
    },
    CopyToHostU8 {
        src: GpuRef<u8>,
        dst: HostBuf<u8>,
        reply: oneshot::Sender<Result<HostBuf<u8>, GpuError>>,
    },
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
        // The forwarding logic is mechanical but expansive: 10 alloc
        // variants + 10 copy variants. We unfold them inline rather
        // than via macros — Rust's struct-init syntax doesn't work
        // through `path`-fragment macro params, and `tt`-only macros
        // would lose error locations.
        let ready = self.context_ref.is_some() && self.children.is_some();

        match msg {
            DeviceMsg::Allocate { len, reply } | DeviceMsg::AllocateF32 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateF32 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateF32 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateF64 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateF64 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateF64 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateI8 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateI8 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateI8 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateI32 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateI32 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateI32 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateI64 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateI64 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateI64 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateU8 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateU8 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateU8 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateU32 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateU32 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateU32 { len, reply })
                    })));
                }
            }
            DeviceMsg::AllocateU64 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateU64 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateU64 { len, reply })
                    })));
                }
            }
            #[cfg(feature = "f16")]
            DeviceMsg::AllocateF16 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateF16 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateF16 { len, reply })
                    })));
                }
            }
            #[cfg(feature = "f16")]
            DeviceMsg::AllocateBf16 { len, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::AllocateBf16 { len, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _b| {
                        c.tell(ContextMsg::AllocateBf16 { len, reply })
                    })));
                }
            }

            DeviceMsg::CopyToHostF32 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyToHostF32 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyToHostF32 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyFromHostF32 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyFromHostF32 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyFromHostF32 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyToHostF64 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyToHostF64 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyToHostF64 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyFromHostF64 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyFromHostF64 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyFromHostF64 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyToHostI32 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyToHostI32 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyToHostI32 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyFromHostI32 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyFromHostI32 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyFromHostI32 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyToHostU32 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyToHostU32 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyToHostU32 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyFromHostU32 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyFromHostU32 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyFromHostU32 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyToHostU8 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyToHostU8 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyToHostU8 { src, dst, reply })
                    })));
                }
            }
            DeviceMsg::CopyFromHostU8 { src, dst, reply } => {
                if ready {
                    self.context_ref
                        .as_ref()
                        .unwrap()
                        .tell(ContextMsg::CopyFromHostU8 { src, dst, reply });
                } else {
                    self.enqueue_pending(WorkRequest::Boxed(Box::new(move |c, _| {
                        c.tell(ContextMsg::CopyFromHostU8 { src, dst, reply })
                    })));
                }
            }

            DeviceMsg::Sgemm(req) => match &self.children {
                Some(c) => c.blas.tell(BlasMsg::Sgemm(req)),
                None => self.enqueue_pending(WorkRequest::Sgemm(req)),
            },

            DeviceMsg::SnapshotContext { reply } => {
                let _ = reply.send(self.state.current_context());
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
mod tests {
    use super::*;
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
        let restored = EnabledLibraries::from_bits(bits)
            .expect("known bits round-trip through from_bits");
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
        assert_eq!(
            lookup::<TensorRtRef>(&extras),
            Some(TensorRtRef("trt"))
        );
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
        assert_eq!(
            children.extra::<FakeCutlassRef>(),
            Some(FakeCutlassRef(42))
        );
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
}
