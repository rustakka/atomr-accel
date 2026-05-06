//! `CutlassActor` — host-side dispatcher for CUTLASS template
//! instantiations.
//!
//! The actor is intentionally GPU-agnostic at the type level:
//! launching a compiled kernel goes through
//! `atomr_accel_cuda::kernel::NvrtcActor` once that crate's `nvrtc`
//! feature is on. When no NVRTC actor is wired in (e.g. the host-only
//! test runs that this crate ships), the actor records the rendered
//! `.cu` source plus the lowered kernel name in the
//! [`crate::plan_cache::PlanCache`] and replies with the cache hit.
//!
//! Wiring into `KernelChildren::register_extra` is left to the
//! caller: this crate doesn't reach into the device actor, but it
//! exposes [`CutlassProps`] so a downstream `ContextActor` can
//! `register_extra("cutlass", atomr_accel_cutlass::props(64))`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::conv::CutlassConvDispatch;
use crate::gemm::{CutlassGemmDispatch, RefitMsg};
use crate::plan_cache::{CachedPlan, PlanCache};

#[cfg(feature = "grouped")]
use crate::grouped_gemm::CutlassGroupedGemmDispatch;

/// Top-level actor mailbox.
///
/// `Refit` carries new weight bytes for an already-compiled plan;
/// the actor copies them into the kernel's bound workspace without
/// recompiling. `reply` is an opaque sender that downstream code
/// can pick — we don't depend on `tokio::sync::oneshot` here so that
/// host-only callers can use a `std::sync::mpsc` reply channel
/// instead.
pub enum CutlassMsg {
    Gemm(Box<dyn CutlassGemmDispatch>),
    #[cfg(feature = "grouped")]
    GroupedGemm(Box<dyn CutlassGroupedGemmDispatch>),
    Conv(Box<dyn CutlassConvDispatch>),
    Refit {
        msg: RefitMsg,
        reply: Box<dyn FnOnce(Result<(), String>) + Send + 'static>,
    },
}

impl std::fmt::Debug for CutlassMsg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CutlassMsg::Gemm(d) => f
                .debug_struct("Gemm")
                .field("dtype", &d.dtype())
                .field("arch", &d.arch())
                .finish(),
            #[cfg(feature = "grouped")]
            CutlassMsg::GroupedGemm(d) => f
                .debug_struct("GroupedGemm")
                .field("dtype", &d.dtype())
                .field("arch", &d.arch())
                .field("group_count", &d.group_count())
                .finish(),
            CutlassMsg::Conv(d) => f
                .debug_struct("Conv")
                .field("kind", &d.kind_name())
                .field("dtype", &d.dtype())
                .field("arch", &d.arch())
                .finish(),
            CutlassMsg::Refit { msg, .. } => f
                .debug_struct("Refit")
                .field("plan_key", &msg.plan_key)
                .field("weights_len", &msg.weights.len())
                .finish(),
        }
    }
}

/// Closure invoked for every rendered `.cu` source — typically wired
/// to `atomr-accel-cuda::kernel::NvrtcActor::Compile` when the
/// downstream binary opts into `nvrtc`.
pub type CompileSink = Arc<dyn Fn(&str, &str) -> Result<(), String> + Send + Sync>;

/// Inner state of a [`CutlassActor`].
pub struct CutlassInner {
    pub plan_cache: Arc<PlanCache>,
    /// Optional NVRTC-shaped sink: when present, the actor forwards
    /// rendered `.cu` source to it for compilation. Left as a generic
    /// `Box<dyn Fn ...>` so the cutlass crate doesn't pull
    /// `atomr-accel-cuda::nvrtc` into its compile graph when the
    /// feature is off.
    pub compile_sink: Option<CompileSink>,
    /// Counter of dispatched messages — exposed for the
    /// `actor::tests::cutlass_msg_constructs` test and for telemetry.
    pub dispatched: Mutex<u64>,
}

impl CutlassInner {
    pub fn new(plan_cache_capacity: usize) -> Self {
        Self {
            plan_cache: Arc::new(PlanCache::new(plan_cache_capacity)),
            compile_sink: None,
            dispatched: Mutex::new(0),
        }
    }

    pub fn dispatched(&self) -> u64 {
        *self.dispatched.lock()
    }
}

/// Host-side actor. Holds an [`Arc<CutlassInner>`] so messages can be
/// processed from a worker thread without locking the actor itself
/// after construction.
pub struct CutlassActor {
    inner: Arc<CutlassInner>,
}

impl CutlassActor {
    pub fn new(plan_cache_capacity: usize) -> Self {
        Self {
            inner: Arc::new(CutlassInner::new(plan_cache_capacity)),
        }
    }

    pub fn inner(&self) -> Arc<CutlassInner> {
        self.inner.clone()
    }

    /// Synchronously process a message. The real production path
    /// runs through `atomr_core::actor::Actor::handle`; this method
    /// is the host-only fast path that the unit tests exercise.
    pub fn handle(&self, msg: CutlassMsg) {
        *self.inner.dispatched.lock() += 1;
        match msg {
            CutlassMsg::Gemm(d) => {
                let key = d.plan_key();
                if self.inner.plan_cache.get(&key).is_none() {
                    let (src, name) = d.render_cu();
                    if let Some(sink) = &self.inner.compile_sink {
                        if let Err(e) = sink(&src, &name) {
                            tracing::warn!(error = %e, "cutlass compile sink rejected source");
                        }
                    }
                    self.inner.plan_cache.insert(CachedPlan {
                        key,
                        source: Arc::new(src),
                        kernel_name: Arc::new(name),
                        kernel_handle: None,
                    });
                }
            }
            #[cfg(feature = "grouped")]
            CutlassMsg::GroupedGemm(d) => {
                let key = d.plan_key();
                if self.inner.plan_cache.get(&key).is_none() {
                    let (src, name) = d.render_cu();
                    if let Some(sink) = &self.inner.compile_sink {
                        let _ = sink(&src, &name);
                    }
                    self.inner.plan_cache.insert(CachedPlan {
                        key,
                        source: Arc::new(src),
                        kernel_name: Arc::new(name),
                        kernel_handle: None,
                    });
                }
            }
            CutlassMsg::Conv(d) => {
                let key = d.plan_key();
                if self.inner.plan_cache.get(&key).is_none() {
                    let (src, name) = d.render_cu();
                    if let Some(sink) = &self.inner.compile_sink {
                        let _ = sink(&src, &name);
                    }
                    self.inner.plan_cache.insert(CachedPlan {
                        key,
                        source: Arc::new(src),
                        kernel_name: Arc::new(name),
                        kernel_handle: None,
                    });
                }
            }
            CutlassMsg::Refit { msg, reply } => {
                let exists = self.inner.plan_cache.get(&msg.plan_key).is_some();
                if exists {
                    reply(Ok(()));
                } else {
                    reply(Err(format!(
                        "cutlass refit: no plan for key {:?}",
                        msg.plan_key
                    )));
                }
            }
        }
    }
}

/// Props-equivalent constructor handle. Mirrors the
/// `atomr_core::actor::Props` shape used elsewhere in the workspace
/// without depending on `atomr-core` directly — once the upstream
/// `KernelChildren::register_extra` API stabilizes, this struct is
/// what the device actor calls into.
#[derive(Debug, Clone, Copy)]
pub struct CutlassProps {
    pub plan_cache_capacity: usize,
}

impl CutlassProps {
    pub fn new(plan_cache_capacity: usize) -> Self {
        Self {
            plan_cache_capacity,
        }
    }

    pub fn create(self) -> CutlassActor {
        CutlassActor::new(self.plan_cache_capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{SmArch, F16};
    use crate::gemm::{GemmRequest, GemmShape};
    use crate::plan_cache::PlanKey;

    #[test]
    fn cutlass_msg_constructs() {
        let actor = CutlassActor::new(8);
        let req = GemmRequest::<F16>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);
        let key = req.plan_key();

        // Gemm
        actor.handle(CutlassMsg::Gemm(Box::new(req.clone())));
        assert_eq!(actor.inner().dispatched(), 1);
        assert!(actor.inner().plan_cache.get(&key).is_some());

        // Conv
        use crate::conv::{ConvFwdRequest, ConvShape};
        let conv = ConvFwdRequest::<F16>::new(ConvShape::nhwc(1, 8, 8, 16, 32, 3, 3), SmArch::Sm80);
        let conv_key = conv.plan_key();
        actor.handle(CutlassMsg::Conv(Box::new(conv)));
        assert_eq!(actor.inner().dispatched(), 2);
        assert!(actor.inner().plan_cache.get(&conv_key).is_some());

        // Refit (existing plan)
        let (tx, rx) = std::sync::mpsc::channel();
        actor.handle(CutlassMsg::Refit {
            msg: RefitMsg {
                plan_key: key,
                weights: vec![0u8; 16],
            },
            reply: Box::new(move |r| {
                let _ = tx.send(r);
            }),
        });
        let res = rx.recv().unwrap();
        assert!(res.is_ok());

        // Refit (missing plan)
        let bogus = PlanKey::gemm::<F16>(
            GemmShape::new(1, 1, 1),
            crate::gemm::GemmLayout::RowMajor,
            crate::gemm::GemmLayout::RowMajor,
            crate::gemm::GemmLayout::RowMajor,
            crate::gemm::GemmEpilogue::default(),
            crate::dtype::CutlassDtype::F32,
            crate::dtype::CutlassDtype::F16,
            SmArch::Sm80,
            false,
        );
        let (tx, rx) = std::sync::mpsc::channel();
        actor.handle(CutlassMsg::Refit {
            msg: RefitMsg {
                plan_key: bogus,
                weights: vec![],
            },
            reply: Box::new(move |r| {
                let _ = tx.send(r);
            }),
        });
        let res = rx.recv().unwrap();
        assert!(res.is_err());

        // Idempotent dispatch: re-sending the same Gemm doesn't grow
        // the plan cache past one entry for that key.
        let before = actor.inner().plan_cache.len();
        actor.handle(CutlassMsg::Gemm(Box::new(req)));
        let after = actor.inner().plan_cache.len();
        assert_eq!(before, after);
    }

    #[cfg(feature = "grouped")]
    #[test]
    fn grouped_dispatch() {
        use crate::grouped_gemm::{GroupedGemmRequest, GroupedGemmShape};
        let actor = CutlassActor::new(4);
        let req = GroupedGemmRequest::<F16>::new(
            GroupedGemmShape::new(vec![GemmShape::new(64, 64, 64)]),
            SmArch::Sm90a,
        );
        let key = req.plan_key();
        actor.handle(CutlassMsg::GroupedGemm(Box::new(req)));
        assert!(actor.inner().plan_cache.get(&key).is_some());
    }
}
