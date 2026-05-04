//! `PerActorAllocator` — F2 default. Each `KernelActor` gets a fresh
//! `CudaStream`, maximising kernel concurrency at the cost of one
//! stream per actor (§5.7 default).
//!
//! Two construction modes:
//!
//! - [`PerActorAllocator::new`] — F1 back-compat, takes a pre-created
//!   stream and hands the same stream to every `acquire()` call.
//!   Equivalent to the F1 single-stream behaviour. Kept so the
//!   foundation tests / `BlasActor::props_legacy` keep working.
//! - [`PerActorAllocator::with_context`] — F2 fresh-stream mode.
//!   Holds an `Arc<CudaContext>` and mints a brand-new stream on
//!   every `acquire()`. cudarc 0.19 does not expose
//!   `cuStreamCreateWithPriority` at the safe layer, so the
//!   `Priority` hint is currently a record-only field; future
//!   cudarc versions will let us honour it without changing this
//!   surface.

use std::sync::{Arc, Weak};

use cudarc::driver::{CudaContext, CudaStream};

use super::{ActorHints, StreamAllocator};

#[derive(Clone)]
pub struct PerActorAllocator {
    inner: Arc<PerActorInner>,
}

enum PerActorInner {
    /// Hand the same stream to every caller. F1 behaviour.
    Shared { stream: Arc<CudaStream> },
    /// Mint a fresh stream per `acquire()`.
    Fresh {
        ctx: Arc<CudaContext>,
        minted: parking_lot::Mutex<Vec<Weak<CudaStream>>>,
    },
}

impl PerActorAllocator {
    /// F1 back-compat: every `acquire()` returns the supplied stream.
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            inner: Arc::new(PerActorInner::Shared { stream }),
        }
    }

    /// F2 default: each `acquire()` mints a fresh stream on the given
    /// context. Tracks weak refs to the minted streams for diagnostics.
    pub fn with_context(ctx: Arc<CudaContext>) -> Self {
        Self {
            inner: Arc::new(PerActorInner::Fresh {
                ctx,
                minted: parking_lot::Mutex::new(Vec::new()),
            }),
        }
    }

    /// Number of live streams this allocator has minted (Fresh mode
    /// only). Returns 1 for Shared mode.
    pub fn live_streams(&self) -> usize {
        match self.inner.as_ref() {
            PerActorInner::Shared { .. } => 1,
            PerActorInner::Fresh { minted, .. } => {
                let mut g = minted.lock();
                g.retain(|w| w.strong_count() > 0);
                g.len()
            }
        }
    }
}

impl StreamAllocator for PerActorAllocator {
    fn acquire(&self, _hints: ActorHints) -> Arc<CudaStream> {
        match self.inner.as_ref() {
            PerActorInner::Shared { stream } => stream.clone(),
            PerActorInner::Fresh { ctx, minted } => {
                // cudarc 0.19 doesn't expose stream priority at the
                // safe layer; we ignore `_hints.priority` for now.
                let s = ctx
                    .new_stream()
                    .unwrap_or_else(|e| panic!("ContextPoisoned: new_stream: {e}"));
                minted.lock().push(Arc::downgrade(&s));
                s
            }
        }
    }
}
