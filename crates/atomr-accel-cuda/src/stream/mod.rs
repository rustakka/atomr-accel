//! Stream allocation strategies (§5.7).
//!
//! `StreamAllocator` controls whether each `KernelActor` owns its own
//! stream, shares from a pool, or runs everything on a single stream.
//! F1 ships only the `PerActor` default (zero contention, max
//! concurrency); the other three strategies are present as trait impls
//! so the surface is exercised, but their full behaviour is F2 work.

mod per_actor;
mod pool;
mod single;

pub use per_actor::PerActorAllocator;
pub use pool::PooledAllocator;
pub use single::SingleStreamAllocator;

use std::sync::Arc;

/// Hints the allocator may use when assigning a stream. Forward-compatible
/// — F1 ignores both fields, F2+ allocators (priority-pooled) consume
/// them.
#[derive(Debug, Clone, Copy)]
pub struct ActorHints {
    pub priority: Priority,
    pub workload: WorkloadKind,
}

impl Default for ActorHints {
    fn default() -> Self {
        Self {
            priority: Priority::Normal,
            workload: WorkloadKind::ShortLatencyBound,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Low,
    Normal,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    ShortLatencyBound,
    LongThroughputBound,
}

/// Pluggable stream allocator (§5.7). Implemented by `PerActorAllocator`
/// (F1 default) and three stubs.
pub trait StreamAllocator: Send + Sync {
    /// Acquire a stream for a `KernelActor` that's just starting.
    fn acquire(&self, hints: ActorHints) -> Arc<cudarc::driver::CudaStream>;

    /// Release the stream when the `KernelActor` stops. Default no-op
    /// because most strategies treat streams as owned by the allocator,
    /// not the caller.
    fn release(&self, _stream: Arc<cudarc::driver::CudaStream>) {}
}
