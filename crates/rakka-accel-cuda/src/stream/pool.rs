//! `PooledAllocator` (§5.7) — bounded round-robin across N streams.
//!
//! Each `KernelActor` constructed with this allocator picks the next
//! stream from a fixed-size pool. Trade-off vs. `PerActorAllocator`:
//! capped stream count (less driver overhead, less memory) at the
//! cost of cross-actor stream contention.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use super::{ActorHints, StreamAllocator};

pub struct PooledAllocator {
    pool: Vec<Arc<CudaStream>>,
    cursor: parking_lot::Mutex<usize>,
}

impl PooledAllocator {
    /// Construct a pool from a vector of pre-existing streams. All
    /// streams must belong to the same context.
    pub fn new(streams: Vec<Arc<CudaStream>>) -> Self {
        assert!(
            !streams.is_empty(),
            "PooledAllocator requires at least one stream"
        );
        Self {
            pool: streams,
            cursor: parking_lot::Mutex::new(0),
        }
    }

    /// Construct a pool by minting `count` fresh streams on `ctx`.
    /// Panics with the `ContextPoisoned` tag if any stream creation
    /// fails so the parent supervisor can restart the actor.
    pub fn with_size(ctx: &Arc<CudaContext>, count: usize) -> Self {
        assert!(count > 0, "PooledAllocator requires count >= 1");
        let mut streams = Vec::with_capacity(count);
        for _ in 0..count {
            let s = ctx
                .new_stream()
                .unwrap_or_else(|e| panic!("ContextPoisoned: new_stream: {e}"));
            streams.push(s);
        }
        Self::new(streams)
    }

    pub fn size(&self) -> usize {
        self.pool.len()
    }
}

impl StreamAllocator for PooledAllocator {
    fn acquire(&self, _hints: ActorHints) -> Arc<CudaStream> {
        let mut cur = self.cursor.lock();
        let idx = *cur % self.pool.len();
        *cur = cur.wrapping_add(1);
        self.pool[idx].clone()
    }
}
