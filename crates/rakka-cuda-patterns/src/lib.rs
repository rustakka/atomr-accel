//! Universal GPU actor blueprints (§7.2).
//!
//! F2 ships:
//! - [`batching::DynamicBatchingServer`] — collects requests up to
//!   (max_batch, max_wait), then dispatches a single batched GPU call
//!   via a user-supplied [`batching::BatchFn`].
//! - [`mock::GpuMockActor`] — CPU-only stand-in for kernel actors,
//!   lets pattern-level tests run on CI without a GPU.
//!
//! F3 adds: [`cascade::InferenceCascade`].
//! F4 adds: [`replica_pool::ModelReplicaPool`].
//! F5 adds: [`scheduler::FairShareScheduler`], [`hot_swap::ModelHotSwapServer`].

pub mod batching;
pub mod cascade;
pub mod hot_swap;
pub mod mock;
pub mod moe;
pub mod replica_pool;
pub mod scheduler;
pub mod speculative;
