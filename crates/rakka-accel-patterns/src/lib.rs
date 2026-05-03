//! Universal GPU actor blueprints for [`rakka-accel`][rakka_accel].
//!
//! Each module is independently usable; `prelude` bundles the
//! canonical types for terse imports.
//!
//! ```ignore
//! use rakka_accel_patterns::prelude::*;
//! ```
//!
//! - [`batching::DynamicBatchingServer`] — collects requests up to
//!   (max_batch, max_wait), then dispatches a single batched GPU call
//!   via a user-supplied [`batching::BatchFn`].
//! - [`cascade::InferenceCascade`] — N-stage early-exit cascade.
//! - [`replica_pool::ModelReplicaPool`] — round-robin / least-loaded
//!   routing across replica actors.
//! - [`scheduler::FairShareScheduler`] — weighted fair queueing.
//! - [`hot_swap::ModelHotSwapServer`] — atomic backend swap.
//! - [`speculative::SpeculativeDecoder`] — draft + verifier loop.
//! - [`moe::MoeRouter`] — softmax-gated mixture of experts.
//! - [`mock::GpuMockActor`] — CPU-only stand-in for kernel actors.

pub mod batching;
pub mod cascade;
pub mod hot_swap;
pub mod mock;
pub mod moe;
pub mod replica_pool;
pub mod scheduler;
pub mod speculative;

pub mod prelude {
    //! Canonical re-exports. `use rakka_accel_patterns::prelude::*;`.
    pub use crate::batching::{
        BatchOverflow, BatchingConfig, BatchingMsg, BatchingStats, DynamicBatchingServer,
    };
    pub use crate::cascade::{
        CascadeConfig, CascadeMsg, CascadeReply, CascadeStageEntry, InferenceCascade,
    };
    pub use crate::hot_swap::{HotSwapMsg, HotSwapStats, ModelHotSwapServer};
    pub use crate::mock::{
        GpuMockActor, GpuMockMsg, MockConv, MockFftR2C, MockRngFill, MockSgemm,
    };
    pub use crate::moe::{MoeConfig, MoeMsg, MoeRouter};
    pub use crate::replica_pool::{ModelReplicaPool, ReplicaPoolConfig, ReplicaPoolMsg, RoutingPolicy};
    pub use crate::scheduler::{
        FairShareConfig, FairShareMsg, FairShareScheduler, FairShareStats, TenantConfig, TenantId,
    };
    pub use crate::speculative::{DecodeStats, SpecMsg, SpeculativeConfig, SpeculativeDecoder};
}
