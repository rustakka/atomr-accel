//! Host-side support: pinned (page-locked) memory pool + `PinnedBuf<T>`.
//!
//! Pinned memory enables true asynchronous H2D / D2H copy. Without it
//! cudarc's `memcpy_*_async` falls back to a synchronous host copy.
//!
//! The pool is an actor at the [`crate::device::DeviceActor`] tier
//! (sibling to [`crate::device::ContextActor`]) so it survives
//! context restarts. Acquires a fresh [`PinnedBuf<T>`] from the
//! free-list; on `Drop` the buffer is returned to the pool via an
//! actor message so the mailbox is the single owner of the
//! free-list — no global locks.

mod pinned;

pub use pinned::{
    PinnedBuf, PinnedBufferPool, PinnedBufferPoolConfig, PinnedPoolMsg, PinnedPoolStats,
};
