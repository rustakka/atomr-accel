//! Managed (unified) memory + Phase 3 driver-API helpers.
//!
//! - [`managed`] — `cudaMallocManaged` actor (`ManagedAllocatorActor`,
//!   `ManagedRef<T>`, `ManagedFlags`, `ManagedStats`,
//!   `PrefetchTarget`).
//! - [`prefetch`] — `cuMemPrefetchAsync` wrapper.
//! - [`advise`] — `cuMemAdvise` wrapper + the [`advise::MemAdvice`]
//!   typed enum.
//! - [`ipc`] — `cuIpcGetMemHandle` / `cuIpcOpenMemHandle` /
//!   `cuIpcCloseMemHandle` (gated `cuda-ipc`).

pub mod advise;
#[cfg(feature = "cuda-ipc")]
pub mod ipc;
pub mod managed;
pub mod prefetch;

pub use advise::MemAdvice;
pub use managed::{
    ManagedAllocatorActor, ManagedFlags, ManagedMsg, ManagedRef, ManagedStats, PrefetchTarget,
};
#[cfg(feature = "cuda-ipc")]
pub use ipc::{IpcMemHandle, OpenedMem};
