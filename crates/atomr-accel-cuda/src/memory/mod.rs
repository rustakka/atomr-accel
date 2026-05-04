//! Managed (unified) memory.
//!
//! Memory accessible from host and any device with no explicit copy.
//! Used for shared agent state across multiple devices.
//!
//! cudarc 0.19's runtime safe layer exposes the runtime API only
//! at the `sys` level (`cudaMallocManaged`, `cudaMemPrefetchAsync`).
//! F4 ships the actor + `ManagedRef<T>` surface; the actual
//! `cudaMallocManaged` wrapping is the F4.x follow-up.

mod managed;

pub use managed::{ManagedAllocatorActor, ManagedFlags, ManagedMsg, ManagedRef, ManagedStats};
