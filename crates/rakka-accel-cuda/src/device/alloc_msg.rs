//! Typed-allocation + memcpy support types for `DeviceMsg`.
//!
//! F1 hard-coded `DeviceMsg::Allocate` to f32. F2 adds per-dtype
//! variants ([`device_actor::DeviceMsg::AllocateF32`],
//! `AllocateF64`, …). Each preserves `GpuRef<T>` static typing on the
//! receive side — a runtime-tagged `DType` enum would erase that.
//!
//! Supported dtypes:
//! - `f32`, `f64` — primary scientific computing types
//! - `i8`, `i32`, `i64` — signed integer
//! - `u8`, `u32`, `u64` — unsigned integer
//! - `f16`, `bf16` — gated on the `f16` cargo feature

use crate::host::PinnedBuf;

/// Host-side buffer surface. Owned `Vec<T>` for low-volume
/// convenience; [`PinnedBuf<T>`] for async-overlappable transfers
/// sourced from a [`crate::host::PinnedBufferPool`].
pub enum HostBuf<T> {
    Owned(Vec<T>),
    Pinned(PinnedBuf<T>),
}

impl<T> HostBuf<T> {
    pub fn len(&self) -> usize {
        match self {
            HostBuf::Owned(v) => v.len(),
            HostBuf::Pinned(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for HostBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostBuf::Owned(v) => f.debug_tuple("HostBuf::Owned").field(&v.len()).finish(),
            HostBuf::Pinned(p) => f.debug_tuple("HostBuf::Pinned").field(&p.len()).finish(),
        }
    }
}

/// Per-device load snapshot returned by [`DeviceMsg::Stats`]. Used by
/// the F5 [`crate::placement::PlacementActor`] for least-loaded
/// scheduling.
#[derive(Debug, Clone, Copy)]
pub struct DeviceLoad {
    pub free_bytes: usize,
    pub total_bytes: usize,
    pub active_streams: u32,
    pub queue_depth: u32,
    pub compute_cap: (i32, i32),
}
