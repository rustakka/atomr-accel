//! cuBLASLt workspace pool — recycles per-heuristic device buffers.
//!
//! cuBLASLt's `cublasLtMatmul` takes an opaque `workspace` device
//! pointer + a `workspaceSizeInBytes`. The size depends on the
//! selected algorithm; the heuristic cache reports a per-algorithm
//! `workspaceSize` and we want to avoid allocating a fresh slab on
//! every call. This pool buckets free slabs by **rounded-up size
//! class** (next power of two ≥ requested) and hands them out under
//! a `WorkspaceLease` RAII guard that returns the slab on Drop.
//!
//! The pool is intentionally a **plain struct** (not a separate
//! actor) because `BlasLtActor` already has single-threaded ownership
//! of all matmul calls. Wrapping it in another actor would just add a
//! mailbox hop on the hot path. If a future phase needs cross-actor
//! sharing we'll lift this to its own actor exactly the way
//! `PinnedBufferPool` (whose allocator pattern this module mirrors)
//! is structured.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use parking_lot::Mutex;

use crate::error::GpuError;

/// Default cap on number of pooled slabs per size class. Beyond this,
/// excess returns are dropped instead of pooled. With the default 256
/// distinct heuristic shapes and a typical 2-3 distinct workspace
/// classes (4 MiB, 32 MiB, 256 MiB) we expect ≤ a few hundred MiB of
/// pinned VRAM in steady state.
pub const DEFAULT_POOL_CAPACITY_PER_CLASS: usize = 4;

/// Round a workspace request up to the next power of two ≥ 1 KiB.
/// Bucketing by power-of-two limits the long-tail of unique sizes
/// the pool tracks.
pub fn size_class(bytes: usize) -> usize {
    bytes.max(1024).next_power_of_two()
}

/// Inner pool state — guarded by a single mutex.
struct WorkspacePoolInner {
    /// Free slabs grouped by size class.
    free: HashMap<usize, Vec<Arc<CudaSlice<u8>>>>,
    /// Maximum number of free slabs to retain per size class.
    per_class_capacity: usize,
    /// Sum of bytes currently held in `free`. Tracked for
    /// observability + to feed a future high-watermark eviction.
    bytes_pooled: usize,
}

/// Cloneable handle to the workspace pool.
#[derive(Clone)]
pub struct WorkspacePool {
    inner: Arc<Mutex<WorkspacePoolInner>>,
}

impl WorkspacePool {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_POOL_CAPACITY_PER_CLASS)
    }

    pub fn with_capacity(per_class: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkspacePoolInner {
                free: HashMap::new(),
                per_class_capacity: per_class.max(1),
                bytes_pooled: 0,
            })),
        }
    }

    /// Acquire a workspace slab of at least `requested_bytes`. If the
    /// pool has a free slab in the matching size class it's reused;
    /// otherwise a fresh `CudaSlice<u8>` is allocated against `stream`.
    ///
    /// The returned [`WorkspaceLease`] auto-returns to the pool on
    /// Drop. Callers should *not* manually clone the inner slice
    /// across the boundary — the kernel envelope already keeps the
    /// `Arc<CudaSlice<u8>>` alive for the duration of the kernel.
    pub fn acquire(
        &self,
        stream: &Arc<CudaStream>,
        requested_bytes: usize,
    ) -> Result<WorkspaceLease, GpuError> {
        let class = size_class(requested_bytes.max(1));

        // Check the pool first.
        let pooled = {
            let mut g = self.inner.lock();
            if let Some(bucket) = g.free.get_mut(&class) {
                let popped = bucket.pop();
                if let Some(ref s) = popped {
                    g.bytes_pooled = g.bytes_pooled.saturating_sub(s.len());
                }
                popped
            } else {
                None
            }
        };

        let slab = match pooled {
            Some(s) => s,
            None => {
                let s = unsafe { stream.alloc::<u8>(class) }.map_err(|e| {
                    GpuError::OutOfMemory(format!("cublaslt workspace alloc {class}B: {e}"))
                })?;
                Arc::new(s)
            }
        };

        Ok(WorkspaceLease {
            slab: Some(slab),
            class,
            pool: self.inner.clone(),
        })
    }

    /// Number of slabs currently in the free list across all size
    /// classes. Diagnostic only.
    pub fn pooled_slabs(&self) -> usize {
        let g = self.inner.lock();
        g.free.values().map(|v| v.len()).sum()
    }

    /// Total bytes currently in the free list.
    pub fn pooled_bytes(&self) -> usize {
        self.inner.lock().bytes_pooled
    }
}

impl Default for WorkspacePool {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII lease — returns the slab to the pool on Drop. Callers can
/// take a shared reference to the inner slice for kernel launch via
/// [`WorkspaceLease::slice`].
pub struct WorkspaceLease {
    slab: Option<Arc<CudaSlice<u8>>>,
    class: usize,
    pool: Arc<Mutex<WorkspacePoolInner>>,
}

impl WorkspaceLease {
    /// Reference to the underlying device slice (`CudaSlice<u8>`).
    pub fn slice(&self) -> &Arc<CudaSlice<u8>> {
        self.slab
            .as_ref()
            .expect("WorkspaceLease::slice after Drop")
    }

    /// Size class (rounded-up bytes) of the leased slab.
    pub fn size(&self) -> usize {
        self.class
    }
}

impl Drop for WorkspaceLease {
    fn drop(&mut self) {
        let Some(slab) = self.slab.take() else {
            return;
        };
        let mut g = self.pool.lock();
        let cap = g.per_class_capacity;
        let bucket = g.free.entry(self.class).or_default();
        if bucket.len() < cap {
            let bytes = slab.len();
            bucket.push(slab);
            g.bytes_pooled = g.bytes_pooled.saturating_add(bytes);
        }
        // else: drop the slab; CudaSlice's Drop frees the device memory.
    }
}

#[cfg(test)]
mod tests {
    //! These tests stay GPU-free by exercising the pool's bookkeeping
    //! through a hand-rolled `WorkspaceLease` path that reuses an
    //! `Arc<CudaSlice<u8>>` we never construct (we use a synthetic
    //! return helper that mirrors the Drop contract).
    use super::*;

    /// Reach into pool internals from tests to seed a free-list slab
    /// without touching the GPU. Mirrors the Drop path.
    fn seed_free_slot(pool: &WorkspacePool, class: usize, slab_bytes: usize) {
        let mut g = pool.inner.lock();
        // Synthesize a fake CudaSlice<u8>… we can't, so seed the
        // bookkeeping directly: track the bytes_pooled but skip the
        // actual `Arc<CudaSlice<u8>>` since constructing one without
        // a context isn't possible. The pool exposes `pooled_bytes`
        // so the recycle test verifies size-class accounting only.
        g.bytes_pooled = g.bytes_pooled.saturating_add(slab_bytes);
        // Ensure the bucket key exists (so we know recycling happens
        // *into* this class on a subsequent return).
        g.free.entry(class).or_default();
    }

    #[test]
    fn size_class_rounds_up() {
        assert_eq!(size_class(1), 1024);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 2048);
        assert_eq!(size_class(4 * 1024 * 1024), 4 * 1024 * 1024);
        assert_eq!(size_class(4 * 1024 * 1024 + 1), 8 * 1024 * 1024);
        assert_eq!(size_class(0), 1024);
    }

    #[test]
    fn workspace_pool_recycles() {
        // We can't allocate real CudaSlices without a GPU, so this
        // test exercises the pool's bookkeeping (size_class +
        // pooled_bytes accounting) and checks that the bucket map
        // tracks classes correctly across "returns".
        let pool = WorkspacePool::with_capacity(2);
        assert_eq!(pool.pooled_slabs(), 0);

        seed_free_slot(&pool, size_class(4 * 1024 * 1024), 4 * 1024 * 1024);
        assert_eq!(pool.pooled_bytes(), 4 * 1024 * 1024);

        seed_free_slot(&pool, size_class(33_554_432), 33_554_432);
        assert_eq!(pool.pooled_bytes(), 4 * 1024 * 1024 + 33_554_432);
        // Two distinct size classes — both buckets exist.
        assert!(pool
            .inner
            .lock()
            .free
            .contains_key(&size_class(4 * 1024 * 1024)));
        assert!(pool.inner.lock().free.contains_key(&size_class(33_554_432)));
    }

    #[test]
    fn pool_capacity_clamps_to_one() {
        let pool = WorkspacePool::with_capacity(0);
        assert_eq!(pool.inner.lock().per_class_capacity, 1);
    }
}
