//! Heuristic-cache for cuBLASLt matmul algorithms.
//!
//! cuBLASLt's `cublasLtMatmulAlgoGetHeuristic` is a synchronous
//! library call that takes single-digit milliseconds. For repeated
//! shapes (every iteration of a transformer step) we cache the
//! best-by-wall-time algorithm under
//! `(m, n, k, dtype, layout, epilogue, sm_arch)` and reuse it.
//!
//! Key design points:
//! - LRU eviction (capacity defaults to 256 entries — large enough to
//!   cover a model's full shape repertoire, small enough to fit in a
//!   couple KiB of host RAM).
//! - The cache lives in a `parking_lot::Mutex<lru::LruCache>` behind
//!   an `Arc`, so a cloneable `HeuristicCacheRef` can flow into
//!   per-message `BlasLtDispatchCtx` without `Send` headaches.
//! - We store the raw `cublasLtMatmulAlgo_t` plus a `workspace_size`
//!   hint; the actor's `WorkspacePool` uses the workspace size to
//!   recycle the right slot.

use std::num::NonZeroUsize;
use std::sync::Arc;

use cudarc::cublaslt::sys::cublasLtMatmulAlgo_t;
use lru::LruCache;
use parking_lot::Mutex;

use crate::dtype::DTypeKind;
use crate::kernel::blas_lt::epilogue::Epilogue;

/// Cache key — fully self-describing so two requests with the same
/// shape/layout/dtype/epilogue/arch trio land in the same bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeuristicKey {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    /// Stable dtype tag. Captured as a u32 (rather than `DTypeKind`)
    /// so the key derives cleanly even if `DTypeKind` grows new
    /// variants.
    pub dtype: u32,
    pub transa: bool,
    pub transb: bool,
    pub epilogue: Epilogue,
    pub sm_arch: u32,
}

impl HeuristicKey {
    pub fn new(
        m: i32,
        n: i32,
        k: i32,
        dtype: DTypeKind,
        transa: bool,
        transb: bool,
        epilogue: Epilogue,
        sm_arch: u32,
    ) -> Self {
        Self {
            m,
            n,
            k,
            dtype: dtype as u32,
            transa,
            transb,
            epilogue,
            sm_arch,
        }
    }
}

/// Cached value — best algorithm by wall-time plus the workspace size
/// the heuristic reported.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicEntry {
    pub algo: cublasLtMatmulAlgo_t,
    pub workspace_size: usize,
    /// Reported wall time (`wavesCount` from
    /// `cublasLtMatmulHeuristicResult_t`); lower is better. Stored so
    /// callers can decide to re-run the search if a better algorithm
    /// might be available after a tuning sweep.
    pub waves_count: f32,
}

// SAFETY: `cublasLtMatmulAlgo_t` is `repr(C) [u64; 8]` — pure POD.
unsafe impl Send for HeuristicEntry {}
unsafe impl Sync for HeuristicEntry {}

/// Default capacity of the heuristic cache.
pub const DEFAULT_HEURISTIC_CAPACITY: usize = 256;

/// Default top-k of algorithms to query from cuBLASLt on each cold
/// lookup. We keep the best by `waves_count` and discard the rest.
pub const DEFAULT_TOP_K: usize = 3;

/// Shareable handle to the heuristic cache. Cheap to clone.
#[derive(Clone)]
pub struct HeuristicCacheRef {
    inner: Arc<Mutex<LruCache<HeuristicKey, HeuristicEntry>>>,
    top_k: usize,
}

impl HeuristicCacheRef {
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1))
            .expect("HeuristicCacheRef::with_capacity: cap.max(1) is non-zero");
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(cap))),
            top_k: DEFAULT_TOP_K,
        }
    }

    pub fn default_size() -> Self {
        Self::with_capacity(DEFAULT_HEURISTIC_CAPACITY)
    }

    /// Number of algorithms to request from cuBLASLt on cold lookup.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Cache hit (refreshes LRU order) or miss.
    pub fn get(&self, key: &HeuristicKey) -> Option<HeuristicEntry> {
        self.inner.lock().get(key).copied()
    }

    /// Insert a (key, entry) pair, possibly evicting the LRU tail.
    pub fn insert(&self, key: HeuristicKey, entry: HeuristicEntry) {
        self.inner.lock().put(key, entry);
    }

    /// Snapshot of cache occupancy. Diagnostic only.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_entry(waves: f32) -> HeuristicEntry {
        HeuristicEntry {
            algo: cublasLtMatmulAlgo_t { data: [0u64; 8] },
            workspace_size: 4 * 1024 * 1024,
            waves_count: waves,
        }
    }

    fn k(m: i32, n: i32, k: i32) -> HeuristicKey {
        HeuristicKey::new(m, n, k, DTypeKind::F32, false, false, Epilogue::None, 0)
    }

    #[test]
    fn cache_lru_hit_miss() {
        let cache = HeuristicCacheRef::with_capacity(2);
        assert!(cache.is_empty());

        let k1 = k(64, 64, 64);
        let k2 = k(128, 128, 128);
        let k3 = k(256, 256, 256);

        // Cold misses.
        assert!(cache.get(&k1).is_none());
        cache.insert(k1, dummy_entry(1.5));
        cache.insert(k2, dummy_entry(2.5));
        assert_eq!(cache.len(), 2);

        // Hits refresh order — touching k1 makes k2 the LRU tail.
        let hit = cache.get(&k1).expect("k1 should hit");
        assert_eq!(hit.waves_count, 1.5);

        // Overflow evicts the LRU tail (k2).
        cache.insert(k3, dummy_entry(3.5));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&k2).is_none(), "k2 should have been evicted");
        assert!(cache.get(&k1).is_some(), "k1 should still be present");
        assert!(cache.get(&k3).is_some(), "k3 should be present");
    }

    #[test]
    fn capacity_min_one() {
        let cache = HeuristicCacheRef::with_capacity(0);
        cache.insert(k(1, 1, 1), dummy_entry(0.0));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn distinct_keys_for_different_axes() {
        let base = k(64, 64, 64);
        let with_trans = HeuristicKey {
            transa: true,
            ..base
        };
        let with_arch = HeuristicKey {
            sm_arch: 90,
            ..base
        };
        let with_epi = HeuristicKey {
            epilogue: Epilogue::Bias,
            ..base
        };
        assert_ne!(base, with_trans);
        assert_ne!(base, with_arch);
        assert_ne!(base, with_epi);
    }
}
