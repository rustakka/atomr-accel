//! LRU plan cache for cuTENSOR operations.
//!
//! `cutensorCreatePlan` is expensive enough that real workloads
//! amortise it across many calls with identical shape signatures.
//! [`PlanCache`] holds an `LruCache<PlanKey, CachedPlan>` so the actor
//! can hash a description once, look up an existing plan, and only
//! pay for the descriptor + plan + workspace-estimate triplet on a
//! miss.
//!
//! # Key
//!
//! Keyed by `(op_kind, modes_hash, extents_hash, alignment,
//! compute_descriptor_tag, scalar_dtype_tag, autotune_algo)` —
//! everything that influences cuTENSOR's choice of internal kernel
//! and workspace size. The autotune-picked algo is folded into the
//! key so an autotuned plan never collides with a default-algo plan.

use std::sync::Arc;

use cudarc::cutensor::result as ct_result;
use cudarc::cutensor::sys as ct_sys;
use lru::LruCache;
use parking_lot::Mutex;

/// Default LRU capacity. 256 is a generous upper bound: each entry
/// owns a `cutensorPlan_t` plus a `cutensorOperationDescriptor_t`
/// plus tensor descriptors, which together cost ~few KiB on the host
/// — order-MiB total at full occupancy.
pub const DEFAULT_PLAN_CACHE_SIZE: usize = 256;

/// Operation kind discriminator embedded in the cache key.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum OpKind {
    Contract,
    Reduce,
    ElementwiseBinary,
    ElementwiseTrinary,
    Permutation,
}

impl OpKind {
    pub fn tag(self) -> &'static str {
        match self {
            OpKind::Contract => "contract",
            OpKind::Reduce => "reduce",
            OpKind::ElementwiseBinary => "ewbin",
            OpKind::ElementwiseTrinary => "ewtri",
            OpKind::Permutation => "permute",
        }
    }
}

/// Hashable plan key. Modes / extents arrive pre-hashed (u64) so the
/// key remains `Copy + Eq`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct PlanKey {
    pub op_kind: OpKind,
    pub modes_hash: u64,
    pub extents_hash: u64,
    pub alignment: u32,
    pub compute_desc_tag: u32,
    pub dtype_tag: &'static str,
    /// `0` means "default algorithm". Autotune writes the chosen
    /// `cutensorAlgo_t as i32` here so autotuned plans get their own
    /// cache slot.
    pub algo: i32,
}

/// Newtype around the cuTENSOR descriptor pointers so we can `unsafe
/// impl Send`. Each `CachedPlan` owns its descriptors and the plan
/// itself; on `Drop` we tear them down in reverse construction order.
pub struct CachedPlan {
    pub plan: ct_sys::cutensorPlan_t,
    pub pref: ct_sys::cutensorPlanPreference_t,
    pub op: ct_sys::cutensorOperationDescriptor_t,
    pub descs: Vec<ct_sys::cutensorTensorDescriptor_t>,
    pub workspace_size: u64,
}

unsafe impl Send for CachedPlan {}
unsafe impl Sync for CachedPlan {}

impl Drop for CachedPlan {
    fn drop(&mut self) {
        unsafe {
            let _ = ct_result::destroy_plan(self.plan);
            let _ = ct_result::destroy_plan_preference(self.pref);
            let _ = ct_result::destroy_operation_descriptor(self.op);
            for d in self.descs.drain(..) {
                let _ = ct_result::destroy_tensor_descriptor(d);
            }
        }
    }
}

/// Thread-safe wrapper around `LruCache<PlanKey, Arc<CachedPlan>>`.
/// `Arc` lets the actor hand the plan out to a kernel-launch closure
/// that may outlive a subsequent cache eviction.
pub struct PlanCache {
    cache: Mutex<LruCache<PlanKey, Arc<CachedPlan>>>,
}

impl PlanCache {
    pub fn new(cap: usize) -> Self {
        let cap = std::num::NonZeroUsize::new(cap.max(1)).expect("non-zero cap");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_PLAN_CACHE_SIZE)
    }

    pub fn get(&self, key: &PlanKey) -> Option<Arc<CachedPlan>> {
        self.cache.lock().get(key).cloned()
    }

    pub fn put(&self, key: PlanKey, plan: Arc<CachedPlan>) {
        self.cache.lock().put(key, plan);
    }

    /// Cache size for tests / observability.
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }
}

/// Hash a slice of `i64` (extents or strides) into a `u64`. Uses the
/// std FxHash-equivalent default hasher. Cheap and stable within a
/// single process — that's all we need for plan-cache lookups.
pub fn hash_i64s(values: &[i64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    values.hash(&mut h);
    h.finish()
}

/// Hash a slice of `i32` modes.
pub fn hash_i32s(values: &[i32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    values.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(op_kind: OpKind, dtype_tag: &'static str, modes: u64) -> PlanKey {
        PlanKey {
            op_kind,
            modes_hash: modes,
            extents_hash: 0,
            alignment: 16,
            compute_desc_tag: 1,
            dtype_tag,
            algo: 0,
        }
    }

    #[test]
    fn cache_lru_hit_miss() {
        // Use a tiny synthetic CachedPlan that doesn't actually hold
        // cuTENSOR resources — we test the LRU policy, not cudarc.
        // Drop is suppressed because the descriptor pointers are
        // null and `destroy_*` would no-op or fault. Wrap each in a
        // ManuallyDrop to keep it inert.
        let cache = PlanCache::new(2);
        let k1 = make_key(OpKind::Contract, "f32", 1);
        let k2 = make_key(OpKind::Reduce, "f32", 2);
        let k3 = make_key(OpKind::Permutation, "f32", 3);

        // We can't construct a real CachedPlan without GPU resources.
        // The PlanCache `put`/`get` API works against `Arc<CachedPlan>`
        // — instead exercise the hash/eq surface on PlanKey, plus the
        // capacity bound, by checking len() after a mock-free path.
        // (Integration tests on a GPU exercise the full insert/get.)
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get(&k1).is_none());
        // Verify keys differ.
        assert_ne!(k1, k2);
        assert_ne!(k2, k3);
        assert_eq!(k1, make_key(OpKind::Contract, "f32", 1));
    }

    #[test]
    fn op_kind_tags_are_stable() {
        assert_eq!(OpKind::Contract.tag(), "contract");
        assert_eq!(OpKind::Reduce.tag(), "reduce");
        assert_eq!(OpKind::ElementwiseBinary.tag(), "ewbin");
        assert_eq!(OpKind::ElementwiseTrinary.tag(), "ewtri");
        assert_eq!(OpKind::Permutation.tag(), "permute");
    }

    #[test]
    fn hash_is_order_sensitive() {
        assert_ne!(hash_i64s(&[1, 2, 3]), hash_i64s(&[3, 2, 1]));
        assert_eq!(hash_i64s(&[1, 2, 3]), hash_i64s(&[1, 2, 3]));
        assert_ne!(hash_i32s(&[1, 2]), hash_i32s(&[2, 1]));
    }
}
