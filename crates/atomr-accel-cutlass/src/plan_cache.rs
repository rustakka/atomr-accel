//! Plan cache: in-process LRU keyed by template instantiation
//! identity. Backs the `CutlassActor` so that repeated GEMM messages
//! with identical `(template_id, shape, dtype, arch)` skip both the
//! Rust-side template render and the NVRTC compile.
//!
//! The on-disk cache used by `NvrtcActor` (Phase 0.6) takes care of
//! cross-process / cross-restart caching; this LRU is purely an
//! in-process win to avoid rendering the same `.cu` source string
//! repeatedly.

use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use crate::conv::{ConvKind, ConvLayout, ConvShape};
use crate::dtype::{CutlassDtype, GemmSupported, SmArch};
use crate::gemm::{GemmEpilogue, GemmLayout, GemmShape};
#[cfg(feature = "grouped")]
use crate::grouped_gemm::GroupedLayout;

/// Plan-cache key. `u128` is chosen so all Phase 6 templates fit
/// without collisions while staying small enough to copy by value.
///
/// The key is opaque on purpose: callers should construct it via the
/// dedicated `PlanKey::gemm` / `PlanKey::grouped_gemm` / `PlanKey::conv`
/// constructors so we control the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanKey {
    /// Discriminator: 1 = gemm, 2 = grouped_gemm, 3 = conv.
    template_id: u8,
    /// Packed shape/layout/dtype/arch identity. The exact layout is
    /// internal to this module; callers that need to inspect the key
    /// use the read-only accessors below.
    payload: [u64; 3],
}

impl PlanKey {
    /// Element size of the cache key in bytes — used by the cache size
    /// reporter and by callers that want to budget the LRU.
    pub const SIZE_BYTES: usize = core::mem::size_of::<PlanKey>();

    pub fn template_id(&self) -> u8 {
        self.template_id
    }

    /// Build a key for a single GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<T: GemmSupported>(
        shape: GemmShape,
        layout_a: GemmLayout,
        layout_b: GemmLayout,
        layout_c: GemmLayout,
        epilogue: GemmEpilogue,
        accum: CutlassDtype,
        out: CutlassDtype,
        arch: SmArch,
        persistent: bool,
    ) -> Self {
        let mut h = Hasher::new();
        h.add_u32(shape.m);
        h.add_u32(shape.n);
        h.add_u32(shape.k);
        h.add_u8(layout_a as u8);
        h.add_u8(layout_b as u8);
        h.add_u8(layout_c as u8);
        h.add_str(T::DTYPE.short_name());
        h.add_str(accum.short_name());
        h.add_str(out.short_name());
        h.add_str(arch.short_name());
        h.add_u8(persistent as u8);
        h.add_str(epilogue.short_name());
        match epilogue {
            GemmEpilogue::Linear { alpha, beta }
            | GemmEpilogue::LinearReLU { alpha, beta }
            | GemmEpilogue::LinearGelu { alpha, beta } => {
                h.add_u32(alpha.to_bits());
                h.add_u32(beta.to_bits());
            }
        }
        Self {
            template_id: 1,
            payload: h.finish(),
        }
    }

    /// Build a key for a grouped GEMM. `shape_summary` is the
    /// `(max_m, max_n, max_k, group_count)` tuple from
    /// `GroupedGemmShape::summary`.
    #[cfg(feature = "grouped")]
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_gemm<T: GemmSupported>(
        shape_summary: (u32, u32, u32, usize),
        layout_a: GemmLayout,
        layout_b: GemmLayout,
        layout_c: GemmLayout,
        grouped_layout: GroupedLayout,
        epilogue: GemmEpilogue,
        accum: CutlassDtype,
        out: CutlassDtype,
        arch: SmArch,
        persistent: bool,
    ) -> Self {
        let mut h = Hasher::new();
        h.add_u32(shape_summary.0);
        h.add_u32(shape_summary.1);
        h.add_u32(shape_summary.2);
        h.add_u32(shape_summary.3 as u32);
        h.add_u8(layout_a as u8);
        h.add_u8(layout_b as u8);
        h.add_u8(layout_c as u8);
        h.add_str(grouped_layout.short_name());
        h.add_str(T::DTYPE.short_name());
        h.add_str(accum.short_name());
        h.add_str(out.short_name());
        h.add_str(arch.short_name());
        h.add_u8(persistent as u8);
        h.add_str(epilogue.short_name());
        Self {
            template_id: 2,
            payload: h.finish(),
        }
    }

    /// Stub overload kept available even when the `grouped` feature
    /// is off, so callers that conditionally produce grouped keys
    /// still link. The non-`grouped` build returns a deterministic
    /// placeholder that distinguishes itself from the gemm/conv keys.
    #[cfg(not(feature = "grouped"))]
    #[allow(dead_code)]
    pub(crate) fn grouped_gemm_unsupported() -> Self {
        Self {
            template_id: 2,
            payload: [0, 0, 0],
        }
    }

    pub(crate) fn conv<T: GemmSupported>(
        kind: ConvKind,
        shape: ConvShape,
        layout: ConvLayout,
        accum: CutlassDtype,
        out: CutlassDtype,
        arch: SmArch,
    ) -> Self {
        let mut h = Hasher::new();
        h.add_str(kind.short_name());
        h.add_u32(shape.n);
        h.add_u32(shape.h);
        h.add_u32(shape.w);
        h.add_u32(shape.c);
        h.add_u32(shape.k);
        h.add_u32(shape.r);
        h.add_u32(shape.s);
        h.add_u32(shape.pad_h);
        h.add_u32(shape.pad_w);
        h.add_u32(shape.stride_h);
        h.add_u32(shape.stride_w);
        h.add_u32(shape.dil_h);
        h.add_u32(shape.dil_w);
        h.add_str(layout.short_name());
        h.add_str(T::DTYPE.short_name());
        h.add_str(accum.short_name());
        h.add_str(out.short_name());
        h.add_str(arch.short_name());
        Self {
            template_id: 3,
            payload: h.finish(),
        }
    }
}

/// Compact hashing helper. Splits the SipHash output into three u64
/// lanes so the resulting key is `Eq` on the underlying bytes
/// without collisions across template kinds.
struct Hasher {
    a: std::collections::hash_map::DefaultHasher,
    b: std::collections::hash_map::DefaultHasher,
    c: std::collections::hash_map::DefaultHasher,
}

impl Hasher {
    fn new() -> Self {
        use std::hash::Hasher as _;
        let mut a = std::collections::hash_map::DefaultHasher::new();
        let mut b = std::collections::hash_map::DefaultHasher::new();
        let mut c = std::collections::hash_map::DefaultHasher::new();
        a.write_u64(0xA5A5_A5A5_A5A5_A5A5);
        b.write_u64(0x5A5A_5A5A_5A5A_5A5A);
        c.write_u64(0xC3C3_C3C3_C3C3_C3C3);
        Self { a, b, c }
    }

    fn add_u8(&mut self, v: u8) {
        use std::hash::Hasher as _;
        self.a.write_u8(v);
        self.b.write_u8(v.wrapping_add(0x55));
        self.c.write_u8(v.wrapping_add(0xAA));
    }

    fn add_u32(&mut self, v: u32) {
        use std::hash::Hasher as _;
        self.a.write_u32(v);
        self.b.write_u32(v.rotate_left(11));
        self.c.write_u32(v.rotate_left(23));
    }

    fn add_str(&mut self, s: &str) {
        use std::hash::Hasher as _;
        self.a.write(s.as_bytes());
        self.b.write(s.as_bytes());
        self.c.write(s.as_bytes());
    }

    fn finish(self) -> [u64; 3] {
        use std::hash::Hasher as _;
        [self.a.finish(), self.b.finish(), self.c.finish()]
    }
}

/// Cached plan entry. The payload is type-erased so a single cache
/// can hold gemm / grouped-gemm / conv plans side by side.
pub struct CachedPlan {
    pub key: PlanKey,
    pub source: Arc<String>,
    pub kernel_name: Arc<String>,
    /// Optional opaque payload, e.g. the post-NVRTC `KernelHandle`.
    pub kernel_handle: Option<Arc<dyn Any + Send + Sync>>,
}

impl core::fmt::Debug for CachedPlan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CachedPlan")
            .field("key", &self.key)
            .field("kernel_name", &*self.kernel_name)
            .field("source_len", &self.source.len())
            .field("has_kernel_handle", &self.kernel_handle.is_some())
            .finish()
    }
}

/// LRU plan cache. Default capacity is 64 entries — a single device
/// rarely keeps more than that many distinct CUTLASS template
/// instantiations live at once, and at ~64 KiB / cached `.cu` source
/// the cache stays well under a megabyte.
pub struct PlanCache {
    inner: Mutex<LruCache<PlanKey, Arc<CachedPlan>>>,
    capacity: usize,
}

impl PlanCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("capacity > 0");
        Self {
            inner: Mutex::new(LruCache::new(cap)),
            capacity: cap.get(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, key: &PlanKey) -> Option<Arc<CachedPlan>> {
        self.inner.lock().get(key).cloned()
    }

    pub fn insert(&self, plan: CachedPlan) -> Arc<CachedPlan> {
        let key = plan.key;
        let arc = Arc::new(plan);
        self.inner.lock().put(key, arc.clone());
        arc
    }

    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::F16;
    use crate::gemm::{GemmLayout, GemmShape};

    fn k(m: u32) -> PlanKey {
        PlanKey::gemm::<F16>(
            GemmShape::new(m, 64, 64),
            GemmLayout::RowMajor,
            GemmLayout::RowMajor,
            GemmLayout::RowMajor,
            GemmEpilogue::default(),
            CutlassDtype::F32,
            CutlassDtype::F16,
            SmArch::Sm80,
            false,
        )
    }

    #[test]
    fn plan_cache_lru_round_trip() {
        let cache = PlanCache::new(2);
        assert_eq!(cache.capacity(), 2);
        assert!(cache.is_empty());

        let p1 = cache.insert(CachedPlan {
            key: k(1),
            source: Arc::new("a".into()),
            kernel_name: Arc::new("k1".into()),
            kernel_handle: None,
        });
        let p2 = cache.insert(CachedPlan {
            key: k(2),
            source: Arc::new("b".into()),
            kernel_name: Arc::new("k2".into()),
            kernel_handle: None,
        });
        assert_eq!(cache.len(), 2);

        // Hit on p1 promotes it; inserting p3 must evict p2.
        let _ = cache.get(&p1.key).unwrap();
        let _ = cache.insert(CachedPlan {
            key: k(3),
            source: Arc::new("c".into()),
            kernel_name: Arc::new("k3".into()),
            kernel_handle: None,
        });
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&p2.key).is_none());
        assert!(cache.get(&p1.key).is_some());

        // PlanKey size is 4 bytes (template_id padded) + 24 bytes
        // payload — guarded so a future struct-layout change can't
        // accidentally explode the cache memory budget.
        assert!(PlanKey::SIZE_BYTES <= 64);

        // Distinct shapes -> distinct keys.
        assert_ne!(k(1), k(2));

        // Clear empties the cache.
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn plan_keys_distinct_across_template_kinds() {
        let gemm = k(1);
        let conv = PlanKey::conv::<F16>(
            ConvKind::Fprop,
            ConvShape::nhwc(1, 1, 1, 1, 1, 1, 1),
            ConvLayout::Nhwc,
            CutlassDtype::F32,
            CutlassDtype::F16,
            SmArch::Sm80,
        );
        assert_ne!(gemm, conv);
        assert_ne!(gemm.template_id(), conv.template_id());
    }
}
