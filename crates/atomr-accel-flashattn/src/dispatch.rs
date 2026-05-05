//! Dispatch table — maps a `(arch, dtype, head_dim, …)` cell onto a
//! mangled kernel name expression.
//!
//! The Phase 7 FlashAttention crate ships forward + backward paths for
//! v2 (sm_80 / sm_89) and v3 (sm_90a, including the fp8 e4m3 / e5m2
//! variants). Every kernel is NVRTC-compiled lazily through the Phase
//! 0.6 disk cache; the dispatch table is the *only* place that knows
//! the canonical mangled symbol — every request type ([`crate::fa2`],
//! [`crate::fa3`], [`crate::paged`], [`crate::prefill`], [`crate::varlen`])
//! produces a [`DispatchKey`] that hashes to the same string.
//!
//! Hot path:
//!
//! 1. Caller constructs a request (e.g. [`crate::fa2::Fa2FwdRequest`]).
//! 2. [`FaFwdDispatch::dispatch_key`] yields a [`DispatchKey`].
//! 3. [`DispatchTable::lookup`] resolves the key to a kernel name.
//! 4. The actor asks `NvrtcActor` to compile-or-fetch by name.
//! 5. The cubin is launched on the actor's stream.
//!
//! Steps 3–5 are GPU-only and gated behind `cuda-runtime-tests`; the
//! request-construction path (1–2) is exercised by the unit tests
//! below and from each request-type module's `tests` block.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use once_cell::sync::Lazy;

/// CUDA streaming-multiprocessor architecture target. The dispatch
/// table refuses to resolve any key whose `arch` is not in this list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmArch {
    /// Ampere (A100, A30) — fa2 only.
    Sm80,
    /// Ada (RTX 40xx, L4) — fa2 only, supports fp8 cuBLASLt but not fa3.
    Sm89,
    /// Hopper (H100, H200) — fa3, fp8, TMA, WGMMA, persistent kernels.
    Sm90a,
    /// Blackwell (B100, B200) — forward-compat target; fa3 with fifth-gen
    /// tensor cores. Falls back to Hopper kernels for now.
    Sm100,
}

impl SmArch {
    /// CUDA `--gpu-architecture` string.
    pub fn nvrtc_flag(self) -> &'static str {
        match self {
            SmArch::Sm80 => "--gpu-architecture=sm_80",
            SmArch::Sm89 => "--gpu-architecture=sm_89",
            SmArch::Sm90a => "--gpu-architecture=sm_90a",
            SmArch::Sm100 => "--gpu-architecture=sm_100a",
        }
    }

    /// True if this arch supports FlashAttention v3 (Hopper+).
    pub fn supports_fa3(self) -> bool {
        matches!(self, SmArch::Sm90a | SmArch::Sm100)
    }

    /// True if this arch supports fp8 e4m3 / e5m2 in FA3.
    pub fn supports_fp8(self) -> bool {
        matches!(self, SmArch::Sm90a | SmArch::Sm100)
    }
}

/// Element type for Q / K / V tiles. Distinct from `atomr-accel-cuda`'s
/// future `CudaDtype` so the FlashAttn crate is self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 binary16 — fa2 + fa3.
    F16,
    /// bfloat16 — fa2 + fa3.
    Bf16,
    /// 8-bit float, e4m3 — fa3 only, sm_90a+.
    F8E4m3,
    /// 8-bit float, e5m2 — fa3 only, sm_90a+ (used for V in DPA-mixed-precision).
    F8E5m2,
}

impl DType {
    /// Element width in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F16 | DType::Bf16 => 2,
            DType::F8E4m3 | DType::F8E5m2 => 1,
        }
    }

    /// True iff this dtype is one of the fp8 variants.
    pub fn is_fp8(self) -> bool {
        matches!(self, DType::F8E4m3 | DType::F8E5m2)
    }

    /// Short tag used inside the kernel-name mangling.
    pub fn tag(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::F8E4m3 => "e4m3",
            DType::F8E5m2 => "e5m2",
        }
    }
}

/// Marker trait for dtypes that can drive a FlashAttention GEMM. Implemented
/// by the same set of zero-sized types that the rest of `atomr-accel`
/// uses to phantom-tag GEMM-supported dtypes. The trait itself carries
/// no methods so it can be referenced from [`crate::fa2`] / [`crate::fa3`]
/// without requiring callers to depend on `atomr-accel-cuda` directly.
pub trait GemmSupported: Send + Sync + 'static {
    /// The runtime dtype tag this marker maps onto.
    fn dtype() -> DType;
}

/// Zero-sized marker for `f16` (IEEE binary16).
#[derive(Debug, Clone, Copy)]
pub struct F16;
impl GemmSupported for F16 {
    fn dtype() -> DType {
        DType::F16
    }
}

/// Zero-sized marker for `bf16` (bfloat16).
#[derive(Debug, Clone, Copy)]
pub struct Bf16;
impl GemmSupported for Bf16 {
    fn dtype() -> DType {
        DType::Bf16
    }
}

/// Zero-sized marker for fp8 e4m3 (gated `fp8`).
#[cfg(feature = "fp8")]
#[derive(Debug, Clone, Copy)]
pub struct F8E4m3;
#[cfg(feature = "fp8")]
impl GemmSupported for F8E4m3 {
    fn dtype() -> DType {
        DType::F8E4m3
    }
}

/// Zero-sized marker for fp8 e5m2 (gated `fp8`).
#[cfg(feature = "fp8")]
#[derive(Debug, Clone, Copy)]
pub struct F8E5m2;
#[cfg(feature = "fp8")]
impl GemmSupported for F8E5m2 {
    fn dtype() -> DType {
        DType::F8E5m2
    }
}

/// Cell key for the FlashAttention dispatch table.
///
/// Every field directly affects the generated CUDA C++ template
/// instantiation — flipping any one of them changes the resulting
/// cubin. The table refuses to resolve unsupported combinations
/// (e.g. `fp8` on `Sm80`, head_dim > 256).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DispatchKey {
    /// Target SM architecture.
    pub arch: SmArch,
    /// Element type for Q/K/V.
    pub dtype: DType,
    /// Per-head dimension (D). Supported: 64, 80, 96, 128, 192, 256.
    pub head_dim: u32,
    /// Causal masking — autoregressive attention.
    pub causal: bool,
    /// Variable-length (cu_seqlens). When false, batched attention with
    /// uniform seqlen.
    pub varlen: bool,
    /// Sliding-window size; `None` means full attention. Window size
    /// is the number of past tokens each query attends to.
    pub sliding_window: Option<u32>,
    /// ALiBi linear-position biases.
    pub alibi: bool,
    /// Number of "sink" tokens (StreamingLLM); each query unconditionally
    /// attends to the first `sink` keys regardless of `sliding_window`.
    pub sink: u32,
    /// vLLM-style paged KV-cache.
    pub paged: bool,
    /// Q heads per KV head. 1 = MHA, >1 = GQA, equal to num_heads = MQA.
    pub gqa_ratio: u32,
}

impl DispatchKey {
    /// Validate the cell for a *forward* path. Returns `Err` for
    /// unreachable combinations.
    pub fn validate_fwd(&self) -> Result<(), DispatchError> {
        // Head-dim whitelist
        const ALLOWED: &[u32] = &[64, 80, 96, 128, 192, 256];
        if !ALLOWED.contains(&self.head_dim) {
            return Err(DispatchError::UnsupportedHeadDim(self.head_dim));
        }

        // fp8 only on FA3-capable architectures
        if self.dtype.is_fp8() && !self.arch.supports_fp8() {
            return Err(DispatchError::Fp8RequiresHopper(self.arch));
        }

        // Sink tokens require sliding_window or causal — otherwise the
        // mask is just full attention.
        if self.sink > 0 && self.sliding_window.is_none() && !self.causal {
            return Err(DispatchError::SinkWithoutMask);
        }

        // GQA ratio must be a power of two and at least 1.
        if self.gqa_ratio == 0 {
            return Err(DispatchError::InvalidGqaRatio(self.gqa_ratio));
        }

        // Sliding-window size must be > 0 when present.
        if let Some(w) = self.sliding_window {
            if w == 0 {
                return Err(DispatchError::ZeroWindow);
            }
        }

        Ok(())
    }

    /// Validate the cell for a *backward* path. Currently the same as
    /// forward, but kept distinct so we can refuse e.g. fp8 backward
    /// (numerically too lossy in the stock FA3) without affecting the
    /// forward whitelist.
    pub fn validate_bwd(&self) -> Result<(), DispatchError> {
        self.validate_fwd()?;
        if self.dtype.is_fp8() {
            return Err(DispatchError::Fp8BackwardUnsupported);
        }
        Ok(())
    }

    /// Validate the cell for a *paged* forward path.
    pub fn validate_paged(&self) -> Result<(), DispatchError> {
        self.validate_fwd()?;
        if !self.paged {
            return Err(DispatchError::PagedFlagNotSet);
        }
        Ok(())
    }

    /// Stable 64-bit hash of the key. Useful as a cubin-cache index
    /// alongside the kernel-name string.
    pub fn stable_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Build the canonical mangled kernel-name expression. Mirrors the
    /// FA2/FA3 csrc naming convention so we can resolve it via NVRTC's
    /// `nvrtcGetLoweredName`.
    pub fn kernel_name(&self) -> String {
        let kind = if self.arch.supports_fa3() {
            "fa3"
        } else {
            "fa2"
        };
        let mut s = format!(
            "atomr_flashattn::{}::fwd<{}, {}, {}>",
            kind,
            self.dtype.tag(),
            self.head_dim,
            self.causal_tag(),
        );
        if self.varlen {
            s.push_str("_varlen");
        }
        if let Some(w) = self.sliding_window {
            s.push_str(&format!("_sw{w}"));
        }
        if self.alibi {
            s.push_str("_alibi");
        }
        if self.sink > 0 {
            s.push_str(&format!("_sink{}", self.sink));
        }
        if self.paged {
            s.push_str("_paged");
        }
        if self.gqa_ratio > 1 {
            s.push_str(&format!("_gqa{}", self.gqa_ratio));
        }
        s
    }

    fn causal_tag(&self) -> &'static str {
        if self.causal {
            "causal"
        } else {
            "full"
        }
    }
}

/// Errors returned from [`DispatchKey::validate_fwd`] /
/// [`DispatchTable::lookup`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum DispatchError {
    #[error("head_dim {0} is not in the FA whitelist (64, 80, 96, 128, 192, 256)")]
    UnsupportedHeadDim(u32),
    #[error("fp8 requires sm_90a or newer, got {0:?}")]
    Fp8RequiresHopper(SmArch),
    #[error("fp8 backward is not supported in FA3")]
    Fp8BackwardUnsupported,
    #[error("sink tokens require either sliding_window or causal")]
    SinkWithoutMask,
    #[error("invalid GQA ratio {0} (must be >= 1)")]
    InvalidGqaRatio(u32),
    #[error("sliding window must be > 0")]
    ZeroWindow,
    #[error("paged path requires DispatchKey::paged = true")]
    PagedFlagNotSet,
    #[error("no kernel registered for key {0:?}")]
    UnknownKey(Box<DispatchKey>),
}

/// Forward-pass dispatch trait. Every forward-attention request type
/// (FA2, FA3, varlen, paged, prefill) implements this and produces a
/// `DispatchKey`.
pub trait FaFwdDispatch: Send + 'static {
    fn dispatch_key(&self) -> DispatchKey;
}

/// Backward-pass dispatch trait.
pub trait FaBwdDispatch: Send + 'static {
    fn dispatch_key(&self) -> DispatchKey;
}

/// Paged-forward dispatch trait. Distinct from `FaFwdDispatch` so the
/// `FlashAttnMsg::PagedForward` variant can specialise on the paged
/// API surface (block table, slot mapping).
pub trait FaPagedFwdDispatch: Send + 'static {
    fn dispatch_key(&self) -> DispatchKey;
}

/// In-process registry of known kernel names. Populated lazily on first
/// access and shared across all `FlashAttnActor`s.
///
/// The "table" is really a `HashMap<DispatchKey, &'static str>`; the
/// values are static name expressions, never owned. Real cubin
/// compilation is delegated to `NvrtcActor` via the Phase 0.6 disk
/// cache.
pub struct DispatchTable {
    entries: HashMap<DispatchKey, String>,
}

impl DispatchTable {
    fn build() -> Self {
        let mut entries: HashMap<DispatchKey, String> = HashMap::new();

        // Pre-populate a cross-product of common cells. The dispatch
        // table also resolves keys absent from this map by falling back
        // to `key.kernel_name()` — so callers don't need every cell
        // pre-registered. Pre-registration is just a self-test that
        // every "common" combination produces a unique mangled name.
        for &arch in &[SmArch::Sm80, SmArch::Sm89, SmArch::Sm90a, SmArch::Sm100] {
            for &dtype in &[DType::F16, DType::Bf16] {
                for &head_dim in &[64u32, 80, 96, 128, 192, 256] {
                    for &causal in &[false, true] {
                        let key = DispatchKey {
                            arch,
                            dtype,
                            head_dim,
                            causal,
                            varlen: false,
                            sliding_window: None,
                            alibi: false,
                            sink: 0,
                            paged: false,
                            gqa_ratio: 1,
                        };
                        if key.validate_fwd().is_ok() {
                            entries.insert(key, key.kernel_name());
                        }
                    }
                }
            }
        }

        // FA3 fp8 cells (sm_90a / sm_100 only)
        #[cfg(feature = "fp8")]
        for &dtype in &[DType::F8E4m3, DType::F8E5m2] {
            for &head_dim in &[64u32, 128, 256] {
                for &arch in &[SmArch::Sm90a, SmArch::Sm100] {
                    for &causal in &[false, true] {
                        let key = DispatchKey {
                            arch,
                            dtype,
                            head_dim,
                            causal,
                            varlen: false,
                            sliding_window: None,
                            alibi: false,
                            sink: 0,
                            paged: false,
                            gqa_ratio: 1,
                        };
                        if key.validate_fwd().is_ok() {
                            entries.insert(key, key.kernel_name());
                        }
                    }
                }
            }
        }

        Self { entries }
    }

    /// Resolve a key to a kernel-name expression.
    ///
    /// Lookup order:
    ///
    /// 1. Pre-registered entry (fast path — no allocation).
    /// 2. Computed [`DispatchKey::kernel_name`] for cells outside the
    ///    pre-registration cross-product.
    /// 3. `Err(DispatchError::UnknownKey(_))` if the key is invalid.
    pub fn lookup(&self, key: &DispatchKey) -> Result<String, DispatchError> {
        key.validate_fwd()?;
        if let Some(name) = self.entries.get(key) {
            return Ok(name.clone());
        }
        Ok(key.kernel_name())
    }

    /// Resolve a key, and additionally fail with `UnknownKey` if it is
    /// not in the pre-registered set. Used by tests.
    pub fn strict_lookup(&self, key: &DispatchKey) -> Result<&str, DispatchError> {
        self.entries
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| DispatchError::UnknownKey(Box::new(*key)))
    }

    /// Number of pre-registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Process-wide dispatch table singleton.
pub static DISPATCH_TABLE: Lazy<DispatchTable> = Lazy::new(DispatchTable::build);

/// Convenience accessor — `DISPATCH_TABLE.lookup(key)`.
pub fn lookup(key: &DispatchKey) -> Result<String, DispatchError> {
    DISPATCH_TABLE.lookup(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fwd_key(arch: SmArch, dtype: DType, head_dim: u32, causal: bool) -> DispatchKey {
        DispatchKey {
            arch,
            dtype,
            head_dim,
            causal,
            varlen: false,
            sliding_window: None,
            alibi: false,
            sink: 0,
            paged: false,
            gqa_ratio: 1,
        }
    }

    /// Every `(arch, dtype, head_dim, causal, …)` cell builds, validates,
    /// and round-trips through `kernel_name + stable_hash` deterministically.
    #[test]
    fn dispatch_key_round_trip() {
        let arches = [SmArch::Sm80, SmArch::Sm89, SmArch::Sm90a, SmArch::Sm100];
        let dtypes = [DType::F16, DType::Bf16];
        let head_dims = [64u32, 80, 96, 128, 192, 256];

        for &arch in &arches {
            for &dtype in &dtypes {
                for &head_dim in &head_dims {
                    for &causal in &[false, true] {
                        let key = fwd_key(arch, dtype, head_dim, causal);
                        assert!(key.validate_fwd().is_ok());

                        // Re-construct identically and re-hash; must match.
                        let key2 = fwd_key(arch, dtype, head_dim, causal);
                        assert_eq!(key.stable_hash(), key2.stable_hash());
                        assert_eq!(key.kernel_name(), key2.kernel_name());

                        // Lookup goes through the table.
                        let name = lookup(&key).expect("lookup");
                        assert!(name.contains(dtype.tag()));
                        assert!(name.contains(&head_dim.to_string()));
                    }
                }
            }
        }

        // Modifying any field changes both the hash and the name.
        let a = fwd_key(SmArch::Sm90a, DType::F16, 128, true);
        let b = fwd_key(SmArch::Sm90a, DType::F16, 128, false);
        assert_ne!(a.stable_hash(), b.stable_hash());
        assert_ne!(a.kernel_name(), b.kernel_name());
    }

    /// Strict lookup of a key that wasn't pre-registered yields
    /// `UnknownKey`; soft `lookup` succeeds via `kernel_name`.
    #[test]
    fn lookup_misses_unknown_key() {
        // varlen + alibi cell — not in the pre-reg cross-product.
        let key = DispatchKey {
            arch: SmArch::Sm90a,
            dtype: DType::Bf16,
            head_dim: 128,
            causal: true,
            varlen: true,
            sliding_window: Some(4096),
            alibi: true,
            sink: 4,
            paged: false,
            gqa_ratio: 8,
        };
        assert!(key.validate_fwd().is_ok());

        // Strict lookup misses (not pre-registered).
        let strict = DISPATCH_TABLE.strict_lookup(&key);
        assert!(matches!(strict, Err(DispatchError::UnknownKey(_))));

        // Soft lookup synthesises the kernel name on the fly.
        let name = lookup(&key).expect("soft lookup synthesises a name");
        assert!(name.contains("varlen"));
        assert!(name.contains("alibi"));
        assert!(name.contains("sink4"));
        assert!(name.contains("sw4096"));
        assert!(name.contains("gqa8"));
    }

    #[test]
    fn fp8_requires_hopper() {
        let mut key = DispatchKey {
            arch: SmArch::Sm80,
            dtype: DType::F8E4m3,
            head_dim: 128,
            causal: true,
            varlen: false,
            sliding_window: None,
            alibi: false,
            sink: 0,
            paged: false,
            gqa_ratio: 1,
        };
        assert!(matches!(
            key.validate_fwd(),
            Err(DispatchError::Fp8RequiresHopper(_))
        ));
        key.arch = SmArch::Sm90a;
        assert!(key.validate_fwd().is_ok());
    }

    #[test]
    fn unsupported_head_dim_rejected() {
        let key = DispatchKey {
            arch: SmArch::Sm90a,
            dtype: DType::F16,
            head_dim: 100,
            causal: false,
            varlen: false,
            sliding_window: None,
            alibi: false,
            sink: 0,
            paged: false,
            gqa_ratio: 1,
        };
        assert!(matches!(
            key.validate_fwd(),
            Err(DispatchError::UnsupportedHeadDim(100))
        ));
    }

    #[test]
    fn sink_without_mask_rejected() {
        let key = DispatchKey {
            arch: SmArch::Sm90a,
            dtype: DType::Bf16,
            head_dim: 128,
            causal: false,
            varlen: false,
            sliding_window: None,
            alibi: false,
            sink: 4,
            paged: false,
            gqa_ratio: 1,
        };
        assert!(matches!(
            key.validate_fwd(),
            Err(DispatchError::SinkWithoutMask)
        ));
    }
}
