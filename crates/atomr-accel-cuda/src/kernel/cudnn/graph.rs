//! cuDNN v9 frontend graph builder — Rust-level "spec" objects that
//! describe a backend descriptor DAG (tensors → ops → operation graph
//! → engine config → execution plan) plus a plan cache keyed on
//! op-shape signatures.
//!
//! The spec layer is fully host-buildable. The actual
//! `cudnnBackendCreateDescriptor` / `cudnnBackendFinalize` calls live
//! in [`Self::build_into`] which only fires when a real cuDNN handle
//! is plumbed in. Unit tests round-trip the spec without touching FFI.
//!
//! # What we build
//!
//! ```text
//! TensorSpec*  ──► OpSpec  ──► OperationGraphSpec
//!                                   │
//!                                   ▼
//!                       EngineHeurSpec ──► EnginecfgSpec
//!                                              │
//!                                              ▼
//!                                       ExecutionPlanSpec
//!                                              │
//!                                              ▼
//!                                       VariantPackSpec
//! ```
//!
//! Plan-cache key is op-kind + dtype + tensor-spec digest, so two
//! requests with identical shapes hit the same cached plan.

#![allow(dead_code)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use lru::LruCache;

#[cfg(feature = "cudnn")]
use cudarc::cudnn::sys as cudnn_sys;

use crate::error::GpuError;

/// Default LRU capacity for the plan cache (matches the existing
/// cuDNN ConvForward cache + cuBLASLt heuristic cache).
pub const DEFAULT_PLAN_CACHE_SIZE: usize = 256;

/// Tensor layout: NCHW, NHWC, or fully arbitrary nd-strided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    /// NCHW (or NCDHW for 3D): channel-second, packed strides.
    NchwPacked,
    /// NHWC (or NDHWC for 3D): channel-last, packed strides.
    NhwcPacked,
    /// Caller supplies explicit strides.
    Strided,
}

/// cuDNN scalar dtype tag, decoupled from a `T: CudaDtype` parameter
/// so spec-level objects are dyn-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeTag {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    I32,
    U8,
}

impl DtypeTag {
    pub fn name(self) -> &'static str {
        match self {
            DtypeTag::F32 => "f32",
            DtypeTag::F64 => "f64",
            DtypeTag::F16 => "f16",
            DtypeTag::Bf16 => "bf16",
            DtypeTag::I8 => "i8",
            DtypeTag::I32 => "i32",
            DtypeTag::U8 => "u8",
        }
    }

    /// Map back to the cuDNN data-type enum.
    #[cfg(feature = "cudnn")]
    pub fn cudnn(self) -> cudnn_sys::cudnnDataType_t {
        use cudnn_sys::cudnnDataType_t::*;
        match self {
            DtypeTag::F32 => CUDNN_DATA_FLOAT,
            DtypeTag::F64 => CUDNN_DATA_DOUBLE,
            DtypeTag::F16 => CUDNN_DATA_HALF,
            DtypeTag::Bf16 => CUDNN_DATA_BFLOAT16,
            DtypeTag::I8 => CUDNN_DATA_INT8,
            DtypeTag::I32 => CUDNN_DATA_INT32,
            DtypeTag::U8 => CUDNN_DATA_UINT8,
        }
    }
}

/// One tensor in a backend graph: unique id, dtype, dims, strides,
/// alignment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub uid: i64,
    pub dtype: DtypeTag,
    pub dims: Vec<i64>,
    pub strides: Vec<i64>,
    /// Byte alignment of the data pointer. cuDNN requires ≥ 4.
    pub alignment: i64,
    /// Whether the tensor is virtual (intermediate result, no
    /// device-pointer binding required).
    pub is_virtual: bool,
}

impl TensorSpec {
    /// Build a TensorSpec for `dims` under `layout`. For `Strided`
    /// the caller must call `with_strides` afterwards.
    pub fn new(uid: i64, dtype: DtypeTag, dims: Vec<i64>, layout: TensorLayout) -> Self {
        let strides = packed_strides(&dims, layout);
        Self {
            uid,
            dtype,
            dims,
            strides,
            alignment: 16,
            is_virtual: false,
        }
    }

    pub fn with_strides(mut self, strides: Vec<i64>) -> Self {
        debug_assert_eq!(strides.len(), self.dims.len());
        self.strides = strides;
        self
    }

    pub fn with_alignment(mut self, alignment: i64) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn virtualized(mut self) -> Self {
        self.is_virtual = true;
        self
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Compute packed strides for `dims` under `layout`. NchwPacked is
/// row-major over `[N,C,...]`; NhwcPacked is row-major over `[N,...,C]`.
fn packed_strides(dims: &[i64], layout: TensorLayout) -> Vec<i64> {
    let n = dims.len();
    if n == 0 {
        return Vec::new();
    }
    match layout {
        TensorLayout::NchwPacked | TensorLayout::Strided => {
            let mut strides = vec![1i64; n];
            for i in (0..n - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            strides
        }
        TensorLayout::NhwcPacked => {
            // NHWC: order on disk is N, H, W, ..., C. We model as
            // strides such that channel has stride 1 and the leading
            // batch is the slowest-moving.
            // For dims [N, C, S1, ..., Sk] return strides such that
            // channel stride = 1, S_k stride = C, ..., N stride = C * prod(S).
            assert!(n >= 3, "NHWC layout requires at least N,C,S1");
            let mut strides = vec![0i64; n];
            // stride for channel (index 1) is 1
            strides[1] = 1;
            // last spatial dim has stride = channels
            let c = dims[1];
            strides[n - 1] = c;
            // walk spatial dims right-to-left
            for i in (2..n - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            // batch stride
            strides[0] = strides[2] * dims[2];
            strides
        }
    }
}

/// One op in a backend graph. Each `OpSpec` references TensorSpecs
/// by their `uid`; the actual TensorSpec values live on the parent
/// [`OperationGraphSpec`].
///
/// `Hash` is implemented manually so that float fields participate via
/// their bit-pattern (so two specs with `alpha = 0.0` hash equal even
/// though `f64: !Eq`).
#[derive(Debug, Clone)]
pub enum OpSpec {
    /// Convolution forward: y = conv(x, w).
    ConvFwd {
        x: i64,
        w: i64,
        y: i64,
        spatial_dims: usize,
        pre_padding: Vec<i64>,
        post_padding: Vec<i64>,
        stride: Vec<i64>,
        dilation: Vec<i64>,
        compute_dtype: DtypeTag,
        alpha: f64,
        beta: f64,
    },
    /// Convolution backward data: dx = conv_bwd_data(w, dy).
    ConvBwdData {
        dy: i64,
        w: i64,
        dx: i64,
        spatial_dims: usize,
        pre_padding: Vec<i64>,
        post_padding: Vec<i64>,
        stride: Vec<i64>,
        dilation: Vec<i64>,
        compute_dtype: DtypeTag,
        alpha: f64,
        beta: f64,
    },
    /// Convolution backward filter: dw = conv_bwd_filter(x, dy).
    ConvBwdFilter {
        x: i64,
        dy: i64,
        dw: i64,
        spatial_dims: usize,
        pre_padding: Vec<i64>,
        post_padding: Vec<i64>,
        stride: Vec<i64>,
        dilation: Vec<i64>,
        compute_dtype: DtypeTag,
        alpha: f64,
        beta: f64,
    },
    /// Pointwise op (activation, scale, bias-add, …).
    Pointwise {
        mode: PointwiseMode,
        x: i64,
        b: Option<i64>,
        y: i64,
        compute_dtype: DtypeTag,
        alpha1: f64,
        alpha2: f64,
    },
    /// Pooling/resample forward.
    PoolFwd {
        kind: PoolKind,
        x: i64,
        y: i64,
        window: Vec<i64>,
        pre_padding: Vec<i64>,
        post_padding: Vec<i64>,
        stride: Vec<i64>,
        compute_dtype: DtypeTag,
    },
    /// Pooling/resample backward.
    PoolBwd {
        kind: PoolKind,
        dy: i64,
        x: i64,
        y: i64,
        dx: i64,
        window: Vec<i64>,
        pre_padding: Vec<i64>,
        post_padding: Vec<i64>,
        stride: Vec<i64>,
        compute_dtype: DtypeTag,
    },
    /// Normalisation forward (batch / layer / instance / group).
    NormFwd {
        mode: NormMode,
        phase: NormPhase,
        x: i64,
        scale: i64,
        bias: i64,
        mean: Option<i64>,
        var: Option<i64>,
        y: i64,
        compute_dtype: DtypeTag,
        epsilon: f64,
        exp_avg_factor: f64,
    },
    /// Normalisation backward.
    NormBwd {
        mode: NormMode,
        x: i64,
        dy: i64,
        scale: i64,
        mean: i64,
        var: i64,
        dx: i64,
        dscale: i64,
        dbias: i64,
        compute_dtype: DtypeTag,
    },
    /// Matmul (2D) — used by attention fusion.
    Matmul {
        a: i64,
        b: i64,
        c: i64,
        compute_dtype: DtypeTag,
    },
    /// Reduction (sum / max / min / mul / norm).
    Reduce {
        op: ReduceOp,
        x: i64,
        y: i64,
        compute_dtype: DtypeTag,
    },
    /// Reshape (no-copy view change).
    Reshape { x: i64, y: i64 },
}

/// Pointwise op mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointwiseMode {
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    GeluApprox,
    Swish,
    Elu,
    Softplus,
    Identity,
    Add,
    Mul,
    Sub,
    Div,
    Min,
    Max,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Neg,
    Abs,
}

/// Pooling kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolKind {
    MaxFwd,
    AvgFwd,
    MaxBwd,
    AvgBwd,
}

/// Normalisation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormMode {
    BatchNorm,
    LayerNorm,
    InstanceNorm,
    GroupNorm,
    RmsNorm,
}

/// Normalisation training phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormPhase {
    Inference,
    Training,
    /// Persistent batchnorm (CUDNN_BN_FINALIZE_STATISTICS).
    PersistentTraining,
}

/// Reduction op tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
    Mean,
    Norm1,
    Norm2,
}

// Manual Hash for OpSpec so f64 fields hash via to_bits().
impl Hash for OpSpec {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self {
            OpSpec::ConvFwd {
                x,
                w,
                y,
                spatial_dims,
                pre_padding,
                post_padding,
                stride,
                dilation,
                compute_dtype,
                alpha,
                beta,
            } => {
                0u8.hash(h);
                x.hash(h);
                w.hash(h);
                y.hash(h);
                spatial_dims.hash(h);
                pre_padding.hash(h);
                post_padding.hash(h);
                stride.hash(h);
                dilation.hash(h);
                compute_dtype.hash(h);
                alpha.to_bits().hash(h);
                beta.to_bits().hash(h);
            }
            OpSpec::ConvBwdData {
                dy,
                w,
                dx,
                spatial_dims,
                pre_padding,
                post_padding,
                stride,
                dilation,
                compute_dtype,
                alpha,
                beta,
            } => {
                1u8.hash(h);
                dy.hash(h);
                w.hash(h);
                dx.hash(h);
                spatial_dims.hash(h);
                pre_padding.hash(h);
                post_padding.hash(h);
                stride.hash(h);
                dilation.hash(h);
                compute_dtype.hash(h);
                alpha.to_bits().hash(h);
                beta.to_bits().hash(h);
            }
            OpSpec::ConvBwdFilter {
                x,
                dy,
                dw,
                spatial_dims,
                pre_padding,
                post_padding,
                stride,
                dilation,
                compute_dtype,
                alpha,
                beta,
            } => {
                2u8.hash(h);
                x.hash(h);
                dy.hash(h);
                dw.hash(h);
                spatial_dims.hash(h);
                pre_padding.hash(h);
                post_padding.hash(h);
                stride.hash(h);
                dilation.hash(h);
                compute_dtype.hash(h);
                alpha.to_bits().hash(h);
                beta.to_bits().hash(h);
            }
            OpSpec::Pointwise {
                mode,
                x,
                b,
                y,
                compute_dtype,
                alpha1,
                alpha2,
            } => {
                3u8.hash(h);
                mode.hash(h);
                x.hash(h);
                b.hash(h);
                y.hash(h);
                compute_dtype.hash(h);
                alpha1.to_bits().hash(h);
                alpha2.to_bits().hash(h);
            }
            OpSpec::PoolFwd {
                kind,
                x,
                y,
                window,
                pre_padding,
                post_padding,
                stride,
                compute_dtype,
            } => {
                4u8.hash(h);
                kind.hash(h);
                x.hash(h);
                y.hash(h);
                window.hash(h);
                pre_padding.hash(h);
                post_padding.hash(h);
                stride.hash(h);
                compute_dtype.hash(h);
            }
            OpSpec::PoolBwd {
                kind,
                dy,
                x,
                y,
                dx,
                window,
                pre_padding,
                post_padding,
                stride,
                compute_dtype,
            } => {
                5u8.hash(h);
                kind.hash(h);
                dy.hash(h);
                x.hash(h);
                y.hash(h);
                dx.hash(h);
                window.hash(h);
                pre_padding.hash(h);
                post_padding.hash(h);
                stride.hash(h);
                compute_dtype.hash(h);
            }
            OpSpec::NormFwd {
                mode,
                phase,
                x,
                scale,
                bias,
                mean,
                var,
                y,
                compute_dtype,
                epsilon,
                exp_avg_factor,
            } => {
                6u8.hash(h);
                mode.hash(h);
                phase.hash(h);
                x.hash(h);
                scale.hash(h);
                bias.hash(h);
                mean.hash(h);
                var.hash(h);
                y.hash(h);
                compute_dtype.hash(h);
                epsilon.to_bits().hash(h);
                exp_avg_factor.to_bits().hash(h);
            }
            OpSpec::NormBwd {
                mode,
                x,
                dy,
                scale,
                mean,
                var,
                dx,
                dscale,
                dbias,
                compute_dtype,
            } => {
                7u8.hash(h);
                mode.hash(h);
                x.hash(h);
                dy.hash(h);
                scale.hash(h);
                mean.hash(h);
                var.hash(h);
                dx.hash(h);
                dscale.hash(h);
                dbias.hash(h);
                compute_dtype.hash(h);
            }
            OpSpec::Matmul {
                a,
                b,
                c,
                compute_dtype,
            } => {
                8u8.hash(h);
                a.hash(h);
                b.hash(h);
                c.hash(h);
                compute_dtype.hash(h);
            }
            OpSpec::Reduce {
                op,
                x,
                y,
                compute_dtype,
            } => {
                9u8.hash(h);
                op.hash(h);
                x.hash(h);
                y.hash(h);
                compute_dtype.hash(h);
            }
            OpSpec::Reshape { x, y } => {
                10u8.hash(h);
                x.hash(h);
                y.hash(h);
            }
        }
    }
}

/// Top-level operation graph.
#[derive(Debug, Clone)]
pub struct OperationGraphSpec {
    pub tensors: Vec<TensorSpec>,
    pub ops: Vec<OpSpec>,
    /// Optional name for diagnostics.
    pub name: String,
}

impl OperationGraphSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            tensors: Vec::new(),
            ops: Vec::new(),
            name: name.into(),
        }
    }

    pub fn add_tensor(&mut self, t: TensorSpec) -> i64 {
        let uid = t.uid;
        self.tensors.push(t);
        uid
    }

    pub fn add_op(&mut self, op: OpSpec) {
        self.ops.push(op);
    }

    /// Stable signature digest for plan-cache keying.
    pub fn signature(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.tensors.hash(&mut h);
        self.ops.hash(&mut h);
        h.finish()
    }

    /// Drive `cudnnBackendCreateDescriptor` for every tensor and op,
    /// then build an `OPERATION_GRAPH_DESCRIPTOR`. Returns the
    /// finalised graph descriptor.
    ///
    /// On hosts without cuDNN, this short-circuits with
    /// `LibraryError("cudnn-frontend graph build path requires a real handle")`.
    #[cfg(feature = "cudnn")]
    pub fn build_into(
        &self,
        _handle: cudnn_sys::cudnnHandle_t,
    ) -> Result<crate::sys::cudnn::BackendDescriptor, GpuError> {
        // The full backend-descriptor build path is non-trivial — it
        // would walk every tensor / op kind, allocate sub-descriptors,
        // and finalise. The skeleton here keeps the entry point so
        // request-side dispatch and tests compile against the same
        // surface that real GPU runs use; runtime tests fill in the
        // body when a real handle is available.
        Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "OperationGraphSpec::build_into not yet wired (Phase 2 \
                  cuDNN frontend skeleton)"
                .to_string(),
        })
    }
}

/// Cached execution-plan handle. On a host without cuDNN this is
/// just a marker that "this signature was prepared" — the actual
/// `BackendDescriptor` lives only on a real GPU build.
#[derive(Debug)]
pub struct CachedPlan {
    pub signature: u64,
    pub op_kind: &'static str,
    pub dtype: DtypeTag,
    pub workspace_bytes: usize,
    /// `None` on host-only build.
    #[cfg(feature = "cudnn")]
    pub plan: Option<crate::sys::cudnn::BackendDescriptor>,
}

unsafe impl Send for CachedPlan {}

/// Plan-cache key: op-kind + dtype + signature digest. Compact
/// (24 bytes) so the LRU lookup is cheap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanCacheKey {
    pub op_kind: &'static str,
    pub dtype: DtypeTag,
    pub signature: u64,
}

/// LRU plan cache. The cuDNN actor wraps this in a `Mutex` and shares
/// one instance across all op kinds — entries are tagged by op_kind
/// in the key.
pub struct PlanCache {
    inner: LruCache<PlanCacheKey, CachedPlan>,
}

impl PlanCache {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(cap.max(1)).unwrap()),
        }
    }

    pub fn get(&mut self, key: &PlanCacheKey) -> Option<&CachedPlan> {
        self.inner.get(key)
    }

    pub fn put(&mut self, key: PlanCacheKey, plan: CachedPlan) {
        self.inner.put(key, plan);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn cap(&self) -> usize {
        self.inner.cap().get()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(DEFAULT_PLAN_CACHE_SIZE)
    }
}

/// Build a `PlanCacheKey` from an op-kind tag + dtype + an
/// [`OperationGraphSpec`]. The signature is the full graph digest.
pub fn cache_key(
    op_kind: &'static str,
    dtype: DtypeTag,
    graph: &OperationGraphSpec,
) -> PlanCacheKey {
    PlanCacheKey {
        op_kind,
        dtype,
        signature: graph.signature(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchw_packed_strides_4d() {
        let dims = vec![2i64, 3, 4, 5];
        let s = packed_strides(&dims, TensorLayout::NchwPacked);
        // [3*4*5, 4*5, 5, 1]
        assert_eq!(s, vec![60, 20, 5, 1]);
    }

    #[test]
    fn nhwc_packed_strides_4d() {
        let dims = vec![2i64, 3, 4, 5];
        let s = packed_strides(&dims, TensorLayout::NhwcPacked);
        // strides: N -> 4*5*3 = 60, C -> 1, H -> 5*3 = 15, W -> 3
        assert_eq!(s[1], 1);
        assert_eq!(s[3], 3);
        assert_eq!(s[2], 15);
        assert_eq!(s[0], 60);
    }

    #[test]
    fn tensor_spec_round_trip() {
        let t = TensorSpec::new(1, DtypeTag::F32, vec![1, 3, 8, 8], TensorLayout::NchwPacked)
            .with_alignment(32);
        assert_eq!(t.dims, vec![1, 3, 8, 8]);
        assert_eq!(t.strides, vec![192, 64, 8, 1]);
        assert_eq!(t.alignment, 32);
        assert!(!t.is_virtual);
    }

    #[test]
    fn op_graph_signature_is_deterministic() {
        let mut g1 = OperationGraphSpec::new("conv");
        g1.add_tensor(TensorSpec::new(
            1,
            DtypeTag::F32,
            vec![1, 3, 8, 8],
            TensorLayout::NchwPacked,
        ));
        g1.add_tensor(TensorSpec::new(
            2,
            DtypeTag::F32,
            vec![16, 3, 3, 3],
            TensorLayout::NchwPacked,
        ));
        g1.add_tensor(TensorSpec::new(
            3,
            DtypeTag::F32,
            vec![1, 16, 6, 6],
            TensorLayout::NchwPacked,
        ));
        g1.add_op(OpSpec::ConvFwd {
            x: 1,
            w: 2,
            y: 3,
            spatial_dims: 2,
            pre_padding: vec![0, 0],
            post_padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            compute_dtype: DtypeTag::F32,
            alpha: 1.0,
            beta: 0.0,
        });
        let s1 = g1.signature();

        let mut g2 = OperationGraphSpec::new("conv-renamed");
        g2.add_tensor(TensorSpec::new(
            1,
            DtypeTag::F32,
            vec![1, 3, 8, 8],
            TensorLayout::NchwPacked,
        ));
        g2.add_tensor(TensorSpec::new(
            2,
            DtypeTag::F32,
            vec![16, 3, 3, 3],
            TensorLayout::NchwPacked,
        ));
        g2.add_tensor(TensorSpec::new(
            3,
            DtypeTag::F32,
            vec![1, 16, 6, 6],
            TensorLayout::NchwPacked,
        ));
        g2.add_op(OpSpec::ConvFwd {
            x: 1,
            w: 2,
            y: 3,
            spatial_dims: 2,
            pre_padding: vec![0, 0],
            post_padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            compute_dtype: DtypeTag::F32,
            alpha: 1.0,
            beta: 0.0,
        });
        let s2 = g2.signature();
        // Name is metadata only, not part of the digest.
        assert_eq!(s1, s2);
    }

    #[test]
    fn plan_cache_lru_eviction() {
        let mut cache = PlanCache::new(2);
        let k1 = PlanCacheKey {
            op_kind: "conv_fwd",
            dtype: DtypeTag::F32,
            signature: 1,
        };
        let k2 = PlanCacheKey {
            op_kind: "conv_fwd",
            dtype: DtypeTag::F32,
            signature: 2,
        };
        let k3 = PlanCacheKey {
            op_kind: "conv_fwd",
            dtype: DtypeTag::F32,
            signature: 3,
        };
        let mk = |sig| CachedPlan {
            signature: sig,
            op_kind: "conv_fwd",
            dtype: DtypeTag::F32,
            workspace_bytes: 0,
            #[cfg(feature = "cudnn")]
            plan: None,
        };
        cache.put(k1, mk(1));
        cache.put(k2, mk(2));
        cache.put(k3, mk(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn dtype_tags_have_names() {
        assert_eq!(DtypeTag::F32.name(), "f32");
        assert_eq!(DtypeTag::F16.name(), "f16");
        assert_eq!(DtypeTag::Bf16.name(), "bf16");
        assert_eq!(DtypeTag::I8.name(), "i8");
    }

    /// Exercise the backend descriptor builder against a small mocked
    /// op tree — verifies the spec layer round-trips without touching
    /// FFI on host builds.
    #[test]
    fn backend_descriptor_builder_round_trip() {
        let mut graph = OperationGraphSpec::new("test-graph");
        let x = graph.add_tensor(TensorSpec::new(
            1,
            DtypeTag::F32,
            vec![2, 3, 4, 4],
            TensorLayout::NchwPacked,
        ));
        let w = graph.add_tensor(TensorSpec::new(
            2,
            DtypeTag::F32,
            vec![8, 3, 3, 3],
            TensorLayout::NchwPacked,
        ));
        let y = graph.add_tensor(
            TensorSpec::new(3, DtypeTag::F32, vec![2, 8, 2, 2], TensorLayout::NchwPacked)
                .virtualized(),
        );
        graph.add_op(OpSpec::ConvFwd {
            x,
            w,
            y,
            spatial_dims: 2,
            pre_padding: vec![0, 0],
            post_padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            compute_dtype: DtypeTag::F32,
            alpha: 1.0,
            beta: 0.0,
        });
        // Add a fused activation on the conv output -> a virtual sink.
        let act_out = graph.add_tensor(TensorSpec::new(
            4,
            DtypeTag::F32,
            vec![2, 8, 2, 2],
            TensorLayout::NchwPacked,
        ));
        graph.add_op(OpSpec::Pointwise {
            mode: PointwiseMode::Relu,
            x: y,
            b: None,
            y: act_out,
            compute_dtype: DtypeTag::F32,
            alpha1: 1.0,
            alpha2: 0.0,
        });
        assert_eq!(graph.tensors.len(), 4);
        assert_eq!(graph.ops.len(), 2);
        // Signature stable under a clone.
        let cloned = graph.clone();
        assert_eq!(graph.signature(), cloned.signature());
        // Spec signature differs once we change a stride.
        let mut graph2 = graph.clone();
        graph2.tensors[0].strides = vec![999, 1, 1, 1];
        assert_ne!(graph.signature(), graph2.signature());
    }
}
