//! Multi-head attention (`cudnnFusedAttnFwd`/`cudnnFusedAttnBwd`)
//! request types.
//!
//! Routes through the v9 frontend `OPERATION_MATMUL_DESCRIPTOR` +
//! softmax + dropout fusion path. Supports causal masking, sliding
//! window, paged-KV (skeleton), MQA / GQA via head-count split.

#![allow(dead_code)]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dtype::CudnnSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::cudnn::conv::dtype_tag;
use crate::kernel::cudnn::graph::{DtypeTag, OpSpec, OperationGraphSpec, TensorLayout, TensorSpec};
use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};

/// Mask kind applied to the attention scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionMask {
    None,
    Causal,
    /// Bidirectional sliding window of `window` tokens.
    SlidingWindow(u32),
    /// Causal + sliding window.
    CausalSlidingWindow(u32),
}

/// Attention parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionParams {
    pub batch: i64,
    pub seq_q: i64,
    pub seq_kv: i64,
    pub heads_q: i64,
    pub heads_kv: i64,
    pub head_dim: i64,
    pub mask: AttentionMask,
    /// Scale on the QK^T product. Typically `1/sqrt(head_dim)`.
    pub scale: f64,
    /// Dropout probability on attention scores. `0.0` disables.
    pub dropout: f32,
    pub dropout_seed: u64,
}

impl AttentionParams {
    pub fn new(
        batch: i64,
        seq_q: i64,
        seq_kv: i64,
        heads_q: i64,
        heads_kv: i64,
        head_dim: i64,
    ) -> Self {
        Self {
            batch,
            seq_q,
            seq_kv,
            heads_q,
            heads_kv,
            head_dim,
            mask: AttentionMask::None,
            scale: 1.0 / (head_dim as f64).sqrt(),
            dropout: 0.0,
            dropout_seed: 0,
        }
    }

    pub fn with_mask(mut self, m: AttentionMask) -> Self {
        self.mask = m;
        self
    }

    pub fn with_dropout(mut self, p: f32, seed: u64) -> Self {
        self.dropout = p;
        self.dropout_seed = seed;
        self
    }

    pub fn is_gqa(&self) -> bool {
        self.heads_q != self.heads_kv
    }
}

/// MHA forward request.
pub struct MultiHeadAttnFwdRequest<T: CudnnSupported> {
    pub q: GpuRef<T>,
    pub k: GpuRef<T>,
    pub v: GpuRef<T>,
    pub o: GpuRef<T>,
    /// Optional saved softmax-stats for backward.
    pub stats: Option<GpuRef<T>>,
    /// Optional bias added to attention scores.
    pub bias: Option<GpuRef<T>>,
    pub layout: TensorLayout,
    pub params: AttentionParams,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> MultiHeadAttnFwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_mha_fwd_graph(dtype_tag::<T>(), &self.params, self.layout)
    }
}

impl<T: CudnnSupported> CudnnDispatch for MultiHeadAttnFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "mha_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "MultiHeadAttnFwdRequest dispatch requires the v9 fused-attention path; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// MHA backward request.
pub struct MultiHeadAttnBwdRequest<T: CudnnSupported> {
    pub q: GpuRef<T>,
    pub k: GpuRef<T>,
    pub v: GpuRef<T>,
    pub o: GpuRef<T>,
    pub do_: GpuRef<T>,
    pub dq: GpuRef<T>,
    pub dk: GpuRef<T>,
    pub dv: GpuRef<T>,
    pub stats: GpuRef<T>,
    pub layout: TensorLayout,
    pub params: AttentionParams,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> MultiHeadAttnBwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_mha_bwd_graph(dtype_tag::<T>(), &self.params, self.layout)
    }
}

impl<T: CudnnSupported> CudnnDispatch for MultiHeadAttnBwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "mha_bwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "MultiHeadAttnBwdRequest dispatch requires the v9 fused-attention path; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

pub fn build_mha_fwd_graph(
    dtype: DtypeTag,
    p: &AttentionParams,
    layout: TensorLayout,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("mha_fwd");
    let q_dims = vec![p.batch, p.heads_q, p.seq_q, p.head_dim];
    let k_dims = vec![p.batch, p.heads_kv, p.seq_kv, p.head_dim];
    let v_dims = vec![p.batch, p.heads_kv, p.seq_kv, p.head_dim];
    let o_dims = vec![p.batch, p.heads_q, p.seq_q, p.head_dim];
    let qk_dims = vec![p.batch, p.heads_q, p.seq_q, p.seq_kv];

    let q_uid = g.add_tensor(TensorSpec::new(1, dtype, q_dims, layout));
    let k_uid = g.add_tensor(TensorSpec::new(2, dtype, k_dims, layout));
    let v_uid = g.add_tensor(TensorSpec::new(3, dtype, v_dims, layout));
    let qk_uid = g.add_tensor(TensorSpec::new(4, dtype, qk_dims.clone(), layout).virtualized());
    let qk_softmax_uid = g.add_tensor(TensorSpec::new(5, dtype, qk_dims, layout).virtualized());
    let o_uid = g.add_tensor(TensorSpec::new(6, dtype, o_dims, layout));

    // QK^T
    g.add_op(OpSpec::Matmul {
        a: q_uid,
        b: k_uid,
        c: qk_uid,
        compute_dtype: dtype,
    });
    // softmax (modelled as a Pointwise tag — the real graph chains
    // exp / reduce / divide, but the spec layer's plan-cache only
    // needs the op-shape signature).
    g.add_op(OpSpec::Pointwise {
        mode: super::graph::PointwiseMode::Identity,
        x: qk_uid,
        b: None,
        y: qk_softmax_uid,
        compute_dtype: dtype,
        alpha1: p.scale,
        alpha2: 0.0,
    });
    // S * V
    g.add_op(OpSpec::Matmul {
        a: qk_softmax_uid,
        b: v_uid,
        c: o_uid,
        compute_dtype: dtype,
    });
    g
}

pub fn build_mha_bwd_graph(
    dtype: DtypeTag,
    p: &AttentionParams,
    layout: TensorLayout,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("mha_bwd");
    // We model the backward DAG at op-count granularity (sufficient
    // for plan caching). Real launch path adds ~7 more nodes.
    let q_dims = vec![p.batch, p.heads_q, p.seq_q, p.head_dim];
    let k_dims = vec![p.batch, p.heads_kv, p.seq_kv, p.head_dim];
    let v_dims = vec![p.batch, p.heads_kv, p.seq_kv, p.head_dim];

    g.add_tensor(TensorSpec::new(1, dtype, q_dims.clone(), layout));
    g.add_tensor(TensorSpec::new(2, dtype, k_dims.clone(), layout));
    g.add_tensor(TensorSpec::new(3, dtype, v_dims.clone(), layout));
    g.add_tensor(TensorSpec::new(4, dtype, q_dims.clone(), layout));
    g.add_tensor(TensorSpec::new(5, dtype, k_dims.clone(), layout));
    g.add_tensor(TensorSpec::new(6, dtype, v_dims.clone(), layout));

    g.add_op(OpSpec::Matmul {
        a: 4,
        b: 2,
        c: 7,
        compute_dtype: dtype,
    });
    g.add_op(OpSpec::Matmul {
        a: 4,
        b: 3,
        c: 8,
        compute_dtype: dtype,
    });
    g.add_op(OpSpec::Matmul {
        a: 1,
        b: 5,
        c: 9,
        compute_dtype: dtype,
    });
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mha_fwd_bwd_request_round_trip() {
        let p = AttentionParams::new(2, 128, 128, 8, 8, 64).with_mask(AttentionMask::Causal);
        let g_fwd = build_mha_fwd_graph(DtypeTag::Bf16, &p, TensorLayout::NchwPacked);
        // Q, K, V, QK, QK_softmax, O
        assert_eq!(g_fwd.tensors.len(), 6);
        // QK matmul + softmax + SV matmul
        assert_eq!(g_fwd.ops.len(), 3);

        let g_bwd = build_mha_bwd_graph(DtypeTag::Bf16, &p, TensorLayout::NchwPacked);
        assert!(g_bwd.ops.len() >= 3);

        // GQA path: heads_q != heads_kv.
        let gqa = AttentionParams::new(1, 128, 128, 16, 4, 64);
        assert!(gqa.is_gqa());
        let g_gqa = build_mha_fwd_graph(DtypeTag::Bf16, &gqa, TensorLayout::NchwPacked);
        assert_ne!(g_fwd.signature(), g_gqa.signature());

        // Different mask -> same graph signature on the spec layer
        // (mask wires into the variant pack at execute time, not the
        // descriptor digest). Verify the params struct still records
        // it.
        let p2 =
            AttentionParams::new(2, 128, 128, 8, 8, 64).with_mask(AttentionMask::SlidingWindow(64));
        assert!(matches!(p2.mask, AttentionMask::SlidingWindow(64)));
    }
}
