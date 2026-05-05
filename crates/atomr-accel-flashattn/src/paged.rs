//! Paged KV-cache attention (vLLM-style) — gated by feature `paged`.
//!
//! Each sequence's KV cache is broken into fixed-size blocks (the
//! `block_size`, typically 16 or 32 tokens) referenced through a
//! per-sequence block table. The attention kernel walks the block
//! table, fetching K / V tiles from non-contiguous memory.
//!
//! Layout matches PagedAttention:
//!
//! ```text
//! K_cache: [num_blocks, num_kv_heads, head_dim, block_size]   (or block_size first)
//! V_cache: [num_blocks, num_kv_heads, head_dim, block_size]
//! block_tables: [num_seqs, max_blocks_per_seq]   (i32 block indices)
//! seq_lens: [num_seqs]                            (i32 actual KV length)
//! ```
//!
//! Kernel choice depends on `(arch, dtype, head_dim, block_size,
//! sliding_window, causal)`.

#![cfg(feature = "paged")]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dispatch::{DispatchKey, FaPagedFwdDispatch, GemmSupported, SmArch};
use crate::fa2::{MaskKind, PositionBias};
use crate::FlashAttnError;

/// Layout descriptor for the paged KV cache.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    /// Number of KV blocks allocated globally.
    pub num_blocks: u32,
    /// Tokens per block. Typical: 16, 32, 64.
    pub block_size: u32,
    /// Number of KV heads (post-GQA folding).
    pub num_kv_heads: u32,
    /// Head dimension (D).
    pub head_dim: u32,
    /// Maximum number of blocks any single sequence might reference.
    pub max_blocks_per_seq: u32,
}

impl PagedKvCache {
    pub fn new(
        num_blocks: u32,
        block_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
        max_blocks_per_seq: u32,
    ) -> Result<Self, FlashAttnError> {
        if num_blocks == 0 || block_size == 0 || num_kv_heads == 0 {
            return Err(FlashAttnError::EmptyPagedCache);
        }
        if !matches!(block_size, 8 | 16 | 32 | 64 | 128) {
            return Err(FlashAttnError::InvalidPagedBlockSize(block_size));
        }
        if max_blocks_per_seq == 0 {
            return Err(FlashAttnError::EmptyPagedCache);
        }
        Ok(Self {
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            max_blocks_per_seq,
        })
    }
}

/// Paged-attention forward request.
pub struct PagedAttentionRequest<T: GemmSupported> {
    pub arch: SmArch,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub mask: MaskKind,
    pub bias: PositionBias,
    pub sink_tokens: u32,
    pub softmax_scale: f32,
    pub cache: PagedKvCache,
    /// Number of sequences in the request batch (== `block_tables.shape[0]`).
    pub num_seqs: u32,
    /// Number of query tokens per sequence (1 for pure decode, > 1 for
    /// chunked-prefill speculative decoding).
    pub q_tokens_per_seq: u32,
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<T>,
}

impl<T: GemmSupported> PagedAttentionRequest<T> {
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        bias: PositionBias,
        sink_tokens: u32,
        softmax_scale: f32,
        cache: PagedKvCache,
        num_seqs: u32,
        q_tokens_per_seq: u32,
    ) -> Result<(Self, oneshot::Receiver<Result<(), FlashAttnError>>), FlashAttnError> {
        if num_seqs == 0 || q_tokens_per_seq == 0 {
            return Err(FlashAttnError::EmptyBatch);
        }
        // The cache's head_dim must match the per-request head_dim.
        if cache.head_dim != head_dim {
            return Err(FlashAttnError::PagedHeadDimMismatch {
                cache: cache.head_dim,
                req: head_dim,
            });
        }
        let (tx, rx) = oneshot::channel();
        let req = Self {
            arch,
            head_dim,
            gqa_ratio,
            mask,
            bias,
            sink_tokens,
            softmax_scale,
            cache,
            num_seqs,
            q_tokens_per_seq,
            reply: tx,
            _marker: PhantomData,
        };
        let key = req.compute_key();
        key.validate_paged().map_err(FlashAttnError::Dispatch)?;
        Ok((req, rx))
    }

    fn compute_key(&self) -> DispatchKey {
        DispatchKey {
            arch: self.arch,
            dtype: T::dtype(),
            head_dim: self.head_dim,
            causal: self.mask.causal(),
            varlen: false,
            sliding_window: self.mask.sliding_window(),
            alibi: self.bias.requires_alibi_flag(),
            sink: self.sink_tokens,
            paged: true,
            gqa_ratio: self.gqa_ratio,
        }
    }

    /// True iff this request is a pure-decode call (q_tokens_per_seq == 1).
    pub fn is_pure_decode(&self) -> bool {
        self.q_tokens_per_seq == 1
    }
}

impl<T: GemmSupported> FaPagedFwdDispatch for PagedAttentionRequest<T> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Bf16, DType};

    #[test]
    fn paged_kv_cache_request_round_trip() {
        let cache = PagedKvCache::new(8192, 16, 8, 128, 256).unwrap();
        let (req, _rx) = PagedAttentionRequest::<Bf16>::new(
            SmArch::Sm90a,
            128,
            8,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            cache.clone(),
            32,
            1,
        )
        .expect("valid paged request");
        assert!(req.is_pure_decode());

        let key = req.dispatch_key();
        assert!(key.paged);
        assert!(key.causal);
        assert_eq!(key.dtype, DType::Bf16);
        assert_eq!(key.gqa_ratio, 8);

        let kernel_name = crate::dispatch::lookup(&key).unwrap();
        assert!(kernel_name.contains("paged"));
        assert!(kernel_name.contains("gqa8"));

        // Bad block size.
        let err = PagedKvCache::new(8, 7, 1, 128, 16)
            .err()
            .expect("expected an error");
        assert!(matches!(err, FlashAttnError::InvalidPagedBlockSize(7)));

        // Empty cache.
        let err = PagedKvCache::new(0, 16, 1, 128, 16)
            .err()
            .expect("expected an error");
        assert!(matches!(err, FlashAttnError::EmptyPagedCache));

        // Cache head_dim mismatch.
        let cache_bad = PagedKvCache::new(8, 16, 1, 64, 16).unwrap();
        let err = PagedAttentionRequest::<Bf16>::new(
            SmArch::Sm90a,
            128,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            cache_bad,
            1,
            1,
        )
        .err()
        .expect("expected an error");
        assert!(matches!(
            err,
            FlashAttnError::PagedHeadDimMismatch {
                cache: 64,
                req: 128
            }
        ));

        // num_seqs = 0 must fail.
        let err = PagedAttentionRequest::<Bf16>::new(
            SmArch::Sm90a,
            128,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            cache,
            0,
            1,
        )
        .err()
        .expect("expected an error");
        assert!(matches!(err, FlashAttnError::EmptyBatch));
    }
}
