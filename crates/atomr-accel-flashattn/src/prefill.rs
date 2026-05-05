//! Chunked-prefill helpers.
//!
//! Chunked prefill processes a long prefix in fixed-size chunks while
//! incrementally appending the produced K / V tiles to an underlying
//! KV cache. It is the path that long-context inference servers use
//! to keep memory bounded — the alternative is to materialise the
//! full prefix attention in one launch, which is infeasible past a
//! few thousand tokens.
//!
//! This module ships a request type that combines varlen layout with
//! a per-chunk KV-cache append. It is the most general non-paged
//! prefill request; for paged caches use [`crate::paged`] with
//! `q_tokens_per_seq > 1`.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dispatch::{DispatchKey, FaFwdDispatch, GemmSupported, SmArch};
use crate::fa2::{MaskKind, PositionBias};
use crate::varlen::CumulativeSeqlens;
use crate::FlashAttnError;

/// Per-chunk metadata describing where in the underlying cache the new
/// K / V tiles should land.
#[derive(Debug, Clone)]
pub struct ChunkLayout {
    /// Length of this chunk in query tokens.
    pub chunk_q_tokens: u32,
    /// Length of the cumulative prefix already written to the cache
    /// before this chunk (i.e. the kv "context" the new q tokens
    /// attend over, in addition to themselves).
    pub prefix_kv_tokens: u32,
    /// Total chunk count for this sequence (used to set up the
    /// causal-on-causal mask).
    pub total_chunks: u32,
    /// 0-indexed chunk number within `total_chunks`.
    pub chunk_index: u32,
}

impl ChunkLayout {
    pub fn new(
        chunk_q_tokens: u32,
        prefix_kv_tokens: u32,
        total_chunks: u32,
        chunk_index: u32,
    ) -> Result<Self, FlashAttnError> {
        if chunk_q_tokens == 0 {
            return Err(FlashAttnError::ZeroSeqlen);
        }
        if total_chunks == 0 {
            return Err(FlashAttnError::ZeroSeqlen);
        }
        if chunk_index >= total_chunks {
            return Err(FlashAttnError::ChunkIndexOutOfRange {
                index: chunk_index,
                total: total_chunks,
            });
        }
        Ok(Self {
            chunk_q_tokens,
            prefix_kv_tokens,
            total_chunks,
            chunk_index,
        })
    }

    /// True iff this is the final chunk for the sequence.
    pub fn is_final(&self) -> bool {
        self.chunk_index + 1 == self.total_chunks
    }
}

/// Chunked-prefill forward request.
pub struct ChunkedPrefillRequest<T: GemmSupported> {
    pub arch: SmArch,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub mask: MaskKind,
    pub bias: PositionBias,
    pub sink_tokens: u32,
    pub softmax_scale: f32,
    pub seqlens: CumulativeSeqlens,
    pub chunk: ChunkLayout,
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<T>,
}

impl<T: GemmSupported> ChunkedPrefillRequest<T> {
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        bias: PositionBias,
        sink_tokens: u32,
        softmax_scale: f32,
        seqlens: CumulativeSeqlens,
        chunk: ChunkLayout,
    ) -> Result<(Self, oneshot::Receiver<Result<(), FlashAttnError>>), FlashAttnError> {
        let (tx, rx) = oneshot::channel();
        let req = Self {
            arch,
            head_dim,
            gqa_ratio,
            mask,
            bias,
            sink_tokens,
            softmax_scale,
            seqlens,
            chunk,
            reply: tx,
            _marker: PhantomData,
        };
        let key = req.compute_key();
        key.validate_fwd().map_err(FlashAttnError::Dispatch)?;
        Ok((req, rx))
    }

    fn compute_key(&self) -> DispatchKey {
        // Chunked prefill is a varlen path with the additional chunk
        // metadata reflected at runtime; the dispatch table uses the
        // varlen flag.
        DispatchKey {
            arch: self.arch,
            dtype: T::dtype(),
            head_dim: self.head_dim,
            causal: self.mask.causal(),
            varlen: true,
            sliding_window: self.mask.sliding_window(),
            alibi: self.bias.requires_alibi_flag(),
            sink: self.sink_tokens,
            paged: false,
            gqa_ratio: self.gqa_ratio,
        }
    }
}

impl<T: GemmSupported> FaFwdDispatch for ChunkedPrefillRequest<T> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Bf16;

    #[test]
    fn chunked_prefill_request_round_trip() {
        let seqlens = CumulativeSeqlens::new(2, 2048, 16384, 4096, 32768).unwrap();
        let chunk = ChunkLayout::new(2048, 14336, 8, 7).unwrap();
        assert!(chunk.is_final());

        let (req, _rx) = ChunkedPrefillRequest::<Bf16>::new(
            SmArch::Sm90a,
            128,
            8,
            MaskKind::SlidingCausal { window: 4096 },
            PositionBias::None,
            4,
            1.0 / (128f32).sqrt(),
            seqlens,
            chunk,
        )
        .expect("valid chunked prefill");

        let key = req.dispatch_key();
        assert!(key.varlen);
        assert!(key.causal);
        assert_eq!(key.sliding_window, Some(4096));
        assert_eq!(key.sink, 4);
        assert_eq!(key.gqa_ratio, 8);

        let kernel_name = crate::dispatch::lookup(&key).unwrap();
        assert!(kernel_name.contains("varlen"));
        assert!(kernel_name.contains("sw4096"));
        assert!(kernel_name.contains("sink4"));
        assert!(kernel_name.contains("gqa8"));

        // Out-of-range chunk index is rejected.
        let err = ChunkLayout::new(1024, 0, 4, 4).unwrap_err();
        assert!(matches!(
            err,
            FlashAttnError::ChunkIndexOutOfRange { index: 4, total: 4 }
        ));

        // Zero-token chunk is rejected.
        let err = ChunkLayout::new(0, 0, 1, 0).unwrap_err();
        assert!(matches!(err, FlashAttnError::ZeroSeqlen));
    }
}
