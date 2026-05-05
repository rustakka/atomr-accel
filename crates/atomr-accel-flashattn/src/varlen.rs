//! Variable-length attention — packs sequences of different lengths
//! into a single batch tensor with a `cu_seqlens` cumulative offset
//! array.
//!
//! The dispatch key has `varlen = true`; layout-wise the kernels read
//! Q / K / V as flat `[total_tokens, num_heads, head_dim]` and use the
//! `cu_seqlens_q[i] .. cu_seqlens_q[i+1]` range to bound work for
//! sequence `i`. This is the format produced by HuggingFace's
//! `attention_mask` when packed via `pad_input` / `unpad_input`.
//!
//! `CumulativeSeqlens` is the request-side description of the layout;
//! the actual GPU buffers (`cu_seqlens_q`, `cu_seqlens_kv`,
//! `max_seqlen_q`, `max_seqlen_kv`) are looked up from device memory
//! at launch time.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dispatch::{DispatchKey, FaFwdDispatch, GemmSupported, SmArch};
use crate::fa2::{MaskKind, PositionBias};
use crate::FlashAttnError;

/// Cumulative-seqlen layout descriptor.
#[derive(Debug, Clone)]
pub struct CumulativeSeqlens {
    /// Number of sequences packed into the batch.
    pub batch_size: u32,
    /// `max_seqlen_q` — maximum query length across the batch. Used to
    /// pick the kernel's tile shape.
    pub max_seqlen_q: u32,
    /// `max_seqlen_kv` — maximum key/value length.
    pub max_seqlen_kv: u32,
    /// Total tokens across all queries (== `cu_seqlens_q[batch_size]`).
    pub total_q_tokens: u32,
    /// Total key/value tokens.
    pub total_kv_tokens: u32,
}

impl CumulativeSeqlens {
    pub fn new(
        batch_size: u32,
        max_seqlen_q: u32,
        max_seqlen_kv: u32,
        total_q_tokens: u32,
        total_kv_tokens: u32,
    ) -> Result<Self, FlashAttnError> {
        if batch_size == 0 {
            return Err(FlashAttnError::EmptyBatch);
        }
        if max_seqlen_q == 0 || max_seqlen_kv == 0 {
            return Err(FlashAttnError::ZeroSeqlen);
        }
        if total_q_tokens > batch_size * max_seqlen_q {
            return Err(FlashAttnError::SeqlenOverflow);
        }
        if total_kv_tokens > batch_size * max_seqlen_kv {
            return Err(FlashAttnError::SeqlenOverflow);
        }
        Ok(Self {
            batch_size,
            max_seqlen_q,
            max_seqlen_kv,
            total_q_tokens,
            total_kv_tokens,
        })
    }
}

/// Variable-length forward request.
pub struct VarlenFwdRequest<T: GemmSupported> {
    pub arch: SmArch,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub mask: MaskKind,
    pub bias: PositionBias,
    pub sink_tokens: u32,
    pub softmax_scale: f32,
    pub seqlens: CumulativeSeqlens,
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<T>,
}

impl<T: GemmSupported> VarlenFwdRequest<T> {
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        bias: PositionBias,
        sink_tokens: u32,
        softmax_scale: f32,
        seqlens: CumulativeSeqlens,
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
            reply: tx,
            _marker: PhantomData,
        };
        let key = req.compute_key();
        key.validate_fwd().map_err(FlashAttnError::Dispatch)?;
        Ok((req, rx))
    }

    fn compute_key(&self) -> DispatchKey {
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

impl<T: GemmSupported> FaFwdDispatch for VarlenFwdRequest<T> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Bf16, DType};

    #[test]
    fn cumulative_seqlen_request_round_trip() {
        let seqlens = CumulativeSeqlens::new(4, 1024, 1024, 4 * 1024, 4 * 1024).unwrap();
        let (req, _rx) = VarlenFwdRequest::<Bf16>::new(
            SmArch::Sm90a,
            128,
            8,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            seqlens.clone(),
        )
        .expect("valid varlen fwd");

        let key = req.dispatch_key();
        assert!(key.varlen);
        assert!(key.causal);
        assert_eq!(key.dtype, DType::Bf16);
        assert_eq!(key.gqa_ratio, 8);

        let kernel_name = crate::dispatch::lookup(&key).unwrap();
        assert!(kernel_name.contains("varlen"));

        // Empty batch must fail.
        let err = CumulativeSeqlens::new(0, 1, 1, 0, 0).err().unwrap();
        assert!(matches!(err, FlashAttnError::EmptyBatch));

        // Zero seqlen must fail.
        let err = CumulativeSeqlens::new(2, 0, 1, 0, 0).err().unwrap();
        assert!(matches!(err, FlashAttnError::ZeroSeqlen));

        // Overflow must fail.
        let err = CumulativeSeqlens::new(2, 4, 4, 100, 0).err().unwrap();
        assert!(matches!(err, FlashAttnError::SeqlenOverflow));

        // Invalid head dim flows through.
        let seqlens = CumulativeSeqlens::new(2, 16, 16, 32, 32).unwrap();
        let res = VarlenFwdRequest::<Bf16>::new(
            SmArch::Sm90a,
            33,
            1,
            MaskKind::Full,
            PositionBias::None,
            0,
            1.0,
            seqlens,
        );
        match res {
            Err(FlashAttnError::Dispatch(crate::dispatch::DispatchError::UnsupportedHeadDim(
                33,
            ))) => {}
            Err(other) => panic!("expected UnsupportedHeadDim(33), got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
