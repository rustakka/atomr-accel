//! FlashAttention v2 — forward + backward request types.
//!
//! v2 targets sm_80 / sm_89 (Ampere / Ada). The kernels are
//! NVRTC-compiled lazily through the Phase 0.6 cubin cache; the
//! request types defined here capture every parameter the kernels
//! need plus the [`DispatchKey`] that picks the right cubin.
//!
//! Each request is generic over a [`GemmSupported`] dtype marker. fp8
//! markers ([`crate::dispatch::F8E4m3`], [`crate::dispatch::F8E5m2`])
//! are *not* implemented for FA2 — only fa3 ships fp8.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dispatch::{
    DType, DispatchError, DispatchKey, FaBwdDispatch, FaFwdDispatch, GemmSupported, SmArch,
};
use crate::FlashAttnError;

/// Position-bias mode applied to the attention scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionBias {
    /// No bias.
    None,
    /// ALiBi linear-position bias. The slopes are encoded in the head
    /// metadata supplied at request time.
    Alibi,
}

impl PositionBias {
    /// True iff this bias requires the alibi flag in the dispatch key.
    pub fn requires_alibi_flag(self) -> bool {
        matches!(self, PositionBias::Alibi)
    }
}

/// Mask family applied to the attention logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskKind {
    /// Full attention.
    Full,
    /// Standard upper-triangular causal mask.
    Causal,
    /// Causal + sliding-window. Each query attends to the last `window`
    /// keys (plus any sink tokens).
    SlidingCausal { window: u32 },
    /// Sliding window without causal. Used by some non-autoregressive
    /// long-context inference paths.
    SlidingFull { window: u32 },
}

impl MaskKind {
    pub fn causal(self) -> bool {
        matches!(self, MaskKind::Causal | MaskKind::SlidingCausal { .. })
    }

    pub fn sliding_window(self) -> Option<u32> {
        match self {
            MaskKind::SlidingCausal { window } | MaskKind::SlidingFull { window } => Some(window),
            _ => None,
        }
    }
}

/// Request payload for a FlashAttention v2 forward pass.
///
/// `T` is the Q/K/V element type; the output is the same dtype. The
/// shape is `[batch, seqlen_q, num_heads, head_dim]` for Q and
/// `[batch, seqlen_kv, num_kv_heads, head_dim]` for K/V (with
/// `gqa_ratio = num_heads / num_kv_heads`).
pub struct Fa2FwdRequest<T: GemmSupported> {
    /// Target architecture. Validated as one of `Sm80` / `Sm89` at
    /// `dispatch_key()` time — fa2 won't run on Hopper without a
    /// fa3 fallback.
    pub arch: SmArch,
    /// Per-head dimension (D).
    pub head_dim: u32,
    /// Q heads per KV head. 1 = MHA.
    pub gqa_ratio: u32,
    /// Mask family.
    pub mask: MaskKind,
    /// Position-bias family.
    pub bias: PositionBias,
    /// Sink token count (StreamingLLM). 0 disables sink behaviour.
    pub sink_tokens: u32,
    /// Softmax scale. Conventionally `1.0 / sqrt(head_dim)`.
    pub softmax_scale: f32,
    /// Optional dropout probability. 0.0 disables dropout. The kernel
    /// uses a Philox4x32 RNG keyed by `dropout_seed`.
    pub dropout_p: f32,
    /// Per-call seed for the Philox RNG. Ignored if `dropout_p == 0.0`.
    pub dropout_seed: u64,
    /// `oneshot` reply channel — receives `Ok(())` after stream
    /// completion or an error if the launch fails.
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<T>,
}

impl<T: GemmSupported> Fa2FwdRequest<T> {
    /// Construct a forward request, returning the matching reply
    /// receiver. Validates the dispatch cell up front.
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        bias: PositionBias,
        sink_tokens: u32,
        softmax_scale: f32,
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
            dropout_p: 0.0,
            dropout_seed: 0,
            reply: tx,
            _marker: PhantomData,
        };
        // Validate the dispatch key.
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
            varlen: false,
            sliding_window: self.mask.sliding_window(),
            alibi: self.bias.requires_alibi_flag(),
            sink: self.sink_tokens,
            paged: false,
            gqa_ratio: self.gqa_ratio,
        }
    }

    /// Resolve the dispatch key to a kernel-name expression.
    pub fn resolve_kernel(&self) -> Result<String, DispatchError> {
        crate::dispatch::lookup(&self.compute_key())
    }
}

impl<T: GemmSupported> FaFwdDispatch for Fa2FwdRequest<T> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

/// Request payload for a FlashAttention v2 backward pass.
///
/// Computes `dQ`, `dK`, `dV` given the saved softmax statistics from
/// the forward pass. The arch / dtype / head_dim must match the
/// forward request that produced the activations.
pub struct Fa2BwdRequest<T: GemmSupported> {
    pub arch: SmArch,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub mask: MaskKind,
    pub bias: PositionBias,
    pub sink_tokens: u32,
    pub softmax_scale: f32,
    /// Whether the backward should rerun softmax (memory-saving) or
    /// reuse the stored intermediates.
    pub recompute: bool,
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<T>,
}

impl<T: GemmSupported> Fa2BwdRequest<T> {
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        bias: PositionBias,
        sink_tokens: u32,
        softmax_scale: f32,
        recompute: bool,
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
            recompute,
            reply: tx,
            _marker: PhantomData,
        };
        // fp8 backward is rejected by validate_bwd, which also catches
        // the head-dim whitelist + sink/mask rule.
        let key = req.compute_key();
        key.validate_bwd().map_err(FlashAttnError::Dispatch)?;
        if T::dtype() == DType::F8E4m3 || T::dtype() == DType::F8E5m2 {
            return Err(FlashAttnError::Dispatch(
                crate::dispatch::DispatchError::Fp8BackwardUnsupported,
            ));
        }
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
            paged: false,
            gqa_ratio: self.gqa_ratio,
        }
    }
}

impl<T: GemmSupported> FaBwdDispatch for Fa2BwdRequest<T> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Bf16, F16};

    #[test]
    fn fa2_fwd_request_round_trip_f16_bf16() {
        // f16 / sm_80 / causal / head_dim 128
        let (req_f16, _rx) = Fa2FwdRequest::<F16>::new(
            SmArch::Sm80,
            128,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
        )
        .expect("valid fa2 fwd");
        let key = req_f16.dispatch_key();
        assert_eq!(key.arch, SmArch::Sm80);
        assert_eq!(key.dtype, DType::F16);
        assert!(key.causal);
        assert_eq!(key.head_dim, 128);
        assert!(req_f16.resolve_kernel().is_ok());

        // bf16 / sm_89 / sliding window 4096 / alibi / sink=8 / gqa=8
        let (req_bf16, _rx) = Fa2FwdRequest::<Bf16>::new(
            SmArch::Sm89,
            128,
            8,
            MaskKind::SlidingCausal { window: 4096 },
            PositionBias::Alibi,
            8,
            1.0 / (128f32).sqrt(),
        )
        .expect("valid fa2 fwd bf16");
        let key = req_bf16.dispatch_key();
        assert_eq!(key.dtype, DType::Bf16);
        assert!(key.alibi);
        assert_eq!(key.sliding_window, Some(4096));
        assert_eq!(key.sink, 8);
        assert_eq!(key.gqa_ratio, 8);

        let kernel_name = req_bf16.resolve_kernel().expect("resolve");
        assert!(kernel_name.contains("bf16"));
        assert!(kernel_name.contains("sw4096"));
        assert!(kernel_name.contains("alibi"));
        assert!(kernel_name.contains("sink8"));
        assert!(kernel_name.contains("gqa8"));

        // Hash determinism: re-build identical requests, compare keys.
        let (req_a, _) = Fa2FwdRequest::<F16>::new(
            SmArch::Sm80,
            64,
            1,
            MaskKind::Full,
            PositionBias::None,
            0,
            1.0 / 8.0,
        )
        .unwrap();
        let (req_b, _) = Fa2FwdRequest::<F16>::new(
            SmArch::Sm80,
            64,
            1,
            MaskKind::Full,
            PositionBias::None,
            0,
            1.0 / 8.0,
        )
        .unwrap();
        assert_eq!(req_a.dispatch_key(), req_b.dispatch_key());
        assert_eq!(
            req_a.dispatch_key().stable_hash(),
            req_b.dispatch_key().stable_hash()
        );

        // Invalid head_dim is rejected before constructing.
        let err = Fa2FwdRequest::<F16>::new(
            SmArch::Sm80,
            123,
            1,
            MaskKind::Full,
            PositionBias::None,
            0,
            1.0,
        )
        .err()
        .expect("expected an error");
        match err {
            FlashAttnError::Dispatch(DispatchError::UnsupportedHeadDim(123)) => {}
            other => panic!("expected UnsupportedHeadDim(123), got {other:?}"),
        }
    }

    #[test]
    fn fa2_bwd_request_round_trip() {
        // Backward bf16 / sm_80 / causal / recompute true.
        let (req, _rx) = Fa2BwdRequest::<Bf16>::new(
            SmArch::Sm80,
            128,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            true,
        )
        .expect("valid fa2 bwd");
        let key = req.dispatch_key();
        assert_eq!(key.dtype, DType::Bf16);
        assert!(key.causal);
        assert_eq!(key.head_dim, 128);

        // Same arch with an unsupported head-dim must fail.
        let err = Fa2BwdRequest::<Bf16>::new(
            SmArch::Sm80,
            7,
            1,
            MaskKind::Full,
            PositionBias::None,
            0,
            1.0,
            false,
        )
        .err()
        .expect("expected an error");
        match err {
            FlashAttnError::Dispatch(DispatchError::UnsupportedHeadDim(7)) => {}
            other => panic!("expected UnsupportedHeadDim(7), got {other:?}"),
        }
    }
}
