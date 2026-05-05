//! FlashAttention v3 — forward request types for Hopper/Blackwell.
//!
//! v3 ships the persistent / warp-specialised kernel shapes plus the
//! fp8 e4m3 / e5m2 paths. Backward is currently shared with FA2 — the
//! v3 backward kernels in the upstream csrc are not stable enough to
//! vendor for general use; callers fall through to [`crate::fa2::Fa2BwdRequest`]
//! against `Sm90a` (which the dispatch layer correctly resolves to the
//! fa3 cubin via [`crate::dispatch::SmArch::supports_fa3`]).
//!
//! The fp8 variants ([`Fa3FwdFp8Request`]) require feature `fp8`.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dispatch::{DType, DispatchKey, FaFwdDispatch, GemmSupported, SmArch};
use crate::fa2::{MaskKind, PositionBias};
use crate::FlashAttnError;

/// Persistence mode for FA3. The v3 kernels can run as a single
/// "persistent" grid that consumes a stream of work tiles, or as a
/// classic per-tile grid. Persistent mode wins for short seqlens and
/// loses for very long seqlens — callers pick based on workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistentMode {
    /// Classic grid — one block per (batch, head, q-tile).
    Grid,
    /// Persistent — `num_sms` blocks; each consumes a tile-queue.
    Persistent { num_sms: u32 },
}

/// Request payload for a FlashAttention v3 forward pass (non-fp8).
///
/// Mirrors [`crate::fa2::Fa2FwdRequest`] but with the FA3-only
/// [`PersistentMode`] knob. Validates that `arch.supports_fa3()`.
pub struct Fa3FwdRequest<T: GemmSupported> {
    pub arch: SmArch,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub mask: MaskKind,
    pub bias: PositionBias,
    pub sink_tokens: u32,
    pub softmax_scale: f32,
    pub persistent: PersistentMode,
    /// FP16 / bfloat16 only at this entry point. Use [`Fa3FwdFp8Request`]
    /// for the fp8 paths.
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<T>,
}

impl<T: GemmSupported> Fa3FwdRequest<T> {
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        bias: PositionBias,
        sink_tokens: u32,
        softmax_scale: f32,
        persistent: PersistentMode,
    ) -> Result<(Self, oneshot::Receiver<Result<(), FlashAttnError>>), FlashAttnError> {
        if !arch.supports_fa3() {
            return Err(FlashAttnError::Fa3RequiresHopper(arch));
        }
        if T::dtype() == DType::F8E4m3 || T::dtype() == DType::F8E5m2 {
            return Err(FlashAttnError::Fp8MustUseFp8Request);
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
            persistent,
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
            varlen: false,
            sliding_window: self.mask.sliding_window(),
            alibi: self.bias.requires_alibi_flag(),
            sink: self.sink_tokens,
            paged: false,
            gqa_ratio: self.gqa_ratio,
        }
    }

    /// True iff this request runs in persistent mode.
    pub fn is_persistent(&self) -> bool {
        matches!(self.persistent, PersistentMode::Persistent { .. })
    }
}

impl<T: GemmSupported> FaFwdDispatch for Fa3FwdRequest<T> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

/// Request payload for FA3 fp8 forward. Q is fp8 (`TQ`), K/V can be a
/// distinct fp8 type (`TKV`) — DPA-mixed-precision uses e4m3 for Q/K
/// and e5m2 for V.
#[cfg(feature = "fp8")]
pub struct Fa3FwdFp8Request<TQ: GemmSupported, TKV: GemmSupported> {
    pub arch: SmArch,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub mask: MaskKind,
    pub sink_tokens: u32,
    pub softmax_scale: f32,
    /// Per-tensor descale factor for Q. Required because fp8 storage
    /// precision can't represent the dequantised range without an
    /// out-of-band scale.
    pub q_scale: f32,
    /// Per-tensor descale factor for K.
    pub k_scale: f32,
    /// Per-tensor descale factor for V.
    pub v_scale: f32,
    pub persistent: PersistentMode,
    pub reply: oneshot::Sender<Result<(), FlashAttnError>>,
    _marker: PhantomData<(TQ, TKV)>,
}

#[cfg(feature = "fp8")]
impl<TQ: GemmSupported, TKV: GemmSupported> Fa3FwdFp8Request<TQ, TKV> {
    pub fn new(
        arch: SmArch,
        head_dim: u32,
        gqa_ratio: u32,
        mask: MaskKind,
        sink_tokens: u32,
        softmax_scale: f32,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        persistent: PersistentMode,
    ) -> Result<(Self, oneshot::Receiver<Result<(), FlashAttnError>>), FlashAttnError> {
        if !arch.supports_fp8() {
            return Err(FlashAttnError::Dispatch(
                crate::dispatch::DispatchError::Fp8RequiresHopper(arch),
            ));
        }
        if !TQ::dtype().is_fp8() || !TKV::dtype().is_fp8() {
            return Err(FlashAttnError::Fp8MustUseFp8Request);
        }
        let (tx, rx) = oneshot::channel();
        let req = Self {
            arch,
            head_dim,
            gqa_ratio,
            mask,
            sink_tokens,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
            persistent,
            reply: tx,
            _marker: PhantomData,
        };
        // Validate against the Q dtype's cell. KV dtype rides along in
        // the kernel-name expression but doesn't change the dispatch
        // table key shape.
        let key = req.compute_key();
        key.validate_fwd().map_err(FlashAttnError::Dispatch)?;
        Ok((req, rx))
    }

    fn compute_key(&self) -> DispatchKey {
        DispatchKey {
            arch: self.arch,
            dtype: TQ::dtype(),
            head_dim: self.head_dim,
            causal: self.mask.causal(),
            varlen: false,
            sliding_window: self.mask.sliding_window(),
            alibi: false,
            sink: self.sink_tokens,
            paged: false,
            gqa_ratio: self.gqa_ratio,
        }
    }

    /// Convenience: returns `(q_dtype, kv_dtype)` for the kernel-name
    /// suffix the runtime appends.
    pub fn fp8_dtypes(&self) -> (DType, DType) {
        (TQ::dtype(), TKV::dtype())
    }
}

#[cfg(feature = "fp8")]
impl<TQ: GemmSupported, TKV: GemmSupported> FaFwdDispatch for Fa3FwdFp8Request<TQ, TKV> {
    fn dispatch_key(&self) -> DispatchKey {
        self.compute_key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Bf16, F16};

    #[test]
    fn fa3_fwd_request_requires_hopper() {
        // sm_80 must be rejected.
        let err = Fa3FwdRequest::<F16>::new(
            SmArch::Sm80,
            128,
            1,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            PersistentMode::Grid,
        )
        .err()
        .expect("expected an error");
        assert!(matches!(err, FlashAttnError::Fa3RequiresHopper(_)));

        // sm_90a is fine.
        let (req, _rx) = Fa3FwdRequest::<Bf16>::new(
            SmArch::Sm90a,
            128,
            8,
            MaskKind::Causal,
            PositionBias::None,
            0,
            1.0 / (128f32).sqrt(),
            PersistentMode::Persistent { num_sms: 132 },
        )
        .expect("fa3 fwd on hopper");
        assert!(req.is_persistent());
        let key = req.dispatch_key();
        assert_eq!(key.arch, SmArch::Sm90a);
        assert_eq!(key.dtype, DType::Bf16);
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn fa3_fwd_fp8_request_round_trip() {
        use crate::dispatch::{F8E4m3, F8E5m2};

        let (req, _rx) = Fa3FwdFp8Request::<F8E4m3, F8E5m2>::new(
            SmArch::Sm90a,
            128,
            8,
            MaskKind::Causal,
            0,
            1.0 / (128f32).sqrt(),
            1.0,
            1.0,
            1.0,
            PersistentMode::Persistent { num_sms: 132 },
        )
        .expect("fp8 fwd on hopper");
        let key = req.dispatch_key();
        assert_eq!(key.arch, SmArch::Sm90a);
        assert_eq!(key.dtype, DType::F8E4m3);
        assert!(key.causal);
        assert_eq!(key.head_dim, 128);
        assert_eq!(key.gqa_ratio, 8);

        let (q_t, kv_t) = req.fp8_dtypes();
        assert_eq!(q_t, DType::F8E4m3);
        assert_eq!(kv_t, DType::F8E5m2);

        // Non-fp8 marker types must be rejected.
        let err = Fa3FwdFp8Request::<F16, F8E5m2>::new(
            SmArch::Sm90a,
            128,
            1,
            MaskKind::Full,
            0,
            1.0 / (128f32).sqrt(),
            1.0,
            1.0,
            1.0,
            PersistentMode::Grid,
        )
        .err()
        .expect("expected an error");
        assert!(matches!(err, FlashAttnError::Fp8MustUseFp8Request));

        // Non-Hopper must be rejected.
        let err = Fa3FwdFp8Request::<F8E4m3, F8E4m3>::new(
            SmArch::Sm80,
            128,
            1,
            MaskKind::Full,
            0,
            1.0 / (128f32).sqrt(),
            1.0,
            1.0,
            1.0,
            PersistentMode::Grid,
        )
        .err()
        .expect("expected an error");
        assert!(matches!(
            err,
            FlashAttnError::Dispatch(crate::dispatch::DispatchError::Fp8RequiresHopper(_))
        ));
    }
}
