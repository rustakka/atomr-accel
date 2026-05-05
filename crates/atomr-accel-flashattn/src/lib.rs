//! # atomr-accel-flashattn
//!
//! FlashAttention v2 + v3 kernel templates for atomr-accel.
//!
//! Provides forward + backward attention via NVRTC-compiled CUDA C++
//! kernels with the full feature matrix:
//!
//! | Feature                 | FA2 (sm_80 / sm_89) | FA3 (sm_90a / sm_100) |
//! |-------------------------|---------------------|------------------------|
//! | f16, bf16               | ✔                   | ✔                      |
//! | fp8 e4m3 / e5m2         |                     | ✔ (`fp8`)              |
//! | causal                  | ✔                   | ✔                      |
//! | varlen (cu_seqlens)     | ✔                   | ✔                      |
//! | sliding window          | ✔                   | ✔                      |
//! | sink tokens             | ✔                   | ✔                      |
//! | ALiBi                   | ✔                   | ✔                      |
//! | MQA / GQA               | ✔                   | ✔                      |
//! | paged KV-cache          | ✔ (`paged`)         | ✔ (`paged`)            |
//! | chunked prefill         | ✔                   | ✔                      |
//! | persistent kernels      |                     | ✔                      |
//! | backward                | ✔ (f16/bf16 only)   | falls through to fa2   |
//!
//! ## Architecture
//!
//! Each request type ([`fa2::Fa2FwdRequest`], [`fa3::Fa3FwdRequest`],
//! [`varlen::VarlenFwdRequest`], [`paged::PagedAttentionRequest`],
//! [`prefill::ChunkedPrefillRequest`]) is generic over a
//! [`dispatch::GemmSupported`] dtype marker and produces a
//! [`dispatch::DispatchKey`] — the canonical (arch, dtype, head_dim,
//! causal, varlen, sliding_window, alibi, sink, paged, gqa_ratio)
//! tuple that picks one of FA2 / FA3 cubins.
//!
//! At runtime, the Phase 0.6 NVRTC disk cache compiles the matching
//! cubin lazily; the dispatch table maps a [`dispatch::DispatchKey`]
//! to the canonical mangled kernel-name expression. The hot path for
//! steady-state inference is: `dispatch_key()` → cache hit → cubin
//! launch.
//!
//! ## Cargo features
//!
//! - `fp8` — enables the FA3 fp8 (`F8E4m3` / `F8E5m2`) request types.
//! - `paged` — enables the [`paged`] module and the paged dispatch
//!   key cells.
//! - `cuda-runtime-tests` — gates the real-GPU example + the
//!   `Real` actor variant. Off by default so the crate builds and
//!   unit-tests on hosts without CUDA.

#![allow(clippy::module_name_repetitions, clippy::too_many_arguments)]

pub mod actor;
pub mod dispatch;
pub mod fa2;
pub mod fa3;
#[cfg(feature = "paged")]
pub mod paged;
pub mod prefill;
pub mod varlen;

#[cfg(feature = "cuda-runtime-tests")]
mod cuda_real {
    //! Real-GPU types referenced from [`crate::actor::FlashAttnInner::Real`].
    //!
    //! Defined behind `cuda-runtime-tests` so the crate builds without
    //! a working CUDA driver.

    /// Opaque reference to the host's `NvrtcActor`. The real type lives
    /// in `atomr-accel-cuda::kernel::NvrtcActor`; the `FlashAttnActor`
    /// only needs to forward `Compile { … }` / `Launch { … }` messages
    /// to it, so a newtype around `ActorRef<NvrtcMsg>` is enough.
    pub struct NvrtcRef {
        // The concrete `ActorRef<NvrtcMsg>` is constructed by callers
        // and embedded by `FlashAttnActor::props`. We keep the field
        // pub(crate) so this stub can be replaced once the runtime
        // launch path lands.
        pub(crate) _opaque: (),
    }
}

pub use actor::{FlashAttnActor, FlashAttnInner, FlashAttnMsg, FlashAttnProps};
pub use dispatch::{
    lookup, Bf16, DType, DispatchError, DispatchKey, DispatchTable, FaBwdDispatch, FaFwdDispatch,
    FaPagedFwdDispatch, GemmSupported, SmArch, DISPATCH_TABLE, F16,
};

#[cfg(feature = "fp8")]
pub use dispatch::{F8E4m3, F8E5m2};

pub use fa2::{Fa2BwdRequest, Fa2FwdRequest, MaskKind, PositionBias};
#[cfg(feature = "fp8")]
pub use fa3::Fa3FwdFp8Request;
pub use fa3::{Fa3FwdRequest, PersistentMode};

#[cfg(feature = "paged")]
pub use paged::{PagedAttentionRequest, PagedKvCache};

pub use prefill::{ChunkLayout, ChunkedPrefillRequest};
pub use varlen::{CumulativeSeqlens, VarlenFwdRequest};

/// Errors surfaced by the FlashAttention crate. Most are construction-
/// time validation failures; a small set are runtime launch errors
/// produced by the actor (and kept here so callers can pattern-match
/// without depending on the rest of `atomr-accel-cuda`).
#[derive(Debug, Clone, thiserror::Error)]
pub enum FlashAttnError {
    /// Validation against the dispatch table failed.
    #[error("dispatch error: {0}")]
    Dispatch(#[from] DispatchError),

    /// A FlashAttention v3 request targeted a non-Hopper arch.
    #[error("FA3 requires sm_90a or newer, got {0:?}")]
    Fa3RequiresHopper(SmArch),

    /// An fp8 dtype was passed to a non-fp8 request type, or vice
    /// versa.
    #[error("fp8 dtypes must use Fa3FwdFp8Request and vice versa")]
    Fp8MustUseFp8Request,

    /// Variable-length / paged batch is empty.
    #[error("attention batch must contain at least one sequence")]
    EmptyBatch,

    /// Sequence length is zero.
    #[error("seqlen must be > 0")]
    ZeroSeqlen,

    /// Cumulative seqlens overflow `batch_size * max_seqlen`.
    #[error("cumulative seqlens overflow batch_size * max_seqlen")]
    SeqlenOverflow,

    /// Paged KV cache is empty / zero-sized.
    #[error("paged KV cache must be non-empty")]
    EmptyPagedCache,

    /// Paged KV-cache block size not in the supported set.
    #[error("paged block_size {0} is not in (8, 16, 32, 64, 128)")]
    InvalidPagedBlockSize(u32),

    /// Paged cache head_dim doesn't match the request head_dim.
    #[error("paged cache head_dim {cache} != request head_dim {req}")]
    PagedHeadDimMismatch { cache: u32, req: u32 },

    /// Chunked-prefill chunk index is out of range.
    #[error("chunk_index {index} >= total_chunks {total}")]
    ChunkIndexOutOfRange { index: u32, total: u32 },

    /// Mock-mode actor saw a launch it can't honour.
    #[error("flashattn actor is in mock mode (no GPU)")]
    MockMode,
}
