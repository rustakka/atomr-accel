//! Per-op recorders that implement [`super::GraphOp`].
//!
//! Each submodule owns the op type for one CUDA library or
//! mechanism. New ops added in later phases (cuBLASLt epilogues,
//! cuSPARSE, cuTENSOR, NCCL, FlashAttention, …) drop in as
//! additional submodules here without editing any central enum.

pub mod memcpy;
pub mod sgemm;

#[cfg(feature = "cufft")]
pub mod fft_r2c;

#[cfg(feature = "curand")]
pub mod rng_fill_uniform;
