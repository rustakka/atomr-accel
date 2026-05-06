//! Per-op recorders that implement [`super::GraphOp`].
//!
//! Each submodule owns the op type for one CUDA library or
//! mechanism. New ops added in later phases drop in as additional
//! submodules here without editing any central enum.

pub mod memcpy;
pub mod sgemm;

#[cfg(feature = "cudnn")]
pub mod cudnn;
#[cfg(feature = "cusparse")]
pub mod cusparse;
#[cfg(feature = "cufft")]
pub mod fft_r2c;
#[cfg(feature = "curand")]
pub mod rng_fill_uniform;
// cutensor/nccl/nvrtc record adapters are stubbed for a follow-up PR
// (the Phase 3 agent ran out of capacity before writing the bodies).
