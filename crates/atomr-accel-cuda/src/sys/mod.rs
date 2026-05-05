//! `sys` — thin Rust wrappers over `cudarc`'s raw `*::sys` FFI for
//! library entry points that aren't yet exposed by the safe layer.
//!
//! Each sub-module is feature-gated to match its cudarc parent so the
//! crate still builds with the corresponding library disabled.

pub mod cublas;
#[cfg(feature = "cublaslt")]
pub mod cublaslt;
pub mod cuda_driver;
#[cfg(feature = "cudnn")]
pub mod cudnn;
#[cfg(feature = "cufft")]
pub mod cufft;
#[cfg(feature = "curand")]
pub mod curand;
#[cfg(feature = "cusolver")]
pub mod cusolver;
#[cfg(feature = "cusparse")]
pub mod cusparse;
#[cfg(feature = "cusparse-lt")]
pub mod cusparse_lt;
#[cfg(feature = "cutensor")]
pub mod cutensor;
