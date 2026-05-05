//! Thin local wrappers over `cudarc::*::sys` for CUDA library entry
//! points cudarc's safe layer doesn't yet expose (Phase 1 cuBLASLt
//! heuristic surface, fp8 scale-pointer setting, etc.).
//!
//! Each sub-module is gated by the same feature flag as its consumer
//! library actor.

#[cfg(feature = "cublaslt")]
pub mod cublaslt;
