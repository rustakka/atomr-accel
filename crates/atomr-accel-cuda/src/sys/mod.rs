//! Local sys-level wrappers for parts of the CUDA library surface that
//! `cudarc` 0.19.4's safe layer does not yet cover.
//!
//! Each submodule is gated on the matching cargo feature so the host
//! compile (no GPU, no library installed) still succeeds when a feature
//! is off.

#[cfg(feature = "cutensor")]
pub mod cutensor;
