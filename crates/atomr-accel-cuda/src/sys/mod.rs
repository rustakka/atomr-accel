//! `sys` — thin Rust wrappers over `cudarc`'s raw `*::sys` FFI for
//! library entry points that aren't yet exposed by the safe layer.
//!
//! Each sub-module is feature-gated to match its cudarc parent so the
//! crate still builds with the corresponding library disabled.

#[cfg(feature = "cufft")]
pub mod cufft;
#[cfg(feature = "curand")]
pub mod curand;
