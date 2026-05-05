//! Local thin FFI wrappers around CUDA library entry points that
//! cudarc 0.19.4 doesn't expose (or only exposes in `unsafe` form
//! without typed shims).
//!
//! Each submodule mirrors the layout of the corresponding cudarc
//! `sys` crate. Symbols here are only ever resolved against the
//! shared-library facade cudarc already pulls in — we never link a
//! second copy.

#[cfg(feature = "cufft")]
pub mod cufft;
