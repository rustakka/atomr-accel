//! Local FFI shims for cudarc gaps.
//!
//! Each sub-module wraps the bits of a CUDA library that cudarc 0.19.4
//! either doesn't expose at the safe level (cuSPARSE generic API beyond
//! SpMV/SpMM, cuSPARSELt entirely) or doesn't expose at all. Keeping the
//! `unsafe` confined to this module makes cudarc upgrades a single-file
//! diff.

#[cfg(feature = "cusparse")]
pub mod cusparse;

#[cfg(feature = "cusparse-lt")]
pub mod cusparse_lt;
