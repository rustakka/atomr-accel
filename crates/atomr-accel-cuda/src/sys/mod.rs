//! Local hand-FFI module — every unsafe wrapper that cudarc 0.19.4
//! does not expose lives here. Per-library sub-modules so a future
//! cudarc upgrade is a single-file diff per library.

#[cfg(feature = "cudnn")]
pub mod cudnn;
