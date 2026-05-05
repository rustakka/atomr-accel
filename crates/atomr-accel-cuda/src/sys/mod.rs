//! Crate-private sys-level FFI wrappers.
//!
//! cudarc 0.19's safe layer covers handle lifecycle but leaves the
//! per-op entry points behind `cusolver::sys::lib::*`. Calling them
//! requires repeated `unsafe`, status decoding, and `i32 → usize`
//! workspace dance — the helpers here factor those into a dtype-
//! generic surface so each actor module stays focused on the actor
//! envelope rather than FFI plumbing.
//!
//! Modules are gated by feature so a build with only a subset of
//! libraries enabled doesn't pay for unused FFI imports.

#[cfg(feature = "cusolver")]
pub mod cusolver;
