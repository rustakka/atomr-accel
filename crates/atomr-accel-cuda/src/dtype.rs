//! `CudaDtype` and per-library element-type marker traits.
//!
//! cudarc exposes its FFI element types as plain Rust types (`f32`,
//! `f64`, …). atomr-accel-cuda uses these markers to make actor
//! requests dtype-generic without leaking the FFI surface — every
//! actor accepts `T: <Library>Supported` instead of hard-coding a
//! single element type per message variant.
//!
//! Phase 1 introduces:
//! - [`CudaDtype`] — base marker for any element type the CUDA
//!   libraries can ingest (real f32/f64 today; complex c32/c64
//!   added incrementally).
//! - [`SolverSupported`] — the subset accepted by `SolverActor`.
//!   cuSOLVER's S/D/C/Z naming gives us per-prefix sys entry points;
//!   this trait dispatches to the right one without each request
//!   having to pattern-match on the type itself.

/// Marker for any scalar element type understood by atomr-accel-cuda.
///
/// Implementors are bytewise-FFI-compatible with their CUDA equivalent
/// (cuSOLVER `S/D/C/Z`, cuBLAS `s/d/c/z`, …). The trait is
/// intentionally narrow — it does not expose dtype enum tags or
/// conversion helpers — so the cost of adding a new dtype downstream
/// is only the new impl plus the per-actor supplemental marker
/// (`SolverSupported`, etc.).
pub trait CudaDtype: Copy + Send + Sync + 'static {}

impl CudaDtype for f32 {}
impl CudaDtype for f64 {}

/// Marker indicating a dtype is supported by `SolverActor`'s dense
/// dispatch tables.
///
/// cuSOLVER's dense entry points come in real (`S`, `D`) and complex
/// (`C`, `Z`) variants. Phase 1 cuSOLVER ships f32 + f64; complex
/// variants are deferred so the request structs can stay
/// `T: SolverSupported` without each impl having to know about
/// complex layouts.
pub trait SolverSupported: CudaDtype {}

impl SolverSupported for f32 {}
impl SolverSupported for f64 {}
