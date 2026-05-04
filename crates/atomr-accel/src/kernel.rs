//! `KernelOp` — marker trait for typed kernel-op envelopes.
//!
//! Each backend ships concrete actors (`BlasActor`, `CudnnActor`,
//! …) whose message enums carry op-specific payloads. The marker
//! trait here lets portable code parameterize over "any op the
//! backend supports", without committing to a specific message set.
//!
//! Backends that share an op vocabulary (e.g. cuBLAS' SGEMM has a
//! direct equivalent in hipBLAS, rocBLAS, MPS) are encouraged to
//! implement `KernelOp` for a shared envelope so portable code can
//! migrate between vendors without rewriting call sites.

/// Marker for any typed op envelope. The default impl is empty —
/// the trait exists to enable bound-by-name patterns in generic
/// pipelines without imposing a specific shape on the op.
pub trait KernelOp: Send + 'static {
    /// Display name for telemetry / tracing. Backends that share
    /// ops across vendors should agree on names (e.g. `"sgemm"`).
    fn op_name(&self) -> &'static str;
}

/// Generic GEMM payload. Backends with cuBLAS-equivalent libraries
/// (cuBLAS, hipBLAS, rocBLAS, MPS) all map this to their underlying
/// SGEMM. The matrices are referred to by backend-typed handles
/// (`AccelRef<f32, B>`); storage layout is column-major to match
/// the BLAS convention.
#[derive(Debug, Clone, Copy)]
pub struct GemmShape {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: f32,
    pub beta: f32,
}

/// Generic dense FFT payload. Maps to cuFFT, rocFFT, MPS Graph FFT,
/// or VkFFT depending on the backend.
#[derive(Debug, Clone, Copy)]
pub enum FftKind {
    Forward1dR2C { n: usize, batch: usize },
    Inverse1dC2R { n: usize, batch: usize },
    Forward2dC2C { ny: usize, nx: usize, batch: usize },
}
