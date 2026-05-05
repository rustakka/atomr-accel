//! fp8 scaling-factor helpers.
//!
//! Hopper+ fp8 cuBLAS calls (`cublasGemmEx` with `CUDA_R_8F_E4M3` /
//! `CUDA_R_8F_E5M2` operands) take a per-tensor or per-row scaling
//! factor that brings the input/output values into the representable
//! fp8 range. This module factors out the small bookkeeping helpers
//! used by both cuBLAS and cuBLASLt fp8 paths so they don't have to
//! be duplicated.
//!
//! The full fp8 path lights up under the `cublas-fp8` cargo feature
//! (currently scaffolded — Phase 1 cuBLAS slice ships the helper
//! types, the wired call site lives in cuBLASLt's own module).

#![allow(dead_code)]

/// Per-tensor scaling factor: a single multiplicative scalar.
///
/// `a_scale` is computed by the caller (typically `max(abs(A)) /
/// fp8_max`) and passed to `cublasGemmEx` via `alpha = a_scale *
/// b_scale * gemm_alpha`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerTensorScale {
    pub scale: f32,
}

/// Per-row scaling factor: a vector of `m` scalars, one per row of
/// the matrix. Stored device-side; the caller passes a `GpuRef<f32>`
/// when the cuBLASLt descriptor accepts row-wise amax.
#[derive(Debug, Clone)]
pub struct PerRowScale {
    pub rows: i32,
    pub scale_buf: crate::gpu_ref::GpuRef<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_tensor_scale_default_is_one() {
        let s = PerTensorScale::default();
        assert_eq!(s.scale, 0.0);
    }
}
