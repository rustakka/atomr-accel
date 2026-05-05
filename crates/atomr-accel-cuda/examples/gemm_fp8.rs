//! `gemm_fp8` — fp8 GEMM placeholder.
//!
//! Phase 1 cuBLAS slice ships the dispatch scaffolding plus the
//! [`atomr_accel_cuda::kernel::blas::scaling`] helpers; the wired
//! end-to-end fp8 GEMM lives in the cuBLASLt slice (parallel agent)
//! since cuBLAS-proper's fp8 path requires `cublasGemmEx` with
//! per-tensor scales which are best driven through cuBLASLt's
//! descriptor API. This example documents the surface area and is
//! gated on `cublas-fp8` so it lights up only on Hopper-capable
//! builds.
//!
//! Run on a Hopper+ GPU host with CUDA 12.x:
//!     cargo run -p atomr-accel-cuda --example gemm_fp8 \
//!         --features cuda-runtime-tests,cublas-fp8

#[cfg(feature = "cublas-fp8")]
fn main() {
    use atomr_accel_cuda::kernel::blas::scaling::PerTensorScale;

    let a_scale = PerTensorScale { scale: 1.0 / 240.0 };
    let b_scale = PerTensorScale { scale: 1.0 / 240.0 };
    println!(
        "fp8 cuBLAS scaffold: a_scale={}, b_scale={} (kernel call routed via cuBLASLt)",
        a_scale.scale, b_scale.scale
    );
}

#[cfg(not(feature = "cublas-fp8"))]
fn main() {
    eprintln!(
        "gemm_fp8 example requires --features cublas-fp8 and a Hopper-class GPU; \
         build with `cargo run --example gemm_fp8 --features cublas-fp8`."
    );
}
