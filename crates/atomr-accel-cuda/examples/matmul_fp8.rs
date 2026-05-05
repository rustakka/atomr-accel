//! Phase 1 cuBLASLt fp8 matmul smoke example.
//!
//! Demonstrates the typed [`MatmulRequest<F8E4m3>`] surface end-to-end.
//! Requires Hopper+ hardware at runtime; the build itself is gated
//! behind `cublas-fp8`. Today the dispatch path replies with
//! `Unrecoverable("matmul<fp8e4m3> not yet implemented")` because
//! cudarc 0.19.4 has no `Matmul<F8E4m3>` impl — this example exists
//! to demonstrate the *call shape* and to wire the example into CI's
//! "build with all features" smoke matrix.
//!
//! Run with:
//!
//! ```text
//! cargo run --example matmul_fp8 \
//!   --features=cuda-runtime-tests,cublaslt,f16,cublas-fp8
//! ```

use atomr_accel_cuda::dtype::F8E4m3;
use atomr_accel_cuda::kernel::blas_lt::{Epilogue, MatmulRequest, ScaleSet};

fn main() {
    // Print the dtype tag and call-shape; no GPU touched.
    println!("atomr-accel-cuda fp8 matmul smoke");
    println!("  dtype       = {}", "F8E4m3");
    println!("  epilogue    = {:?}", Epilogue::GeluBias);
    println!("  scale set   = {:?}", ScaleSet::default());
    println!(
        "  request<T>  type = MatmulRequest<F8E4m3> (size = {} B)",
        std::mem::size_of::<MatmulRequest<F8E4m3>>()
    );
    println!("(no GPU work performed in this smoke binary)");
}
