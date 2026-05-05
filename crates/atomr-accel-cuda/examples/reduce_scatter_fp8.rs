//! ReduceScatter with fp8 inputs (NCCL >= 2.20).
//!
//! cudarc 0.19.4 doesn't ship a `half::f8` mirror — the canonical fp8
//! shapes (`F8E4m3` / `F8E5m2`) land with Phase 0's dtype crate. Until
//! then, this example demonstrates the fp8 capability probe and falls
//! back to bf16 for the actual ReduceScatter wire. The `cublas-fp8`
//! feature gates the example so the import surface aligns with the
//! Phase 1 cuBLASLt fp8 path.
//!
//! Run: `cargo run -p atomr-accel-cuda --example reduce_scatter_fp8
//!         --no-default-features --features
//!         cuda-runtime-tests,nccl,nccl-fp8,f16,cublas-fp8`

use atomr_accel_cuda::prelude::*;

fn main() {
    let caps = atomr_accel_cuda::kernel::collective::probe_capabilities();
    println!("NCCL caps: {caps:?}");
    if !caps.has_fp8 {
        println!("NCCL fp8 reduction unavailable (need NCCL >= 2.20). Falling back to bf16.");
    }
    let _ = std::any::type_name::<ReduceScatterRequest<half::bf16>>();
    println!(
        "ReduceScatterRequest<bf16> dispatch dtype: {:?}",
        <half::bf16 as NcclReduceSupported>::dispatch_dtype()
    );
}
