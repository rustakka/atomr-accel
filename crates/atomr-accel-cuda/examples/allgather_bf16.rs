//! AllGather over `bf16`.
//!
//! Demonstrates the typed `AllGatherRequest<half::bf16>` shape. Runs
//! against the local NCCL world; gated on `cuda-runtime-tests` since
//! it allocates real device memory.
//!
//! Run: `cargo run -p atomr-accel-cuda --example allgather_bf16
//!         --no-default-features --features
//!         cuda-runtime-tests,nccl,f16`

use atomr_accel_cuda::prelude::*;

fn main() {
    // Capability probe is the cheapest demonstration that doesn't
    // require a multi-GPU host. A full multi-rank AllGather demo
    // would spawn N `DeviceActor`s + an `NcclWorldActor`; the
    // dtype-generic surface is exercised here by constructing the
    // request type and inspecting its dispatch dtype.
    let caps = atomr_accel_cuda::kernel::collective::probe_capabilities();
    println!("NCCL caps: {caps:?}");
    let _ = std::any::type_name::<AllGatherRequest<half::bf16>>();
    println!(
        "AllGatherRequest<bf16> dispatch dtype: {:?}",
        <half::bf16 as NcclReduceSupported>::dispatch_dtype()
    );
}
