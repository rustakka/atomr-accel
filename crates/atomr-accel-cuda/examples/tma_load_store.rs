//! `tma_load_store` — Phase 5 demo: build a 2D TMA tile descriptor
//! suitable for an fp16 GEMM A-operand load.
//!
//! Build / run (Hopper-only):
//!     cargo run -p atomr-accel-cuda --example tma_load_store \
//!         --features cuda-runtime-tests,hopper
//!
//! Like `cluster_launch`, this example exercises the host-side
//! descriptor-builder surface. Plumbing the encoded `CUtensorMap` into
//! a real cp.async.bulk-driven kernel ships in Phase 5.1.

use atomr_accel_cuda::hopper::tma::{
    TensorMapDataType, TensorMapDescriptor, TensorMapInterleave, TensorMapL2Promotion,
    TensorMapOobFill, TensorMapSwizzle,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let desc = TensorMapDescriptor {
        data_type: TensorMapDataType::Float16,
        // 16-byte-aligned device pointer — placeholder for the demo;
        // real kernels would pass the address of a cuMemAlloc'd buffer.
        global_address: 0x1_0000,
        // 1024 × 1024 row-major fp16 matrix.
        global_dim: vec![1024, 1024],
        global_strides: vec![1024 * 2], // bytes per row
        // 64 × 64 tile (matches a single wgmma m64n64k16 op-feed).
        box_dim: vec![64, 64],
        element_strides: vec![1, 1],
        interleave: TensorMapInterleave::None,
        swizzle: TensorMapSwizzle::Bytes128,
        l2_promotion: TensorMapL2Promotion::Bytes128,
        oob_fill: TensorMapOobFill::NaZero,
    };

    desc.validate()?;

    println!("TensorMap descriptor (rank {}):", desc.rank());
    println!("  data_type     = {:?}", desc.data_type);
    println!("  global_dim    = {:?}", desc.global_dim);
    println!("  box_dim       = {:?}", desc.box_dim);
    println!("  swizzle       = {:?}", desc.swizzle);
    println!("  l2_promotion  = {:?}", desc.l2_promotion);
    println!("  oob_fill      = {:?}", desc.oob_fill);

    #[cfg(feature = "hopper")]
    {
        // The encode call would invoke `cuTensorMapEncodeTiled` against
        // the active driver. The body stays gated behind `hopper`
        // because the cudarc cu* symbols only resolve under that feature.
        // Uncomment once a live CUDA context is available:
        // let map = desc.encode()?;
        // println!("Encoded CUtensorMap @ {:p}", map.as_ptr());
        println!("(hopper feature on — encode() available)");
    }

    Ok(())
}
