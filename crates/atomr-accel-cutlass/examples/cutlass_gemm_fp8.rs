//! Example: render a Hopper fp8 GEMM template via CutlassActor.
//!
//! Runs entirely on the host — no CUDA toolkit required. Prints the
//! generated `.cu` source and the lowered kernel name, plus the
//! plan-cache hit rate after a second identical dispatch.

use atomr_accel_cutlass::{
    CutlassActor, CutlassMsg, GemmEpilogue, GemmRequest, GemmShape, SmArch,
};
use atomr_accel_cutlass::dtype::F8E4m3;

fn main() {
    let actor = CutlassActor::new(16);
    let req = GemmRequest::<F8E4m3>::new(GemmShape::new(4096, 4096, 4096), SmArch::Sm90a)
        .with_epilogue(GemmEpilogue::LinearReLU { alpha: 1.0, beta: 0.0 });

    println!("plan key: {:?}", req.plan_key());
    let (src, name) = req.render_cu();
    println!("kernel:   {name}");
    println!("--- generated .cu ---");
    println!("{src}");

    actor.handle(CutlassMsg::Gemm(Box::new(req.clone())));
    actor.handle(CutlassMsg::Gemm(Box::new(req)));

    println!("dispatched: {}", actor.inner().dispatched());
    println!("plan cache len: {}", actor.inner().plan_cache.len());
}
