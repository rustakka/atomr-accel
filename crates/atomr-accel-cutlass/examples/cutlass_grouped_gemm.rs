//! Example: grouped GEMM dispatch with three distinct shapes per
//! group on Hopper. Requires the `grouped` cargo feature.

#[cfg(feature = "grouped")]
fn main() {
    use atomr_accel_cutlass::dtype::F16;
    use atomr_accel_cutlass::{
        CutlassActor, CutlassMsg, GemmShape, GroupedGemmRequest, GroupedGemmShape, GroupedLayout,
        SmArch,
    };

    let actor = CutlassActor::new(8);
    let shapes = vec![
        GemmShape::new(64, 64, 64),
        GemmShape::new(128, 128, 64),
        GemmShape::new(64, 256, 32),
    ];
    let req = GroupedGemmRequest::<F16>::new(GroupedGemmShape::new(shapes), SmArch::Sm90a)
        .with_grouped_layout(GroupedLayout::Variable);

    println!("plan key: {:?}", req.plan_key());
    let (_src, name) = req.render_cu();
    println!("kernel:   {name}");

    actor.handle(CutlassMsg::GroupedGemm(Box::new(req)));
    println!("plan cache len: {}", actor.inner().plan_cache.len());
}

#[cfg(not(feature = "grouped"))]
fn main() {
    println!("rebuild with --features grouped to run this example");
}
