//! `fa2_fwd_bf16` — sketch of constructing a FlashAttention v2
//! bf16 forward request and resolving it through the dispatch table.
//!
//! Gated behind `cuda-runtime-tests` because a real launch would
//! require a `CudaContext`, the vendored FA2 csrc compiled, and a
//! live `NvrtcActor`. This example demonstrates the request +
//! dispatch surface only.

use atomr_accel_flashattn::{
    dispatch::{lookup, Bf16, SmArch},
    Fa2FwdRequest, MaskKind, PositionBias,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (req, _rx) = Fa2FwdRequest::<Bf16>::new(
        SmArch::Sm80,
        128,
        1,
        MaskKind::Causal,
        PositionBias::None,
        0,
        1.0 / (128f32).sqrt(),
    )?;

    let key = req.resolve_kernel()?;
    println!("dispatch key resolved to: {key}");

    let cell = req.dispatch_key();
    let name = lookup(&cell)?;
    println!("(via lookup) kernel name = {name}");

    Ok(())
}
