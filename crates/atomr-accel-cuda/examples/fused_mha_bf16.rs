//! `fused_mha_bf16` — Phase 2 cuDNN demo: build a multi-head
//! attention forward request (bf16, causal, GQA) and print the spec
//! summary.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example fused_mha_bf16 \
//!         --features cuda-runtime-tests,cudnn,f16

use atomr_accel_cuda::kernel::cudnn::attention::build_mha_fwd_graph;
use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let system = ActorSystem::create("atomr-accel-mha-bf16", Config::empty()).await?;
    let _device = system.actor_of(
        DeviceActor::props(
            DeviceConfig::new(0).with_libraries(EnabledLibraries::CUDNN),
        ),
        "device-0",
    )?;

    // GQA: 32 query heads, 8 KV heads, head_dim = 128, seq = 2048,
    // causal mask, dropout disabled.
    let p = AttentionParams::new(2, 2048, 2048, 32, 8, 128).with_mask(AttentionMask::Causal);
    let g = build_mha_fwd_graph(
        atomr_accel_cuda::kernel::cudnn::DtypeTag::Bf16,
        &p,
        TensorLayout::NchwPacked,
    );
    println!(
        "MHA fwd: tensors = {}, ops = {} (QK^T + softmax + S*V), is_gqa = {}",
        g.tensors.len(),
        g.ops.len(),
        p.is_gqa(),
    );

    system.terminate().await;
    Ok(())
}
