//! `layernorm` — Phase 2 cuDNN demo: builds a `LayerNormRequest<f16>`
//! op-graph spec and prints the cache key.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example layernorm \
//!         --features cuda-runtime-tests,cudnn,f16

use atomr_accel_cuda::kernel::cudnn::norm::build_norm_fwd_graph;
use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let system = ActorSystem::create("atomr-accel-layernorm", Config::empty()).await?;
    let _device = system.actor_of(
        DeviceActor::props(DeviceConfig::new(0).with_libraries(EnabledLibraries::CUDNN)),
        "device-0",
    )?;

    let graph = build_norm_fwd_graph(
        NormMode::LayerNorm,
        NormPhase::Training,
        atomr_accel_cuda::kernel::cudnn::DtypeTag::F16,
        &[8, 4096, 1, 1],
        TensorLayout::NchwPacked,
        1e-5,
        0.0,
    );
    let key = atomr_accel_cuda::kernel::cudnn::cache_key(
        "layernorm",
        atomr_accel_cuda::kernel::cudnn::DtypeTag::F16,
        &graph,
    );
    println!("layernorm cache key signature = {:#x}", key.signature);

    system.terminate().await;
    Ok(())
}
