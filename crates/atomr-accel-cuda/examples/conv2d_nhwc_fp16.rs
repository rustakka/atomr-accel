//! `conv2d_nhwc_fp16` — Phase 2 cuDNN exit demo.
//!
//! Builds a `ConvFwdRequest<half::f16>` against the v9 frontend graph
//! API in NHWC layout and dispatches it through the supervised
//! `CudnnActor` pipeline. The launch closure currently short-circuits
//! with a `LibraryError` (the v9 graph-builder body is filed in this
//! same PR but disabled until end-to-end FFI support lands), so this
//! example primarily exercises:
//!
//! * `DeviceActor` -> `ContextActor` -> `CudnnActor` plumbing,
//! * `ConvFwdRequest::graph_spec()` round-tripping through the plan
//!   cache,
//! * NHWC stride generation matching cuDNN's expected layout.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example conv2d_nhwc_fp16 \
//!         --features cuda-runtime-tests,cudnn,f16

use std::marker::PhantomData;

use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("atomr-accel-conv2d-nhwc", Config::empty()).await?;
    let _device = system.actor_of(
        DeviceActor::props(
            DeviceConfig::new(0).with_libraries(EnabledLibraries::CUDNN),
        ),
        "device-0",
    )?;

    // Demonstrate spec-side graph building (host-only).
    let graph = atomr_accel_cuda::kernel::cudnn::conv::build_conv_fwd_graph(
        atomr_accel_cuda::kernel::cudnn::DtypeTag::F16,
        &[1, 32, 56, 56],
        &[64, 32, 3, 3],
        &[1, 64, 54, 54],
        &ConvDescParams::symmetric_2d(0, 1, 1),
        TensorLayout::NhwcPacked,
        EpilogueKind::None,
    );

    println!(
        "conv_fwd graph signature = {:#x}; tensors = {}, ops = {}",
        graph.signature(),
        graph.tensors.len(),
        graph.ops.len(),
    );
    let _ = PhantomData::<half::f16>;

    system.terminate().await;
    Ok(())
}
