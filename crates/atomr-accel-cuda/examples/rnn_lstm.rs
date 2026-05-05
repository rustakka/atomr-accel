//! `rnn_lstm` — Phase 2 cuDNN demo: build an LSTM forward request
//! spec for plan-cache keying.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example rnn_lstm \
//!         --features cuda-runtime-tests,cudnn,f16

use atomr_accel_cuda::kernel::cudnn::rnn::build_rnn_fwd_spec;
use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let system = ActorSystem::create("atomr-accel-rnn-lstm", Config::empty()).await?;
    let _device = system.actor_of(
        DeviceActor::props(
            DeviceConfig::new(0).with_libraries(EnabledLibraries::CUDNN),
        ),
        "device-0",
    )?;

    let p = RnnParams::new(
        RnnMode::Lstm,
        RnnDirection::Bidirectional,
        2,
        512,
        1024,
        128,
        16,
    );
    let g = build_rnn_fwd_spec(
        atomr_accel_cuda::kernel::cudnn::DtypeTag::F16,
        &p,
        TensorLayout::NchwPacked,
    );
    println!(
        "lstm tensors = {} (incl. cell-state pair), output_size = {}",
        g.tensors.len(),
        p.output_size()
    );

    system.terminate().await;
    Ok(())
}
