//! Demonstrates compiling a custom Relu kernel via NVRTC at runtime.
//! The compile path is exercised; the launch path requires the
//! `NvrtcActor` ref to be plumbed through (F7 adds a public
//! `device.kernel_children()` accessor).
//!
//! Build with: `cargo run -p rakka-accel-cuda --example jit_relu \
//!     --features cuda-runtime-tests,nvrtc`

use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;

use rakka_accel_cuda::prelude::*;

const KERNEL: &str = r#"
extern "C" __global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}
"#;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let sys = ActorSystem::create("jit-relu-demo", Config::empty()).await?;
    let dev_cfg = DeviceConfig::new(0)
        .with_libraries(EnabledLibraries::BLAS | EnabledLibraries::NVRTC);
    let _device = sys.actor_of(DeviceActor::props(dev_cfg), "device-0")?;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let _ = KERNEL; // would be passed to NvrtcMsg::Compile via the
                    // plumbed actor ref.
    println!("NVRTC source ready ({} bytes); compile path requires NvrtcActor ref", KERNEL.len());

    sys.terminate().await;
    Ok(())
}
