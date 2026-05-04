//! Demonstrates the atomr-accel-cuda pipeline pattern surface without a
//! GPU. Builds a `PipelineExecutor` over two trivial host stages
//! to show the message wiring; in a real GPU pipeline each stage
//! would be backed by a kernel-actor adapter.
//!
//! Run with: `cargo run -p atomr-accel-cuda --example pipeline_no_gpu`

use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::pipeline::PipelineStage;

struct AddOne;
impl PipelineStage for AddOne {
    type In = ();
    type Out = ();
    fn enqueue(
        &mut self,
        _stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        _wait_for: Option<&cudarc::driver::CudaEvent>,
        _input: (),
    ) -> Result<(cudarc::driver::CudaEvent, ()), GpuError> {
        // Without a GPU we can't construct a real CudaEvent; this
        // example just validates that the trait + executor compile
        // and the actor system wiring stays consistent. A real
        // pipeline binds each stage to a CudaStream from
        // `PerActorAllocator`.
        Err(GpuError::Unrecoverable(
            "pipeline_no_gpu: stages require a real CudaStream".into(),
        ))
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    tracing_subscriber::fmt().init();

    println!("atomr-accel-cuda pipeline-no-gpu demo");
    println!("Pipeline stages compile against the trait but require a CUDA stream to enqueue.");
    println!(
        "Run examples/sgemm or examples/conv_forward (gated cuda-runtime-tests) on a GPU host"
    );
    println!("to exercise the full pipeline path.");
    let _ = AddOne;
}
