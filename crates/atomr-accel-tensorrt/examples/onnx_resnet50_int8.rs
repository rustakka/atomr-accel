//! End-to-end skeleton: import a ResNet-50 ONNX file, run INT8 PTQ,
//! and emit an engine plan. Runs without a GPU under
//! `--features=tensorrt-onnx,tensorrt-int8` (no `tensorrt-link`); the
//! actual `enqueue` step needs `tensorrt-link` plus a TensorRT install.

use std::sync::Arc;

use atomr_accel_tensorrt::calibration::{CalibrationBinding, MinMaxCalibrator};
use atomr_accel_tensorrt::onnx::OnnxMsg;
use atomr_accel_tensorrt::{IBuilderConfig, NetworkSource, Precision, TrtActor, TrtMsg};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Pretend we read a ResNet-50 ONNX file off disk. In a real
    // pipeline this is `tokio::fs::read("resnet50.onnx").await?`.
    let onnx_bytes: Arc<Vec<u8>> = Arc::new(b"<-- onnx blob would go here -->".to_vec());

    // Build a calibrator that yields N batches of 224x224x3 images.
    // For the demo we just emit empty placeholder bindings; the
    // production code wires `device_ptr` to a real CUDA buffer.
    let batches = (0..16usize)
        .map(|i| {
            vec![CalibrationBinding {
                name: "input".into(),
                device_ptr: 0xCAFE_0000 + i as u64,
                bytes: 224 * 224 * 3 * 4,
            }]
        })
        .collect();
    let _calibrator = MinMaxCalibrator::new(batches);

    // Construct the build config: INT8 + sparsity + a 2 GiB workspace.
    let config = IBuilderConfig::new()
        .with_precision(Precision::Int8)
        .with_sparsity(true)
        .with_workspace_bytes(2 << 30);

    // Build via TrtMsg::Build — host-side construction only, no GPU
    // call, no libnvinfer dep.
    let actor = TrtActor::new();
    let (tx, rx) = oneshot::channel();
    let _build_msg = TrtMsg::Build {
        source: NetworkSource::Onnx(onnx_bytes.as_ref().clone()),
        config: Box::new(config.clone()),
        reply: tx,
    };

    // The OnnxMsg::Parse variant is the same shape via the dedicated
    // ONNX parser actor — useful when the parser lives on its own
    // tokio task.
    let (otx, _orx) = oneshot::channel();
    let _onnx_msg = OnnxMsg::Parse {
        bytes: onnx_bytes.clone(),
        config: Box::new(config),
        reply: otx,
    };

    // Without `tensorrt-link`, `actor.ensure_runtime()` reports
    // `NotLinked` and the example exits gracefully. Real builds
    // would `actor.handle(...)` the message and wait on `rx`.
    if let Err(e) = actor.ensure_runtime() {
        tracing::info!(error = %e, "TensorRT not linked: skipping build");
        drop(rx);
        println!("onnx_resnet50_int8 demo wired up; run with --features tensorrt-link plus a real .onnx file for end-to-end build");
        return;
    }
    drop(rx);

    #[cfg(all(feature = "tensorrt-link", feature = "tensorrt-onnx"))]
    {
        // The placeholder bytes won't parse — TRT will report a
        // protobuf decode error. Demonstrates the round-trip wiring:
        // `actor.build_from_onnx` returns a structured `TrtError`
        // rather than panicking, proving the libnvonnxparser link
        // chain is intact.
        let plan = actor.build_from_onnx(onnx_bytes.as_ref(), &IBuilderConfig::new());
        match plan {
            Ok(p) => println!("plan bytes: {}", p.as_slice().len()),
            Err(e) => println!("expected ONNX parse failure on placeholder bytes: {e}"),
        }
    }
}
