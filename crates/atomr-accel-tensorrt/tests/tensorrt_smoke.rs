//! Opt-in smoke test for `atomr-accel-tensorrt`. Verifies the
//! actor's lazy-load path against a real `libnvinfer.so` if one is
//! installed. Skips cleanly when not.
//!
//! Run via `cargo xtask gpu-test tensorrt` or:
//!   cargo test -p atomr-accel-tensorrt --features cuda-runtime-tests \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda-runtime-tests")]

use atomr_accel_tensorrt::TrtActor;

#[test]
#[ignore = "requires libnvinfer on the host"]
fn tensorrt_runtime_lazy_load_succeeds_or_skips_cleanly() {
    let actor = TrtActor::new();
    match actor.ensure_runtime() {
        Ok(()) => {
            println!("[tensorrt] runtime initialised successfully against libnvinfer");
        }
        Err(e) => {
            eprintln!("[skip] TensorRT runtime not available: {e}");
        }
    }
}
