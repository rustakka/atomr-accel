//! Smoke test for the `tensorrt-link` build path. Compiles only when
//! the `tensorrt-link` feature is enabled (otherwise the `extern`
//! symbols don't exist). Verifies the C++ shim links cleanly against
//! libnvinfer by creating a `TrtRuntime`, installing the logger, and
//! tearing down.
//!
//! Run on a TRT-equipped host:
//!
//! ```text
//! LIBNVINFER_PATH=/usr/lib/x86_64-linux-gnu \
//! TENSORRT_INCLUDE_PATH=/usr/include/x86_64-linux-gnu \
//! CUDA_PATH=/usr/local/cuda \
//! cargo test -p atomr-accel-tensorrt \
//!     --features tensorrt-link,cuda-runtime-tests \
//!     -- --ignored --nocapture
//! ```

#![cfg(all(feature = "tensorrt-link", feature = "cuda-runtime-tests"))]

use atomr_accel_tensorrt::TrtRuntime;

#[test]
#[ignore = "requires libnvinfer at runtime"]
fn create_and_drop_runtime() {
    atomr_accel_tensorrt::init_logger();
    atomr_accel_tensorrt::init_logger(); // idempotent
    let _rt = TrtRuntime::new().expect("TrtRuntime::new");
    // Drop ends the test.
}
