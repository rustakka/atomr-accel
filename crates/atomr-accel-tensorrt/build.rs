//! Build script for `atomr-accel-tensorrt`.
//!
//! With the `tensorrt-link` feature **off** (the default) this script
//! is a no-op so the crate builds on hosts without TensorRT installed.
//!
//! With the feature **on**, the script would normally probe for
//! `libnvinfer.so` and emit `cargo:rustc-link-lib=dylib=nvinfer` (and
//! `nvonnxparser` when `tensorrt-onnx` is also on). That code path is
//! currently disabled: the `atomr_trt_*` C-ABI shim symbols declared
//! in `src/sys.rs` have no implementation yet (the hand-written
//! `nvinfer_shim.cpp` has not landed), so emitting link directives
//! would only produce an opaque "undefined reference" linker error.
//! The real diagnostic is the `compile_error!` in `src/lib.rs`, which
//! fires before any link step. Tracked by
//! <https://github.com/rustakka/atomr-accel/issues/6>; the probe
//! helpers below are kept verbatim so re-enabling the feature in the
//! shim PR is a one-line change.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBNVINFER_PATH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TENSORRT_LINK");

    // Both branches are intentionally a no-op while issue #6 is open.
    // The `compile_error!` in `src/lib.rs` carries the user-facing
    // message; a build-script panic here would only race that error
    // with a misleading "set LIBNVINFER_PATH" hint.
}

#[allow(dead_code)]
fn probe_libnvinfer() -> Option<PathBuf> {
    if let Ok(env_path) = env::var("LIBNVINFER_PATH") {
        let p = PathBuf::from(&env_path);
        if dir_contains_libnvinfer(&p) {
            return Some(p);
        }
        // If a full path to libnvinfer.so was given, return its parent.
        if p.is_file()
            && p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libnvinfer"))
                .unwrap_or(false)
        {
            return p.parent().map(Path::to_path_buf);
        }
    }

    for candidate in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
    ] {
        let p = PathBuf::from(candidate);
        if dir_contains_libnvinfer(&p) {
            return Some(p);
        }
    }

    None
}

#[allow(dead_code)]
fn dir_contains_libnvinfer(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in rd.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if name.starts_with("libnvinfer") && name.contains(".so") {
                return true;
            }
        }
    }
    false
}
