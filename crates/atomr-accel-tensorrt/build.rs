//! Build script for `atomr-accel-tensorrt`.
//!
//! When the `tensorrt-link` feature is **off** (the default), this
//! script is a no-op so the crate builds on hosts without TensorRT
//! installed. When the feature is **on**, the script probes for
//! `libnvinfer.so` in this order:
//!
//! 1. `$LIBNVINFER_PATH` — explicit override; may be a directory or a
//!    full path to `libnvinfer.so`.
//! 2. `/usr/lib/x86_64-linux-gnu` — Debian/Ubuntu system path.
//! 3. `/usr/local/cuda/lib64` — Common CUDA toolkit install.
//! 4. `/usr/local/lib` — Linux generic prefix.
//!
//! If none match, the script panics with a clear "set LIBNVINFER_PATH
//! or install TensorRT" message. Real linking only happens through this
//! gate; with the feature off, the crate's `sys.rs` exposes opaque
//! types and stub function signatures that never get linked.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBNVINFER_PATH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TENSORRT_LINK");

    let link_enabled = env::var_os("CARGO_FEATURE_TENSORRT_LINK").is_some();
    if !link_enabled {
        // Feature off → do nothing. The `sys` module compiles to a set
        // of opaque types and never references libnvinfer symbols, so
        // no link probe is needed.
        return;
    }

    let onnx_enabled = env::var_os("CARGO_FEATURE_TENSORRT_ONNX").is_some();

    let probe = probe_libnvinfer().unwrap_or_else(|| {
        panic!(
            "atomr-accel-tensorrt: libnvinfer.so not found.\n\
             Set LIBNVINFER_PATH (directory or full path), or install TensorRT \
             from https://developer.nvidia.com/tensorrt and ensure libnvinfer.so \
             is on the system library path. Probed: $LIBNVINFER_PATH, \
             /usr/lib/x86_64-linux-gnu, /usr/local/cuda/lib64, /usr/local/lib."
        )
    });

    println!("cargo:rustc-link-search=native={}", probe.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    if onnx_enabled {
        println!("cargo:rustc-link-lib=dylib=nvonnxparser");
    }
}

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
