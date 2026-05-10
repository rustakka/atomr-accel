//! Build script for `atomr-accel-tensorrt`.
//!
//! With the `tensorrt-link` feature **off** (the default) this script
//! is a no-op so the crate builds on hosts without TensorRT installed.
//!
//! With the feature **on**, the script:
//!   1. probes for libnvinfer + the TRT headers + the CUDA runtime
//!      headers via `LIBNVINFER_PATH` / `TENSORRT_INCLUDE_PATH` /
//!      `CUDA_PATH` (with sensible defaults under `/usr` and
//!      `/usr/local/cuda`),
//!   2. compiles `csrc/nvinfer_shim.cpp` (and `csrc/onnx_parser.cpp`
//!      under `tensorrt-onnx`, `csrc/plugin_proxy.cpp` under
//!      `tensorrt-plugin`) into a static `atomr_trt_shim` library,
//!   3. emits `cargo:rustc-link-lib=dylib=nvinfer` and friends so the
//!      final binary resolves the shim against libnvinfer.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=LIBNVINFER_PATH");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TENSORRT_LINK");

    if env::var_os("CARGO_FEATURE_TENSORRT_LINK").is_none() {
        return;
    }

    let lib_dir = probe_libnvinfer().unwrap_or_else(|| {
        panic!(
            "atomr-accel-tensorrt: tensorrt-link enabled but libnvinfer not found. \
             Install TensorRT 10.x (`apt install libnvinfer-dev`) or set LIBNVINFER_PATH \
             to the directory containing libnvinfer.so."
        )
    });
    let include_dir = probe_tensorrt_include().unwrap_or_else(|| {
        panic!(
            "atomr-accel-tensorrt: tensorrt-link enabled but TensorRT headers not found. \
             Install libnvinfer-headers-dev or set TENSORRT_INCLUDE_PATH to the directory \
             containing NvInfer.h."
        )
    });
    let cuda_inc = probe_cuda_include().unwrap_or_else(|| {
        panic!(
            "atomr-accel-tensorrt: tensorrt-link enabled but CUDA runtime headers not found. \
             Install the CUDA toolkit or set CUDA_PATH."
        )
    });

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("csrc/nvinfer_shim.cpp")
        .include(&include_dir)
        .include(&cuda_inc)
        .include("csrc")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-Wno-unused-parameter");

    if env::var_os("CARGO_FEATURE_TENSORRT_ONNX").is_some() {
        let onnx = Path::new("csrc/onnx_parser.cpp");
        if onnx.exists() {
            build.file(onnx).define("ATOMR_TRT_ONNX", None);
        }
    }
    if env::var_os("CARGO_FEATURE_TENSORRT_PLUGIN").is_some() {
        let plugin = Path::new("csrc/plugin_proxy.cpp");
        if plugin.exists() {
            build.file(plugin).define("ATOMR_TRT_PLUGIN", None);
        }
    }

    build.compile("atomr_trt_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    if env::var_os("CARGO_FEATURE_TENSORRT_ONNX").is_some() {
        println!("cargo:rustc-link-lib=dylib=nvonnxparser");
    }
    if env::var_os("CARGO_FEATURE_TENSORRT_PLUGIN").is_some() {
        println!("cargo:rustc-link-lib=dylib=nvinfer_plugin");
    }
    // libstdc++ for the C++ glue. cc::Build handles this on most Linux
    // hosts but emit it explicitly for forward-compat with stricter
    // toolchains (Alpine / musl).
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn probe_libnvinfer() -> Option<PathBuf> {
    if let Ok(env_path) = env::var("LIBNVINFER_PATH") {
        let p = PathBuf::from(&env_path);
        if dir_contains_libnvinfer(&p) {
            return Some(p);
        }
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

fn probe_tensorrt_include() -> Option<PathBuf> {
    if let Ok(p) = env::var("TENSORRT_INCLUDE_PATH") {
        let pb = PathBuf::from(&p);
        if pb.join("NvInfer.h").is_file() {
            return Some(pb);
        }
    }
    for candidate in [
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        "/opt/tensorrt/include",
        "/usr/local/cuda/include",
    ] {
        let pb = PathBuf::from(candidate);
        if pb.join("NvInfer.h").is_file() {
            return Some(pb);
        }
    }
    None
}

fn probe_cuda_include() -> Option<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        let pb = PathBuf::from(&p).join("include");
        if pb.join("cuda_runtime.h").is_file() {
            return Some(pb);
        }
    }
    for candidate in [
        "/usr/local/cuda/include",
        "/usr/include",
        "/usr/include/cuda",
    ] {
        let pb = PathBuf::from(candidate);
        if pb.join("cuda_runtime.h").is_file() {
            return Some(pb);
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
