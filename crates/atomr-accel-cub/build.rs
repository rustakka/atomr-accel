//! `atomr-accel-cub` build script.
//!
//! Discovers the CUDA toolkit include directory at build time and
//! exposes two `rustc-env=` variables consumed by
//! `crates/atomr-accel-cub/src/dispatch.rs` when assembling the NVRTC
//! `--include-path=…` flags. CUB is header-only and ships with the
//! CUDA toolkit; we just need to know where the headers live.
//!
//! ## Probe order
//!
//! 1. `CUDA_PATH` env var (`{root}/include`)
//! 2. `CUDA_HOME` env var (`{root}/include`)
//! 3. `/usr/local/cuda/include`
//! 4. PATH-relative `nvcc` (resolved up two levels — `…/bin/nvcc` →
//!    `…/include`)
//!
//! ## CUDA 12 vs CUDA 13 layout
//!
//! CUDA 12 puts CUB at `<root>/include/cub/cub.cuh`. CUDA 13
//! reorganized the CCCL libraries under `<root>/include/cccl/`, so CUB
//! is at `<root>/include/cccl/cub/cub.cuh`. Both directories appear on
//! the include path so the vendored `atomr_cub_kernels.cuh`'s
//! `#include <cub/...>` lines resolve regardless of toolkit version.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let cuda_root = locate_cuda();
    if let Some(root) = cuda_root.as_ref() {
        let inc = root.join("include");
        if inc.is_dir() {
            println!("cargo:rustc-env=ATOMR_CUB_CUDA_INCLUDE={}", inc.display());
        }
        // CUDA 13+ moved CCCL (CUB / Thrust / libcu++) into a sub-tree.
        let cccl = inc.join("cccl");
        if cccl.is_dir() {
            println!("cargo:rustc-env=ATOMR_CUB_CCCL_INCLUDE={}", cccl.display());
        }
    } else {
        println!(
            "cargo:warning=atomr-accel-cub: CUDA toolkit not found via CUDA_PATH / CUDA_HOME / \
             /usr/local/cuda / nvcc on PATH. CUB kernel compiles will fail at runtime; install \
             the CUDA toolkit (12.0+) or set CUDA_PATH."
        );
    }

    let crate_inc = env::current_dir()
        .ok()
        .map(|p| p.join("include").join("cub_kernels"))
        .filter(|p| p.is_dir());
    if let Some(p) = crate_inc {
        println!("cargo:rustc-env=ATOMR_CUB_INCLUDE={}", p.display());
    }
}

fn locate_cuda() -> Option<PathBuf> {
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(p) = env::var(var) {
            let pb = PathBuf::from(&p);
            if pb.is_dir() {
                return Some(pb);
            }
        }
    }
    let fallback = PathBuf::from("/usr/local/cuda");
    if fallback.is_dir() {
        return Some(fallback);
    }
    if let Ok(path) = env::var("PATH") {
        for entry in env::split_paths(&path) {
            let nvcc = entry.join("nvcc");
            if nvcc.is_file() {
                // …/bin/nvcc → take the toolkit root two levels up.
                if let Some(root) = nvcc.parent().and_then(Path::parent) {
                    return Some(root.to_path_buf());
                }
            }
        }
    }
    None
}
