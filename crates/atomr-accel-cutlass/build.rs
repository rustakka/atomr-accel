//! `atomr-accel-cutlass` build script.
//!
//! Strategy A (default): no-op. The vendored CUTLASS headers are
//! resolved at runtime by `NvrtcActor` via `--include-path=...`; we
//! emit `cargo:rerun-if-changed` lines so cargo invalidates the build
//! when a header changes.
//!
//! Strategy B (`cutlass-prebuilt` feature): probe for `nvcc` and run a
//! small generator that emits a static library of pre-instantiated
//! kernels. The generator itself is a follow-up — this script
//! currently:
//!
//!   * detects nvcc on `$CUDA_PATH/bin` or `/usr/local/cuda/bin`,
//!   * prints `cargo:warning=` if nvcc is missing so the build
//!     surfaces the contract requirement,
//!   * emits `cargo:rustc-cfg=cutlass_prebuilt_active` when nvcc is
//!     found so downstream code can `cfg!` against actual prebuilt
//!     availability (separate from the cargo feature gate, which only
//!     opts in to the *intent*).
//!
//! Once the generator lands, the same probe drives `cc::Build`-style
//! compilation of the emitted `.cu` files into a `staticlib` linked
//! into the crate.

use std::path::Path;

fn main() {
    // Track header changes so cargo re-runs the build script when
    // the vendored CUTLASS subset gets bumped.
    let header_dir = Path::new("cutlass/include");
    println!("cargo:rerun-if-changed=cutlass/include");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // Strategy A is the default path. We still expose the include
    // directory to dependents so `atomr-accel-cuda::nvrtc` can pick
    // it up via DEP_ATOMR_ACCEL_CUTLASS_INCLUDE without hard-coding
    // the relative path.
    if header_dir.is_dir() {
        if let Ok(canon) = header_dir.canonicalize() {
            println!("cargo:include={}", canon.display());
        }
    }

    let prebuilt_requested = std::env::var_os("CARGO_FEATURE_CUTLASS_PREBUILT").is_some();

    if !prebuilt_requested {
        // Strategy A — nothing more to do.
        return;
    }

    // Strategy B path. Probe for nvcc.
    let nvcc = locate_nvcc();
    match nvcc {
        Some(path) => {
            println!(
                "cargo:warning=atomr-accel-cutlass: cutlass-prebuilt feature enabled \
                 — nvcc found at {} (generator stub: prebuilt static lib not yet emitted)",
                path.display()
            );
            println!("cargo:rustc-cfg=cutlass_prebuilt_active");
        }
        None => {
            println!(
                "cargo:warning=atomr-accel-cutlass: cutlass-prebuilt feature requested \
                 but nvcc not found on PATH or under $CUDA_PATH / $CUDA_HOME. \
                 Falling back to Strategy A (NVRTC at runtime)."
            );
        }
    }
}

fn locate_nvcc() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    // 1. Explicit override via CUDA_PATH / CUDA_HOME.
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = std::env::var(var) {
            let candidate = PathBuf::from(root).join("bin").join(nvcc_name());
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    // 2. /usr/local/cuda fallback.
    let fallback = PathBuf::from("/usr/local/cuda/bin").join(nvcc_name());
    if fallback.is_file() {
        return Some(fallback);
    }
    // 3. PATH lookup.
    if let Ok(path) = std::env::var("PATH") {
        for entry in std::env::split_paths(&path) {
            let candidate = entry.join(nvcc_name());
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(windows)]
fn nvcc_name() -> &'static str {
    "nvcc.exe"
}

#[cfg(not(windows))]
fn nvcc_name() -> &'static str {
    "nvcc"
}
