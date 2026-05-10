//! `atomr-accel-cutlass` build script.
//!
//! Strategy A (default): no-op. The vendored CUTLASS headers are
//! resolved at runtime by `NvrtcActor` via `--include-path=...`; we
//! emit `cargo:rerun-if-changed` lines so cargo invalidates the build
//! when a header changes.
//!
//! Strategy B (`cutlass-prebuilt` feature): probes `nvcc`, generates
//! a `.cu` per cell in [`prebuilt::CELLS`], compiles each into an
//! object file, then bundles them into `libatomr_cutlass_prebuilt.a`
//! via `nvcc --lib`. Emits link directives + sets the
//! `cutlass_prebuilt_active` cfg flag so the dispatcher can skip the
//! NVRTC compile path on a hit. Falls back gracefully (warning, no
//! cfg flag) when nvcc is missing — the runtime NVRTC path still
//! works.
//!
//! Phase 6.1 ships ONE canonical cell (`fp32 GEMM, RowMajor x ColMajor
//! → RowMajor, sm_80, no epilogue`) as proof-of-infrastructure.
//! Adding more cells is a matter of extending [`prebuilt::CELLS`];
//! the rest of this file is dimension-agnostic.

use std::path::{Path, PathBuf};
use std::process::Command;

mod prebuilt {
    /// One per (template, shape, dtype, layout, arch) prebuilt
    /// instantiation. Source is generated inline rather than
    /// templated through the runtime emitter so the build script
    /// stays free of cross-module Rust code dependencies.
    pub struct Cell {
        pub kernel_name: &'static str,
        pub arch: &'static str,
        pub source: &'static str,
    }

    pub const CELLS: &[Cell] = &[Cell {
        kernel_name: "atomr_cutlass_gemm_fp32_sm80_canonical",
        arch: "sm_80",
        // The body intentionally avoids the full CUTLASS GEMM
        // template (multi-minute compile) for Phase 6.1; it is a
        // typed `__global__` placeholder that the dispatcher never
        // actually dispatches to. The point is to prove that nvcc +
        // cc::Build wiring produces a usable static lib.
        source: r#"
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

extern "C" __global__ void atomr_cutlass_gemm_fp32_sm80_canonical(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k)
{
    // Phase 6.1 placeholder: a one-thread no-op so the symbol
    // exists in the static lib. Phase 6.2 swaps in the real
    // CUTLASS template instantiation.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Deliberately empty.
        (void)A; (void)B; (void)C; (void)m; (void)n; (void)k;
    }
}
"#,
    }];
}

fn main() {
    let header_dir = Path::new("cutlass/include");
    println!("cargo:rerun-if-changed=cutlass/include");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    // Declare the cfg flag so downstream `cfg!(cutlass_prebuilt_active)`
    // checks don't trip the `unexpected_cfgs` lint.
    println!("cargo:rustc-check-cfg=cfg(cutlass_prebuilt_active)");

    if header_dir.is_dir() {
        if let Ok(canon) = header_dir.canonicalize() {
            println!("cargo:include={}", canon.display());
        }
    }

    let prebuilt_requested = std::env::var_os("CARGO_FEATURE_CUTLASS_PREBUILT").is_some();
    if !prebuilt_requested {
        return;
    }

    let nvcc = match locate_nvcc() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=atomr-accel-cutlass: cutlass-prebuilt feature requested \
                 but nvcc not found on PATH or under $CUDA_PATH / $CUDA_HOME. \
                 Falling back to Strategy A (NVRTC at runtime)."
            );
            return;
        }
    };

    if let Err(e) = build_prebuilt_lib(&nvcc, header_dir) {
        println!(
            "cargo:warning=atomr-accel-cutlass: cutlass-prebuilt build failed ({e}); \
             falling back to Strategy A (NVRTC at runtime)."
        );
        return;
    }

    println!("cargo:rustc-cfg=cutlass_prebuilt_active");
}

fn build_prebuilt_lib(nvcc: &Path, header_dir: &Path) -> Result<(), String> {
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set".to_string())?);
    let header_inc = header_dir
        .canonicalize()
        .map_err(|e| format!("canonicalize cutlass/include: {e}"))?;

    let mut object_paths: Vec<PathBuf> = Vec::new();
    for cell in prebuilt::CELLS {
        let src_path = out_dir.join(format!("{}.cu", cell.kernel_name));
        std::fs::write(&src_path, cell.source.trim_start())
            .map_err(|e| format!("write {}: {}", src_path.display(), e))?;

        let obj_path = out_dir.join(format!("{}.o", cell.kernel_name));
        let arch_flag = format!("--gpu-architecture={}", cell.arch);
        let status = Command::new(nvcc)
            .args([
                "-c",
                "--std=c++17",
                "-O3",
                "-Xcompiler=-fPIC",
                &arch_flag,
                "--include-path",
                header_inc
                    .to_str()
                    .ok_or_else(|| "non-utf8 cutlass include path".to_string())?,
            ])
            .arg(&src_path)
            .arg("-o")
            .arg(&obj_path)
            .status()
            .map_err(|e| format!("invoke nvcc: {e}"))?;
        if !status.success() {
            return Err(format!("nvcc exit {} for {}", status, cell.kernel_name));
        }
        object_paths.push(obj_path);
    }

    let lib_path = out_dir.join("libatomr_cutlass_prebuilt.a");
    let mut lib_cmd = Command::new(nvcc);
    lib_cmd.args(["--lib", "-o"]).arg(&lib_path);
    for obj in &object_paths {
        lib_cmd.arg(obj);
    }
    let status = lib_cmd
        .status()
        .map_err(|e| format!("invoke nvcc --lib: {e}"))?;
    if !status.success() {
        return Err(format!("nvcc --lib exit {status}"));
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // CUDA runtime library lives next to nvcc — emit its parent's
    // `lib64` so the linker can resolve `-lcudart` / `-lcudart_static`.
    if let Some(cuda_root) = nvcc.parent().and_then(Path::parent) {
        let lib64 = cuda_root.join("lib64");
        if lib64.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
        }
        let targets_lib = cuda_root.join("targets/x86_64-linux/lib");
        if targets_lib.is_dir() {
            println!("cargo:rustc-link-search=native={}", targets_lib.display());
        }
    }
    println!("cargo:rustc-link-lib=static=atomr_cutlass_prebuilt");
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}

fn locate_nvcc() -> Option<PathBuf> {
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = std::env::var(var) {
            let candidate = PathBuf::from(root).join("bin").join(nvcc_name());
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let fallback = PathBuf::from("/usr/local/cuda/bin").join(nvcc_name());
    if fallback.is_file() {
        return Some(fallback);
    }
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
