//! `xtask` — developer tooling for the atomr-accel workspace.
//!
//! Subcommands:
//! * `bump <patch|minor|major>` — bump the workspace version, refresh
//!   `Cargo.lock`, mirror to `crates/atomr-accel-py/pyproject.toml`,
//!   rewrite internal-path-dep version pins inside
//!   `[workspace.dependencies]`, and rewrite per-crate inline
//!   `path = "../atomr-accel-*"`-with-`version = "..."` pins so
//!   sibling-crate deps don't drift.
//! * `bump --pre <id>` / `bump --set <ver>` — variants for
//!   pre-release tags and exact-version overrides.
//! * `verify` — local mirror of the release-pipeline gate:
//!   `cargo fmt --check` → `cargo clippy -D warnings` →
//!   `cargo test --workspace --no-default-features` →
//!   `cargo check --workspace --features atomr-accel-cuda/full-cuda`.
//!
//! The pattern is borrowed from the sibling atomr workspace's xtask.

use std::env;
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "bump" => bump(args.collect()),
        "verify" => verify(),
        "gpu-probe" => gpu_probe(),
        "gpu-test" => gpu_test(args.collect()),
        "gpu-bench" => gpu_bench(args.collect()),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(anyhow!("unknown xtask subcommand: {other}")),
    }
}

fn print_help() {
    println!("atomr-accel xtask");
    println!();
    println!("USAGE:");
    println!("  cargo xtask <subcommand>");
    println!();
    println!("SUBCOMMANDS:");
    println!(
        "  bump <patch|minor|major>      bump workspace version + python version + Cargo.lock"
    );
    println!("  bump --pre <id>               append a pre-release identifier (e.g. rc.1)");
    println!(
        "  bump --set <version>          set an exact version (used by Release-As: overrides)"
    );
    println!("  verify                        run fmt + clippy + test (release-pipeline gate)");
    println!();
    println!("GPU INTEGRATION (opt-in, not part of CI):");
    println!("  gpu-probe                     report CUDA availability + device list");
    println!("  gpu-test [SUITE...]           run GPU integration tests against the local driver");
    println!("                                  SUITE in: cublas, cublaslt, cudnn, cufft, curand,");
    println!(
        "                                            cusolver, cusparse, cutensor, nccl, nvrtc,"
    );
    println!(
        "                                            graph, event, memory, cub, cutlass, flashattn,"
    );
    println!("                                            tensorrt, telemetry, all (default)");
    println!("  gpu-bench [BENCH...]          run criterion GPU benches (perf regression)");
    println!();
    println!("  help                          print this help");
}

// ─── bump ───────────────────────────────────────────────────────────────

fn bump(args: Vec<String>) -> Result<()> {
    let mut iter = args.into_iter();
    let arg = iter.next().ok_or_else(|| {
        anyhow!("usage: bump <patch|minor|major> | bump --pre <id> | bump --set <version>")
    })?;
    let cargo_toml = Path::new("Cargo.toml");
    let pyproject = Path::new("crates/atomr-accel-py/pyproject.toml");
    let current = read_workspace_version(cargo_toml)?;
    let next = match arg.as_str() {
        "patch" => semver_bump(&current, BumpKind::Patch)?,
        "minor" => semver_bump(&current, BumpKind::Minor)?,
        "major" => semver_bump(&current, BumpKind::Major)?,
        "--pre" => {
            let id = iter.next().ok_or_else(|| anyhow!("--pre requires <id>"))?;
            semver_bump(&current, BumpKind::Pre(id))?
        }
        "--set" => iter
            .next()
            .ok_or_else(|| anyhow!("--set requires <version>"))?,
        other => return Err(anyhow!("unknown bump arg: {other}")),
    };
    println!("{} -> {}", current, next);
    write_workspace_version(cargo_toml, &next)?;
    write_workspace_deps_versions(cargo_toml, &current, &next)?;
    write_member_inline_deps_versions(&current, &next)?;
    if pyproject.exists() {
        write_pyproject_version(pyproject, &next)?;
    }
    // Refresh Cargo.lock so the workspace builds against the new version.
    let _ = Command::new(env!("CARGO"))
        .args(["update", "--workspace"])
        .status();
    println!("ATOMR_ACCEL_NEW_VERSION={next}");
    Ok(())
}

#[derive(Debug)]
enum BumpKind {
    Patch,
    Minor,
    Major,
    Pre(String),
}

fn semver_bump(current: &str, kind: BumpKind) -> Result<String> {
    let (core, _pre) = match current.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (current, None),
    };
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() != 3 {
        return Err(anyhow!("version `{current}` is not MAJOR.MINOR.PATCH"));
    }
    let mut major: u64 = parts[0].parse().context("major")?;
    let mut minor: u64 = parts[1].parse().context("minor")?;
    let mut patch: u64 = parts[2].parse().context("patch")?;
    let next = match kind {
        BumpKind::Patch => {
            patch += 1;
            format!("{major}.{minor}.{patch}")
        }
        BumpKind::Minor => {
            minor += 1;
            patch = 0;
            format!("{major}.{minor}.{patch}")
        }
        BumpKind::Major => {
            major += 1;
            minor = 0;
            patch = 0;
            format!("{major}.{minor}.{patch}")
        }
        BumpKind::Pre(id) => format!("{major}.{minor}.{patch}-{id}"),
    };
    Ok(next)
}

fn read_workspace_version(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path)?;
    let block_start = text
        .find("[workspace.package]")
        .ok_or_else(|| anyhow!("no [workspace.package] block in {}", path.display()))?;
    let block_end = text[block_start..]
        .find("\n[")
        .map(|i| block_start + i)
        .unwrap_or(text.len());
    let block = &text[block_start..block_end];
    for line in block.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("version") {
            let after_eq = rest.split_once('=').map(|(_, v)| v.trim()).unwrap_or("");
            let value = after_eq.trim_matches('"').trim_matches('\'');
            return Ok(value.to_string());
        }
    }
    Err(anyhow!("no version key in [workspace.package]"))
}

fn write_workspace_version(path: &Path, version: &str) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let block_start = text
        .find("[workspace.package]")
        .ok_or_else(|| anyhow!("no [workspace.package] block"))?;
    let after_block = &text[block_start..];
    let local_idx = after_block
        .find("version = ")
        .ok_or_else(|| anyhow!("no version line"))?;
    let abs = block_start + local_idx;
    let line_end = text[abs..]
        .find('\n')
        .map(|i| abs + i)
        .unwrap_or(text.len());
    let new_line = format!("version = \"{version}\"");
    let mut out = String::with_capacity(text.len() + new_line.len());
    out.push_str(&text[..abs]);
    out.push_str(&new_line);
    out.push_str(&text[line_end..]);
    std::fs::write(path, out)?;
    Ok(())
}

/// Bump every internal-path-dep `version = "<prev>"` pin inside
/// `[workspace.dependencies]`. Sub-crates that path-depend on
/// `atomr-accel-cuda` (e.g. patterns / train / agents / realtime / py)
/// declare `atomr-accel-cuda = { path = "...", version = "..." }` and
/// crates.io resolves against the version pin on publish — they
/// must move in lockstep with the workspace version.
fn write_workspace_deps_versions(path: &Path, prev: &str, next: &str) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let block_start = match text.find("[workspace.dependencies]") {
        Some(i) => i,
        None => return Ok(()),
    };
    let after = &text[block_start + "[workspace.dependencies]".len()..];
    let block_len = after.find("\n[").map(|i| i + 1).unwrap_or(after.len());
    let head = &text[..block_start];
    let block = &text[block_start..block_start + "[workspace.dependencies]".len() + block_len];
    let tail = &text[block_start + "[workspace.dependencies]".len() + block_len..];

    let needle = format!("version = \"{prev}\"");
    let replacement = format!("version = \"{next}\"");
    let mut new_block = String::with_capacity(block.len());
    for line in block.split_inclusive('\n') {
        // Only rewrite intra-workspace path-deps. Don't touch external
        // deps that happen to share the previous version string.
        let is_internal_path =
            line.contains("path = \"crates/") || line.contains("path = \"../atomr/crates/");
        if is_internal_path && line.contains(&needle) {
            new_block.push_str(&line.replace(&needle, &replacement));
        } else {
            new_block.push_str(line);
        }
    }
    let mut out = String::with_capacity(text.len());
    out.push_str(head);
    out.push_str(&new_block);
    out.push_str(tail);
    std::fs::write(path, out)?;
    Ok(())
}

/// Bump every per-crate inline `path = "../atomr-accel-*"`-with-`version = "<prev>"`
/// pin in member Cargo.tomls. xtask's workspace-deps rewrite covers
/// `[workspace.dependencies]` only; sibling-crate deps that the
/// member crates declare directly (e.g. patterns/train/agents/realtime/py
/// → atomr-accel-cuda) drift on every bump otherwise, breaking
/// `cargo publish` with "no matching package" errors.
fn write_member_inline_deps_versions(prev: &str, next: &str) -> Result<()> {
    let crates_dir = Path::new("crates");
    if !crates_dir.exists() {
        return Ok(());
    }
    let needle = format!("version = \"{prev}\"");
    let replacement = format!("version = \"{next}\"");
    for entry in std::fs::read_dir(crates_dir)? {
        let entry = entry?;
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&manifest)?;
        let mut out = String::with_capacity(text.len());
        let mut changed = false;
        for line in text.split_inclusive('\n') {
            // Only rewrite lines that pin a sibling atomr-accel-* path
            // dep at the previous workspace version.
            let is_sibling_path = line.contains("path = \"../atomr-accel");
            if is_sibling_path && line.contains(&needle) {
                out.push_str(&line.replace(&needle, &replacement));
                changed = true;
            } else {
                out.push_str(line);
            }
        }
        if changed {
            std::fs::write(&manifest, out)?;
        }
    }
    Ok(())
}

fn write_pyproject_version(path: &Path, version: &str) -> Result<()> {
    // pyproject.toml in maturin mode declares `dynamic = ["version"]`
    // — the wheel version comes from Cargo.toml. We still update the
    // `version` line if one is present (some setups pin it).
    let text = std::fs::read_to_string(path)?;
    if !text.contains("version") {
        return Ok(());
    }
    let mut replaced = false;
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        if !replaced
            && trimmed.starts_with("version")
            && trimmed.contains('=')
            && !trimmed.starts_with("dynamic")
        {
            // Skip the `dynamic = ["version"]` declaration itself.
            out.push_str(&format!("version = \"{version}\"\n"));
            replaced = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if replaced {
        std::fs::write(path, out)?;
    }
    Ok(())
}

// ─── verify ─────────────────────────────────────────────────────────────

fn verify() -> Result<()> {
    let cargo = env!("CARGO");
    let steps: Vec<(&str, &[&str])> = vec![
        ("fmt", &["fmt", "--all", "--", "--check"]),
        (
            "clippy (no-default-features)",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--no-default-features",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "clippy (core-libs)",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--features",
                "atomr-accel-cuda/core-libs",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "test (no-default-features)",
            &["test", "--workspace", "--no-default-features"],
        ),
        (
            "check (training-libs)",
            &[
                "check",
                "--workspace",
                "--features",
                "atomr-accel-cuda/training-libs",
            ],
        ),
        (
            "check (full-cuda)",
            &[
                "check",
                "--workspace",
                "--features",
                "atomr-accel-cuda/full-cuda",
            ],
        ),
        (
            "check (replay)",
            &[
                "check",
                "--workspace",
                "--features",
                "atomr-accel-cuda/replay",
            ],
        ),
        (
            "check (cluster)",
            &[
                "check",
                "--workspace",
                "--features",
                "atomr-accel-cuda/cluster",
            ],
        ),
        (
            "check (streams)",
            &[
                "check",
                "--workspace",
                "--features",
                "atomr-accel-cuda/streams",
            ],
        ),
        (
            "check (telemetry)",
            &[
                "check",
                "--workspace",
                "--features",
                "atomr-accel-cuda/telemetry",
            ],
        ),
        ("doc", &["doc", "--workspace", "--no-deps"]),
    ];
    for (label, args) in steps {
        println!("==> verify: {label}");
        let status = Command::new(cargo).args(args).status().with_context(|| {
            format!(
                "spawning cargo {} for verify step `{label}`",
                args.join(" ")
            )
        })?;
        if !status.success() {
            return Err(anyhow!("verify step `{label}` failed ({status})"));
        }
    }
    println!("==> verify: all steps passed");
    Ok(())
}

// ─── GPU integration tests (opt-in, not in CI) ───────────────────────

/// Probe the local machine for CUDA availability and report device + lib state.
/// This is read-only and safe to run on hosts without CUDA installed.
fn gpu_probe() -> Result<()> {
    println!("==> gpu-probe: scanning local CUDA install");
    println!();

    // 1. nvcc presence
    print!("  nvcc: ");
    match Command::new("nvcc").arg("--version").output() {
        Ok(out) if out.status.success() => {
            let txt = String::from_utf8_lossy(&out.stdout);
            let version = txt
                .lines()
                .find(|l| l.contains("release"))
                .map(|l| l.trim().to_string())
                .unwrap_or_else(|| "(version line not found)".into());
            println!("FOUND — {}", version);
        }
        _ => println!("not found on PATH"),
    }

    // 2. CUDA driver via libcuda.so probe (no link)
    print!("  libcuda.so.1: ");
    let driver = Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so.1").exists()
        || Path::new("/usr/lib/wsl/lib/libcuda.so.1").exists()
        || Path::new("/usr/lib64/libcuda.so.1").exists();
    println!("{}", if driver { "FOUND" } else { "not found" });

    // 3. nvidia-smi device list (safe — no allocations)
    print!("  nvidia-smi: ");
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,compute_cap",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(out) if out.status.success() => {
            println!("FOUND");
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                println!("    {}", line.trim());
            }
        }
        _ => println!("not found"),
    }

    // 4. Per-library .so probes
    println!();
    println!("  optional libraries:");
    let libs = [
        ("cuBLAS", "libcublas.so.12"),
        ("cuBLASLt", "libcublasLt.so.12"),
        ("cuDNN", "libcudnn.so.9"),
        ("cuFFT", "libcufft.so.11"),
        ("cuRAND", "libcurand.so.10"),
        ("cuSOLVER", "libcusolver.so.11"),
        ("cuSPARSE", "libcusparse.so.12"),
        ("cuSPARSELt", "libcusparseLt.so.0"),
        ("cuTENSOR", "libcutensor.so.2"),
        ("NCCL", "libnccl.so.2"),
        ("NVRTC", "libnvrtc.so.12"),
        ("CUPTI", "libcupti.so.12"),
        ("NVML", "libnvidia-ml.so.1"),
        ("TensorRT", "libnvinfer.so.10"),
    ];
    for (name, soname) in libs {
        let found = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
            "/usr/lib64",
        ]
        .iter()
        .any(|p| Path::new(p).join(soname).exists());
        println!(
            "    {:<14} {}  ({})",
            name,
            if found { "FOUND" } else { "—" },
            soname,
        );
    }

    println!();
    println!("Run `cargo xtask gpu-test` to execute the GPU integration suite.");
    Ok(())
}

/// Run GPU integration tests for one or more suites. Suites map to
/// `cargo test` invocations against `tests/gpu_*.rs` files (or the
/// existing `tests/*_e2e.rs` set) with the right feature flags.
///
/// Each suite gates on `cuda-runtime-tests` plus its library feature,
/// so a host without CUDA still compiles cleanly — the tests just
/// don't run.
fn gpu_test(args: Vec<String>) -> Result<()> {
    let mut suites: Vec<&str> = if args.is_empty() {
        vec!["all"]
    } else {
        args.iter().map(|s| s.as_str()).collect()
    };
    if suites.contains(&"all") {
        suites = vec![
            "cublas",
            "cublaslt",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "cutensor",
            "nccl",
            "nvrtc",
            "graph",
            "event",
            "memory",
            "cub",
            "cutlass",
            "flashattn",
            "tensorrt",
            "telemetry",
        ];
    }
    println!("==> gpu-test: {} suite(s)", suites.len());
    let mut failed: Vec<String> = Vec::new();
    for suite in suites {
        let plan = gpu_test_plan(suite);
        let Some(plan) = plan else {
            eprintln!("  [skip] unknown suite `{}`", suite);
            failed.push(format!("{} (unknown)", suite));
            continue;
        };
        println!(
            "  [{}] {} {}",
            suite,
            plan.crate_name,
            plan.features.join(",")
        );
        let status = Command::new(env!("CARGO"))
            .args([
                "test",
                "-p",
                plan.crate_name,
                "--no-default-features",
                "--features",
                &plan.features.join(","),
            ])
            .args(plan.test_filter.iter().map(String::as_str))
            .args(["--", "--ignored", "--nocapture"])
            .status()
            .with_context(|| format!("spawn cargo test for suite `{}`", suite))?;
        if !status.success() {
            failed.push(suite.into());
        }
    }
    if !failed.is_empty() {
        return Err(anyhow!(
            "gpu-test: {} suite(s) failed: {}",
            failed.len(),
            failed.join(", ")
        ));
    }
    println!("==> gpu-test: all suites passed");
    Ok(())
}

/// Run criterion GPU benchmarks for a named set or all of them.
fn gpu_bench(args: Vec<String>) -> Result<()> {
    let benches: Vec<String> = if args.is_empty() {
        vec!["sgemm_overhead".into(), "rng_throughput".into()]
    } else {
        args
    };
    println!("==> gpu-bench: {} bench(es)", benches.len());
    for b in benches {
        let features = match b.as_str() {
            "rng_throughput" => "cuda-runtime-tests,curand",
            _ => "cuda-runtime-tests",
        };
        println!("  [{}] features={}", b, features);
        let status = Command::new(env!("CARGO"))
            .args([
                "bench",
                "-p",
                "atomr-accel-cuda",
                "--no-default-features",
                "--features",
                features,
                "--bench",
                &b,
            ])
            .status()
            .with_context(|| format!("spawn cargo bench `{}`", b))?;
        if !status.success() {
            return Err(anyhow!("bench `{}` failed", b));
        }
    }
    Ok(())
}

struct GpuTestPlan {
    crate_name: &'static str,
    features: Vec<&'static str>,
    test_filter: Vec<String>,
}

fn gpu_test_plan(suite: &str) -> Option<GpuTestPlan> {
    let (crate_name, feats, filter): (&str, Vec<&str>, &str) = match suite {
        "cublas" => ("atomr-accel-cuda", vec!["cuda-runtime-tests"], "sgemm_e2e"),
        "cublaslt" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cublaslt", "f16"],
            "",
        ),
        "cudnn" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cudnn", "f16"],
            "",
        ),
        "cufft" => ("atomr-accel-cuda", vec!["cuda-runtime-tests", "cufft"], ""),
        "curand" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "curand"],
            "rng_fill_e2e",
        ),
        "cusolver" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cusolver"],
            "",
        ),
        "cusparse" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cusparse"],
            "spmv_e2e",
        ),
        "cutensor" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cutensor"],
            "contract_e2e",
        ),
        "nccl" => ("atomr-accel-cuda", vec!["cuda-runtime-tests", "nccl"], ""),
        "nvrtc" => ("atomr-accel-cuda", vec!["cuda-runtime-tests", "nvrtc"], ""),
        "graph" => ("atomr-accel-cuda", vec!["cuda-runtime-tests"], "graph"),
        "event" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cuda-ipc"],
            "event",
        ),
        "memory" => (
            "atomr-accel-cuda",
            vec!["cuda-runtime-tests", "cuda-managed"],
            "pinned_memcpy_e2e",
        ),
        "cub" => ("atomr-accel-cub", vec!["cuda-runtime-tests"], ""),
        "cutlass" => ("atomr-accel-cutlass", vec!["cuda-runtime-tests"], ""),
        "flashattn" => ("atomr-accel-flashattn", vec!["cuda-runtime-tests"], ""),
        "tensorrt" => ("atomr-accel-tensorrt", vec!["cuda-runtime-tests"], ""),
        "telemetry" => ("atomr-accel-telemetry", vec!["nvtx", "nvml", "cupti"], ""),
        _ => return None,
    };
    Some(GpuTestPlan {
        crate_name,
        features: feats,
        test_filter: if filter.is_empty() {
            Vec::new()
        } else {
            vec![filter.into()]
        },
    })
}
