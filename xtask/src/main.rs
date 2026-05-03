//! `xtask` — developer tooling for the rakka-accel workspace.
//!
//! Subcommands:
//! * `bump <patch|minor|major>` — bump the workspace version, refresh
//!   `Cargo.lock`, mirror to `crates/rakka-accel-py/pyproject.toml`,
//!   and rewrite internal-path-dep version pins inside
//!   `[workspace.dependencies]`.
//! * `bump --pre <id>` / `bump --set <ver>` — variants for
//!   pre-release tags and exact-version overrides.
//! * `verify` — local mirror of the release-pipeline gate:
//!   `cargo fmt --check` → `cargo clippy -D warnings` →
//!   `cargo test --workspace --no-default-features` →
//!   `cargo check --workspace --features rakka-accel-cuda/full-cuda`.
//!
//! The pattern is borrowed from the sibling rakka workspace's xtask.

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
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(anyhow!("unknown xtask subcommand: {other}")),
    }
}

fn print_help() {
    println!("rakka-accel xtask");
    println!();
    println!("USAGE:");
    println!("  cargo xtask <subcommand>");
    println!();
    println!("SUBCOMMANDS:");
    println!("  bump <patch|minor|major>      bump workspace version + python version + Cargo.lock");
    println!("  bump --pre <id>               append a pre-release identifier (e.g. rc.1)");
    println!("  bump --set <version>          set an exact version (used by Release-As: overrides)");
    println!("  verify                        run fmt + clippy + test (release-pipeline gate)");
    println!("  help                          print this help");
}

// ─── bump ───────────────────────────────────────────────────────────────

fn bump(args: Vec<String>) -> Result<()> {
    let mut iter = args.into_iter();
    let arg = iter.next().ok_or_else(|| {
        anyhow!("usage: bump <patch|minor|major> | bump --pre <id> | bump --set <version>")
    })?;
    let cargo_toml = Path::new("Cargo.toml");
    let pyproject = Path::new("crates/rakka-accel-py/pyproject.toml");
    let current = read_workspace_version(cargo_toml)?;
    let next = match arg.as_str() {
        "patch" => semver_bump(&current, BumpKind::Patch)?,
        "minor" => semver_bump(&current, BumpKind::Minor)?,
        "major" => semver_bump(&current, BumpKind::Major)?,
        "--pre" => {
            let id = iter.next().ok_or_else(|| anyhow!("--pre requires <id>"))?;
            semver_bump(&current, BumpKind::Pre(id))?
        }
        "--set" => iter.next().ok_or_else(|| anyhow!("--set requires <version>"))?,
        other => return Err(anyhow!("unknown bump arg: {other}")),
    };
    println!("{} -> {}", current, next);
    write_workspace_version(cargo_toml, &next)?;
    write_workspace_deps_versions(cargo_toml, &current, &next)?;
    if pyproject.exists() {
        write_pyproject_version(pyproject, &next)?;
    }
    // Refresh Cargo.lock so the workspace builds against the new version.
    let _ = Command::new(env!("CARGO")).args(["update", "--workspace"]).status();
    println!("RAKKA_CUDA_NEW_VERSION={next}");
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
    let block_end = text[block_start..].find("\n[").map(|i| block_start + i).unwrap_or(text.len());
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
    let block_start =
        text.find("[workspace.package]").ok_or_else(|| anyhow!("no [workspace.package] block"))?;
    let after_block = &text[block_start..];
    let local_idx = after_block.find("version = ").ok_or_else(|| anyhow!("no version line"))?;
    let abs = block_start + local_idx;
    let line_end = text[abs..].find('\n').map(|i| abs + i).unwrap_or(text.len());
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
/// `rakka-accel-cuda` (e.g. patterns / train / agents / realtime / py)
/// declare `rakka-accel-cuda = { path = "...", version = "..." }` and
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
            line.contains("path = \"crates/") || line.contains("path = \"../rakka/crates/");
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
                "rakka-accel-cuda/core-libs",
                "--",
                "-D",
                "warnings",
            ],
        ),
        ("test (no-default-features)", &["test", "--workspace", "--no-default-features"]),
        (
            "check (training-libs)",
            &["check", "--workspace", "--features", "rakka-accel-cuda/training-libs"],
        ),
        ("check (full-cuda)", &["check", "--workspace", "--features", "rakka-accel-cuda/full-cuda"]),
        ("check (replay)", &["check", "--workspace", "--features", "rakka-accel-cuda/replay"]),
        ("check (cluster)", &["check", "--workspace", "--features", "rakka-accel-cuda/cluster"]),
        ("check (streams)", &["check", "--workspace", "--features", "rakka-accel-cuda/streams"]),
        ("check (telemetry)", &["check", "--workspace", "--features", "rakka-accel-cuda/telemetry"]),
        ("doc", &["doc", "--workspace", "--no-deps"]),
    ];
    for (label, args) in steps {
        println!("==> verify: {label}");
        let status = Command::new(cargo)
            .args(args)
            .status()
            .with_context(|| format!("spawning cargo {} for verify step `{label}`", args.join(" ")))?;
        if !status.success() {
            return Err(anyhow!("verify step `{label}` failed ({status})"));
        }
    }
    println!("==> verify: all steps passed");
    Ok(())
}
