---
name: version-bump
description: |
  Decide the right SemVer bump (patch / minor / major / pre-release)
  for the changes in the current atomr-accel working tree, then apply
  the bump via `cargo xtask bump`. Use whenever the user asks "bump
  the version", "cut a release", "release this as <kind>", or after
  a substantive change has been reviewed and is ready to ship.
---

# version-bump (atomr-accel)

Single-purpose skill: pick a SemVer bump for the staged/working
changes in the atomr-accel workspace and apply it. Mirrors the
sibling atomr workspace's skill but with atomr-accel-specific crate
list and GPU-domain notes.

## Decision rule (Conventional Commits → SemVer)

Walk the commits since the last `v*` tag (or the working diff if no
tag exists yet) and choose the **strongest** bump implied:

| Found in any commit                       | Bump  |
|-------------------------------------------|-------|
| `BREAKING CHANGE:` / `<type>!:` / typed-API removal in `atomr-accel-cuda`, `atomr-accel-patterns`, `atomr-accel-train`, `atomr-accel-agents`, `atomr-accel-cuda-realtime` | **major** |
| `feat:` / new public type, trait method, or `DeviceMsg` variant | **minor** |
| `fix:` / `perf:` / `revert:` / internal-only fix | **patch** |
| `chore:` / `docs:` / `ci:` / `test:` / `refactor:` / `style:` / `build:` only | **skip** (don't bump) |

If unsure between minor and patch, prefer **minor** when any `pub`
item was added/changed, **patch** otherwise.

For pre-1.0 releases (`0.x.y`), treat breaking changes as **minor**
(per Cargo SemVer conventions for `0.x`).

## What's *not* a breaking change

GPU-domain things that look scary but don't break the public API:

- Switching a kernel actor from `cudarc::*::sys` to a safe-layer
  implementation (when cudarc upstream catches up). The actor
  surface stays identical.
- Wiring a previously-empty feature-gated module to a real impl
  (e.g. when a cuSPARSE / cuTENSOR placeholder lights up). New
  capability, not a break — that's a **minor**.
- Adding a new `feature` flag or `prelude` re-export.
- Promoting a CPU-reference actor to use its bundled NVRTC kernel
  (the message API doesn't change).

## What's *always* a breaking change

- Renaming or removing a public type / trait / function.
- Adding a required field to a public struct (use builders).
- Adding a non-default-constructible variant to a `#[non_exhaustive]`
  enum that callers match exhaustively.
- Tightening a `Send`/`Sync` bound that downstream actors relied on.
- Changing the layout of `GpuRef<T>` (fortunately stable since F1).

## What to do

1. **Survey the changes.** Look at:
   - `git log $(git describe --tags --abbrev=0 2>/dev/null)..HEAD --pretty=format:%s%n%b`
     (or `git diff --name-only` if no tag exists).
   - For each modified `crates/*/src/*.rs` — look for added / removed
     `pub` items, signature changes, trait additions.
   - `docs/features-matrix.md` if a new feature flag was added.
   - The `## Status` section in `README.md` for context on the
     current F-phase.

2. **Pick the bump.** State the decision in one sentence:
   *"This is a **minor** bump because phase D.6 added a new public
   `observability::install` API."*

3. **Apply via xtask.** Always use the workspace tool — never edit
   `Cargo.toml` / `pyproject.toml` by hand:

   ```bash
   cargo xtask bump <patch|minor|major>
   # or
   cargo xtask bump --pre rc.1
   # or
   cargo xtask bump --set 1.0.0-rc.1
   ```

   The xtask:
   - Bumps `[workspace.package].version` in the root `Cargo.toml`.
   - Walks `[workspace.dependencies]` rewriting internal-path-dep
     `version = "<old>"` pins to the new version (sub-crates need
     this in lockstep so crates.io can resolve them on publish).
   - Mirrors the new version into
     `crates/atomr-accel-py/pyproject.toml` if it has a static
     `version` line.
   - Refreshes `Cargo.lock` via `cargo update --workspace`.

4. **Stage the bump.** `git add Cargo.toml Cargo.lock
   crates/atomr-accel-py/pyproject.toml` and prepare a commit titled
   `chore(release): vX.Y.Z`. Don't push or tag from this skill —
   that's the user's call (or
   `.github/workflows/version-bump.yml`'s job on `main`, which fires
   `release.yml` on the resulting tag).

5. **Sanity-check.** Run `cargo xtask verify` to confirm the
   workspace still builds clean across every feature combination.

## When *not* to bump

- The change is purely doc / CI / refactor (no public-API delta).
- The user is mid-development and the public API isn't ready.
- A bump landed within the last few commits and no new public-API
  delta has accumulated since.

In those cases, say so explicitly and skip the bump.

## Tooling reference

- `cargo xtask bump <patch|minor|major>` — strip pre-release tags,
  bump the requested component, update Cargo.lock, update
  pyproject.toml. Prints `RAKKA_CUDA_NEW_VERSION=<x.y.z>`.
- `cargo xtask bump --pre <id>` — append a pre-release identifier
  (e.g. `rc.1`) without changing the numeric core.
- `cargo xtask bump --set <ver>` — set an exact version. Used by
  the `Release-As: <ver>` commit-trailer override in CI.
- `cargo xtask verify` — fmt + clippy + workspace test + every
  feature-combo check + doc. The release-pipeline gate.
- `.github/workflows/version-bump.yml` runs the same logic on every
  `main` push (Conventional-Commit-driven, autocommits + tags
  `vX.Y.Z`, which fires `release.yml`).
- `.github/workflows/release.yml` is the downstream pipeline:
  verify → build wheels + sdist → GitHub Release →
  publish-crates → publish-pypi.

## Example invocations

> "Cut a 0.1.0 release."
1. Run `cargo xtask bump --set 0.1.0`.
2. Stage + commit `chore(release): v0.1.0`.
3. Tell the user the tag will be created on push (or instruct them
   to `git tag v0.1.0 && git push --follow-tags`).

> "I added a public `Device.compile_kernel` API in the Python bridge;
> bump appropriately."
1. State: *"new public method on a published Python class →
   **minor** bump (or **patch** while pre-1.0 since the wheel
   surface is still maturing)."*
2. Run `cargo xtask bump minor`.

> "Release the cuTENSOR wrapper as a real version."
1. Confirm `kernel/tensor.rs` is no longer empty and tests pass.
2. State: *"first real cuTENSOR support → **minor** bump (new
   capability, no API removal)."*
3. Run `cargo xtask bump minor`.
