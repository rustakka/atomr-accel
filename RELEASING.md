# Releasing rakka-accel

A single tag → both **crates.io** (five Rust crates in dep order)
and **PyPI** (one Python wheel + sdist). Mirrors the rakka workspace's
release flow.

## TL;DR — fully automated path

```bash
# Land your changes on main with Conventional-Commit subjects
# (feat: / fix: / etc.). The version-bump workflow on main will
# auto-decide the SemVer kind, bump the workspace, commit
# `chore(release): vX.Y.Z`, tag `vX.Y.Z`, and push the tag — which
# fires release.yml.
git push origin main
```

Force a specific bump by adding `Release-As: 1.0.0-rc.1` to a
commit footer, or trigger
`Actions → Version bump + tag → Run workflow` with `force=major`.

## Manual path

If you want to drive the bump locally:

```bash
cargo xtask bump <patch|minor|major>            # auto-pick by SemVer
cargo xtask bump --pre rc.1                     # pre-release tag
cargo xtask bump --set 1.0.0                    # exact version
git commit -am "chore(release): v$(grep -A1 '\[workspace.package\]' Cargo.toml | grep version | sed -E 's/.*"(.*)".*/\1/')"
git tag vX.Y.Z
git push origin main --follow-tags
```

The xtask updates `[workspace.package].version`, rewrites internal
path-dep version pins inside `[workspace.dependencies]`, refreshes
`Cargo.lock`, and mirrors the version into
`crates/rakka-accel-py/pyproject.toml`.

## Workflows

| Workflow                         | Trigger                       | What it does |
|----------------------------------|-------------------------------|--------------|
| `ci.yml`                         | PRs + pushes to `main`        | fmt + clippy + test (8 feature configs) + verify gate + semver-checks (PR-only) + Python build/test + doc upload |
| `version-bump.yml`               | Push to `main`                | Conventional-Commits → `cargo xtask bump` → commit `chore(release)` → tag `vX.Y.Z` |
| `release.yml`                    | Tag `v*` push                 | xtask verify → build wheels (manylinux/macOS/Windows) + sdist → GitHub Release with auto-notes → publish-crates → publish-pypi |
| `docs.yml`                       | Push to `main`, tag `v*`      | rustdoc → GitHub Pages |

`release.yml` honors three `workflow_dispatch` inputs for testing:
`dry_run`, `skip_python`, `skip_crates`. The dry-run path uses
TestPyPI for wheel uploads.

## Pre-flight checklist

- [ ] All CI gates green on `main`.
- [ ] `CHANGELOG.md` updated (or rely on auto-generated release notes).
- [ ] `cargo xtask verify` passes locally (mirrors the
      release-pipeline gate).
- [ ] `(cd crates/rakka-accel-py && maturin develop --release && pytest tests/)`
      passes.

## After a release

1. Verify on [crates.io](https://crates.io/crates/rakka-accel) and
   [PyPI](https://pypi.org/project/rakka-accel/).
2. `pip install --upgrade rakka-accel && python -c "import
   rakka_accel; print(rakka_accel.__version__)"`.
3. `cargo install --version <new> rakka-accel-cuda` (sanity check).
4. Bump the workspace version one minor / patch ahead on `main` to
   start the next development cycle (mirrors the rakka pattern):
   ```toml
   version = "0.0.3-dev"
   ```

## Secrets

- **`CARGO_REGISTRY_TOKEN`** — repo settings → secrets →
  `crates-io` environment. Generate at
  https://crates.io/me with publish-only scope on the rakka-accel
  crates.
- **`PYPI_API_TOKEN`** — repo settings → secrets → `pypi`
  environment. Project-scoped token from
  https://pypi.org/manage/account/token/. Prefer trusted publishing
  via OIDC if your fork allows it; the workflow keeps token-auth as a
  fallback.

## Yanking a bad release

```bash
# crates.io
cargo yank --vers 0.0.2 rakka-accel
cargo yank --vers 0.0.2 rakka-accel-patterns
# ...etc

# PyPI: open the project page → Manage → Releases → Yank.
```

## Manually publishing one crate

If you need to publish a single crate out-of-band (rare, but
sometimes happens when a downstream user reports a bug):

```bash
cargo publish -p rakka-accel-patterns --token "$CARGO_REGISTRY_TOKEN"
```

The `cargo publish --wait-for-publish-timeout 120` flag is useful
when chaining publishes that depend on each other.
