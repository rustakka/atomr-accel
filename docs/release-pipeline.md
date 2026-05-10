# Release pipeline

> **See also:** [release-process.md](release-process.md) — the
> operator-facing reference (how to ship, conventional-commit rules,
> trampoline architecture, troubleshooting). This document focuses on
> workflow internals: jobs, matrix entries, build commands.

`/.github/workflows/release.yml` ships atomr-accel to three places on
every `v*` tag:

1. **GitHub Releases** — a release page with auto-generated notes;
   the wheel and sdist artifacts are attached.
2. **crates.io** — every publishable Rust crate, in dependency order.
3. **PyPI** — platform-specific wheels (Linux x86_64/aarch64, Linux
   musl x86_64/aarch64, macOS universal2, Windows x86_64) and an sdist.

The pipeline mirrors the sibling [atomr](https://github.com/rustakka/atomr)
workspace's `release.yml`. Differences are confined to the crate list,
the absence of pre-built CLI binaries, and the sibling-repo checkout
required for atomr-accel's path-deps.

## Triggering

There are three paths into this pipeline; they all converge on the
same publish jobs.

* **Direct tag push** (`git push origin vX.Y.Z`) — fires
  `on: push: tags`. Use this when a human is cutting a release
  outside of the auto-bump flow.
* **Auto-bump trampoline** — `version-bump.yml` runs on every push
  to `main` and decides a SemVer bump from Conventional-Commit
  subjects (`feat:` → minor, `fix:`/`perf:`/`revert:` → patch,
  `!:`/`BREAKING CHANGE` → major; everything else — including
  `build:`, `chore:`, `docs:`, `ci:`, `test:`, `refactor:`,
  `style:` — is `skip`). When it decides to bump, it commits the
  version change, tags it, pushes, **and then explicitly dispatches
  `release.yml`** via `gh workflow run release.yml --ref vX.Y.Z
  -f dry_run=false`. The explicit dispatch is required because tag
  events authored by the default `GITHUB_TOKEN` do not fire downstream
  workflows.
* **Manual `workflow_dispatch`** — choose `dry_run=true` for a
  rehearsal that publishes to TestPyPI and runs `cargo publish --dry-run`.
  Toggle `skip_python` / `skip_crates` to ship to only one registry.
  A manual dispatch with `dry_run=false` against a `v*` tag ref also
  performs a real publish (this is the same path the trampoline takes).

### What gets published when

| Trigger | verify | wheels | sdist | GitHub Release | crates.io | PyPI |
|---|---|---|---|---|---|---|
| `push` on `v*` tag | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `workflow_dispatch` ref=`v*` `dry_run=false` | ✓ | ✓ | ✓ | ✓ | ✓ (unless `skip_crates`) | ✓ (unless `skip_python`) |
| `workflow_dispatch` `dry_run=true` | ✓ | ✓ | ✓ | — | dry-run only | TestPyPI |

The publish jobs guard on `startsWith(github.ref, 'refs/tags/v')`, so
a `workflow_dispatch` against a branch ref will only run the verify
gate and (optionally) dry-run jobs — never a real publish.

## Sibling-repo checkout

Every job that builds Rust code checks out two repos side-by-side
into the runner workspace:

```yaml
- name: Checkout sibling atomr
  uses: actions/checkout@v5
  with:
    repository: rustakka/atomr
    path: atomr
- name: Checkout atomr-accel
  uses: actions/checkout@v5
  with:
    path: atomr-accel
```

After this, `$GITHUB_WORKSPACE` looks like:

```
$GITHUB_WORKSPACE/
├── atomr/         # rustakka/atomr — sibling, path-dep target
└── atomr-accel/   # this repo
```

The `atomr-accel/Cargo.toml` workspace pins `atomr-core`,
`atomr-config`, `atomr-macros`, `atomr-persistence`,
`atomr-cluster-sharding`, `atomr-streams`, `atomr-telemetry`, and
`atomr-testkit` via `path = "../atomr/crates/..."`. Without the
sibling checkout, `cargo` would fail to resolve those path-deps.

Subsequent steps run with `working-directory: atomr-accel` so cargo
sees the workspace root.

## What gets built

### Wheels (`build-wheels`)

Built via `PyO3/maturin-action`. The action runs each target inside the
appropriate `manylinux` / `musllinux` container; the action's
`--interpreter` flag builds a wheel per CPython ABI (3.10 – 3.13)
under the abi3 stable interface.

| OS | Target | Wheel tag |
|---|---|---|
| Ubuntu | `x86_64-unknown-linux-gnu` | `manylinux_2_17_x86_64` |
| Ubuntu (`ubuntu-22.04-arm`) | `aarch64-unknown-linux-gnu` | `manylinux_2_17_aarch64` |
| Ubuntu | `x86_64-unknown-linux-musl` | `musllinux_1_2_x86_64` |
| Ubuntu (`ubuntu-22.04-arm`) | `aarch64-unknown-linux-musl` | `musllinux_1_2_aarch64` |
| macOS | `universal2-apple-darwin` | `macosx_*_universal2` (fat: x86_64 + arm64) |
| Windows | `x86_64-pc-windows-msvc` | `win_amd64` |

aarch64 Linux wheels are built natively on a GitHub-hosted ARM runner
(`ubuntu-22.04-arm`) instead of cross-compiled inside the x86_64
manylinux container. This avoids a cross-compile of `cudarc` /
native-deps that historically blocked the aarch64 wheel — the same
pattern atomr uses for the Python bindings.

### sdist (`build-sdist`)

A single source distribution `atomr_accel-X.Y.Z.tar.gz`, used by PyPI
for platforms that have no pre-built wheel.

### Binaries — not built

Unlike the upstream atomr workspace, atomr-accel does not currently
ship pre-built CLI binaries. There are no equivalents to atomr's
`atomr-dashboard` or `atomr-profiler`. If a binary tool is added in
the future, slot in a `build-binaries` job mirroring atomr's
release.yml: cross-platform matrix (Linux x86_64/aarch64 via `cross`,
macOS x86_64/aarch64, Windows x86_64), `cargo build --release`,
artifact-upload, attach to GitHub Release.

## Required secrets / config

| Secret | Where | Used by |
|---|---|---|
| `CRATES_IO_TOKEN` | `crates-io` GitHub environment | `publish-crates` |
| PyPI Trusted Publisher | configured on PyPI itself, **not** as a GitHub secret | `publish-pypi` |
| TestPyPI Trusted Publisher | configured on TestPyPI itself | `publish-pypi-dry-run` |

### PyPI Trusted Publishing setup

Trusted publishing avoids long-lived API tokens. One-time setup:

1. Create the project on https://pypi.org/manage/projects/ (or run a
   manual upload first).
2. Go to *Manage → Publishing → Add a new publisher → GitHub*.
3. Fill in:
   * Owner: `rustakka`
   * Repository: `atomr-accel`
   * Workflow name: `release.yml`
   * Environment: `pypi`
4. Repeat for TestPyPI with environment `testpypi`.

The `publish-pypi` job already declares `permissions: id-token: write`
and `environment: pypi` so the OIDC handshake works once you've
registered the publisher. atomr-accel's TestPyPI publisher is already
configured; the production PyPI publisher is the only one that
needs the steps above the first time the project ships to pypi.org.

If you'd rather use an API token, replace the
`pypa/gh-action-pypi-publish` action's `with:` block with:

```yaml
with:
  packages-dir: upload
  password: ${{ secrets.PYPI_API_TOKEN }}
  skip-existing: true
```

## Crates published

The `publish-crates` job walks every publishable crate in dependency
order. Adding a new crate? Slot it into the earliest layer whose
prerequisites have already been published, and pin its intra-workspace
deps with `{ workspace = true }` (NOT a hand-written `version = "..."`
literal) so the next bump doesn't leave a stale pin behind.

Current order (top to bottom):

1. `atomr-accel` (umbrella core; no intra-workspace deps)
2. `atomr-accel-telemetry`
3. `atomr-accel-flashattn`
4. `atomr-accel-tensorrt`
5. `atomr-accel-cutlass`
6. `atomr-accel-cuda` (depends on layers 1–5)
7. `atomr-accel-patterns` (depends on `atomr-accel-cuda`)
8. `atomr-accel-agents` (depends on `atomr-accel-cuda`)
9. `atomr-accel-cuda-realtime` (depends on `atomr-accel-cuda`)
10. `atomr-accel-train` (depends on `atomr-accel-patterns`)

Workspace members deliberately excluded: `xtask` (`publish = false`),
`crates/atomr-accel-py` (`publish = false`; ships as the single
`atomr-accel` PyPI wheel via maturin), and `crates/atomr-accel-cub`
(`publish = false` until Phase 5.1 lands the NVRTC kernel emitters
that back its dispatch surface).

## Cross-publishing constraints

* **crates.io publishes are sequential** — every dependent crate
  must wait for its dependencies to be visible. The `publish-crates`
  job orders them deliberately; if you add a new crate, slot it into
  the matching layer of that block.
* **`already uploaded` is treated as success** — re-tagging the same
  version (after fixing one mid-pipeline crate) is cheap; previously-
  uploaded crates skip in <1s.
* **Rate limiting** — each successful publish sleeps 30s; `429 Too
  Many Requests` triggers exponential backoff up to 6 attempts. With
  ~10 crates this caps the publish-crates job around 7 minutes.
* **Wheel ABI tags** are baked in by maturin from the build container,
  so each matrix entry produces a different wheel tag. If you need
  more (e.g. PyPy), add another matrix line.
* **Universal2 macOS wheels** cover both Intel and Apple Silicon in a
  single artifact; that's why we don't run separate macOS x86_64 and
  aarch64 wheel builds.
* **musllinux** is Alpine-friendly; if you don't ship to Alpine, drop
  those matrix rows to halve Linux build time.
* **Sibling atomr version skew** — atomr-accel pins specific
  upstream-atomr versions in `[workspace.dependencies]`. If the
  matching atomr release isn't on crates.io yet, `publish-crates`
  will block on the dep resolution. The `version-bump` skill bumps
  the atomr pin in lockstep when needed.

## Verifying a release locally

Dry-run a release by triggering the workflow manually:

```
gh workflow run release.yml -f dry_run=true
```

This runs the verify gate, builds every wheel + sdist, and uploads
to TestPyPI (`https://test.pypi.org/p/atomr-accel`) without touching
crates.io or production PyPI. The artifacts also land on the
workflow's *Artifacts* panel so you can download and smoke-test
before tagging.

For local-only verification (no remote pipeline):

```
cargo xtask verify
(cd crates/atomr-accel-py && maturin develop --release && pytest tests/)
```
