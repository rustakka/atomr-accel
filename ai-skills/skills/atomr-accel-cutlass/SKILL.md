---
name: atomr-accel-cutlass
description: Use when wiring or extending CUTLASS kernel templates through `atomr-accel-cutlass` — the `CutlassActor`, `GemmRequest<T>` / `GroupedGemmRequest<T>` / `ConvFwdRequest<T>` / `Dgrad` / `Wgrad`, the EVT (epilogue visitor tree) emitter, the `(template, shape, dtype, arch)` plan cache, and the Strategy A (NVRTC at runtime) vs Strategy B (`cutlass-prebuilt`, nvcc at build time) compilation choice. Triggers on adding a CUTLASS template, picking arch×dtype, hitting a plan-cache miss, choosing fp8 vs fp4, or fitting an EVT chain.
---

# CUTLASS templates

This skill covers the Phase 6 sibling crate. Enable the `cutlass`
feature on `atomr-accel-cuda` and `CutlassActor` becomes available
alongside the other kernel actors. For the per-library kernel
actor pattern see [`atomr-accel-kernels`](../atomr-accel-kernels/SKILL.md);
for portable trait surface considerations see
[`atomr-accel-backends`](../atomr-accel-backends/SKILL.md).

## Compilation strategies

| Strategy | When | Trade-off |
|---|---|---|
| **A — NVRTC at runtime** (default) | First call to a new `(template, shape, dtype, arch)` triggers an NVRTC compile, then the cubin is cached on disk via the Phase 0.6 cache. Subsequent calls are warm. | First-call latency 30–60s per kernel; downstream builds run on no-GPU hosts. |
| **B — nvcc at build time** (`cutlass-prebuilt` feature) | `build.rs` walks a generator and emits a static archive of pre-instantiated kernels for a fixed `(op × dtype × arch)` matrix. | Fast cold start, no NVRTC at runtime. Requires `nvcc` on the build host — CI on no-GPU runners breaks. |

Default to A. Switch to B for production deployments where every
serving instance hits the same kernel matrix.

## Cargo features

Add to `atomr-accel-cuda` features:

```toml
features = ["cutlass", "f16"]                                  # GEMM only
features = ["cutlass", "cutlass-grouped", "f16"]               # + grouped GEMM
features = ["cutlass", "cutlass-evt", "f16"]                   # + EVT epilogues
features = ["cutlass", "cutlass-prebuilt", "f16"]              # Strategy B
```

## arch × dtype support matrix

| dtype | sm_80 | sm_86 | sm_89 | sm_90a | sm_100 |
|---|:-:|:-:|:-:|:-:|:-:|
| f32, f64, f16, bf16 | ✔ | ✔ | ✔ | ✔ | ✔ |
| fp8 e4m3 / e5m2 | | | ✔ | ✔ | ✔ |
| fp4 e2m1 | | | | | ✔ |
| int8 → int32 | ✔ | ✔ | ✔ | ✔ | ✔ |

Use `is_supported_for(dtype, arch)` (or `is_fp8_supported` /
`is_fp4_supported`) before constructing a request — building a
`GemmRequest` in an unsupported cell still succeeds, but the
NVRTC compile will reject the template instantiation.

## Request types

Every request is generic over `T: GemmSupported` (currently `f32`,
`f64`, `f16`, `bf16`, plus the fp8 / fp4 markers under the matching
feature) and produces a `PlanKey` for the plan cache.

| Module | Request | Dispatch trait | Gate |
|---|---|---|---|
| `gemm` | `GemmRequest<T>` | `CutlassGemmDispatch` | always-on |
| `grouped_gemm` | `GroupedGemmRequest<T>` | `CutlassGroupedGemmDispatch` | `grouped` |
| `conv` | `ConvFwdRequest<T>` / `ConvDgradRequest<T>` / `ConvWgradRequest<T>` | `CutlassConvDispatch` | always-on |
| `evt` | `EpilogueVisitorTree`, `EvtBuilder`, `EpilogueOp` | n/a (composes onto `GemmRequest`) | `evt` |

## A simple GEMM

```rust
use atomr_accel_cutlass::{
    CutlassMsg, GemmEpilogue, GemmLayout, GemmRequest, GemmShape, SmArch,
};
use half::f16;

let req = GemmRequest::<f16> {
    arch: SmArch::Sm90a,
    shape: GemmShape::new(4096, 4096, 4096),
    layout_a: GemmLayout::RowMajor,
    layout_b: GemmLayout::ColMajor,
    layout_c: GemmLayout::RowMajor,
    epilogue: GemmEpilogue::LinearReLU { alpha: 1.0, beta: 0.0 },
    /* a/b/c GpuRefs, reply channel … */
};

cutlass.tell(CutlassMsg::Gemm(Box::new(req)));
```

## EVT — fused epilogue chains

`cutlass-evt` unlocks the epilogue visitor tree emitter — the way
to chain post-GEMM ops (bias-add, activation, dropout, scale,
quantize, reduce) into a single launch. Build with `EvtBuilder`:

```rust
#[cfg(feature = "cutlass-evt")]
use atomr_accel_cutlass::{EpilogueOp, EpilogueVisitorTree, EvtBuilder};

let tree: EpilogueVisitorTree = EvtBuilder::new()
    .scale(1.0 / 8.0)
    .add_bias(/* bias GpuRef */)
    .activation(EpilogueOp::Gelu)
    .quantize_to_fp8()
    .build()?;

let req = GemmRequest { /* … */, epilogue: tree.into_epilogue() };
```

Each EVT chain produces a unique `PlanKey` — the cache discriminates
GEMM-with-EVT-A from GEMM-with-EVT-B without collision.

## The plan cache

`PlanCache` (LRU, capacity set at `CutlassActor` construction)
stores rendered `.cu` source + lowered kernel name keyed by
`(template_id, shape, dtype, arch, layout, epilogue)`. The cache
saves the per-call NVRTC compile — under Strategy A a warm cache
hit is microseconds, a miss is tens of seconds.

```rust
let props = atomr_accel_cutlass::props(/* plan_cache_capacity */ 256);
let cutlass: ActorRef<CutlassMsg> = system.actor_of(props, "cutlass");
```

The cache is **per-actor**, not global. If you spawn multiple
`CutlassActor`s for parallelism, each gets its own cache. The
underlying NVRTC disk cache is shared (Phase 0.6), so the second
actor's first call reads from disk — fast, but not as fast as an
in-process LRU hit.

## Refitting weights without recompile

```rust
use atomr_accel_cutlass::{CutlassMsg, RefitMsg};

cutlass.tell(CutlassMsg::Refit {
    msg: RefitMsg {
        plan_key: cached_key,    // from a previous Gemm dispatch
        weights: new_bytes,      // host-side; the actor stages them
    },
    reply: Box::new(|res| { /* … */ }),
});
```

Refit is for already-compiled plans. The plan key carries the
template + shape + dtype + arch fingerprint; new weight bytes are
copied into the kernel's bound workspace. No NVRTC pass.

## Wiring into `ContextActor`

```rust
let cutlass = system.actor_of(atomr_accel_cutlass::props(64), "cutlass");
context.tell(ContextMsg::RegisterExtra {
    name: "cutlass",
    actor: cutlass.clone().into_dyn(),
});
```

`KernelChildren::register_extra` exists exactly for siblings like
this — the cutlass actor lives next to `BlasActor` / `CudnnActor`
and dies with them when the context rebuilds.

## Mock vs real

`CutlassInner::compile_sink` is `Option<...>` so the actor records
rendered `.cu` source + lowered kernel name into the plan cache
even without an NVRTC actor wired in. This is the host-only test
path — the smoke test exercises plan-cache discrimination without a
GPU. In production set `compile_sink` to a closure that forwards
to `atomr_accel_cuda::kernel::NvrtcActor`.

## Canonical references

- `crates/atomr-accel-cutlass/src/lib.rs` — public surface,
  Strategy A/B explainer, arch×dtype matrix.
- `crates/atomr-accel-cutlass/src/{gemm,grouped_gemm,conv,evt}.rs`
  — one request type per file.
- `crates/atomr-accel-cutlass/src/plan_cache.rs` — `PlanCache`
  + `PlanKey` (`(template_id, shape, dtype, arch, layout,
  epilogue)`).
- `crates/atomr-accel-cutlass/src/dtype.rs` — `CutlassDtype`,
  `is_supported_for`, `GemmSupported`, `SmArch`.
- `crates/atomr-accel-cutlass/cutlass/include/` — vendored CUTLASS
  headers (BSD-3-Clause).
- `crates/atomr-accel-cutlass/tests/cutlass_smoke.rs` — arch×dtype
  smoke test (host-only).
- [`docs/features-matrix.md`](../../../docs/features-matrix.md) §
  `atomr-accel-cutlass` — feature flags + transitive deps.

## Common pitfalls

- **Cold-start latency under Strategy A.** The first call to a new
  shape kicks off a 30–60s NVRTC compile. Pre-warm at startup by
  issuing a no-op `GemmRequest` for each canonical shape, or
  switch to Strategy B if your shape catalogue is fixed.
- **Forgetting `cutlass-prebuilt` requires nvcc.** CI fails on
  no-GPU runners. Either keep Strategy A in CI and B in production,
  or self-host a CUDA-equipped builder.
- **Mixing fp8 with sm_80 / sm_86.** `is_fp8_supported(arch)` is
  false there. The smoke test enforces this; production code
  should call `is_supported_for` before submitting.
- **fp4 outside Blackwell.** Only sm_100 / sm_120 supports
  `F4E2m1`. `is_fp4_supported(arch)` returns false elsewhere.
- **EVT without the feature.** Building an `EvtBuilder` chain
  errors at compile time when `cutlass-evt` is off — it's not
  plumbed through plain `GemmEpilogue`. Add the feature explicitly.
- **Plan-cache reuse across GPUs of different arch.** `PlanKey`
  includes `arch`, so swapping a sm_80 cubin into a sm_90a context
  is a cache miss (correctly). Don't try to lift a cached plan to
  a different arch by editing the key.
- **Holding a `PlanKey` past a context rebuild.** Same `KernelHandle`
  story as NVRTC actor — re-resolve through the actor after
  `ContextReady` cycles.
