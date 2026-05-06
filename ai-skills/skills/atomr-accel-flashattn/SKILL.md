---
name: atomr-accel-flashattn
description: Use when wiring or extending FlashAttention v2 / v3 through `atomr-accel-flashattn` — the `FlashAttnActor`, the `(arch, dtype, head_dim, causal, varlen, sliding_window, alibi, sink, paged, gqa)` dispatch table, paged KV cache (vLLM-style), chunked prefill, varlen, and the FA2-vs-FA3 / fp16-vs-bf16-vs-fp8 picking matrix. Triggers on adding an attention call, choosing fa2 vs fa3, reasoning about dispatch-key collisions, or paged KV-cache shape mistakes.
---

# FlashAttention v2 + v3

This skill covers the Phase 7 sibling crate. It plugs into
`atomr-accel-cuda` via the `flashattn` cargo feature; turn that on
and `FlashAttnActor` becomes available alongside the cuBLAS / cuDNN
actors as a child of `ContextActor`. For the per-library kernel
actor pattern, see [`atomr-accel-kernels`](../atomr-accel-kernels/SKILL.md).

## Picking fa2 vs fa3

| GPU | Kernel family | Notes |
|---|---|---|
| Ampere (sm_80, A100/A30) | fa2 | f16 / bf16, fwd + bwd. No fp8. |
| Ada (sm_89, RTX 40xx, L4) | fa2 | f16 / bf16, fwd + bwd. fp8 on cuBLASLt only — fa2 itself is half-precision. |
| Hopper (sm_90a, H100/H200) | **fa3** | f16 / bf16 / fp8 e4m3 / fp8 e5m2 fwd. Backward falls through to fa2. Persistent kernels available. |
| Blackwell (sm_100, B100/B200) | fa3 | Same as Hopper for now; future fifth-gen tensor-core kernels will land here. |

`SmArch::supports_fa3()` and `SmArch::supports_fp8()` encode this.
Constructing an `Fa3FwdRequest` against a non-Hopper arch returns
`FlashAttnError::Fa3RequiresHopper`.

## Cargo features

Add to `atomr-accel-cuda` features:

```toml
features = ["flashattn", "f16"]                # fa2/fa3 fp16+bf16, no paged
features = ["flashattn", "flashattn-fp8", "f16", "f8"]  # + fa3 fp8 (sm_90a)
features = ["flashattn", "flashattn-paged", "f16"]      # + paged KV-cache
```

`flashattn-paged` is required to construct a `PagedKvCache` /
`PagedAttentionRequest`; the `paged` module is `#[cfg]`-stripped
otherwise.

## Request types

Each request module produces a [`DispatchKey`] via `dispatch_key()`,
which the actor resolves to a kernel-name expression and forwards to
`NvrtcActor`. The hot path is dispatch-table lookup → cubin-cache
hit → launch.

| Module | Request | Dispatch trait |
|---|---|---|
| `fa2` | `Fa2FwdRequest<T>`, `Fa2BwdRequest<T>` | `FaFwdDispatch`, `FaBwdDispatch` |
| `fa3` | `Fa3FwdRequest<T>`, `Fa3FwdFp8Request` (gated `fp8`) | `FaFwdDispatch` |
| `varlen` | `VarlenFwdRequest<T>` | `FaFwdDispatch` |
| `paged` | `PagedAttentionRequest<T>` (gated `paged`) | `FaPagedFwdDispatch` |
| `prefill` | `ChunkedPrefillRequest<T>` | `FaFwdDispatch` |

`T` is bound by `dispatch::GemmSupported` — implemented for `F16`,
`Bf16`, and (under `fp8`) `F8E4m3` / `F8E5m2`. fa2 requests reject
fp8 markers at request-construction time
(`Fp8MustUseFp8Request`).

## Sending a forward pass

```rust
use atomr_accel_flashattn::{Fa2FwdRequest, FlashAttnMsg, MaskKind, PositionBias, SmArch, F16};
use std::time::Duration;

let req = Fa2FwdRequest::<F16> {
    arch: SmArch::Sm80,
    head_dim: 128,
    gqa_ratio: 1,
    mask: MaskKind::Causal,
    bias: PositionBias::None,
    sink_tokens: 0,
    softmax_scale: 1.0 / (128.0_f32).sqrt(),
    /* Q/K/V/O GpuRefs, batch metadata, reply channel … */
    _phantom: std::marker::PhantomData,
    /* … */
};

flashattn.tell(FlashAttnMsg::Forward(Box::new(req)));
```

## Paged KV-cache (vLLM-style)

```rust
use atomr_accel_flashattn::{PagedAttentionRequest, PagedKvCache, SmArch, F16};

// 8 / 16 / 32 / 64 / 128 are the only supported block sizes.
let cache = PagedKvCache::new(
    /* num_blocks      */ 4096,
    /* block_size      */ 16,
    /* num_kv_heads    */ 8,
    /* head_dim        */ 128,
    /* max_blocks_per_seq */ 256,
)?;

let req = PagedAttentionRequest::<F16> {
    arch: SmArch::Sm90a,
    head_dim: 128,
    gqa_ratio: 4,           // GQA: 32 Q heads / 8 KV heads
    /* mask, bias, sink_tokens, softmax_scale … */
    cache,
    /* block_tables: GpuRef<i32>, seq_lens: GpuRef<i32>, etc. */
};

flashattn.tell(FlashAttnMsg::PagedForward(Box::new(req)));
```

Cache layout (must match what your serving loop fills):

```text
K_cache: [num_blocks, num_kv_heads, head_dim, block_size]
V_cache: [num_blocks, num_kv_heads, head_dim, block_size]
block_tables: [num_seqs, max_blocks_per_seq]   i32
seq_lens:     [num_seqs]                        i32
```

## Chunked prefill

`ChunkedPrefillRequest` carries a `ChunkLayout` describing how a
long prompt is split across kernel launches — `chunk_index`,
`total_chunks`, the per-chunk Q range. Combine with
`PagedAttentionRequest` for online prefill / decode interleaving in
serving loops.

## Varlen batching

`VarlenFwdRequest` packs `[total_tokens, num_heads, head_dim]` Q/K/V
tensors plus `cu_seqlens_q` / `cu_seqlens_kv` (`CumulativeSeqlens` —
must be `[0, len_0, len_0+len_1, …]`, monotonically non-decreasing).
`SeqlenOverflow` errors fire when `cu_seqlens` overflows
`batch_size * max_seqlen`.

## Dispatch-key key fields

A miss in `DISPATCH_TABLE` is never silent — `lookup` returns
`DispatchError` naming the missing cell. The cell is exactly:

```text
(arch, dtype, head_dim, causal, varlen, sliding_window, alibi, sink, paged, gqa_ratio)
```

When extending the table, name your new cubin so its mangled symbol
expression is unique across all 10 axes — adding a new `head_dim`
or new `gqa_ratio` is the most common reason to extend.

## Mock vs real

`FlashAttnActor::mock_props` returns an actor that replies
`Err(FlashAttnError::MockMode)` for every message. It exists so the
crate builds + unit-tests on hosts without CUDA. The real actor
(`FlashAttnActor::real_props`) is gated behind `cuda-runtime-tests`
and requires both an `NvrtcActor` ref and a `CudaStream`.

In production, install the real props in your bootstrap path; in
unit tests, prefer the mock variant + the dispatch-table assertions.

## Canonical references

- `crates/atomr-accel-flashattn/src/lib.rs` — public module surface
  + the supported-features matrix.
- `crates/atomr-accel-flashattn/src/dispatch.rs` —
  `SmArch`, `DType`, `DispatchKey`, `DISPATCH_TABLE`, `lookup`.
- `crates/atomr-accel-flashattn/src/{fa2,fa3,varlen,paged,prefill}.rs`
  — one request type per file.
- `crates/atomr-accel-flashattn/tests/flashattn_smoke.rs` — covers
  the canonical (arch, dtype, head_dim, causal, varlen) cells.
- [`docs/gpu-testing.md`](../../../docs/gpu-testing.md) — opt-in GPU
  test gating used by the smoke suite.
- [`docs/features-matrix.md`](../../../docs/features-matrix.md) §
  `atomr-accel-flashattn` — feature flags + transitive deps.

## Common pitfalls

- **Mixing fp8 with `Fa2FwdRequest`.** fa2 has no fp8 path. Use
  `Fa3FwdFp8Request` on Hopper or `Fa3FwdRequest<F16>` /
  `Fa3FwdRequest<Bf16>` if you don't need fp8.
- **Backward on Hopper / Blackwell.** Backward falls through to fa2
  for now. If you build a graph that expects fa3 backward, the
  dispatch lookup will fail; either run the backward on an Ampere
  GPU or accept the fa2 fallback.
- **Reusing a dispatch key after rebuilding the context.** The
  cubin cache survives, but the `KernelHandle` does not — the
  envelope re-resolves through `NvrtcActor`, which checks generation.
  Never cache `KernelHandle` across `ContextReady` transitions.
- **Paged block size outside `(8, 16, 32, 64, 128)`.** The Phase 7
  cubins are templated only on those values.
  `InvalidPagedBlockSize` is a fatal request-construction error.
- **Forgetting `flashattn-paged`.** Without the feature, the
  `paged` module is gone — `PagedAttentionRequest` won't resolve.
  The compiler error mentions `paged`; do not chase it as a missing
  import.
- **GQA shape mistakes.** `gqa_ratio = num_q_heads / num_kv_heads`,
  *not* the absolute KV head count. A 32-head Q + 8-head KV is
  `gqa_ratio = 4`.
- **FA3 backward.** There is none — the actor rejects
  `Backward(Box::new(Fa3 …))` at construction time. Use fa2 backward.
