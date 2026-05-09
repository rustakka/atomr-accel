---
name: atomr-accel-tensorrt
description: Use when wiring or extending TensorRT inference through `atomr-accel-tensorrt` — the `TrtActor` lifecycle (`Build` / `Deserialize` / `CreateContext` / `EnqueueOnStream` / `Refit`), `IBuilderConfig` knobs (precision / DLA / sparsity / tactic sources / timing cache), ONNX import, INT8 / FP8 PTQ calibration, IPluginV3 trampolines, and stream sharing with `DeviceActor`. Triggers on building or refitting a TensorRT engine, picking precision, calibrating quantization, or integrating with the device actor's stream.
---

# TensorRT engines + runtime

This skill covers the Phase 8 sibling crate. Enable the `tensorrt`
feature on `atomr-accel-cuda` and `TrtActor` becomes available
**alongside** `DeviceActor` (not as a child) — TensorRT manages its
own resources but shares CUDA streams with the device actor so
inference rides the same execution timeline. For per-library kernel
actor patterns see [`atomr-accel-kernels`](../atomr-accel-kernels/SKILL.md).

## Library link is opt-in

`libnvinfer.so` is **proprietary** and not vendored. Two layers
control whether it's loaded:

1. **`atomr-accel-cuda` feature `tensorrt`** — pulls in
   `atomr-accel-tensorrt` so `TrtActor` types are in scope.
2. **`atomr-accel-tensorrt/tensorrt-link` feature** — _currently
   disabled_. Enabling it triggers a `compile_error!` until the
   `nvinfer_shim.cpp` C-ABI shim lands; tracked by
   <https://github.com/rustakka/atomr-accel/issues/6>. The eventual
   shipping behaviour is: `build.rs` probes `LIBNVINFER_PATH`
   first, then standard libdirs, and panics with a clear hint if
   `libnvinfer.so` is missing.

Without `tensorrt-link` the crate compiles and unit-tests on hosts
without TensorRT — `TrtActor::ensure_runtime` returns
`TrtError::LibraryUnavailable` and tests skip cleanly.

## Cargo features

```toml
[dependencies.atomr-accel-cuda]
version  = "0.1"
features = ["tensorrt", "tensorrt-onnx", "tensorrt-int8"]

[dependencies.atomr-accel-tensorrt]
version  = "0.1"
features = ["tensorrt-link"]   # disabled until nvinfer_shim.cpp lands (issue #6)
```

Per-feature add-ons:

| Feature | Adds |
|---|---|
| `tensorrt-link` | _disabled_ — see [issue #6](https://github.com/rustakka/atomr-accel/issues/6); will link `libnvinfer.so` once the C++ shim lands |
| `tensorrt-onnx` | `OnnxParser` + `onnx_resnet50_int8` example |
| `tensorrt-int8` | INT8 entropy / minmax PTQ calibrator helpers |
| `tensorrt-fp8` | FP8 PTQ helpers (Hopper-class GPUs) |
| `tensorrt-plugin` | IPluginV3 Rust trampolines |

## Actor lifecycle

`TrtMsg` covers the full engine + runtime cycle:

| Message | What it does |
|---|---|
| `Build { source, config, reply }` | Drive `IBuilder::buildSerializedNetwork`, return an `EnginePlan` |
| `Deserialize { plan, reply }` | Load a previously built plan into a shared `Arc<TrtEngine>` |
| `CreateContext { engine, reply }` | Create a fresh `ExecutionContext` |
| `EnqueueOnStream { stream, context, bindings, reply }` | `enqueueV3` on a `DeviceActor`-owned stream |
| `Refit { weights, reply }` | Patch engine weights (requires `RefitPolicy::OnDemand` or `WeightsStreaming` at build time) |

Engines are kept in `Arc<TrtEngine>` so multiple
`ExecutionContext`s can share one. The actor serialises access to
each context (`ExecutionContext` is `Send + Sync` only because the
actor mailbox owns it for life).

## Building from ONNX

```rust
use atomr_accel_tensorrt::{
    BuilderFlags, IBuilderConfig, NetworkSource, Precision, RefitPolicy, TacticSources, TrtMsg,
};

let onnx_bytes: Vec<u8> = std::fs::read("model.onnx")?;

let mut config = IBuilderConfig::default();
config.precision = Precision::Best;       // FP16 | BF16 | INT8 | FP8 | TF32
config.refit = RefitPolicy::OnDemand;     // allow Refit later
config.tactic_sources = TacticSources::CUBLAS_LT | TacticSources::CUDNN;
config.set_workspace_bytes(2 * 1024 * 1024 * 1024); // 2 GiB

trt.tell(TrtMsg::Build {
    source: NetworkSource::Onnx(onnx_bytes),
    config: Box::new(config),
    reply: build_tx,
});

let plan: EnginePlan = build_rx.await??;
```

`Precision::Best` lets the builder pick per-layer; use a specific
arm (`Fp16`, `Int8`, `Fp8`) when you need a hard precision floor.
TF32 is on by default.

## Sharing the device stream

`EnqueueOnStream` accepts an `Arc<cudarc::driver::CudaStream>` —
the same stream type carried by `atomr_accel_cuda::DeviceActor`.
Pass the stream from a `DeviceMsg::SnapshotStream` reply and the
TensorRT runtime joins the device's execution timeline:

```rust
let stream = device.ask_with(
    |tx| DeviceMsg::SnapshotStream { reply: tx },
    Duration::from_secs(5),
).await??;

trt.tell(TrtMsg::EnqueueOnStream {
    stream,
    context: ctx,
    bindings,
    reply: enqueue_tx,
});
```

No cross-stream sync, no extra event hops. Completion is observed
through the same `CompletionStrategy` your `ContextActor` is
configured with.

## Bindings — names + device pointers

`ExecutionBindings` is `name → device_ptr (u64)` plus per-input
`TensorShape` (`nvinfer1::Dims`-shaped, max 8 dims). Pointers are
raw `u64`s so the message stays `Send + Sync` without lifetimes
from `Arc<CudaSlice<T>>`:

```rust
use atomr_accel_tensorrt::{ExecutionBindings, TensorShape};

let mut bindings = ExecutionBindings::new();
bindings.bind("input_ids", q_ref.device_ptr_raw());
bindings.bind("logits",     out_ref.device_ptr_raw());
bindings.set_shape("input_ids", TensorShape::new(&[1, 512]));
```

Dynamic shapes are set per-call via `set_shape`; the engine must
have been built with the matching `OptimizationProfile`.

## INT8 / FP8 PTQ calibration

```rust
#[cfg(feature = "tensorrt-int8")]
use atomr_accel_tensorrt::calibration::{Int8EntropyCalibrator, Int8MinmaxCalibrator};

let calib = Int8EntropyCalibrator::new(/* CalibrationData iterator */, "model.cache");
config.precision = Precision::Int8;
config.flags |= BuilderFlags::INT8;
config.set_int8_calibrator(Box::new(calib));
```

FP8 PTQ (Hopper) layers identically through
`tensorrt-fp8`. Calibration data is iterated once at build time;
the cache file lets subsequent builds skip the calibration pass.

## Refitting weights without rebuild

```rust
trt.tell(TrtMsg::Refit {
    weights: vec![RefitWeights {
        name: "encoder.layer.0.weight".into(),
        bytes: new_weight_blob,
        dtype: sys::DataType::FP16,
    }],
    reply: refit_tx,
});
```

The engine must have been built with `RefitPolicy::OnDemand` or
`WeightsStreaming`. Refit is in-place against the engine's bound
workspace — no rebuild, no NVRTC.

## IPluginV3 trampolines

`tensorrt-plugin` exposes a Rust trait surface that conforms to
the `IPluginV3` ABI. Implement the trait, register your factory,
and TensorRT can call into your Rust code from inside an engine —
useful for custom layers that aren't expressible via standard
ONNX ops or built-in TRT plugins.

## Mock vs real

`TrtActor::ensure_runtime` lazily loads `libnvinfer.so` and
returns `TrtError::LibraryUnavailable` when missing. The smoke
test (`tests/tensorrt_smoke.rs`) is gated `#[ignore]` and uses
this path: it passes both with and without TensorRT installed.

## Canonical references

- `crates/atomr-accel-tensorrt/src/lib.rs` — public surface,
  feature-flag matrix.
- `crates/atomr-accel-tensorrt/src/actor.rs` — `TrtActor`,
  `TrtMsg`, lifecycle docs.
- `crates/atomr-accel-tensorrt/src/builder.rs` — `IBuilderConfig`,
  `BuilderFlags`, `Precision`, `RefitPolicy`, `TacticSources`.
- `crates/atomr-accel-tensorrt/src/runtime.rs` —
  `ExecutionContext`, `ExecutionBindings`, `TensorShape`.
- `crates/atomr-accel-tensorrt/src/engine.rs` — `TrtEngine`,
  `EnginePlan`, `TrtRefitter`.
- `crates/atomr-accel-tensorrt/examples/onnx_resnet50_int8.rs` —
  end-to-end ONNX → INT8 → enqueue.
- [`docs/features-matrix.md`](../../../docs/features-matrix.md) §
  `atomr-accel-tensorrt` — feature flags + transitive deps.

## Common pitfalls

- **Forgetting `tensorrt-link`.** Without it the crate compiles
  but `ensure_runtime` always returns `LibraryUnavailable`. For
  CI without TensorRT, leave it off and use the smoke-skip path;
  for production, turn it on and set `LIBNVINFER_PATH` if the
  library isn't on the standard `LD_LIBRARY_PATH`.
- **Calling `Refit` on a non-refit engine.** Builders default to
  `RefitPolicy::Disabled`. Set `OnDemand` (or `WeightsStreaming`)
  at build time or the refit fails fast with
  `TrtError::EngineNotRefittable`.
- **Mixing TensorRT streams with `DeviceActor` streams.** Don't.
  Use `EnqueueOnStream` and pass the device's stream — that's the
  whole point of the cross-actor stream-sharing contract.
- **Ignoring `tensorrt-plugin` ABI.** IPluginV3 has hard ABI
  requirements (versioning, reference counting, lifetime). Read
  `plugin.rs`'s safety notes before implementing — bugs here are
  silent crashes inside `enqueueV3`.
- **Building one engine per request.** `IBuilder` runs are
  expensive (seconds to minutes). Build once, deserialize many
  times (one `ExecutionContext` per concurrent request).
- **`Precision::Best` without a calibrator.** It enables INT8 / FP8
  flags, but without calibration data the builder falls back to
  FP16 / BF16 silently. Verify with `IBuilderConfig::get_quant_flags`
  after build.
- **TensorShape with > 8 dims.** TensorRT enforces a hard cap of
  8; `TensorShape::new` panics on overflow.
