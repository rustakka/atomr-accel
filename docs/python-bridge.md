# Python bridge architecture

`atomr-accel-py` exposes the atomr-accel actor system to Python so
downstream libraries (PyTorch-adjacent runtimes, data pipelines,
research notebooks) can drive CUDA through the same supervision /
generation / completion machinery the Rust API gets.

```
                ┌─────────────────────── Python process ───────────────────────┐
                │                                                              │
                │  ┌── atomr_accel (pure Python facade) ──┐                     │
                │  │ from atomr_accel import System, ...  │                     │
                │  └────────────────┬────────────────────┘                     │
                │                   │                                          │
                │                   ▼                                          │
                │  ┌── atomr_accel._native (PyO3 cdylib) ─────────────────────┐ │
                │  │ class System  ─── ActorSystem                           │ │
                │  │ class Device  ─── ActorRef<DeviceMsg>                   │ │
                │  │ class GpuBuffer{F32,F64,I32,U32,U8} ─ Mutex<Opt<GpuRef>> │ │
                │  │ class Blas    ─── ActorRef<BlasMsg>                     │ │
                │  │ class Cudnn   (feature: cudnn)                          │ │
                │  │ class Fft     (feature: cufft)                          │ │
                │  │ class RngGenerator (feature: curand)                    │ │
                │  │ class Solver  (feature: cusolver)                       │ │
                │  │ class Collective (feature: nccl)                        │ │
                │  │ class NvrtcKernel (feature: nvrtc)                      │ │
                │  │ exceptions: GpuRuntimeError + 6 typed subclasses        │ │
                │  └─────────────────────────────────────────────────────────┘ │
                │                   │                                          │
                │                   ▼                                          │
                │  ┌── shared tokio runtime (process-wide) ──────────────────┐ │
                │  │ pyo3-async-runtimes::tokio::init(...)                   │ │
                │  │ multi-threaded scheduler, all I/O drivers enabled       │ │
                │  └─────────────────────────────────────────────────────────┘ │
                │                   │                                          │
                │                   ▼                                          │
                │  ┌── atomr-accel actor tree ────────────────────────────────┐ │
                │  │ DeviceActor → ContextActor → BlasActor / CudnnActor /   │ │
                │  │ FftActor / RngActor / NvrtcActor / ...                  │ │
                │  └─────────────────────────────────────────────────────────┘ │
                └──────────────────────────────────────────────────────────────┘
```

## Design choices

### Sync API by default; async-ready underneath

`Device.allocate_f32(len)`, `device.sgemm(...)`, `device.copy_to_numpy(buf)`
all **block** the calling Python thread until the underlying actor
replies. The implementation:

1. Captures the typed message and clones the `ActorRef`.
2. Releases the GIL (`py.allow_threads`).
3. Calls `runtime().block_on(async move { actor.tell(msg); rx.await })`
   — the future is driven on the shared tokio runtime, not on the
   Python thread.
4. Reacquires the GIL, maps `Result<_, GpuError>` into either a
   typed Python exception or a Python value.

This keeps Python code straight-line ("I called allocate, I got a
buffer back") while preserving atomr's async semantics inside.
Futures-based wrappers can be added later via
`pyo3_async_runtimes::tokio::future_into_py`; the underlying actor
machinery doesn't change.

### One process-wide tokio runtime

`src/runtime.rs` initializes a multi-threaded tokio runtime the first
time anyone calls `System.open(...)`. Every subsequent `System`
shares it. This is the documented `pyo3-async-runtimes` pattern and
matches the atomr-pycore extension exactly.

You **cannot** create multiple runtimes per process — pyo3-async-runtimes
panics on double-init. If you need isolated systems for testing, use
multiple `System.open(...)` calls; each spawns an independent
`ActorSystem` on the shared runtime.

### Typed errors → typed Python exceptions

`src/errors.rs` declares a class hierarchy:

```
Exception
└── GpuRuntimeError                         (base)
    ├── ContextPoisoned                     (CUDA context poisoned; will restart)
    ├── OutOfMemory                         (allocator OOM; supervisor resumes)
    ├── Unrecoverable                       (hardware fault / past retry budget)
    ├── GpuRefStale                         (buffer used after context rebuild)
    ├── LibraryError                        (cuBLAS/cuDNN/etc. error)
    └── AskTimeout                          (ask exceeded its budget)
```

`map_gpu(GpuError) -> PyErr` pattern-matches the typed enum into the
right subclass. Downstream Python code uses `try / except` against
the specific subclasses to recover (`except OutOfMemory: shrink_batch_size()`).

### `GpuBuffer`: opaque, generation-validated

The Python `GpuBuffer` wraps `Mutex<Option<GpuRef<f32>>>`. The `Option`
exists so future ops can move the underlying `GpuRef` out (e.g. into
an SGEMM keep-alive); the `Mutex` makes the wrapper `Send` for
`#[pyclass]`. `len`, `device_id`, and `is_stale()` are zero-cost
probes.

There's intentionally no way to mint a raw device pointer from
Python — that would defeat the generation-token guarantees. Reads
and writes go through `Device.copy_{to,from}_numpy`.

### Numpy as the data path

`PyReadonlyArray1<'_, f32>` (input) and `PyArray1<f32>` (output) come
from the `numpy` crate. The bridge currently supports `f32`
1D contiguous arrays; broader dtype/shape coverage follows the atomr-accel
typed-allocate matrix (`f64`, `i32`, `u32`, `u8`, plus `f16` / `bf16`
under the `f16` feature).

`copy_from_numpy` materializes `Vec<f32>` from the numpy buffer
on the Python side, then hands ownership to `HostBuf::Owned` — the
actor pipeline then runs an async H2D copy. `copy_to_numpy` does the
reverse: pre-allocates the destination as `Vec<f32>`, awaits the
reply, and constructs a fresh `PyArray1` from the buffer.

For zero-copy or pinned-memory paths, future iterations can route
through `HostBuf::Pinned` against `PinnedBufferPool`. The Python API
stays the same; the implementation chooses the path.

### Feature-gated optional surfaces

`RngGenerator` requires `curand`; `NvrtcKernel` requires `nvrtc`. The
crate's `Cargo.toml` mirrors `atomr-accel-cuda`'s feature flags so a wheel
built with `--features curand,nvrtc` exposes those classes; a minimal
build doesn't. The Python facade gracefully handles either case:

```python
try:
    from atomr_accel._native import RngGenerator
except ImportError:
    RngGenerator = None
```

## Extending the bridge

Adding a new typed message follows a small pattern:

1. Add a method on `PyDevice` (or a new `#[pyclass]` if it's a
   different actor).
2. Inside the method, clone the `ActorRef`, build the message with a
   `oneshot::Sender`, and call `runtime().block_on(...)` inside
   `py.allow_threads`.
3. Map the reply with `errors::map_gpu` for typed errors, or
   construct the Python return value manually.
4. Register the class in `src/lib.rs::_native`.

For an async surface, replace `runtime().block_on` with
`pyo3_async_runtimes::tokio::future_into_py` and have the method
return `Bound<'py, PyAny>` — Python callers `await` the result.

## Why not just call cudarc directly from Python?

Two reasons.

1. **Supervision and recovery.** Direct cudarc bindings give you the
   raw API but no story for context poisoning, sticky errors, or
   handle restart. atomr-accel packages those concerns behind the
   actor surface, and the Python bridge inherits them for free.
2. **One async runtime to rule them all.** Pyo3-async-runtimes plus
   tokio plus atomr actors plus cudarc add up to a single integrated
   stack. Rolling your own `Mutex<CublasHandle>` across `cffi` /
   `ctypes` / `pycuda` is the kind of thing that works fine in a
   notebook and crashes at scale.

## Status

`Phase 1 — structural parity with atomr 0.5.x.` Upstream atomr's "Depth
Wave" (0.5.0) extended Python coverage to the full Rust public surface;
this repo mirrors that organization for the CUDA backend.

**Phase 1 surface (this iteration):**

- `System` / `Device` lifecycle (unchanged).
- Multi-dtype buffers: `GpuBufferF32`, `GpuBufferF64`, `GpuBufferI32`,
  `GpuBufferU32`, `GpuBufferU8`. `GpuBuffer` is a back-compat alias
  for `GpuBufferF32`.
- Typed allocate/copy: `Device.allocate_{f32,f64,i32,u32,u8}` plus
  matching `copy_from_numpy_*` / `copy_to_numpy_*`.
- `Device.libraries_ready()` — probe which kernel actors are live.
- `Device.blas()` → `Blas` handle: `gemm_f32`, `gemm_f64`, `axpy_f32`.
- `Device.cudnn()` → `Cudnn` handle (feat: `cudnn`): `conv2d_fwd_f32`.
- `Device.fft()` → `Fft` handle (feat: `cufft`): structural anchor.
- `Device.rng()` → `RngGenerator` handle (feat: `curand`):
  `set_seed`, `uniform_f32`, `normal_f32`.
- `Solver`, `Collective`, `NvrtcKernel` — handle classes shipped as
  structural anchors; full method coverage in Phase 1.5 (see below).
- Per-domain Python facade modules (`atomr_accel.{system,device,blas,
  cudnn,fft,rng,solver,collective,nvrtc,errors}`).

**Tracked follow-ups** (GitHub issues filed at the same time as this
release):

- *Phase 1.5 — cuBLAS depth.* Strided-batched gemm; the rest of L1
  (`dot`, `nrm2`, `scal`, `asum`, `iamax`, `iamin`, `copy`, `swap`,
  `rot`); L2 (`gemv`, `ger`); L3 (`geam`, `syrk`, `trsm`). All across
  f32/f64/f16/bf16.
- *Phase 1.5 — cuDNN depth.* Pool / batch_norm / layer_norm / RNN /
  attention / dropout, plus backward passes.
- *Phase 1.5 — cuFFT depth.* Typed plans (R2C/C2R/C2C across f32/f64,
  1-D/2-D/3-D, plan-many, callbacks). Requires numpy↔complex
  marshalling.
- *Phase 1.5 — cuRAND depth.* Quasi generators, log-normal / Poisson /
  exponential / beta / Cauchy / gamma / discrete distributions, and
  per-dtype variants (`uniform_f64`, `uniform_u32`, …).
- *Phase 1.5 — cuSOLVER depth.* `Device.solver()` accessor + LU / QR /
  Cholesky / SVD / eigendecomp.
- *Phase 1.5 — NVRTC depth.* `Device.compile_kernel(name, src)` →
  `NvrtcKernel`; `kernel.launch(grid, block, shared, args)` with
  typed `KernelArg` marshalling.
- *Phase 2 — multi-GPU.* `Device.collective()` / NcclWorldActor
  bootstrap; broadcast / all-reduce / all-gather / reduce-scatter /
  all-to-all / send-recv across rank groups.
- *Phase 2 — CUDA graphs + memory ops.* `Device.capture_graph()`
  context-manager, managed allocation, `prefetch` / `advise` / IPC.
- *Phase 3 — patterns / train / agents / telemetry.* Wrap
  `atomr-accel-{patterns,train,agents,telemetry,cuda-realtime}`.
- *Phase 4 — template kernel crates.* Wrap `atomr-accel-{cub,cutlass,
  flashattn,tensorrt}`.
- *Phase 5 — async API.* `pyo3_async_runtimes::tokio::future_into_py`
  variants alongside the existing blocking methods.
