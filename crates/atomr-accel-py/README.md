# atomr-accel (Python)

Python bindings for [atomr-accel](../..) — drive an actor-supervised
NVIDIA CUDA pipeline directly from Python without juggling streams,
contexts, or hand-rolled retry loops.

```python
import numpy as np
import atomr_accel

with atomr_accel.System.open("my-app") as sys:
    dev = sys.spawn_device(device_id=0)         # real CUDA device

    # Allocate two N×N f32 buffers on-device.
    n = 256
    a = dev.allocate_f32(n * n)
    b = dev.allocate_f32(n * n)
    c = dev.allocate_f32(n * n)

    # Upload from numpy.
    dev.copy_from_numpy(a, np.ones(n * n, dtype=np.float32))
    dev.copy_from_numpy(b, np.full(n * n, 2.0, dtype=np.float32))

    # Run cuBLAS SGEMM — the call blocks until the kernel finishes.
    # Either through the legacy alias on Device:
    dev.sgemm(a, b, c, m=n, n=n, k=n, alpha=1.0, beta=0.0)
    # …or the typed Blas handle (also supports gemm_f64, axpy_f32):
    blas = dev.blas()
    blas.gemm_f32(a, b, c, m=n, n=n, k=n)

    # Pull the result back into a fresh numpy array.
    result = dev.copy_to_numpy(c)
    print(result.reshape(n, n))
```

For hosts without a GPU, pass `mock=True` to `spawn_device` and the
device replies `Unrecoverable("...mock mode")` for any kernel call —
useful for testing the surrounding plumbing in CI.

## Install

The wheel builds with [maturin](https://www.maturin.rs/):

```bash
# from this directory
pip install maturin pytest numpy
maturin develop --release       # builds + installs into the active venv
pytest tests/                    # runs the no-GPU smoke suite
```

For a release wheel:

```bash
maturin build --release --no-default-features --features extension-module
# wheel lands in target/wheels/
```

The pure-Python facade (`python/atomr_accel/__init__.py`) re-exports
the native classes and exception types. Downstream libraries import
from `atomr_accel` and treat `atomr_accel._native` as private.

## Feature flags

The Rust crate matches `atomr-accel`'s feature gating so the wheel can
be built minimal or full:

| Feature           | Adds                                            |
|-------------------|-------------------------------------------------|
| (default)         | `System`, `Device`, `GpuBuffer{F32,F64,I32,U32,U8}`, `Blas`, exceptions |
| `cudnn`           | `Cudnn` handle (`Device.cudnn()`, `conv2d_fwd_f32`)  |
| `cufft`           | `Fft` handle (`Device.fft()`, structural anchor)     |
| `curand`          | `RngGenerator` handle (`Device.rng()`, `set_seed`, `uniform_f32`, `normal_f32`) |
| `cusolver`        | `Solver` handle (structural anchor; spawn path tracked) |
| `nccl`            | `Collective` handle (structural anchor; comm-group bootstrap tracked) |
| `nvrtc`           | `NvrtcKernel` (structural anchor; compile/launch tracked) |
| `cublaslt`        | (placeholder; future Python surface)            |
| `core-libs` / `training-libs` / `full-cuda` | aggregates                          |

```bash
maturin develop --features atomr-accel-py/curand,atomr-accel-py/nvrtc
```

## Public API

| Class / function                                    | What it wraps                                     |
|-----------------------------------------------------|---------------------------------------------------|
| `atomr_accel.System.open(name)`                     | A `atomr_core::actor::ActorSystem` lifetime       |
| `system.spawn_device(id, mock=)`                    | A `DeviceActor` (real or mock)                    |
| `device.allocate_{f32,f64,i32,u32,u8}(len)`         | Typed `DeviceMsg::alloc::<T>` → `GpuBuffer{T}`    |
| `device.copy_from_numpy[_T](buf, np)`               | H2D `DeviceMsg::copy_from_host::<T>`              |
| `device.copy_to_numpy[_T](buf)`                     | D2H `DeviceMsg::copy_to_host::<T>` → numpy        |
| `device.sgemm(a,b,c,m,n,k,...)`                     | cuBLAS SGEMM (legacy alias for `blas.gemm_f32`)   |
| `device.stats()`                                    | `DeviceMsg::Stats` → `DeviceLoad`                 |
| `device.libraries_ready()`                          | `KernelChildren` snapshot probe                   |
| `device.blas()` → `Blas`                            | `ActorRef<BlasMsg>` handle                        |
| `blas.gemm_f32 / gemm_f64 / axpy_f32`               | Typed `BlasMsg::Gemm` / `BlasMsg::L1` dispatch    |
| `device.cudnn()` → `Cudnn` (feat: `cudnn`)          | `ActorRef<CudnnMsg>` handle                       |
| `cudnn.conv2d_fwd_f32(x, w, y, ...)`                | `CudnnMsg::Op(ConvFwdRequest::<f32>)`             |
| `device.fft()` → `Fft` (feat: `cufft`)              | `ActorRef<FftMsg>` handle (Phase 1 anchor)        |
| `device.rng()` → `RngGenerator` (feat: `curand`)    | `ActorRef<RngMsg>` handle                         |
| `rng.set_seed / uniform_f32 / normal_f32`           | `RngMsg::SetSeed` / `Fill(FillRequest::<f32>)`    |
| `Solver`, `Collective` (feat-gated)                 | Handle classes; full method coverage in Phase 1.5 |
| `NvrtcKernel` (feat: `nvrtc`)                       | `KernelHandle` probe (Phase 1 stub)               |
| `GpuBuffer{T}.is_stale()` / `.dtype` / `.len`       | Generation token check + dtype tag                |
| `GpuRuntimeError` (and subclasses)                  | Typed `GpuError` mapping                          |

Every method blocks the calling thread until the underlying actor
replies (the GIL is released for the duration via `py.allow_threads`).
Async wrappers can be layered later via
`pyo3_async_runtimes::tokio::future_into_py`.

## How it works

Three pieces:

1. **A shared tokio runtime.** The first call to
   `System.open(...)` initializes a multi-threaded scheduler; every
   subsequent call reuses it. Implemented in `src/runtime.rs` via
   `pyo3-async-runtimes::tokio::init`.
2. **The `_native` extension module.** `src/lib.rs` registers
   `System`, `Device`, `GpuBuffer`, exceptions, and (feature-gated)
   `RngGenerator` / `NvrtcKernel`. Each Python class wraps a typed
   `ActorRef<...>` from atomr-accel and converts replies via
   `errors::map_gpu`.
3. **The pure-Python facade** at `python/atomr_accel/__init__.py`. Hides
   `_native`; documents the API; gives downstream libraries a stable
   import path.

See [docs/python-bridge.md](../../docs/python-bridge.md) for the
full architecture write-up.

## License

Apache-2.0.
