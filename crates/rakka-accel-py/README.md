# rakka-accel (Python)

Python bindings for [rakka-accel](../..) — drive an actor-supervised
NVIDIA CUDA pipeline directly from Python without juggling streams,
contexts, or hand-rolled retry loops.

```python
import numpy as np
import rakka_accel

with rakka_accel.System.open("my-app") as sys:
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
    dev.sgemm(a, b, c, m=n, n=n, k=n, alpha=1.0, beta=0.0)

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

The pure-Python facade (`python/rakka_accel/__init__.py`) re-exports
the native classes and exception types. Downstream libraries import
from `rakka_accel` and treat `rakka_accel._native` as private.

## Feature flags

The Rust crate matches `rakka-accel`'s feature gating so the wheel can
be built minimal or full:

| Feature           | Adds                                            |
|-------------------|-------------------------------------------------|
| (default)         | `System`, `Device`, `GpuBuffer`, exceptions     |
| `curand`          | `RngGenerator`                                  |
| `nvrtc`           | `NvrtcKernel`                                   |
| `cudnn` / `cufft` / `cusolver` / `cublaslt` / `nccl` | placeholder; future Python surfaces |
| `core-libs` / `training-libs` / `full-cuda`         | aggregates                          |

```bash
maturin develop --features rakka-accel-py/curand,rakka-accel-py/nvrtc
```

## Public API

| Class / function                | What it wraps                                       |
|---------------------------------|-----------------------------------------------------|
| `rakka_accel.System.open(name)`  | A `rakka_core::actor::ActorSystem` lifetime         |
| `system.spawn_device(id, mock=)` | A `DeviceActor` (real or mock)                      |
| `device.allocate_f32(len)`      | `DeviceMsg::AllocateF32` → `GpuBuffer`              |
| `device.copy_from_numpy(buf, np)` | H2D `CopyFromHostF32`                              |
| `device.copy_to_numpy(buf)`     | D2H `CopyToHostF32` → numpy `float32` array         |
| `device.sgemm(a,b,c,m,n,k,...)` | cuBLAS SGEMM via `BlasActor`                        |
| `device.stats()`                | `DeviceMsg::Stats` → `DeviceLoad`                   |
| `GpuBuffer.is_stale()`          | Generation token check vs. `DeviceState`            |
| `GpuRuntimeError` (and subclasses) | Typed `GpuError` mapping                         |

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
   `ActorRef<...>` from rakka-accel and converts replies via
   `errors::map_gpu`.
3. **The pure-Python facade** at `python/rakka_accel/__init__.py`. Hides
   `_native`; documents the API; gives downstream libraries a stable
   import path.

See [docs/python-bridge.md](../../docs/python-bridge.md) for the
full architecture write-up.

## License

Apache-2.0.
