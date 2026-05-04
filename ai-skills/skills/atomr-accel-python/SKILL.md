---
name: atomr-accel-python
description: Use when consuming atomr-accel from Python — the `atomr_accel` package, `System`/`Device`/`GpuBuffer` lifecycle, numpy float32 H2D/D2H roundtrip, GIL release in blocking calls, mock-mode pytest patterns, and the typed exception hierarchy. Triggers on `import atomr_accel`, building / installing the wheel via maturin, or wiring a Python service that drives a GPU through atomr.
---

# Driving atomr-accel from Python

This skill covers the Python wheel published as `atomr-accel` on
PyPI, with import name `atomr_accel`. For the Rust API, see the
other atomr-accel skills.

## Mental model

```
Python process
  ┌── atomr_accel (pure Python facade) ──┐
  │  System / Device / GpuBuffer / …     │
  │  exception hierarchy                 │
  └────────────────┬─────────────────────┘
                   ▼
  ┌── atomr_accel._native (PyO3 cdylib) ─┐
  │  thin wrappers around ActorRefs      │
  │  numpy float32 marshalling           │
  └────────────────┬─────────────────────┘
                   ▼
  ┌── shared tokio runtime (process-wide) ┐
  └────────────────┬─────────────────────┘
                   ▼
  ┌── atomr-accel-cuda actor tree ────────┐
  │  DeviceActor → ContextActor → …       │
  └───────────────────────────────────────┘
```

One process-wide tokio runtime initialized lazily on the first
`System.open(...)` call. Every Python class is a thin wrapper
around a typed `ActorRef<...>` from the Rust crate.

## Installing

```bash
pip install atomr-accel
# or, for a local wheel build:
cd /path/to/atomr-accel/crates/atomr-accel-py
maturin develop --release --no-default-features --features extension-module
```

Wheels are abi3-py310, so a single binary covers Python 3.10+ on
each (os, arch) pair. Linux x86_64, macOS universal2, and Windows
x86_64 are published; ARM Linux installs from sdist.

## A complete roundtrip

```python
import numpy as np
import atomr_accel

with atomr_accel.System.open("my-app") as sys:
    dev = sys.spawn_device(device_id=0)        # real CUDA device

    # Allocate a 1024-elem float32 buffer on-device.
    buf = dev.allocate_f32(1024)
    print(buf.len, buf.device_id)              # 1024, 0

    # Upload from numpy.
    dev.copy_from_numpy(buf, np.ones(1024, dtype=np.float32))

    # Download into a fresh numpy array.
    arr = dev.copy_to_numpy(buf)
    assert arr.shape == (1024,) and arr.dtype == np.float32
```

`System.open` is sync (blocks while the actor system spins up),
returns a context manager. `spawn_device` is sync. Every kernel
call is sync from Python's perspective — the GIL is released
during the await via `py.allow_threads`, so other Python threads
keep running.

## Mock mode for no-GPU tests

```python
with atomr_accel.System.open("smoke") as sys:
    dev = sys.spawn_device(device_id=0, mock=True)
    # Every typed allocation replies with Unrecoverable("...mock mode").
    with pytest.raises(atomr_accel.Unrecoverable):
        dev.allocate_f32(16)
    # But Stats always replies:
    load = dev.stats()
    assert load.compute_cap_major == 0
```

This lets the smoke suite run on hosts without a CUDA driver,
exercising every actor pipeline (System spawn → DeviceActor →
ContextActor → BlasActor → reply).

## SGEMM end-to-end

```python
n = 256
a = dev.allocate_f32(n * n)
b = dev.allocate_f32(n * n)
c = dev.allocate_f32(n * n)

dev.copy_from_numpy(a, np.ones(n * n, dtype=np.float32))
dev.copy_from_numpy(b, np.full(n * n, 2.0, dtype=np.float32))

dev.sgemm(a, b, c, m=n, n=n, k=n, alpha=1.0, beta=0.0)

result = dev.copy_to_numpy(c).reshape(n, n)
# result == 2 * n  (each output is sum of n × 1 × 2)
```

`sgemm` blocks until the kernel completes (sub-µs wakeup via
`HostFnCompletion`).

## Exception hierarchy

```text
Exception
└── atomr_accel.GpuRuntimeError                  (base)
    ├── atomr_accel.ContextPoisoned              (will restart)
    ├── atomr_accel.OutOfMemory                  (resume)
    ├── atomr_accel.Unrecoverable                (device stops)
    ├── atomr_accel.GpuRefStale                  (buffer used after rebuild)
    ├── atomr_accel.LibraryError                 (cuBLAS/cuDNN/etc. error)
    └── atomr_accel.AskTimeout                   (ask exceeded budget)
```

```python
try:
    dev.sgemm(a, b, c, m=n, n=n, k=n)
except atomr_accel.OutOfMemory:
    shrink_batch_and_retry()
except atomr_accel.GpuRefStale:
    reallocate_then_retry()
```

The variants map 1:1 to Rust's `GpuError`. Pattern-match on the
specific subclass; don't string-match the message.

## GIL strategy

Every blocking method on `Device` releases the GIL with
`py.allow_threads` before awaiting the actor reply. This means:

- Other Python threads keep running during long-running kernels.
- `dev.sgemm(...)` from a thread won't starve a Flask handler.
- `asyncio` callers wrap with `asyncio.to_thread` to keep the
  event loop responsive — the underlying tokio runtime drives
  the actor work concurrently.

Async wrappers (returning Python coroutines that complete on
kernel finish) are a planned addition; today the API is sync
because Python script ergonomics matter more than async fan-out
for typical GPU use.

## What's available behind feature flags

| Class | Feature | Backed by |
|---|---|---|
| `System` / `Device` / `GpuBuffer` / `DeviceLoad` | always-on | cuBLAS via `BlasActor` |
| `RngGenerator` | `curand` (build flag) | `RngActor` |
| `NvrtcKernel` | `nvrtc` (build flag) | `NvrtcActor` |

Build a richer wheel:

```bash
maturin develop --release --features atomr-accel-py/curand,atomr-accel-py/nvrtc
```

The Python facade in `python/atomr_accel/__init__.py` defensively
imports the optional classes — a minimal build doesn't fail; the
optional names just become `None`.

## Canonical references

- [`docs/python-bridge.md`](../../../docs/python-bridge.md) — the
  full architecture: shared tokio runtime, GIL strategy, error
  mapping, extension recipes.
- [`crates/atomr-accel-py/README.md`](../../../crates/atomr-accel-py/README.md)
  — install + quick-start.
- `crates/atomr-accel-py/python/atomr_accel/__init__.py` — the
  pure-Python facade that re-exports `_native`.
- `crates/atomr-accel-py/tests/test_smoke.py` — pytest patterns
  for mock mode.

## Common pitfalls

- **Using `from atomr_accel._native import …`.** Don't. `_native`
  is private and may change between versions. Always import from
  `atomr_accel`.
- **Calling kernel methods from inside `asyncio` without
  `to_thread`.** The blocking call parks the Python thread, which
  parks the event loop. Wrap with `asyncio.to_thread(dev.sgemm, …)`.
- **Assuming `RngGenerator` is always available.** It only exists
  when the wheel was built with `--features curand`. Check
  `atomr_accel.RngGenerator is not None` before use.
- **Holding a `GpuBuffer` past `System.close()`.** Once the system
  terminates, every buffer is invalid; the next op surfaces
  `GpuRefStale`. Drop buffers before close.
- **Numpy dtype mismatch.** `copy_from_numpy` requires
  `dtype=np.float32` and a 1D contiguous array. Reshape /
  `astype` on the Python side; the Rust crate doesn't auto-cast.
- **Spinning up multiple `System`s on the same tokio runtime.**
  Each `System.open` creates a separate `ActorSystem` on the
  shared runtime, which is intentional. But you cannot
  re-initialize the runtime itself — first `open` wins.
