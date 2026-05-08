"""``atomr_accel.tensorrt`` — TensorRt + TrtEngine handles (NVIDIA TensorRT).

The ``TensorRt`` handle is obtained via ``device.tensorrt()`` when the
wheel was built with ``--features tensorrt``. The wrapped crate's
libnvinfer link is itself gated behind the upstream ``tensorrt-link``
feature so the wheel builds on hosts without TensorRT installed.

Phase 4.5 surface:

- ``TensorRt.runtime_ready()`` — feature-detection probe.
- ``TensorRt.load_engine(path)`` — deserialise a plan file via the
  upstream ``TrtRuntime`` (synchronous safe-Rust path; no actor run
  loop is required). Returns a ``TrtEngine`` Python handle.
- ``TrtEngine.is_loaded()`` / ``TrtEngine.num_io_tensors`` /
  ``__repr__`` — opaque introspection.

Gaps (carried forward to the next Phase 4.5 iteration):

- ``build_engine_from_onnx`` is **not** exposed: the upstream
  ``TrtActor`` mailbox is design-only (no run loop), and the
  ``IBuilder`` / ``nvonnxparser`` FFI shims live behind the upstream
  ``tensorrt-link`` + ``tensorrt-onnx`` cargo features that don't
  pass through to the Python wheel's feature flags.
- ``engine.execute(...)`` is **not** exposed: it requires raw
  ``CUdeviceptr`` access (cudarc keeps it crate-private), an
  ``Arc<CudaStream>`` from ``DeviceActor`` (no Python accessor), and
  ``IExecutionContext::enqueueV3`` (link-gated).

On builds without TensorRT, both ``TensorRt`` and ``TrtEngine`` are
``None``.
"""

try:
    from ._native import TensorRt, TrtEngine
except ImportError:
    TensorRt = None  # type: ignore[assignment]
    TrtEngine = None  # type: ignore[assignment]

__all__ = ["TensorRt", "TrtEngine"]
