"""``atomr_accel.tensorrt`` — TensorRt + TrtEngine handles (NVIDIA TensorRT).

The ``TensorRt`` handle is obtained via ``device.tensorrt()`` when the
wheel was built with ``--features tensorrt``. The wrapped crate's
libnvinfer link is itself gated behind the upstream ``tensorrt-link``
feature so the wheel builds on hosts without TensorRT installed.

Phase 4.5++ surface:

- ``TensorRt.runtime_ready()`` — feature-detection probe.
- ``TensorRt.load_engine(path)`` — deserialise a plan file via the
  upstream ``TrtRuntime`` (synchronous safe-Rust path; no actor run
  loop is required). Returns a ``TrtEngine`` Python handle.
- ``TensorRt.build_engine_from_onnx(onnx_path, output_path,
  fp16=False, int8=False, workspace_bytes=...)`` — parse an ONNX
  model and build a serialised engine plan. Gated on the upstream
  ``tensorrt-link`` + ``tensorrt-onnx`` cargo features (passed
  through from this crate's ``--features tensorrt-onnx``).
- ``TrtEngine.is_loaded()`` / ``TrtEngine.num_io_tensors`` /
  ``TrtEngine.binding_info()`` / ``__repr__`` — opaque introspection.
- ``TrtEngine.execute(device, inputs, outputs, input_shapes=None)``
  — bind every I/O tensor's CUdeviceptr, optionally apply dynamic
  input shapes, then call ``enqueueV3`` on the device's primary
  ``CudaStream``. Gated on ``tensorrt-link`` (otherwise raises a
  ``GpuRuntimeError("libnvinfer not available")``).

On builds without TensorRT, both ``TensorRt`` and ``TrtEngine`` are
``None``.
"""

try:
    from ._native import TensorRt, TrtEngine
except ImportError:
    TensorRt = None  # type: ignore[assignment]
    TrtEngine = None  # type: ignore[assignment]

__all__ = ["TensorRt", "TrtEngine"]
