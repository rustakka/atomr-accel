"""``TensorRt`` + ``TrtEngine`` — surface presence + mock-mode skip.

Phase 4.5++ ships:

- ``TensorRt.runtime_ready`` (feature probe).
- ``TensorRt.load_engine(path)`` — synchronous deserialise via
  ``TrtRuntime`` (returns a ``TrtEngine`` handle).
- ``TensorRt.build_engine_from_onnx(onnx_path, output_path, ...)`` —
  parse ONNX + build engine plan; gated on the upstream
  ``tensorrt-link`` + ``tensorrt-onnx`` cargo features.
- ``TrtEngine.is_loaded`` / ``num_io_tensors`` / ``binding_info`` /
  ``execute(device, inputs, outputs, input_shapes=...)`` — opaque
  introspection + inference dispatch.

The whole module is skipped when the wheel was built without
``--features tensorrt``; build/execute paths are further skipped
when libnvinfer isn't linked in (they would surface a clean
``GpuRuntimeError("libnvinfer not available: ...")``).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from atomr_accel import tensorrt as trt_mod
from atomr_accel.errors import GpuRuntimeError


pytestmark = pytest.mark.skipif(
    trt_mod.TensorRt is None, reason="tensorrt feature not compiled in"
)


def test_tensorrt_class_exists():
    TensorRt = trt_mod.TensorRt
    assert TensorRt is not None
    assert TensorRt.__name__ == "TensorRt"


def test_trt_engine_class_exists():
    TrtEngine = trt_mod.TrtEngine
    assert TrtEngine is not None
    assert TrtEngine.__name__ == "TrtEngine"


def test_tensorrt_method_surface():
    """Phase 4.5++ surface — every method should be reachable from
    Python (most degrade gracefully without ``tensorrt-link`` /
    ``tensorrt-onnx``)."""
    TensorRt = trt_mod.TensorRt
    for attr in (
        "runtime_ready",
        "load_engine",
        "build_engine_from_onnx",
        "__repr__",
    ):
        assert hasattr(TensorRt, attr), attr


def test_trt_engine_method_surface():
    TrtEngine = trt_mod.TrtEngine
    for attr in (
        "is_loaded",
        "num_io_tensors",
        "binding_info",
        "execute",
        "__repr__",
    ):
        assert hasattr(TrtEngine, attr), attr


def test_load_engine_missing_file_raises_gpu_error():
    """`load_engine` surfaces IO failures as `GpuRuntimeError`,
    *not* as a panic / segfault. This test runs even without
    libnvinfer because the file-read happens before any FFI."""
    TensorRt = trt_mod.TensorRt
    with pytest.raises(GpuRuntimeError):
        TensorRt.load_engine("/nonexistent/path/to/engine.plan")


def test_load_engine_unlinked_or_invalid_surfaces_clean_error():
    """When the wheel was built without `tensorrt-link`, even a
    perfectly valid plan file can't be deserialised — the upstream
    `TrtRuntime::new` returns `NotLinked`. We surface that as a
    `GpuRuntimeError` whose message contains "libnvinfer not
    available". On hosts *with* libnvinfer this same call path
    surfaces a runtime / deserialise error for our garbage bytes —
    also a `GpuRuntimeError`. Either way, no panic, no segfault.
    """
    TensorRt = trt_mod.TensorRt
    with tempfile.NamedTemporaryFile(delete=False, suffix=".plan") as f:
        f.write(b"\x00" * 16)
        plan_path = f.name
    try:
        with pytest.raises(GpuRuntimeError) as ei:
            TensorRt.load_engine(plan_path)
        msg = str(ei.value).lower()
        assert any(
            tag in msg
            for tag in ("libnvinfer not available", "tensorrt", "deserialize", "runtime")
        ), f"unexpected error message: {msg!r}"
    finally:
        os.unlink(plan_path)


def test_build_engine_from_onnx_missing_file_raises_gpu_error():
    """`build_engine_from_onnx` surfaces IO failures as `GpuRuntimeError`
    rather than panicking. Runs even without libnvinfer because the
    file-read happens before any FFI."""
    TensorRt = trt_mod.TensorRt
    with tempfile.NamedTemporaryFile(delete=True, suffix=".plan") as out:
        out_path = out.name
    with pytest.raises(GpuRuntimeError):
        TensorRt.build_engine_from_onnx(
            "/nonexistent/path/to/model.onnx", out_path
        )


def test_build_engine_from_onnx_unlinked_surfaces_clean_error():
    """Without `tensorrt-link` + `tensorrt-onnx` the upstream
    `build_from_onnx` helper returns `TrtError::NotLinked`. We
    surface that as a `GpuRuntimeError`. With both features on this
    test would still pass for our garbage ONNX bytes — the parser
    rejects them and we surface `Onnx(...)` — also a `GpuRuntimeError`.
    """
    TensorRt = trt_mod.TensorRt
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as f:
        f.write(b"\x00" * 16)
        onnx_path = f.name
    out_fd, out_path = tempfile.mkstemp(suffix=".plan")
    os.close(out_fd)
    try:
        with pytest.raises(GpuRuntimeError) as ei:
            TensorRt.build_engine_from_onnx(onnx_path, out_path)
        msg = str(ei.value).lower()
        assert any(
            tag in msg
            for tag in (
                "libnvinfer not available",
                "tensorrt",
                "tensorrt-link",
                "tensorrt-onnx",
                "onnx",
                "build",
                "parse",
            )
        ), f"unexpected error message: {msg!r}"
    finally:
        os.unlink(onnx_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
