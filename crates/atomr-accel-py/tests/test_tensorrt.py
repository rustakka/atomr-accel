"""``TensorRt`` + ``TrtEngine`` — surface presence + mock-mode skip.

Phase 4.5 ships:

- ``TensorRt.runtime_ready`` (feature probe).
- ``TensorRt.load_engine(path)`` — synchronous deserialise via
  ``TrtRuntime`` (returns a ``TrtEngine`` handle).
- ``TrtEngine.is_loaded`` / ``TrtEngine.num_io_tensors`` /
  ``__repr__`` — opaque introspection.

Gaps (asserted *not* exposed so the test fails loudly if a future
agent half-wires them): ``build_engine_from_onnx``, ``execute``,
``binding_info``. See ``crates/atomr-accel-py/src/tensorrt.rs`` for
rationale.

The whole module is skipped when the wheel was built without
``--features tensorrt``; the ``load_engine`` test is further skipped
when libnvinfer isn't linked in (it would surface a clean
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
    """Phase 4.5 surface: `runtime_ready`, `load_engine`, `__repr__`.

    Build / Execute / Refit follow in the next Phase 4.5 iteration —
    we assert their absence so a half-wired follow-up trips this test
    rather than shipping a half-finished surface.
    """
    TensorRt = trt_mod.TensorRt
    for attr in ("runtime_ready", "load_engine", "__repr__"):
        assert hasattr(TensorRt, attr), attr

    # Negative assertions — these are the documented gaps.
    for missing in ("build_engine_from_onnx", "execute"):
        assert not hasattr(TensorRt, missing), (
            f"{missing!r} appeared on TensorRt; if you're wiring it, "
            "update this test and the module docstring at the same time."
        )


def test_trt_engine_method_surface():
    TrtEngine = trt_mod.TrtEngine
    for attr in ("is_loaded", "num_io_tensors", "__repr__"):
        assert hasattr(TrtEngine, attr), attr

    # Inference / refit aren't on the engine handle yet either.
    for missing in ("execute", "binding_info", "refit"):
        assert not hasattr(TrtEngine, missing), (
            f"{missing!r} appeared on TrtEngine; update this test alongside the wiring."
        )


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
