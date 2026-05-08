"""``TensorRt`` handle — surface presence + mock-mode skip.

TensorRt is structural-anchor-only in Phase 4 (see
``crates/atomr-accel-py/src/tensorrt.rs`` for rationale). The single
exposed method, ``runtime_ready``, is a feature-detection probe.
"""
from __future__ import annotations

import pytest

from atomr_accel import tensorrt as trt_mod


pytestmark = pytest.mark.skipif(
    trt_mod.TensorRt is None, reason="tensorrt feature not compiled in"
)


def test_tensorrt_class_exists():
    TensorRt = trt_mod.TensorRt
    assert TensorRt is not None
    assert TensorRt.__name__ == "TensorRt"


def test_tensorrt_method_surface():
    """Phase 4 surface: `runtime_ready` + `__repr__`. Build,
    Deserialize, CreateContext, EnqueueOnStream, and Refit follow in
    Phase 4.5."""
    TensorRt = trt_mod.TensorRt
    for attr in ("runtime_ready", "__repr__"):
        assert hasattr(TensorRt, attr), attr
