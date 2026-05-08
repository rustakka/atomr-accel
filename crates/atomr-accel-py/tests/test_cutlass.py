"""``Cutlass`` handle — surface presence + mock-mode skip."""
from __future__ import annotations

import pytest

from atomr_accel import cutlass as cutlass_mod


pytestmark = pytest.mark.skipif(
    cutlass_mod.Cutlass is None, reason="cutlass feature not compiled in"
)


def test_cutlass_class_exists():
    Cutlass = cutlass_mod.Cutlass
    assert Cutlass is not None
    assert Cutlass.__name__ == "Cutlass"


def test_cutlass_method_surface():
    """Phase 4 surface: `gemm_f32_plan`, `gemm_f64_plan`, `dispatched`,
    `plan_cache_len`, `__repr__`. Method-level coverage lands when the
    merge agent wires `device.cutlass()` to a real CutlassActor."""
    Cutlass = cutlass_mod.Cutlass
    for attr in (
        "gemm_f32_plan",
        "gemm_f64_plan",
        "dispatched",
        "plan_cache_len",
        "__repr__",
    ):
        assert hasattr(Cutlass, attr), attr
