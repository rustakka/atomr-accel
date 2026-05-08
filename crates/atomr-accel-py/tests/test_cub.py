"""``Cub`` handle — surface presence + mock-mode skip.

The ``cub`` cargo feature is opt-in; on minimal builds the import
yields ``None``, in which case the whole module is skipped. When the
feature *is* compiled in, the merge-time wiring agent exposes a
``device.cub()`` constructor; until that lands the test simply checks
that the class exists and has the expected method surface.
"""
from __future__ import annotations

import pytest

from atomr_accel import cub as cub_mod


pytestmark = pytest.mark.skipif(
    cub_mod.Cub is None, reason="cub feature not compiled in"
)


def test_cub_class_exists():
    """Cub is registered as a `_native` PyClass with the expected name."""
    Cub = cub_mod.Cub
    assert Cub is not None
    assert Cub.__name__ == "Cub"


def test_cub_method_surface():
    """Phase 4 surface: a single representative `reduce_sum_f32` plus
    `__repr__`. Method-level coverage lands when the merge agent wires
    `device.cub()` and a real CubActor is reachable from Python."""
    Cub = cub_mod.Cub
    # Both are present on the class even before construction.
    for attr in ("reduce_sum_f32", "__repr__"):
        assert hasattr(Cub, attr), attr
