"""``FlashAttn`` handle — surface presence + mock-mode skip."""
from __future__ import annotations

import pytest

from atomr_accel import flashattn as fa_mod


pytestmark = pytest.mark.skipif(
    fa_mod.FlashAttn is None, reason="flashattn feature not compiled in"
)


def test_flashattn_class_exists():
    FlashAttn = fa_mod.FlashAttn
    assert FlashAttn is not None
    assert FlashAttn.__name__ == "FlashAttn"


def test_flashattn_method_surface():
    """Phase 4 surface: `forward_f16` + `__repr__`. FA2 backward, FA3
    forward, varlen, paged, prefill, and the bf16 / fp8 axes follow in
    Phase 4.5."""
    FlashAttn = fa_mod.FlashAttn
    for attr in ("forward_f16", "__repr__"):
        assert hasattr(FlashAttn, attr), attr
