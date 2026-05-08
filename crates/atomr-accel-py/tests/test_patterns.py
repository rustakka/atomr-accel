"""``atomr_accel.patterns`` surface tests.

Phase 2 ships the seven canonical pattern actors as structural
anchors — every one is generic over a Python-typed Req/Resp or
expert protocol that hasn't been bridged yet. These tests verify
the symbol surface and ``__repr__`` shape so downstream Python code
can import the names without crashing.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import patterns


PATTERN_NAMES = [
    "DynamicBatchingServer",
    "InferenceCascade",
    "ModelReplicaPool",
    "FairShareScheduler",
    "HotSwapServer",
    "SpeculativeDecoder",
    "MoeRouter",
]


def test_patterns_module_exposes_handles():
    """Each handle is importable from the facade."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        assert cls is not None, name


def test_patterns_classes_are_types():
    """They're classes, not modules / placeholders."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        assert isinstance(cls, type), name


def test_pattern_handles_have_repr():
    """Every PyClass has a ``__repr__`` method (defined via PyO3)."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        assert hasattr(cls, "__repr__"), name


def test_native_module_has_pattern_classes():
    """The ``_native`` module hosts the same handle names."""
    native = atomr_accel._native  # type: ignore[attr-defined]
    for name in PATTERN_NAMES:
        assert hasattr(native, name), name


def test_pattern_handles_not_constructable_from_python():
    """Phase 2 ships them as structural anchors — no ``__new__``,
    so direct construction must error."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        with pytest.raises(TypeError):
            cls()
