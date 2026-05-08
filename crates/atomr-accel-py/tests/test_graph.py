"""``atomr_accel.graph`` Phase 1.5 surface tests.

The graph module is always-on at build time (no cargo feature),
mirroring ``patterns`` / ``train`` / ``realtime``. If the wheel was
built before Phase 1.5 — e.g. if ``GraphCapture`` is missing from
``_native`` — every test in this file is skipped gracefully via the
module-level guard.

These tests do NOT exercise the methods end-to-end against a real
CUDA driver. ``GraphCapture.spawn`` lands an actor in *mock mode*;
``Record`` and ``Launch`` reply ``Unrecoverable`` regardless of
script contents. Real-mode capture is gated on Phase 5's
``Device.graph()`` accessor; per-op correctness lives in
``atomr-accel-cuda`` integration tests.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import graph as graph_mod


pytestmark = pytest.mark.skipif(
    graph_mod.GraphCapture is None,
    reason="graph wrapper not compiled into this wheel",
)


# ----- module surface -------------------------------------------------


def test_graph_module_exposes_classes():
    """All three Phase 1.5 classes are importable from the facade."""
    assert graph_mod.GraphCapture is not None
    assert graph_mod.GraphHandle is not None
    assert graph_mod.GraphScript is not None


def test_graph_classes_match_native():
    """The facade re-exports the native classes verbatim."""
    from atomr_accel import _native

    assert graph_mod.GraphCapture is _native.GraphCapture
    assert graph_mod.GraphHandle is _native.GraphHandle
    assert graph_mod.GraphScript is _native.GraphScript


# ----- GraphScript builder -------------------------------------------


def test_graph_script_starts_empty():
    s = graph_mod.GraphScript()
    assert len(s) == 0
    assert "ops=0" in repr(s)


def test_graph_script_add_methods_present():
    """``add_memcpy`` and ``add_sgemm`` exist as bound methods."""
    s = graph_mod.GraphScript()
    assert callable(getattr(s, "add_memcpy"))
    assert callable(getattr(s, "add_sgemm"))


# ----- GraphCapture spawn / record / launch --------------------------


def test_graph_capture_spawn_mock_mode():
    """``GraphCapture.spawn`` succeeds against a mock System."""
    with atomr_accel.System.open("graph-spawn") as sys:
        cap = graph_mod.GraphCapture.spawn(sys, name="g0")
        assert "GraphCapture" in repr(cap)


def test_graph_capture_record_empty_script_in_mock_replies_unrecoverable():
    """Mock-mode actor replies ``Unrecoverable`` to any Record — even
    an empty script. The error subclass is the typed
    ``Unrecoverable`` from ``errors.rs``."""
    with atomr_accel.System.open("graph-record") as sys:
        cap = graph_mod.GraphCapture.spawn(sys)
        script = graph_mod.GraphScript()
        with pytest.raises(atomr_accel.Unrecoverable):
            cap.record(script, timeout_secs=2.0)


def test_graph_capture_launch_synthetic_handle_in_mock_replies_unrecoverable():
    """Mock-mode actor replies ``Unrecoverable`` to Launch as well."""
    with atomr_accel.System.open("graph-launch") as sys:
        cap = graph_mod.GraphCapture.spawn(sys)
        h = graph_mod.GraphHandle.synthetic()
        with pytest.raises(atomr_accel.Unrecoverable):
            cap.launch(h, timeout_secs=2.0)


# ----- GraphHandle synthetic + accessors -----------------------------


def test_graph_handle_synthetic_has_zero_generation():
    """``GraphHandle.synthetic()`` carries a zero-generation marker
    (the actor uses generation parity to reject stale launches)."""
    h = graph_mod.GraphHandle.synthetic()
    assert h.generation == 0
    assert "GraphHandle" in repr(h)


def test_graph_handle_export_dot_does_not_panic():
    """``export_dot`` on a synthetic handle either:
      * returns a (possibly empty / whitespace) DOT string when the
        CUDA driver is loadable, OR
      * raises ``Unrecoverable`` / ``LibraryError`` on driverless hosts.
    Either way the call must not panic / segfault."""
    h = graph_mod.GraphHandle.synthetic()
    try:
        out = h.export_dot(flags=0)
    except (atomr_accel.Unrecoverable, atomr_accel.LibraryError):
        return
    # Real driver — accept any returned string (may be empty for a
    # null cu_graph).
    assert isinstance(out, str)
