"""Telemetry bindings tests — NVTX / NVML / CUPTI surface presence.

Mock-mode parity: each test must run on a host without CUDA / NVTX /
NVML / CUPTI installed. Where the underlying library is required,
the constructor must surface a typed exception (``Unrecoverable``)
rather than panic across the FFI boundary.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import telemetry


def test_telemetry_module_importable():
    """The pure-Python facade imports cleanly even on minimal builds
    where every backend is ``None``."""
    assert hasattr(telemetry, "NvtxKernelTrace")
    assert hasattr(telemetry, "NvmlActor")
    assert hasattr(telemetry, "CuptiSession")


@pytest.mark.skipif(
    telemetry.NvtxKernelTrace is None,
    reason="telemetry-nvtx feature not compiled in",
)
def test_nvtx_kernel_trace_is_context_manager():
    """``NvtxKernelTrace`` is a context manager. On hosts without
    ``libnvToolsExt.so`` (the CI default) ``__enter__`` raises
    ``Unrecoverable``; on hosts where NVTX is installed the with-block
    enters and exits cleanly."""
    span = telemetry.NvtxKernelTrace("test-span")
    assert "NvtxKernelTrace" in repr(span)
    try:
        with span:
            pass
    except atomr_accel.Unrecoverable:
        # Expected on mock-mode CI.
        pass


@pytest.mark.skipif(
    telemetry.NvmlActor is None, reason="telemetry-nvml feature not compiled in"
)
def test_nvml_actor_mock_mode_unrecoverable():
    """``NvmlActor()`` on a host without ``libnvidia-ml.so.1`` raises
    ``Unrecoverable``. On a host with NVML the actor spawns and
    ``read()`` returns a dict."""
    try:
        actor = telemetry.NvmlActor(interval_secs=0.5)
    except atomr_accel.Unrecoverable:
        return
    snap = actor.read(timeout_secs=2.0)
    assert "devices" in snap
    assert "generated_at_unix_nanos" in snap


@pytest.mark.skipif(
    telemetry.CuptiSession is None,
    reason="telemetry-cupti feature not compiled in",
)
def test_cupti_session_lifecycle():
    """CUPTI session spawns, accepts ``start`` / ``stop`` / ``drain``
    even on mock-mode hosts (the actor stores categories without
    enabling CUPTI's FFI when libcupti is missing)."""
    sess = telemetry.CuptiSession()
    assert "CuptiSession" in repr(sess)
    sess.start(categories=["kernel", "memcpy"], timeout_secs=2.0)
    sess.stop(timeout_secs=2.0)
    records = sess.drain(timeout_secs=2.0)
    # Mock-mode: no real CUPTI records were captured.
    assert records == []


@pytest.mark.skipif(
    telemetry.CuptiSession is None,
    reason="telemetry-cupti feature not compiled in",
)
def test_cupti_unknown_category_raises():
    """Unknown category names raise a typed Python exception rather
    than crashing the actor."""
    sess = telemetry.CuptiSession()
    with pytest.raises(atomr_accel.GpuRuntimeError):
        sess.start(categories=["not-a-real-category"], timeout_secs=2.0)
