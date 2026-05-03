"""No-GPU smoke tests for the rakka-accel Python bridge.

Exercise the System ↔ Device ↔ actor pipeline in mock mode so the
suite runs on CI hosts without a CUDA driver. Real-GPU tests live
under ``tests/test_gpu.py`` and are skipped by default.

Run with:
    pip install -e .
    pytest tests/
"""
from __future__ import annotations

import pytest

import rakka_accel


def test_module_surface_is_complete():
    """Every public name listed in __init__ resolves to a class or
    string. Catches PyO3 macro regressions immediately."""
    assert isinstance(rakka_accel.__version__, str)
    for name in [
        "System",
        "Device",
        "DeviceLoad",
        "GpuBuffer",
        "GpuRuntimeError",
        "ContextPoisoned",
        "OutOfMemory",
        "Unrecoverable",
        "GpuRefStale",
        "LibraryError",
        "AskTimeout",
    ]:
        assert getattr(rakka_accel, name) is not None, name


def test_exception_hierarchy():
    """Typed `GpuError` variants land as Python exception subclasses
    rooted at GpuRuntimeError."""
    assert issubclass(rakka_accel.ContextPoisoned, rakka_accel.GpuRuntimeError)
    assert issubclass(rakka_accel.OutOfMemory, rakka_accel.GpuRuntimeError)
    assert issubclass(rakka_accel.Unrecoverable, rakka_accel.GpuRuntimeError)
    assert issubclass(rakka_accel.GpuRefStale, rakka_accel.GpuRuntimeError)
    assert issubclass(rakka_accel.LibraryError, rakka_accel.GpuRuntimeError)
    assert issubclass(rakka_accel.AskTimeout, rakka_accel.GpuRuntimeError)


def test_system_open_close():
    """The System lifecycle works as a context manager."""
    with rakka_accel.System.open("smoke") as sys:
        assert sys.name == "smoke"
        assert "smoke" in repr(sys)


def test_mock_device_allocate_returns_unrecoverable():
    """Mock-mode device replies Unrecoverable to every typed
    allocate. This proves the full Python → Rust → rakka actor
    pipeline (System spawn → DeviceActor → ContextActor →
    BlasActor → reply) round-trips without touching CUDA."""
    with rakka_accel.System.open("mock-alloc") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        assert dev.device_id == 0
        with pytest.raises(rakka_accel.Unrecoverable):
            dev.allocate_f32(16, timeout_secs=5.0)


def test_mock_device_stats_returns_load_snapshot():
    """`Device.stats` always replies (no GPU needed)."""
    with rakka_accel.System.open("mock-stats") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        load = dev.stats(timeout_secs=2.0)
        assert load.compute_cap_major == 0
        assert load.compute_cap_minor == 0
        assert load.active_streams == 0
        assert "DeviceLoad" in repr(load)
