"""No-GPU smoke tests for the atomr-accel Python bridge.

Exercise the System ↔ Device ↔ actor pipeline in mock mode so the
suite runs on CI hosts without a CUDA driver. Real-GPU tests live
under ``tests/test_gpu.py`` and are skipped by default.

Run with:
    pip install -e .
    pytest tests/
"""
from __future__ import annotations

import pytest

import atomr_accel


def test_module_surface_is_complete():
    """Every public name listed in __init__ resolves to a class or
    string. Catches PyO3 macro regressions immediately."""
    assert isinstance(atomr_accel.__version__, str)
    # Always-present surface (default features).
    for name in [
        "System",
        "Device",
        "DeviceLoad",
        "GpuBuffer",
        "GpuBufferF32",
        "GpuBufferF64",
        "GpuBufferI32",
        "GpuBufferU32",
        "GpuBufferU8",
        "Blas",
        "GpuRuntimeError",
        "ContextPoisoned",
        "OutOfMemory",
        "Unrecoverable",
        "GpuRefStale",
        "LibraryError",
        "AskTimeout",
    ]:
        assert getattr(atomr_accel, name) is not None, name

    # Optional surface — present iff the matching cargo feature was
    # compiled in. The attribute must always exist (set to None on
    # minimal builds), so users can ``if atomr_accel.Cudnn:`` guard.
    for name in ["Cudnn", "Fft", "RngGenerator", "Solver", "Collective", "NvrtcKernel"]:
        assert hasattr(atomr_accel, name), name


def test_exception_hierarchy():
    """Typed `GpuError` variants land as Python exception subclasses
    rooted at GpuRuntimeError."""
    assert issubclass(atomr_accel.ContextPoisoned, atomr_accel.GpuRuntimeError)
    assert issubclass(atomr_accel.OutOfMemory, atomr_accel.GpuRuntimeError)
    assert issubclass(atomr_accel.Unrecoverable, atomr_accel.GpuRuntimeError)
    assert issubclass(atomr_accel.GpuRefStale, atomr_accel.GpuRuntimeError)
    assert issubclass(atomr_accel.LibraryError, atomr_accel.GpuRuntimeError)
    assert issubclass(atomr_accel.AskTimeout, atomr_accel.GpuRuntimeError)


def test_system_open_close():
    """The System lifecycle works as a context manager."""
    with atomr_accel.System.open("smoke") as sys:
        assert sys.name == "smoke"
        assert "smoke" in repr(sys)


def test_mock_device_allocate_returns_unrecoverable():
    """Mock-mode device replies Unrecoverable to every typed
    allocate. This proves the full Python → Rust → atomr actor
    pipeline (System spawn → DeviceActor → ContextActor →
    BlasActor → reply) round-trips without touching CUDA."""
    with atomr_accel.System.open("mock-alloc") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        assert dev.device_id == 0
        with pytest.raises(atomr_accel.Unrecoverable):
            dev.allocate_f32(16, timeout_secs=5.0)


def test_mock_device_stats_returns_load_snapshot():
    """`Device.stats` always replies (no GPU needed)."""
    with atomr_accel.System.open("mock-stats") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        load = dev.stats(timeout_secs=2.0)
        assert load.compute_cap_major == 0
        assert load.compute_cap_minor == 0
        assert load.active_streams == 0
        assert "DeviceLoad" in repr(load)


def test_per_domain_submodules_importable():
    """The per-domain facade modules import cleanly and re-export the
    expected symbol(s) — even on minimal builds where the optional
    handles resolve to ``None``."""
    from atomr_accel import system, device, blas, errors

    assert system.System is atomr_accel.System
    assert device.Device is atomr_accel.Device
    assert blas.Blas is atomr_accel.Blas
    assert errors.GpuRuntimeError is atomr_accel.GpuRuntimeError

    # Optional ones must at least import without raising.
    from atomr_accel import cudnn, fft, rng, solver, collective, nvrtc  # noqa: F401
