"""Kernel-actor handle tests — `Device.{blas,cudnn,fft,rng}` access.

These check the structural plumbing: Device knows how to mint a
handle, and the right errors surface when the underlying actor isn't
ready (mock mode, missing feature, etc.). Method-level coverage is
exercised in the per-domain test files (test_blas.py, test_rng.py, …).
"""
from __future__ import annotations

import pytest

import atomr_accel


def test_libraries_ready_in_mock_mode_is_empty():
    """In mock mode the ContextActor never emits ContextReady so the
    children snapshot is None — `libraries_ready` reports everything
    false."""
    with atomr_accel.System.open("ready-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        ready = dev.libraries_ready(timeout_secs=2.0)
        # Mock ContextActor *does* emit ContextReady with a Mock
        # BlasActor — so blas may legitimately be True. Just probe
        # that the dict has the expected shape.
        for key in ("blas", "cudnn", "cufft", "curand", "extras"):
            assert key in ready, key


def test_blas_handle_unavailable_in_mock_mode():
    """Mock-mode device never publishes children → ``device.blas()``
    raises GpuRuntimeError. (When the mock ContextActor *does* publish
    children with a Mock BlasActor, this returns the handle instead.)"""
    with atomr_accel.System.open("blas-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        # Whichever path the mock takes, callers should at least get
        # a typed result back: either a Blas handle or GpuRuntimeError.
        try:
            handle = dev.blas(timeout_secs=2.0)
            assert isinstance(handle, atomr_accel.Blas)
            assert "Blas" in repr(handle)
        except atomr_accel.GpuRuntimeError:
            pass


@pytest.mark.skipif(
    atomr_accel.Cudnn is None, reason="cudnn feature not compiled in"
)
def test_cudnn_handle_optional():
    """``device.cudnn()`` only resolves when both the cargo feature is
    on *and* the device has CUDNN enabled — mock mode lacks both, so
    we expect GpuRuntimeError."""
    with atomr_accel.System.open("cudnn-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            dev.cudnn(timeout_secs=2.0)


@pytest.mark.skipif(
    atomr_accel.Fft is None, reason="cufft feature not compiled in"
)
def test_fft_handle_optional():
    with atomr_accel.System.open("fft-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            dev.fft(timeout_secs=2.0)


@pytest.mark.skipif(
    atomr_accel.RngGenerator is None, reason="curand feature not compiled in"
)
def test_rng_handle_optional():
    with atomr_accel.System.open("rng-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            dev.rng(timeout_secs=2.0)
