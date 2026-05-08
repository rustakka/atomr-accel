"""Mock-mode surface tests for the cuFFT Python wrapper.

Phase 1.5++ — `Fft` gains four 1-shot host-driven FFT methods. Mock
mode exercises the *binding surface* (every method is callable with
the right signature, raises a typed `GpuRuntimeError` when the actor
can't reply). Real correctness lives in the Rust integration tests
under ``crates/atomr-accel-cuda/tests/``.

Mock-mode behaviour: the mock `FftActor` drops `FftMsg::Exec` requests
silently (the boxed `FftRequest` is dropped, closing the reply
channel — surfaces as `RecvError` on the Python side). Some calls
fail earlier: the device's mock allocator may itself reply with
`Unrecoverable`. Either path produces a `GpuRuntimeError` subclass,
which is what we assert.

If `device.fft()` itself raises (children not yet ready in mock
mode), the test still passes — the binding-existence check is the
real gate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import atomr_accel


pytestmark = pytest.mark.skipif(
    atomr_accel.Fft is None, reason="cufft feature not compiled in"
)


def _open_fft(name: str):
    """Open a System, spawn a mock device, try to acquire the Fft
    handle. Skips the test if the mock device doesn't publish one."""
    sys_cm = atomr_accel.System.open(name)
    sys = sys_cm.__enter__()
    try:
        dev = sys.spawn_device(device_id=0, mock=True)
        try:
            fft = dev.fft(timeout_secs=2.0)
        except atomr_accel.GpuRuntimeError as e:
            sys_cm.__exit__(None, None, None)
            pytest.skip(f"mock device did not publish an Fft handle: {e}")
            raise  # unreachable
    except BaseException:
        sys_cm.__exit__(None, None, None)
        raise
    return sys_cm, dev, fft


def _expect_runtime_error(call: Callable[[], object]) -> None:
    with pytest.raises(atomr_accel.GpuRuntimeError):
        call()


# ─────────────────────── method existence ───────────────────────


_FFT_METHODS = [
    "forward_1d_r2c_f32",
    "inverse_1d_c2r_f32",
    "exec_1d_c2c_f32",
    "forward_2d_r2c_f32",
    # Phase 1.5++ Path A — typed complex buffer dispatch.
    "exec_typed_f32",
    "exec_typed_f64",
]


@pytest.mark.parametrize("name", _FFT_METHODS)
def test_method_exists(name: str) -> None:
    """Every documented method is reachable on the Fft pyclass.
    This is a pure surface gate: the import + getattr step alone
    catches missing #[pymethods] declarations at collection time."""
    assert hasattr(atomr_accel.Fft, name), f"Fft.{name} missing"


# ─────────────────────── repr / handle plumbing ───────────────────────


def test_fft_handle_repr() -> None:
    sys_cm, _dev, fft = _open_fft("fft-repr")
    try:
        assert "Fft" in repr(fft)
    finally:
        sys_cm.__exit__(None, None, None)


# ─────────────────────── shape / dtype validation ───────────────────────


def test_forward_1d_r2c_rejects_bad_batch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-bad-batch")
    try:
        x = np.zeros(64, dtype=np.float32)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.forward_1d_r2c_f32(x, batch=0)
    finally:
        sys_cm.__exit__(None, None, None)


def test_forward_1d_r2c_rejects_length_mismatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-len-mismatch")
    try:
        x = np.zeros(63, dtype=np.float32)  # not divisible by batch=2
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.forward_1d_r2c_f32(x, batch=2)
    finally:
        sys_cm.__exit__(None, None, None)


def test_inverse_1d_c2r_requires_n() -> None:
    """`inverse_1d_c2r_f32` requires `n` (real-domain length cannot be
    inferred from the half-spectrum input alone)."""
    sys_cm, _dev, fft = _open_fft("fft-c2r-need-n")
    try:
        x = np.zeros(33, dtype=np.complex64)
        with pytest.raises((TypeError, atomr_accel.GpuRuntimeError)):
            # `n` is a required positional in the signature
            fft.inverse_1d_c2r_f32(x)  # type: ignore[call-arg]
    finally:
        sys_cm.__exit__(None, None, None)


def test_inverse_1d_c2r_rejects_negative_n() -> None:
    sys_cm, _dev, fft = _open_fft("fft-c2r-neg-n")
    try:
        x = np.zeros(33, dtype=np.complex64)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.inverse_1d_c2r_f32(x, n=-4)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_1d_c2c_rejects_bad_direction() -> None:
    sys_cm, _dev, fft = _open_fft("fft-bad-dir")
    try:
        x = np.zeros(64, dtype=np.complex64)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_1d_c2c_f32(x, direction="sideways")
    finally:
        sys_cm.__exit__(None, None, None)


def test_forward_2d_r2c_rejects_bad_dims() -> None:
    sys_cm, _dev, fft = _open_fft("fft-2d-bad-dims")
    try:
        x = np.zeros(16, dtype=np.float32)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.forward_2d_r2c_f32(x, nx=0, ny=4)
    finally:
        sys_cm.__exit__(None, None, None)


def test_forward_2d_r2c_rejects_length_mismatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-2d-len")
    try:
        x = np.zeros(15, dtype=np.float32)  # not 4*4
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.forward_2d_r2c_f32(x, nx=4, ny=4)
    finally:
        sys_cm.__exit__(None, None, None)


# ─────────────────────── mock-mode dispatch ───────────────────────


def test_forward_1d_r2c_mock_dispatch() -> None:
    """Valid args reach the actor pipeline, which surfaces some
    `GpuRuntimeError` in mock mode (allocator reply or fft drop)."""
    sys_cm, _dev, fft = _open_fft("fft-r2c-mock")
    try:
        x = np.random.randn(64).astype(np.float32)
        _expect_runtime_error(lambda: fft.forward_1d_r2c_f32(x, n=64, batch=1))
    finally:
        sys_cm.__exit__(None, None, None)


def test_inverse_1d_c2r_mock_dispatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-c2r-mock")
    try:
        x = np.zeros(33, dtype=np.complex64)
        _expect_runtime_error(lambda: fft.inverse_1d_c2r_f32(x, n=64, batch=1))
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_1d_c2c_mock_dispatch_forward() -> None:
    sys_cm, _dev, fft = _open_fft("fft-c2c-fwd-mock")
    try:
        x = np.zeros(32, dtype=np.complex64)
        _expect_runtime_error(
            lambda: fft.exec_1d_c2c_f32(x, direction="forward", n=32, batch=1)
        )
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_1d_c2c_mock_dispatch_inverse() -> None:
    sys_cm, _dev, fft = _open_fft("fft-c2c-inv-mock")
    try:
        x = np.zeros(32, dtype=np.complex64)
        _expect_runtime_error(
            lambda: fft.exec_1d_c2c_f32(x, direction="inverse", n=32, batch=1)
        )
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_1d_c2c_direction_aliases() -> None:
    """'fwd' / 'inv' / 'backward' should parse as valid directions and
    reach the actor (rather than failing at parse time)."""
    sys_cm, _dev, fft = _open_fft("fft-c2c-aliases")
    try:
        x = np.zeros(32, dtype=np.complex64)
        for d in ("forward", "FORWARD", "fwd", "f", "inverse", "INV", "backward", "i"):
            with pytest.raises(atomr_accel.GpuRuntimeError):
                fft.exec_1d_c2c_f32(x, direction=d, n=32, batch=1)
    finally:
        sys_cm.__exit__(None, None, None)


def test_forward_2d_r2c_mock_dispatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-2d-mock")
    try:
        x = np.random.randn(8 * 16).astype(np.float32)
        _expect_runtime_error(lambda: fft.forward_2d_r2c_f32(x, nx=8, ny=16))
    finally:
        sys_cm.__exit__(None, None, None)


# ─────────────────────── batch + reuse ───────────────────────


def test_forward_1d_r2c_with_explicit_n_and_batch() -> None:
    """Explicit `n=` and `batch=` together. The mock device path will
    still error, but the call should reach the actor without raising
    a value-error on shape parsing."""
    sys_cm, _dev, fft = _open_fft("fft-r2c-batched")
    try:
        n, batch = 32, 4
        x = np.random.randn(n * batch).astype(np.float32)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.forward_1d_r2c_f32(x, n=n, batch=batch)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_1d_c2c_batched() -> None:
    sys_cm, _dev, fft = _open_fft("fft-c2c-batched")
    try:
        n, batch = 16, 3
        x = np.zeros(n * batch, dtype=np.complex64)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_1d_c2c_f32(x, direction="forward", n=n, batch=batch)
    finally:
        sys_cm.__exit__(None, None, None)


# ─────────────────────── Path A — typed buffer dispatch ───────────────────────


def test_exec_typed_f32_rejects_f64_kind() -> None:
    """``exec_typed_f32`` covers R2C / C2R / C2C only. Passing a
    double-lane kind (D2Z / Z2D / Z2Z) must error fast at the binding
    layer rather than reaching the actor."""
    sys_cm, _dev, fft = _open_fft("fft-typed-wrong-lane")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="d2z", nx=16)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f64_rejects_f32_kind() -> None:
    sys_cm, _dev, fft = _open_fft("fft-typed-wrong-lane-2")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f64(kind="r2c", nx=16)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f32_rejects_unknown_kind() -> None:
    sys_cm, _dev, fft = _open_fft("fft-typed-bad-kind")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="not-a-kind", nx=16)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f32_requires_nx() -> None:
    sys_cm, _dev, fft = _open_fft("fft-typed-need-nx")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="r2c")
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f32_requires_buffers() -> None:
    """R2C without `real_buf` (input) or `complex_buf` (output) must
    fail with a clear error before reaching the actor."""
    sys_cm, _dev, fft = _open_fft("fft-typed-no-bufs")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="r2c", nx=16)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f32_rejects_bad_batch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-typed-bad-batch")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="r2c", nx=16, batch=0)
    finally:
        sys_cm.__exit__(None, None, None)
