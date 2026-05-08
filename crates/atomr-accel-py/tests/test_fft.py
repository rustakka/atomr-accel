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
    # Phase 1.5++ — cufftPlanMany surface + output-length helpers.
    "exec_plan_many_f32",
    "exec_plan_many_f64",
    "r2c_output_len",
    "r2c_output_len_2d",
    "r2c_output_len_3d",
    "r2c_output_len_many",
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


# ─────────────────────── 3-D plans on exec_typed_* ───────────────────────


def test_exec_typed_f32_3d_dispatch() -> None:
    """All three dims supplied ⇒ a 3-D R2C plan is built. The mock
    actor will surface a runtime error since no buffers are wired, but
    the binding-side validation (rank inference, plan_key build) must
    have succeeded before the call reaches the actor / buffer-required
    branch."""
    sys_cm, _dev, fft = _open_fft("fft-typed-3d")
    try:
        # Missing buffers ⇒ the binding raises a `GpuRuntimeError` after
        # the plan-key has been built. That alone confirms the 3-D
        # dispatch path doesn't reject `(nx, ny, nz)` at parse time.
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="r2c", nx=8, ny=8, nz=8)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f32_3d_rejects_zero_dim() -> None:
    sys_cm, _dev, fft = _open_fft("fft-typed-3d-bad")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="r2c", nx=8, ny=0, nz=8)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f32(kind="r2c", nx=8, ny=8, nz=0)
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_typed_f64_3d_dispatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-typed-3d-f64")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_typed_f64(kind="d2z", nx=8, ny=8, nz=8)
    finally:
        sys_cm.__exit__(None, None, None)


# ─────────────────────── exec_plan_many_* surface ───────────────────────


def test_exec_plan_many_f32_rejects_bad_rank() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-bad-rank")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(rank=0, n=[16], kind="r2c")
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(rank=4, n=[2, 2, 2, 2], kind="r2c")
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f32_rejects_n_length_mismatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-bad-n")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            # rank=2 but n only has one dim
            fft.exec_plan_many_f32(rank=2, n=[16], kind="r2c")
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f32_rejects_inembed_length_mismatch() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-bad-inembed")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(
                rank=2, n=[8, 16], inembed=[8], kind="r2c"
            )
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f32_rejects_bad_stride() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-bad-stride")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(rank=1, n=[16], istride=0, kind="r2c")
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(rank=1, n=[16], ostride=-1, kind="r2c")
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f32_rejects_f64_kind() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-wrong-lane")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(rank=1, n=[16], kind="d2z")
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f64_rejects_f32_kind() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-wrong-lane-2")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f64(rank=1, n=[16], kind="r2c")
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f32_dispatch_no_buffers() -> None:
    """Valid args but no buffers ⇒ the plan-key-many is built first,
    then the binding raises when it discovers `real_buf=None` (R2C).
    Exercises the full validation chain through plan_key_from_many."""
    sys_cm, _dev, fft = _open_fft("fft-many-no-bufs")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f32(
                rank=2,
                n=[8, 16],
                inembed=[8, 16],
                istride=1,
                idist=128,
                onembed=[8, 9],
                ostride=1,
                odist=72,
                batch=4,
                kind="r2c",
            )
    finally:
        sys_cm.__exit__(None, None, None)


def test_exec_plan_many_f64_dispatch_no_buffers() -> None:
    sys_cm, _dev, fft = _open_fft("fft-many-no-bufs-f64")
    try:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            fft.exec_plan_many_f64(
                rank=1,
                n=[32],
                istride=1,
                idist=32,
                ostride=1,
                odist=17,
                batch=2,
                kind="d2z",
            )
    finally:
        sys_cm.__exit__(None, None, None)


# ─────────────────────── output-length helpers ───────────────────────


def test_r2c_output_len_basic() -> None:
    """Static method matches cuFFT's `n // 2 + 1` rule."""
    assert atomr_accel.Fft.r2c_output_len(64) == 33
    assert atomr_accel.Fft.r2c_output_len(7) == 4
    assert atomr_accel.Fft.r2c_output_len(1) == 1


def test_r2c_output_len_rejects_zero() -> None:
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len(0)
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len(-4)


def test_r2c_output_len_2d() -> None:
    assert atomr_accel.Fft.r2c_output_len_2d(8, 16) == 8 * 9
    assert atomr_accel.Fft.r2c_output_len_2d(4, 4) == 4 * 3


def test_r2c_output_len_2d_rejects_zero() -> None:
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len_2d(0, 16)
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len_2d(8, 0)


def test_r2c_output_len_3d() -> None:
    assert atomr_accel.Fft.r2c_output_len_3d(4, 8, 16) == 4 * 8 * 9
    assert atomr_accel.Fft.r2c_output_len_3d(2, 2, 2) == 2 * 2 * 2  # nz//2+1 = 2


def test_r2c_output_len_3d_rejects_zero() -> None:
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len_3d(0, 8, 16)


def test_r2c_output_len_many_natural() -> None:
    """When `odist=0`, the helper falls back to the natural per-batch
    layout: prod(n[:-1]) * (n[-1] // 2 + 1) * batch."""
    # 1-D, batch=3: per-batch = 16//2+1 = 9; total = 27.
    assert atomr_accel.Fft.r2c_output_len_many(rank=1, n=[16], batch=3) == 27
    # 2-D, batch=2: per-batch = 8 * 9 = 72; total = 144.
    assert atomr_accel.Fft.r2c_output_len_many(rank=2, n=[8, 16], batch=2) == 144
    # 3-D, batch=1: per-batch = 4 * 8 * 9 = 288.
    assert atomr_accel.Fft.r2c_output_len_many(rank=3, n=[4, 8, 16]) == 288


def test_r2c_output_len_many_honors_odist() -> None:
    """When `odist >= per-batch natural`, the helper uses
    `odist * batch` (covers padded/strided layouts)."""
    # natural per-batch = 9, odist=16 ⇒ batch * 16 = 32 wins.
    assert (
        atomr_accel.Fft.r2c_output_len_many(rank=1, n=[16], batch=2, odist=16)
        == 32
    )
    # odist smaller than natural ⇒ natural wins (defensive max).
    assert (
        atomr_accel.Fft.r2c_output_len_many(rank=1, n=[16], batch=2, odist=4)
        == 18  # 9 * 2
    )


def test_r2c_output_len_many_rejects_bad_args() -> None:
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len_many(rank=0, n=[])
    with pytest.raises(atomr_accel.GpuRuntimeError):
        atomr_accel.Fft.r2c_output_len_many(rank=4, n=[2, 2, 2, 2])
    with pytest.raises(atomr_accel.GpuRuntimeError):
        # n length doesn't match rank
        atomr_accel.Fft.r2c_output_len_many(rank=2, n=[16])
    with pytest.raises(atomr_accel.GpuRuntimeError):
        # batch must be >=1
        atomr_accel.Fft.r2c_output_len_many(rank=1, n=[16], batch=0)
    with pytest.raises(atomr_accel.GpuRuntimeError):
        # zero dim rejected
        atomr_accel.Fft.r2c_output_len_many(rank=1, n=[0])
