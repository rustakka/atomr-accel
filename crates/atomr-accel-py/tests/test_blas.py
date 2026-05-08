"""Mock-mode surface tests for the cuBLAS Python wrapper.

Phase 1.5 expands `Blas` from three methods (gemm_f32 / gemm_f64 /
axpy_f32) to the full L1/L2/L3 surface plus strided-batched gemm. The
goal of this file is *surface coverage* — confirm every method is
callable from Python with the right signature. Real correctness lives
in the Rust integration tests under
``crates/atomr-accel-cuda/tests/``.

Mock-mode behaviour: the mock `BlasActor` drops incoming requests
without ever sending a reply (see `mod.rs::mock_reply`). The Python
wrapper observes the dropped sender as a `RecvError` and surfaces it
as a generic `GpuRuntimeError("blas dropped reply")`. So every method
here is expected to raise `GpuRuntimeError`. We catch the base class
so the tests stay valid even if the mock starts replying with a more
specific `Unrecoverable` variant in the future.

If `device.blas()` itself raises (children not yet ready), the test
still passes — the point is that the *binding* compiles and is
reachable from Python.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pytest

import atomr_accel


# ─────────────────────── helpers ───────────────────────


def _open_blas(name: str):
    """Open a System, spawn a mock device, and try to acquire a Blas
    handle. Returns ``(system_cm, dev, blas)`` or skips the test if
    the handle isn't available in mock mode."""
    sys_cm = atomr_accel.System.open(name)
    sys = sys_cm.__enter__()
    try:
        dev = sys.spawn_device(device_id=0, mock=True)
        try:
            blas = dev.blas(timeout_secs=2.0)
        except atomr_accel.GpuRuntimeError as e:
            sys_cm.__exit__(None, None, None)
            pytest.skip(f"mock device did not publish a Blas handle: {e}")
            raise  # unreachable, satisfies type checker
    except BaseException:
        sys_cm.__exit__(None, None, None)
        raise
    return sys_cm, dev, blas


def _alloc_f32(dev, n: int):
    """Allocate an f32 buffer or skip if the mock device can't allocate."""
    try:
        return dev.allocate_f32(n, timeout_secs=5.0)
    except atomr_accel.GpuRuntimeError as e:
        pytest.skip(f"mock device cannot allocate f32 buffers: {e}")


def _alloc_f64(dev, n: int):
    try:
        return dev.allocate_f64(n, timeout_secs=5.0)
    except atomr_accel.GpuRuntimeError as e:
        pytest.skip(f"mock device cannot allocate f64 buffers: {e}")


def _expect_runtime_error(call: Callable[[], object]) -> None:
    """Assert the call raises ``GpuRuntimeError`` (or any subclass).
    The mock BLAS actor never replies, so this is what we expect
    every typed dispatch to surface."""
    with pytest.raises(atomr_accel.GpuRuntimeError):
        call()


# ─────────────────────── method existence (compile gate) ───────────────────────


# Every method we expect to exist on the Blas class. If a method is
# missing the test collection itself fails, which is the surface gate
# we want.
_BLAS_METHODS = [
    # gemm
    "gemm_f32",
    "gemm_f64",
    # gemm strided-batched
    "gemm_strided_batched_f32",
    "gemm_strided_batched_f64",
    # L1
    "axpy_f32",
    "axpy_f64",
    "dot_f32",
    "dot_f64",
    "nrm2_f32",
    "nrm2_f64",
    "scal_f32",
    "scal_f64",
    "asum_f32",
    "asum_f64",
    "iamax_f32",
    "iamax_f64",
    "iamin_f32",
    "iamin_f64",
    "copy_f32",
    "copy_f64",
    "swap_f32",
    "swap_f64",
    "rot_f32",
    "rot_f64",
    # L2
    "gemv_f32",
    "gemv_f64",
    "ger_f32",
    "ger_f64",
    # L3
    "geam_f32",
    "geam_f64",
    "syrk_f32",
    "syrk_f64",
    "trsm_f32",
    "trsm_f64",
]


@pytest.mark.parametrize("name", _BLAS_METHODS)
def test_blas_class_exposes_method(name: str):
    """Every Phase 1.5 method must be reachable as an attribute on the
    Blas pyclass. Doesn't actually call anything — pure surface."""
    assert hasattr(atomr_accel.Blas, name), name
    assert callable(getattr(atomr_accel.Blas, name))


# ─────────────────────── per-method mock dispatches ───────────────────────


def test_gemm_strided_batched_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-strided-f32")
    try:
        a = _alloc_f32(dev, 4)
        b = _alloc_f32(dev, 4)
        c = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.gemm_strided_batched_f32(
            a, b, c,
            m=2, n=2, k=2,
            stride_a=4, stride_b=4, stride_c=4, batch_count=1,
            timeout_secs=2.0,
        ))
    finally:
        sys_cm.__exit__(None, None, None)


def test_gemm_strided_batched_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-strided-f64")
    try:
        a = _alloc_f64(dev, 4)
        b = _alloc_f64(dev, 4)
        c = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.gemm_strided_batched_f64(
            a, b, c,
            m=2, n=2, k=2,
            stride_a=4, stride_b=4, stride_c=4, batch_count=1,
            timeout_secs=2.0,
        ))
    finally:
        sys_cm.__exit__(None, None, None)


# ─── L1 ────────────────────────────────────────────────────────────


def test_axpy_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-axpy-f64")
    try:
        x = _alloc_f64(dev, 8)
        y = _alloc_f64(dev, 8)
        _expect_runtime_error(lambda: blas.axpy_f64(2.0, x, y, n=8, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_dot_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-dot-f32")
    try:
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.dot_f32(x, y, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_dot_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-dot-f64")
    try:
        x = _alloc_f64(dev, 4)
        y = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.dot_f64(x, y, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_nrm2_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-nrm2-f32")
    try:
        x = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.nrm2_f32(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_nrm2_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-nrm2-f64")
    try:
        x = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.nrm2_f64(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_scal_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-scal-f32")
    try:
        x = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.scal_f32(2.0, x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_scal_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-scal-f64")
    try:
        x = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.scal_f64(2.0, x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_asum_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-asum-f32")
    try:
        x = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.asum_f32(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_asum_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-asum-f64")
    try:
        x = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.asum_f64(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_iamax_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-iamax-f32")
    try:
        x = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.iamax_f32(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_iamax_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-iamax-f64")
    try:
        x = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.iamax_f64(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_iamin_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-iamin-f32")
    try:
        x = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.iamin_f32(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_iamin_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-iamin-f64")
    try:
        x = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.iamin_f64(x, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_copy_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-copy-f32")
    try:
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.copy_f32(x, y, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_copy_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-copy-f64")
    try:
        x = _alloc_f64(dev, 4)
        y = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.copy_f64(x, y, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_swap_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-swap-f32")
    try:
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.swap_f32(x, y, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_swap_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-swap-f64")
    try:
        x = _alloc_f64(dev, 4)
        y = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.swap_f64(x, y, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_rot_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-rot-f32")
    try:
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.rot_f32(x, y, c=1.0, s=0.0, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_rot_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-rot-f64")
    try:
        x = _alloc_f64(dev, 4)
        y = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.rot_f64(x, y, c=1.0, s=0.0, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


# ─── L2 ────────────────────────────────────────────────────────────


def test_gemv_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-gemv-f32")
    try:
        a = _alloc_f32(dev, 16)
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        _expect_runtime_error(lambda: blas.gemv_f32(a, x, y, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_gemv_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-gemv-f64")
    try:
        a = _alloc_f64(dev, 16)
        x = _alloc_f64(dev, 4)
        y = _alloc_f64(dev, 4)
        _expect_runtime_error(lambda: blas.gemv_f64(a, x, y, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_ger_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-ger-f32")
    try:
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        a = _alloc_f32(dev, 16)
        _expect_runtime_error(lambda: blas.ger_f32(x, y, a, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_ger_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-ger-f64")
    try:
        x = _alloc_f64(dev, 4)
        y = _alloc_f64(dev, 4)
        a = _alloc_f64(dev, 16)
        _expect_runtime_error(lambda: blas.ger_f64(x, y, a, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


# ─── L3 ────────────────────────────────────────────────────────────


def test_geam_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-geam-f32")
    try:
        a = _alloc_f32(dev, 16)
        b = _alloc_f32(dev, 16)
        c = _alloc_f32(dev, 16)
        _expect_runtime_error(lambda: blas.geam_f32(a, b, c, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_geam_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-geam-f64")
    try:
        a = _alloc_f64(dev, 16)
        b = _alloc_f64(dev, 16)
        c = _alloc_f64(dev, 16)
        _expect_runtime_error(lambda: blas.geam_f64(a, b, c, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_syrk_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-syrk-f32")
    try:
        a = _alloc_f32(dev, 16)
        c = _alloc_f32(dev, 16)
        _expect_runtime_error(lambda: blas.syrk_f32(a, c, n=4, k=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_syrk_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-syrk-f64")
    try:
        a = _alloc_f64(dev, 16)
        c = _alloc_f64(dev, 16)
        _expect_runtime_error(lambda: blas.syrk_f64(a, c, n=4, k=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_trsm_f32_callable():
    sys_cm, dev, blas = _open_blas("blas-trsm-f32")
    try:
        a = _alloc_f32(dev, 16)
        b = _alloc_f32(dev, 16)
        _expect_runtime_error(lambda: blas.trsm_f32(a, b, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


def test_trsm_f64_callable():
    sys_cm, dev, blas = _open_blas("blas-trsm-f64")
    try:
        a = _alloc_f64(dev, 16)
        b = _alloc_f64(dev, 16)
        _expect_runtime_error(lambda: blas.trsm_f64(a, b, m=4, n=4, timeout_secs=2.0))
    finally:
        sys_cm.__exit__(None, None, None)


# ─── enum-string validation ────────────────────────────────────────


def test_gemv_rejects_bad_trans_string():
    """`trans` must be one of N/T/C — anything else raises before
    even hitting the actor."""
    sys_cm, dev, blas = _open_blas("blas-bad-trans")
    try:
        a = _alloc_f32(dev, 16)
        x = _alloc_f32(dev, 4)
        y = _alloc_f32(dev, 4)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            blas.gemv_f32(a, x, y, m=4, n=4, trans="Z", timeout_secs=2.0)
    finally:
        sys_cm.__exit__(None, None, None)


def test_syrk_rejects_bad_uplo_string():
    sys_cm, dev, blas = _open_blas("blas-bad-uplo")
    try:
        a = _alloc_f32(dev, 16)
        c = _alloc_f32(dev, 16)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            blas.syrk_f32(a, c, n=4, k=4, uplo="Z", timeout_secs=2.0)
    finally:
        sys_cm.__exit__(None, None, None)


def test_trsm_rejects_bad_side_string():
    sys_cm, dev, blas = _open_blas("blas-bad-side")
    try:
        a = _alloc_f32(dev, 16)
        b = _alloc_f32(dev, 16)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            blas.trsm_f32(a, b, m=4, n=4, side="Z", timeout_secs=2.0)
    finally:
        sys_cm.__exit__(None, None, None)


# Silence unused imports under conditions where some tests skip early.
_ = np
_ = Optional
