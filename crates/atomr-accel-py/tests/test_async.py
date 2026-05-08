"""Mock-mode surface tests for the Phase 5 async API.

Each method on the blocking surface (Device / Blas / Cudnn / Rng /
Cub / agents / realtime / train) has an `_async` counterpart that
returns a Python awaitable. This file exercises a representative
sample of those awaitables under asyncio. We use plain
``asyncio.new_event_loop()`` + ``run_until_complete`` so the tests do
not depend on ``pytest-asyncio`` being installed.

Mock-mode behaviour matches the blocking surface: most ops surface
either ``Unrecoverable`` (when the mock actor replies) or the generic
``GpuRuntimeError("... dropped reply")`` (when the mock actor drops
the boxed reply sender). We catch the base class so the tests stay
valid as the mock surface evolves.
"""
from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest

import atomr_accel


def _run(coro):
    """Run a coroutine on a fresh loop. Avoids any cross-test
    pollution of the asyncio loop and avoids requiring
    ``pytest-asyncio``."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────── Device async surface ───────────────────────


def test_allocate_f32_async_returns_awaitable_and_raises():
    async def go():
        with atomr_accel.System.open("async-alloc-f32") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            coro = dev.allocate_f32_async(16, timeout_secs=5.0)
            assert inspect.isawaitable(coro)
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await coro
    _run(go())


def test_allocate_f64_async_raises():
    async def go():
        with atomr_accel.System.open("async-alloc-f64") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await dev.allocate_f64_async(8, timeout_secs=5.0)
    _run(go())


def test_allocate_i32_async_raises():
    async def go():
        with atomr_accel.System.open("async-alloc-i32") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await dev.allocate_i32_async(4, timeout_secs=5.0)
    _run(go())


def test_copy_from_numpy_async_size_mismatch_is_synchronous():
    """The size-mismatch check happens before the awaitable is
    returned (it's part of the synchronous setup), so it raises
    immediately rather than from `await`."""
    with atomr_accel.System.open("async-copy-mismatch") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        # If allocate fails (mock returns Unrecoverable), there's no
        # buffer to copy into — skip rather than test the wrong path.
        try:
            buf = dev.allocate_f32(4, timeout_secs=5.0)
        except atomr_accel.GpuRuntimeError:
            pytest.skip("mock device cannot allocate f32 buffers")
        host = np.arange(8, dtype=np.float32)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            dev.copy_from_numpy_async(buf, host, timeout_secs=5.0)


def test_copy_to_numpy_async_returns_awaitable():
    async def go():
        with atomr_accel.System.open("async-copy-to") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            try:
                buf = dev.allocate_f32(4, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate f32 buffers")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await dev.copy_to_numpy_async(buf, timeout_secs=5.0)
    _run(go())


def test_sgemm_async_shape_check():
    async def go():
        with atomr_accel.System.open("async-sgemm") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            try:
                a = dev.allocate_f32(4, timeout_secs=5.0)
                b = dev.allocate_f32(4, timeout_secs=5.0)
                c = dev.allocate_f32(4, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate f32 buffers")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await dev.sgemm_async(a, b, c, m=2, n=2, k=2, timeout_secs=5.0)
    _run(go())


def test_stats_async_returns_load_snapshot():
    async def go():
        with atomr_accel.System.open("async-stats") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            load = await dev.stats_async(timeout_secs=2.0)
            assert load.compute_cap_major == 0
            assert load.active_streams == 0
    _run(go())


# ─────────────────────── Blas async surface ─────────────────────────


def _open_blas(name: str):
    sys_cm = atomr_accel.System.open(name)
    sys = sys_cm.__enter__()
    try:
        dev = sys.spawn_device(device_id=0, mock=True)
        try:
            blas = dev.blas(timeout_secs=2.0)
        except atomr_accel.GpuRuntimeError:
            sys_cm.__exit__(None, None, None)
            pytest.skip("mock device did not publish a Blas handle")
            raise
    except BaseException:
        sys_cm.__exit__(None, None, None)
        raise
    return sys_cm, dev, blas


def test_blas_axpy_f32_async():
    async def go():
        sys_cm, dev, blas = _open_blas("async-blas-axpy")
        try:
            try:
                x = dev.allocate_f32(4, timeout_secs=5.0)
                y = dev.allocate_f32(4, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await blas.axpy_f32_async(1.0, x, y, timeout_secs=5.0)
        finally:
            sys_cm.__exit__(None, None, None)
    _run(go())


def test_blas_dot_f32_async():
    async def go():
        sys_cm, dev, blas = _open_blas("async-blas-dot")
        try:
            try:
                x = dev.allocate_f32(4, timeout_secs=5.0)
                y = dev.allocate_f32(4, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await blas.dot_f32_async(x, y, timeout_secs=5.0)
        finally:
            sys_cm.__exit__(None, None, None)
    _run(go())


def test_blas_gemm_f32_async():
    async def go():
        sys_cm, dev, blas = _open_blas("async-blas-gemm")
        try:
            try:
                a = dev.allocate_f32(4, timeout_secs=5.0)
                b = dev.allocate_f32(4, timeout_secs=5.0)
                c = dev.allocate_f32(4, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await blas.gemm_f32_async(a, b, c, m=2, n=2, k=2, timeout_secs=5.0)
        finally:
            sys_cm.__exit__(None, None, None)
    _run(go())


# ─────────────────────── cuRAND async surface ───────────────────────


@pytest.mark.skipif(atomr_accel.RngGenerator is None, reason="curand feature not built")
def test_rng_uniform_f32_async():
    async def go():
        with atomr_accel.System.open("async-rng-uniform") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            try:
                rng = dev.rng(timeout_secs=2.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device did not publish an RngGenerator")
                return
            try:
                buf = dev.allocate_f32(8, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await rng.uniform_f32_async(buf, lo=0.0, hi=1.0, timeout_secs=5.0)
    _run(go())


@pytest.mark.skipif(atomr_accel.RngGenerator is None, reason="curand feature not built")
def test_rng_set_seed_async():
    async def go():
        with atomr_accel.System.open("async-rng-seed") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            try:
                rng = dev.rng(timeout_secs=2.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device did not publish an RngGenerator")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await rng.set_seed_async(42, timeout_secs=5.0)
    _run(go())


# ─────────────────────── cuDNN async surface ────────────────────────


@pytest.mark.skipif(atomr_accel.Cudnn is None, reason="cudnn feature not built")
def test_cudnn_conv2d_fwd_f32_async():
    async def go():
        with atomr_accel.System.open("async-cudnn-conv") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            try:
                cudnn = dev.cudnn(timeout_secs=2.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device did not publish a Cudnn handle")
                return
            try:
                x = dev.allocate_f32(16, timeout_secs=5.0)
                w = dev.allocate_f32(16, timeout_secs=5.0)
                y = dev.allocate_f32(16, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await cudnn.conv2d_fwd_f32_async(
                    x, w, y,
                    x_shape=(1, 1, 4, 4),
                    w_shape=(1, 1, 1, 1),
                    y_shape=(1, 1, 4, 4),
                    timeout_secs=5.0,
                )
    _run(go())


@pytest.mark.skipif(atomr_accel.Cudnn is None, reason="cudnn feature not built")
def test_cudnn_softmax_fwd_f32_async():
    async def go():
        with atomr_accel.System.open("async-cudnn-softmax") as sys:
            dev = sys.spawn_device(device_id=0, mock=True)
            try:
                cudnn = dev.cudnn(timeout_secs=2.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device did not publish a Cudnn handle")
                return
            try:
                x = dev.allocate_f32(8, timeout_secs=5.0)
                y = dev.allocate_f32(8, timeout_secs=5.0)
            except atomr_accel.GpuRuntimeError:
                pytest.skip("mock device cannot allocate")
                return
            with pytest.raises(atomr_accel.GpuRuntimeError):
                await cudnn.softmax_fwd_f32_async(x, y, dims=[1, 2, 2, 2], timeout_secs=5.0)
    _run(go())


# ─────────────────────── method existence (compile-gate) ────────────


def test_async_method_inventory():
    """Smoke-check that the new async methods exist on the relevant
    classes. Catches PyO3 macro regressions cheaply."""
    # Device (always present)
    for name in [
        "allocate_f32_async",
        "allocate_f64_async",
        "allocate_i32_async",
        "allocate_u32_async",
        "allocate_u8_async",
        "copy_from_numpy_async",
        "copy_from_numpy_f64_async",
        "copy_from_numpy_i32_async",
        "copy_from_numpy_u32_async",
        "copy_from_numpy_u8_async",
        "copy_to_numpy_async",
        "copy_to_numpy_f64_async",
        "copy_to_numpy_i32_async",
        "copy_to_numpy_u32_async",
        "copy_to_numpy_u8_async",
        "sgemm_async",
        "stats_async",
    ]:
        assert hasattr(atomr_accel.Device, name), name

    # Blas (always present)
    for name in [
        "gemm_f32_async",
        "gemm_f64_async",
        "axpy_f32_async",
        "axpy_f64_async",
        "dot_f32_async",
        "dot_f64_async",
        "nrm2_f32_async",
        "scal_f32_async",
        "asum_f32_async",
        "iamax_f32_async",
        "iamin_f32_async",
        "copy_f32_async",
        "swap_f32_async",
        "rot_f32_async",
        "gemv_f32_async",
        "ger_f32_async",
        "geam_f32_async",
        "syrk_f32_async",
        "trsm_f32_async",
    ]:
        assert hasattr(atomr_accel.Blas, name), name

    # cuDNN (optional)
    if atomr_accel.Cudnn is not None:
        for name in [
            "conv2d_fwd_f32_async",
            "pool2d_fwd_f32_async",
            "softmax_fwd_f32_async",
            "activation_fwd_f32_async",
            "batch_norm_f32_async",
            "layer_norm_f32_async",
            "instance_norm_f32_async",
            "group_norm_f32_async",
            "dropout_fwd_f32_async",
            "lrn_fwd_f32_async",
            "rnn_fwd_f32_async",
            "multihead_attn_fwd_f32_async",
        ]:
            assert hasattr(atomr_accel.Cudnn, name), name

    # cuRAND (optional)
    if atomr_accel.RngGenerator is not None:
        for name in [
            "set_seed_async",
            "set_generator_async",
            "uniform_f32_async",
            "uniform_f64_async",
            "normal_f32_async",
            "normal_f64_async",
            "log_normal_f32_async",
            "log_normal_f64_async",
            "poisson_f32_async",
            "exponential_f32_async",
            "beta_f32_async",
            "cauchy_f32_async",
            "gamma_f32_async",
            "discrete_f32_async",
            "uniform_u32_async",
        ]:
            assert hasattr(atomr_accel.RngGenerator, name), name
