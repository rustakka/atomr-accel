"""Mock-mode surface checks for ``atomr_accel`` NVRTC bindings.

Phase 1.5++ wires the NVRTC spawn-path (``ContextActor`` mints a
``NvrtcActor`` when both the ``nvrtc`` cargo feature and
``EnabledLibraries::NVRTC`` are on) and exposes:

  * ``Device.compile_kernel(name, src, ...)`` returning an
    :class:`atomr_accel.NvrtcKernel`.
  * :class:`atomr_accel.NvrtcKernel.launch(grid, block, args, ...)`.
  * :class:`atomr_accel.KernelArg` — typed kernel-arg constructors
    covering scalar f32/f64/i32/i64/u32/u64 and device-pointer wrappers
    around every supported ``GpuBuffer*``.

The whole module is skipped when ``atomr_accel.NvrtcKernel is None``
(i.e. the wheel was built without ``--features nvrtc``).
"""
from __future__ import annotations

import pytest

import atomr_accel

pytestmark = pytest.mark.skipif(
    atomr_accel.NvrtcKernel is None, reason="nvrtc feature not compiled in"
)


# ---------------------------------------------------------------------
# Surface-shape checks: NvrtcKernel + KernelArg are exposed and have
# the expected attribute names.
# ---------------------------------------------------------------------


def test_nvrtc_kernel_class_is_exposed():
    assert atomr_accel.NvrtcKernel is not None
    assert hasattr(atomr_accel.NvrtcKernel, "launch")
    assert hasattr(atomr_accel.NvrtcKernel, "name")
    assert hasattr(atomr_accel.NvrtcKernel, "generation")


def test_kernel_arg_class_is_exposed():
    assert atomr_accel.KernelArg is not None
    for name in (
        "buffer_f32",
        "buffer_f64",
        "buffer_i32",
        "buffer_u32",
        "buffer_u8",
        "scalar_i32",
        "scalar_i64",
        "scalar_f32",
        "scalar_f64",
        "scalar_u32",
        "scalar_u64",
    ):
        assert hasattr(atomr_accel.KernelArg, name), name


def test_device_compile_kernel_method_exists():
    """Device.compile_kernel is bound on the Rust-side class."""
    assert hasattr(atomr_accel.Device, "compile_kernel")


# ---------------------------------------------------------------------
# Submodule re-exports: atomr_accel.nvrtc.{NvrtcKernel,KernelArg}.
# ---------------------------------------------------------------------


def test_submodule_reexports():
    from atomr_accel import nvrtc as nvrtc_mod

    assert nvrtc_mod.NvrtcKernel is atomr_accel.NvrtcKernel
    assert nvrtc_mod.KernelArg is atomr_accel.KernelArg


# ---------------------------------------------------------------------
# Scalar KernelArg construction round-trips. These don't require a
# device — they're pure Rust wrappers — so they always run.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctor, value, label",
    [
        ("scalar_i32", 7, "scalar_i32"),
        ("scalar_i64", 9_000_000_000, "scalar_i64"),
        ("scalar_f32", 1.5, "scalar_f32"),
        ("scalar_f64", 2.5, "scalar_f64"),
        ("scalar_u32", 17, "scalar_u32"),
        ("scalar_u64", 18, "scalar_u64"),
    ],
)
def test_kernel_arg_scalar_construction(ctor, value, label):
    arg = getattr(atomr_accel.KernelArg, ctor)(value)
    assert isinstance(arg, atomr_accel.KernelArg)
    assert label in repr(arg)


# ---------------------------------------------------------------------
# Buffer KernelArg construction. Mock-mode allocate raises Unrecoverable,
# so we can't actually build a GpuBufferF32 to wrap unless the mock
# ContextActor surfaced it. The construction site is always reachable
# in principle — guard the test on whether mock allocate succeeds.
# ---------------------------------------------------------------------


def test_kernel_arg_buffer_construction_optional():
    """If we *can* allocate a buffer, then KernelArg.buffer_f32 wraps it
    without error. Otherwise (mock mode) skip — the construction site
    is a pure Rust pass-through."""
    with atomr_accel.System.open("nvrtc-buffer") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        try:
            buf = dev.allocate_f32(16, timeout_secs=2.0)
        except atomr_accel.Unrecoverable:
            pytest.skip("mock-mode allocate cannot mint a buffer")
        arg = atomr_accel.KernelArg.buffer_f32(buf)
        assert "buffer_f32" in repr(arg)


# ---------------------------------------------------------------------
# Device-side compile path. The default mock device does NOT enable
# `EnabledLibraries::NVRTC` so `compile_kernel` raises
# GpuRuntimeError. (When the mock ContextActor *does* mint a NvrtcActor
# child, the actor itself replies Unrecoverable for every Compile.)
# Either branch is acceptable — we just verify the call is dispatched.
# ---------------------------------------------------------------------


def test_compile_kernel_in_mock_mode():
    src = 'extern "C" __global__ void k(float* a) { a[0] = 1.0f; }'
    with atomr_accel.System.open("nvrtc-compile") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            dev.compile_kernel("k", src, timeout_secs=2.0)


def test_libraries_ready_includes_nvrtc_key():
    """``Device.libraries_ready`` advertises an ``nvrtc`` key whether
    or not the actor was spawned."""
    with atomr_accel.System.open("nvrtc-ready") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        ready = dev.libraries_ready(timeout_secs=2.0)
        assert "nvrtc" in ready, ready


# ---------------------------------------------------------------------
# KernelArg consumption semantics — each PyKernelArg is a one-shot
# wrapper. Re-using one in two launches must surface a clear error.
# We can't actually call launch() without a compiled kernel handle, so
# this test only exercises the construction + repr surface.
# ---------------------------------------------------------------------


def test_kernel_arg_repr_stable():
    """The pretty-print form is stable so logs round-trip through it."""
    a = atomr_accel.KernelArg.scalar_f32(3.14)
    r = repr(a)
    # Don't assert on numeric formatting — just on the variant tag.
    assert "scalar_f32" in r
    assert "KernelArg" in r
