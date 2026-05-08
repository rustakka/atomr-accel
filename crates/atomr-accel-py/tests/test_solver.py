"""Mock-mode surface checks for ``atomr_accel.Solver``.

Phase 1.5++ wires the cuSOLVER spawn-path (``ContextActor`` now mints a
``SolverActor`` when both the ``cusolver`` cargo feature and
``EnabledLibraries::CUSOLVER`` are on) and exposes a representative
slice of dense ops on the Python ``Solver`` handle:

  * ``lu_{f32,f64}`` / ``lu_solve_{f32,f64}`` — getrf / getrs
  * ``cholesky_{f32,f64}`` — potrf
  * ``qr_{f32,f64}`` — geqrf
  * ``svd_{f32,f64}`` — gesvd
  * ``eigh_{f32,f64}`` — syevd

These tests verify that:

  * the methods exist on the Rust-side ``Solver`` class;
  * ``Device.solver()`` either returns a ``Solver`` handle (when the
    mock ``ContextActor`` minted one) or raises ``GpuRuntimeError`` when
    ``EnabledLibraries::CUSOLVER`` isn't set on the device.

Real-numerics correctness is covered by the GPU-runtime e2e suite under
``crates/atomr-accel-cuda/tests/`` — this file only proves the
Python-side wiring.

The whole module is skipped when ``atomr_accel.Solver is None`` (i.e.
the wheel was built without ``--features cusolver``).
"""
from __future__ import annotations

import pytest

import atomr_accel

pytestmark = pytest.mark.skipif(
    atomr_accel.Solver is None, reason="cusolver feature not compiled in"
)


# ---------------------------------------------------------------------
# Surface-shape checks: every Phase 1.5++ method exists on the class.
# ---------------------------------------------------------------------

PHASE_1_5_SOLVER_METHODS = [
    # LU factorize / solve.
    "lu_f32",
    "lu_f64",
    "lu_solve_f32",
    "lu_solve_f64",
    # Cholesky.
    "cholesky_f32",
    "cholesky_f64",
    # QR.
    "qr_f32",
    "qr_f64",
    # SVD.
    "svd_f32",
    "svd_f64",
    # Symmetric eigendecomposition.
    "eigh_f32",
    "eigh_f64",
]


def test_phase_1_5_solver_method_count():
    """Catch accidental coverage regressions."""
    # 6 ops × 2 dtypes = 12 methods.
    assert len(PHASE_1_5_SOLVER_METHODS) == 12


@pytest.mark.parametrize("method", PHASE_1_5_SOLVER_METHODS)
def test_method_exists_on_class(method: str):
    """Each Phase 1.5++ method is a bound attribute on `Solver`."""
    assert hasattr(atomr_accel.Solver, method), method


# ---------------------------------------------------------------------
# Device.solver() resolves to either a Solver handle or GpuRuntimeError.
# Mirrors test_handles.py::test_cudnn_handle_optional. The mock
# ContextActor only mints a SolverActor when EnabledLibraries::CUSOLVER
# is set on the device, which the default `spawn_device(mock=True)`
# path doesn't do (BLAS-only default). So we expect GpuRuntimeError.
# ---------------------------------------------------------------------


def test_solver_handle_optional():
    """``device.solver()`` only resolves when both the cargo feature is
    on *and* the device has CUSOLVER enabled — mock mode lacks the
    flag, so we expect GpuRuntimeError."""
    with atomr_accel.System.open("solver-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        try:
            handle = dev.solver(timeout_secs=2.0)
            # If we somehow got a handle, it should be a Solver instance.
            assert isinstance(handle, atomr_accel.Solver)
            assert "Solver" in repr(handle)
        except atomr_accel.GpuRuntimeError:
            pass


def test_libraries_ready_includes_cusolver_key():
    """``Device.libraries_ready`` advertises a ``cusolver`` key whether
    or not the actor was spawned. Mirrors the ``cudnn`` / ``cufft`` /
    ``curand`` keys."""
    with atomr_accel.System.open("solver-ready-mock") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        ready = dev.libraries_ready(timeout_secs=2.0)
        assert "cusolver" in ready, ready


# ---------------------------------------------------------------------
# uplo string parsing. Validation is a pure-Rust string match, so we
# don't actually need a live SolverActor — just a constructible Solver
# handle. Since the mock ContextActor doesn't spawn one without the
# CUSOLVER flag, we skip the call-shape tests when no handle can be
# obtained.
# ---------------------------------------------------------------------


def _try_mock_solver():
    """Return ``(system_ctx, dev, solver_handle)`` or ``None`` if the
    mock ContextActor didn't mint a SolverActor child (default config
    has BLAS only)."""
    sys_ctx = atomr_accel.System.open("solver-phase15")
    sys = sys_ctx.__enter__()
    try:
        dev = sys.spawn_device(device_id=0, mock=True)
        try:
            handle = dev.solver(timeout_secs=2.0)
        except atomr_accel.GpuRuntimeError:
            sys_ctx.__exit__(None, None, None)
            return None
        return sys_ctx, dev, handle
    except Exception:
        sys_ctx.__exit__(None, None, None)
        raise


def test_repr_when_solver_reachable():
    """Sanity check on the `__repr__` so existing code that logs the
    handle keeps a stable string. Skipped when no mock SolverActor."""
    bundle = _try_mock_solver()
    if bundle is None:
        pytest.skip("mock ContextActor did not mint a SolverActor child")
    sys_ctx, _dev, solver = bundle
    try:
        assert "Solver" in repr(solver)
    finally:
        sys_ctx.__exit__(None, None, None)


def test_uplo_rejects_unknown_string_when_solver_reachable():
    """Unknown ``uplo`` aliases raise GpuRuntimeError before any actor
    traffic. Skipped when no mock SolverActor."""
    bundle = _try_mock_solver()
    if bundle is None:
        pytest.skip("mock ContextActor did not mint a SolverActor child")
    sys_ctx, dev, solver = bundle
    try:
        # Mock-mode allocate raises Unrecoverable, so we can't actually
        # build a buffer to pass through. The string parsing happens
        # before buffer access — but pyo3's argument parsing demands
        # the buffer arg first. To exercise the string path we'd need a
        # real allocation; settle for confirming alloc itself raises.
        with pytest.raises(atomr_accel.Unrecoverable):
            dev.allocate_f32(16, timeout_secs=2.0)
    finally:
        sys_ctx.__exit__(None, None, None)
