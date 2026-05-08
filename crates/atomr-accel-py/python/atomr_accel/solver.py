"""``atomr_accel.solver`` — Solver handle (cuSOLVER).

Phase 1 ships the handle class as a structural anchor; the device-side
``solver()`` accessor and the LU / QR / Cholesky / SVD / eigendecomp
methods follow in the Phase 1.5 cuSOLVER tracking issue (the actor is
not auto-spawned by ``ContextActor`` today — it requires a separate
spawn path that's part of the same issue).

On builds without cuSOLVER, ``Solver`` is ``None``.
"""

try:
    from ._native import Solver
except ImportError:
    Solver = None  # type: ignore[assignment]

__all__ = ["Solver"]
