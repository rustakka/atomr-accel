"""``atomr_accel.collective`` — Collective handle (NCCL).

Phase 1 ships the handle class as a structural anchor; the actor lives
in ``atomr-accel-cuda::multi_device`` and is spawned by
``NcclWorldActor`` rather than auto-spawned by a single device's
``ContextActor``. All-reduce / broadcast / all-gather / reduce-scatter
/ all-to-all / send-recv across NcclReduceSupported dtypes follow in
the Phase 2 multi-GPU tracking issue.

On builds without NCCL, ``Collective`` is ``None``.
"""

try:
    from ._native import Collective
except ImportError:
    Collective = None  # type: ignore[assignment]

__all__ = ["Collective"]
