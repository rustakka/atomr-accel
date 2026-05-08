"""``atomr_accel.telemetry`` — NVTX / NVML / CUPTI bindings.

Wraps `atomr-accel-telemetry`. Each backend is feature-gated on the
Rust side (``telemetry-nvtx``, ``telemetry-nvml``, ``telemetry-cupti``);
the corresponding class is ``None`` on builds where the feature was
not compiled in.

Phase 3 surface:

* :class:`NvtxKernelTrace` — context manager that emits an NVTX
  domain range. ``with NvtxKernelTrace("span"): ...``. On hosts
  without ``libnvToolsExt.so`` ``__enter__`` raises ``Unrecoverable``.
* :class:`NvmlActor` — polling actor for power / temperature / clocks
  / memory. ``read()`` returns the latest snapshot as a dict;
  ``power_w()`` is a convenience accessor for device 0's wattage.
  Construction raises ``Unrecoverable`` on hosts without NVML.
* :class:`CuptiSession` — CUPTI activity session. ``start(categories)``
  / ``stop()`` / ``drain()`` lifecycle. On mock-mode hosts the actor
  spawns and accepts messages; ``drain()`` returns an empty list.

Method-level depth — full NVML field projection, per-category
activity decoders, range-profiler metric selection — follows in the
Phase 3.5 telemetry-coverage tracking issue.
"""

try:
    from ._native import NvtxKernelTrace
except ImportError:
    NvtxKernelTrace = None  # type: ignore[assignment]

try:
    from ._native import NvmlActor
except ImportError:
    NvmlActor = None  # type: ignore[assignment]

try:
    from ._native import CuptiSession
except ImportError:
    CuptiSession = None  # type: ignore[assignment]

__all__ = ["NvtxKernelTrace", "NvmlActor", "CuptiSession"]
