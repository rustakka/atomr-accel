"""``atomr_accel.rng`` — RngGenerator handle (cuRAND).

Obtained via ``device.rng()`` when the wheel was built with
``--features curand`` *and* the device's ``EnabledLibraries::CURAND``
flag is set. Phase 1 surface: ``set_seed``, ``uniform_f32``,
``normal_f32``. Quasi generators, log-normal / Poisson / exponential /
beta / Cauchy / gamma / discrete distributions, and per-dtype variants
(``uniform_f64``, ``uniform_u32``, …) follow in the Phase 1.5 cuRAND
tracking issue.

On builds without cuRAND, ``RngGenerator`` is ``None``.
"""

try:
    from ._native import RngGenerator
except ImportError:
    RngGenerator = None  # type: ignore[assignment]

__all__ = ["RngGenerator"]
