"""``atomr_accel.fft`` — Fft handle (cuFFT).

Obtained via ``device.fft()`` when the wheel was built with
``--features cufft`` *and* the device's ``EnabledLibraries::CUFFT``
flag is set. Phase 1 ships the handle as a structural anchor; typed
plans (R2C/C2R/C2C across f32/f64, 1-D / 2-D / 3-D, plan-many,
callbacks) follow in the Phase 1.5 cuFFT tracking issue — they
require numpy↔complex marshalling that's deferred for this PR.

On builds without cuFFT, ``Fft`` is ``None``.
"""

try:
    from ._native import Fft
except ImportError:
    Fft = None  # type: ignore[assignment]

__all__ = ["Fft"]
