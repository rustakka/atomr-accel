"""``atomr_accel.cub`` — Cub handle (CUB device-wide primitives).

Obtained via ``device.cub()`` when the wheel was built with
``--features cub`` *and* the device's ``KernelChildren`` extras slot
has the CUB actor registered. Phase 4 surface: ``reduce_sum_f32``.
Scan / sort / histogram / select / partition / segmented_reduce, plus
the f64 / i32 / u32 / i64 / u64 / i8 / u8 dtype axes follow in the
Phase 4.5 CUB tracking issue.

On builds without CUB, ``Cub`` is ``None``.
"""

try:
    from ._native import Cub
except ImportError:
    Cub = None  # type: ignore[assignment]

__all__ = ["Cub"]
