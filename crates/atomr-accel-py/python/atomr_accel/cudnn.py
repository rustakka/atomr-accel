"""``atomr_accel.cudnn`` — Cudnn handle (cuDNN).

Obtained via ``device.cudnn()`` when the wheel was built with
``--features cudnn`` *and* the device's ``EnabledLibraries::CUDNN``
flag is set. Phase 1 surface: ``conv2d_fwd_f32``. Pool / batch_norm
/ layer_norm / RNN / attention / dropout follow in the Phase 1.5
cuDNN tracking issue.

On builds without cuDNN, ``Cudnn`` is ``None``.
"""

try:
    from ._native import Cudnn
except ImportError:
    Cudnn = None  # type: ignore[assignment]

__all__ = ["Cudnn"]
