"""``atomr_accel.cudnn`` — Cudnn handle (cuDNN).

Obtained via ``device.cudnn()`` when the wheel was built with
``--features cudnn`` *and* the device's ``EnabledLibraries::CUDNN``
flag is set. Phase 1 surface: ``conv2d_fwd_f32``. Pool / batch_norm
/ layer_norm / RNN / attention / dropout follow in the Phase 1.5
cuDNN tracking issue.

``RnnBwdInputs`` and ``MultiHeadAttnBwdInputs`` are builder classes
used to populate the wide-arg backward requests via per-field
setters; pass the populated builder to ``Cudnn.rnn_bwd_f32`` /
``Cudnn.multihead_attn_bwd_f32``.

On builds without cuDNN, ``Cudnn`` / ``RnnBwdInputs`` /
``MultiHeadAttnBwdInputs`` are ``None``.
"""

try:
    from ._native import Cudnn
except ImportError:
    Cudnn = None  # type: ignore[assignment]

try:
    from ._native import RnnBwdInputs
except ImportError:
    RnnBwdInputs = None  # type: ignore[assignment]

try:
    from ._native import MultiHeadAttnBwdInputs
except ImportError:
    MultiHeadAttnBwdInputs = None  # type: ignore[assignment]

__all__ = ["Cudnn", "RnnBwdInputs", "MultiHeadAttnBwdInputs"]
