"""``atomr_accel.tensorrt`` — TensorRt handle (NVIDIA TensorRT).

Obtained via ``device.tensorrt()`` when the wheel was built with
``--features tensorrt``. The wrapped crate's libnvinfer link is itself
gated behind the upstream ``tensorrt-link`` feature so the wheel
builds on hosts without TensorRT installed.

Phase 4 ships this handle as a structural anchor only — the upstream
``TrtMsg`` mailbox surface (Build / Deserialize / CreateContext /
EnqueueOnStream / Refit) needs typed engine handles and CUDA stream
marshalling that don't currently round-trip through PyO3. The single
exposed method, ``runtime_ready``, lets callers feature-detect
TensorRT before any Phase 4.5 typed methods land.

On builds without TensorRT, ``TensorRt`` is ``None``.
"""

try:
    from ._native import TensorRt
except ImportError:
    TensorRt = None  # type: ignore[assignment]

__all__ = ["TensorRt"]
