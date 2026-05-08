"""``atomr_accel.flashattn`` — FlashAttn handle (FlashAttention v2 + v3).

Obtained via ``device.flashattn()`` when the wheel was built with
``--features flashattn``. Phase 4 surface: ``forward_f16``, which
constructs an ``Fa2FwdRequest::<F16>`` and dispatches it through the
actor (request-construction validates the dispatch cell against the
arch / dtype / head_dim / mask / bias / sink / gqa table).

FA2 backward, FA3 forward (Hopper / Blackwell), varlen, paged KV-cache,
chunked prefill, and the bf16 / fp8 dtype axes follow in the Phase 4.5
FlashAttention tracking issue.

On builds without FlashAttention, ``FlashAttn`` is ``None``.
"""

try:
    from ._native import FlashAttn
except ImportError:
    FlashAttn = None  # type: ignore[assignment]

__all__ = ["FlashAttn"]
