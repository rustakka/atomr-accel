"""``atomr_accel.cutlass`` — Cutlass handle (CUTLASS template kernels).

Obtained via ``device.cutlass()`` when the wheel was built with
``--features cutlass``. Phase 4 surface: ``gemm_f32_plan``,
``gemm_f64_plan``, ``dispatched``, ``plan_cache_len``. Each plan call
constructs a typed ``GemmRequest::<T>``, hands it to the actor, and
records the rendered ``.cu`` source in the plan cache; if the device
has an NVRTC ``compile_sink`` wired in, the source is forwarded for
compilation.

Grouped GEMM, conv (fwd / dgrad / wgrad), refit, EVT, and the
fp16 / bf16 / fp8 / fp4 dtype axes follow in the Phase 4.5 CUTLASS
tracking issue.

On builds without CUTLASS, ``Cutlass`` is ``None``.
"""

try:
    from ._native import Cutlass
except ImportError:
    Cutlass = None  # type: ignore[assignment]

__all__ = ["Cutlass"]
