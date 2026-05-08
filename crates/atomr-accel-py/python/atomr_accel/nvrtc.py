"""``atomr_accel.nvrtc`` — NVRTC compile + launch surface.

Phase 1.5++ wires the full path:

  * ``Device.compile_kernel(name, src, ...)`` returns an
    :class:`NvrtcKernel`.
  * :class:`NvrtcKernel.launch(grid, block, args, ...)` dispatches
    typed :class:`KernelArg` payloads (scalar f32/f64/i32/i64/u32/u64
    plus device-pointer wrappers around every supported ``GpuBuffer*``).

On builds without NVRTC, ``NvrtcKernel`` and ``KernelArg`` are ``None``.
"""

try:
    from ._native import NvrtcKernel, KernelArg
except ImportError:
    NvrtcKernel = None  # type: ignore[assignment]
    KernelArg = None  # type: ignore[assignment]

__all__ = ["NvrtcKernel", "KernelArg"]
