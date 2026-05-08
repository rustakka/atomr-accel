"""``atomr_accel.nvrtc`` — NvrtcKernel handle.

Phase 1 keeps the existing ``NvrtcKernel`` stub (``name``, ``generation``).
The device-side ``compile_kernel`` and ``launch`` paths require
``SnapshotChildren`` plumbing plus typed ``KernelArg`` marshalling; both
follow in the Phase 1.5 NVRTC tracking issue.

On builds without NVRTC, ``NvrtcKernel`` is ``None``.
"""

try:
    from ._native import NvrtcKernel
except ImportError:
    NvrtcKernel = None  # type: ignore[assignment]

__all__ = ["NvrtcKernel"]
