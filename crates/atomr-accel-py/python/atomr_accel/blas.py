"""``atomr_accel.blas`` — Blas handle (cuBLAS).

Obtained via ``device.blas()``. Phase 1 surface: ``gemm_f32``,
``gemm_f64``, ``axpy_f32``. Strided-batched gemm and the rest of
L1/L2/L3 follow in the Phase 1.5 cuBLAS tracking issue.
"""

from ._native import Blas

__all__ = ["Blas"]
