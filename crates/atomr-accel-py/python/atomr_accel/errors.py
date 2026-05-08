"""``atomr_accel.errors`` — typed exception hierarchy.

::

    Exception
    └── GpuRuntimeError                (base)
        ├── ContextPoisoned            (CUDA context poisoned; supervisor will restart)
        ├── OutOfMemory                (allocator OOM; supervisor resumes)
        ├── Unrecoverable              (hardware fault / past retry budget)
        ├── GpuRefStale                (buffer used after context rebuild)
        ├── LibraryError               (cuBLAS/cuDNN/etc.)
        └── AskTimeout                 (ask exceeded its budget)

Catch the most specific subclass that fits — e.g.
``except OutOfMemory: shrink_batch_size()``.
"""

from ._native import (
    AskTimeout,
    ContextPoisoned,
    GpuRefStale,
    GpuRuntimeError,
    LibraryError,
    OutOfMemory,
    Unrecoverable,
)

__all__ = [
    "AskTimeout",
    "ContextPoisoned",
    "GpuRefStale",
    "GpuRuntimeError",
    "LibraryError",
    "OutOfMemory",
    "Unrecoverable",
]
