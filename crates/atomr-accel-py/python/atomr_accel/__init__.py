"""atomr-accel — actor-shaped face for NVIDIA CUDA, exposed to Python.

The native extension lives in ``atomr_accel._native``; everything you
typically need is re-exported here. Downstream libraries should import
from ``atomr_accel`` (this module) and treat ``_native`` as private.

Quick start
-----------

>>> import numpy as np
>>> import atomr_accel
>>>
>>> # System lifecycle is sync-friendly. Use as a context manager:
>>> with atomr_accel.System.open("my-app") as sys:
...     dev = sys.spawn_device(device_id=0, mock=True)
...     # mock=True makes this work on hosts without a CUDA driver.
...     try:
...         buf = dev.allocate_f32(16)
...     except atomr_accel.Unrecoverable as e:
...         # Mock mode replies with Unrecoverable; on real hardware
...         # this returns a GpuBuffer.
...         pass
...
"""

from ._native import (  # noqa: F401
    __version__,
    System,
    Device,
    DeviceLoad,
    GpuBuffer,
    GpuRuntimeError,
    ContextPoisoned,
    OutOfMemory,
    Unrecoverable,
    GpuRefStale,
    LibraryError,
    AskTimeout,
)

# Optional surface — present only when the matching cargo feature was
# compiled in. The `try` keeps imports robust on minimal builds.
try:
    from ._native import RngGenerator  # noqa: F401
except ImportError:
    RngGenerator = None  # type: ignore[assignment]

try:
    from ._native import NvrtcKernel  # noqa: F401
except ImportError:
    NvrtcKernel = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "System",
    "Device",
    "DeviceLoad",
    "GpuBuffer",
    "RngGenerator",
    "NvrtcKernel",
    "GpuRuntimeError",
    "ContextPoisoned",
    "OutOfMemory",
    "Unrecoverable",
    "GpuRefStale",
    "LibraryError",
    "AskTimeout",
]
