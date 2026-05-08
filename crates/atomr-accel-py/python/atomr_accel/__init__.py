"""atomr-accel — actor-shaped face for NVIDIA CUDA, exposed to Python.

The native extension lives in ``atomr_accel._native``; the public
surface is re-exported here. Per-domain helpers also live in side
modules (``atomr_accel.system``, ``atomr_accel.device``,
``atomr_accel.blas``, ``atomr_accel.cudnn``, ``atomr_accel.fft``,
``atomr_accel.rng``, ``atomr_accel.solver``, ``atomr_accel.collective``,
``atomr_accel.nvrtc``, ``atomr_accel.errors``). Downstream libraries
should import from ``atomr_accel`` (this module) and treat
``_native`` as private.

Quick start
-----------

>>> import numpy as np
>>> import atomr_accel
>>>
>>> with atomr_accel.System.open("my-app") as sys:
...     dev = sys.spawn_device(device_id=0, mock=True)
...     try:
...         buf = dev.allocate_f32(16)
...     except atomr_accel.Unrecoverable:
...         # mock=True replies Unrecoverable; on real hardware we'd
...         # get a GpuBufferF32 back.
...         pass
...
"""

from ._native import (  # noqa: F401
    __version__,
    System,
    Device,
    DeviceLoad,
    GpuBuffer,
    GpuBufferF32,
    GpuBufferF64,
    GpuBufferI32,
    GpuBufferU32,
    GpuBufferU8,
    Blas,
    GpuRuntimeError,
    ContextPoisoned,
    OutOfMemory,
    Unrecoverable,
    GpuRefStale,
    LibraryError,
    AskTimeout,
)

# Optional surface — present only when the matching cargo feature was
# compiled in. The ``try`` keeps imports robust on minimal builds.
try:
    from ._native import Cudnn  # noqa: F401
except ImportError:
    Cudnn = None  # type: ignore[assignment]

try:
    from ._native import Fft  # noqa: F401
except ImportError:
    Fft = None  # type: ignore[assignment]

try:
    from ._native import RngGenerator  # noqa: F401
except ImportError:
    RngGenerator = None  # type: ignore[assignment]

try:
    from ._native import Solver  # noqa: F401
except ImportError:
    Solver = None  # type: ignore[assignment]

try:
    from ._native import Collective  # noqa: F401
except ImportError:
    Collective = None  # type: ignore[assignment]

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
    "GpuBufferF32",
    "GpuBufferF64",
    "GpuBufferI32",
    "GpuBufferU32",
    "GpuBufferU8",
    "Blas",
    "Cudnn",
    "Fft",
    "RngGenerator",
    "Solver",
    "Collective",
    "NvrtcKernel",
    "GpuRuntimeError",
    "ContextPoisoned",
    "OutOfMemory",
    "Unrecoverable",
    "GpuRefStale",
    "LibraryError",
    "AskTimeout",
]
