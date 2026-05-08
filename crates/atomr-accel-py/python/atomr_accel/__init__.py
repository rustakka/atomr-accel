"""atomr-accel — actor-shaped face for NVIDIA CUDA, exposed to Python.

The native extension lives in ``atomr_accel._native``; the public
surface is re-exported here. Per-domain helpers also live in side
modules (``atomr_accel.{system,device,blas,cudnn,fft,rng,solver,
collective,nvrtc,patterns,train,agents,realtime,telemetry,cub,cutlass,
flashattn,tensorrt,errors}``). Downstream libraries should import from
``atomr_accel`` (this module) and treat ``_native`` as private.

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

# ─── Always-present surface ─────────────────────────────────────────
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
    GpuBufferC64,
    GpuBufferC128,
    Blas,
    GpuRuntimeError,
    ContextPoisoned,
    OutOfMemory,
    Unrecoverable,
    GpuRefStale,
    LibraryError,
    AskTimeout,
    # Phase 1.5 — CUDA graphs + memory ops
    GraphCapture,
    GraphScript,
    GraphHandle,
    Memory,
    ManagedBufferF32,
    # Phase 2 — patterns
    DynamicBatchingServer,
    InferenceCascade,
    ModelReplicaPool,
    FairShareScheduler,
    HotSwapServer,
    SpeculativeDecoder,
    MoeRouter,
    # Phase 2 — train
    AsyncParameterServer,
    DataParallelTrainer,
    PipelineParallelTrainer,
    TensorParallelTrainer,
    # Phase 2 — agents
    SharedGpuStateCoordinator,
    EmbeddingCache,
    CpuVectorIndex,
    RagPipeline,
    LangGraphGpuActor,
    # Phase 2 — cuda-realtime
    ClothSimulationActor,
    FluidSimulationActor,
    ParticleSystemActor,
    SpatialIndexActor,
    GpuHashMapActor,
    ImageFilterPipeline,
)


def _try_import(name):
    """Helper — return the symbol from ``_native`` or ``None`` if the
    matching cargo feature wasn't compiled in."""
    try:
        from . import _native
        return getattr(_native, name)
    except (ImportError, AttributeError):
        return None


# ─── Optional surface (cudnn / cufft / curand / cusolver / nccl /
#     nvrtc / telemetry / cub / cutlass / flashattn / tensorrt) ───
Cudnn = _try_import("Cudnn")
RnnBwdInputs = _try_import("RnnBwdInputs")
MultiHeadAttnBwdInputs = _try_import("MultiHeadAttnBwdInputs")
Fft = _try_import("Fft")
RngGenerator = _try_import("RngGenerator")
Solver = _try_import("Solver")
Collective = _try_import("Collective")
NvrtcKernel = _try_import("NvrtcKernel")
KernelArg = _try_import("KernelArg")

# Phase 1.5 — IPC handles (cfg cuda-ipc)
IpcMemHandle = _try_import("IpcMemHandle")
IpcOpenedMem = _try_import("IpcOpenedMem")

# Phase 3 — telemetry
NvtxKernelTrace = _try_import("NvtxKernelTrace")
NvmlActor = _try_import("NvmlActor")
CuptiSession = _try_import("CuptiSession")

# Phase 4 — template kernel crates
Cub = _try_import("Cub")
Cutlass = _try_import("Cutlass")
FlashAttn = _try_import("FlashAttn")
TensorRt = _try_import("TensorRt")


__all__ = [
    "__version__",
    # Core
    "System",
    "Device",
    "DeviceLoad",
    "GpuBuffer",
    "GpuBufferF32",
    "GpuBufferF64",
    "GpuBufferI32",
    "GpuBufferU32",
    "GpuBufferU8",
    "GpuBufferC64",
    "GpuBufferC128",
    "Blas",
    # Phase 1.5 — graphs + memory
    "GraphCapture",
    "GraphScript",
    "GraphHandle",
    "Memory",
    "ManagedBufferF32",
    "IpcMemHandle",
    "IpcOpenedMem",
    # Errors
    "GpuRuntimeError",
    "ContextPoisoned",
    "OutOfMemory",
    "Unrecoverable",
    "GpuRefStale",
    "LibraryError",
    "AskTimeout",
    # Phase 1 optional cuda-kernel handles
    "Cudnn",
    "RnnBwdInputs",
    "MultiHeadAttnBwdInputs",
    "Fft",
    "RngGenerator",
    "Solver",
    "Collective",
    "NvrtcKernel",
    "KernelArg",
    # Phase 2 — patterns
    "DynamicBatchingServer",
    "InferenceCascade",
    "ModelReplicaPool",
    "FairShareScheduler",
    "HotSwapServer",
    "SpeculativeDecoder",
    "MoeRouter",
    # Phase 2 — train
    "AsyncParameterServer",
    "DataParallelTrainer",
    "PipelineParallelTrainer",
    "TensorParallelTrainer",
    # Phase 2 — agents
    "SharedGpuStateCoordinator",
    "EmbeddingCache",
    "CpuVectorIndex",
    "RagPipeline",
    "LangGraphGpuActor",
    # Phase 2 — cuda-realtime
    "ClothSimulationActor",
    "FluidSimulationActor",
    "ParticleSystemActor",
    "SpatialIndexActor",
    "GpuHashMapActor",
    "ImageFilterPipeline",
    # Phase 3 — telemetry
    "NvtxKernelTrace",
    "NvmlActor",
    "CuptiSession",
    # Phase 4 — template kernel crates
    "Cub",
    "Cutlass",
    "FlashAttn",
    "TensorRt",
]
