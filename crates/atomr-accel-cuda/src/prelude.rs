//! Common imports for users of `atomr-accel-cuda`.

pub use crate::completion::{CompletionStrategy, HostFnCompletion};
pub use crate::device::{
    ContextActor, ContextMsg, DeviceActor, DeviceConfig, DeviceLoad, DeviceMsg, DeviceState,
    EnabledLibraries, HostBuf, KernelChildren, SgemmRequest,
};
pub use crate::dispatcher::GpuDispatcher;
pub use crate::error::{decider, device_supervisor_strategy, DeviceSupervisor, GpuError};
pub use crate::gpu_ref::GpuRef;
pub use crate::graph::{GraphActor, GraphHandle, GraphMsg, GraphOp};
pub use crate::host::{
    PinnedBuf, PinnedBufferPool, PinnedBufferPoolConfig, PinnedPoolMsg, PinnedPoolStats,
};
pub use crate::kernel::envelope;
pub use crate::kernel::record::RecordMode;
pub use crate::kernel::{BlasActor, BlasMsg};
pub use crate::memory::{
    ManagedAllocatorActor, ManagedFlags, ManagedMsg, ManagedRef, ManagedStats,
};
pub use crate::p2p::{P2pGraph, P2pMsg, P2pTopology};
pub use crate::pipeline::{
    run_pipeline, spawn_pipeline, BoxedStage, PipelineExecutor, PipelineExecutorN, PipelineSink,
    PipelineSource, PipelineStage, StageBox,
};
pub use crate::placement::{
    DeviceChoice, LeastLoadedPolicy, PlacementActor, PlacementHints, PlacementMsg, PlacementPolicy,
    RoundRobinPolicy,
};
pub use crate::replay::{
    replay_via_sink, JournalEntry, ReplayHarness, ReplayMode, ReplayMsg, ReplaySink,
};

#[cfg(feature = "cusolver")]
pub use crate::kernel::{SolverActor, SolverMsg, Uplo};

#[cfg(feature = "cusparse")]
pub use crate::kernel::{CsrMatrix, SparseActor, SparseMsg};

#[cfg(feature = "cutensor")]
pub use crate::kernel::{TensorActor, TensorMsg, TensorSpec};

#[cfg(feature = "cublaslt")]
pub use crate::kernel::{Activation, BlasLtActor, BlasLtMsg};

#[cfg(feature = "nvrtc")]
pub use crate::kernel::{KernelArg, KernelHandle, NvrtcActor, NvrtcMsg, NvrtcOpts};

#[cfg(feature = "nccl")]
pub use crate::kernel::{CollectiveActor, CollectiveMsg, ReduceOp};
#[cfg(feature = "nccl")]
pub use crate::multi_device::{NcclWorldActor, NcclWorldConfig, NcclWorldMsg};
pub use crate::stream::{
    ActorHints, PerActorAllocator, PooledAllocator, Priority, SingleStreamAllocator,
    StreamAllocator, WorkloadKind,
};

#[cfg(feature = "cudnn")]
pub use crate::kernel::{
    ActivationKind, ActivationRequest, ConvForwardRequest, ConvParams, CudnnActor, CudnnMsg,
    SoftmaxRequest,
};

#[cfg(feature = "cufft")]
pub use crate::kernel::{FftActor, FftKind, FftMsg, PlanKey};

#[cfg(feature = "curand")]
pub use crate::kernel::{RngActor, RngMsg};

// Phase 9 — observability backends. Re-export the public surface
// of `atomr-accel-telemetry` so callers can `use prelude::*` and
// drop NVTX / NVML / CUPTI handles straight onto their actor
// system.
#[cfg(feature = "nvtx-trace")]
pub use atomr_accel_telemetry::nvtx::{Domain as NvtxDomain, NvtxKernelTrace};

#[cfg(feature = "nvml")]
pub use atomr_accel_telemetry::nvml::{
    register_all as register_nvml_probes, NvmlActor, NvmlConfig, NvmlError, NvmlMsg, NvmlReply,
    NvmlSnapshot, ProbeRegistration as NvmlProbeRegistration,
};

#[cfg(feature = "cupti")]
pub use atomr_accel_telemetry::cupti::{
    Activity, ActivityCategory, CuptiBootstrap, CuptiError, CuptiMsg, CuptiReply, CuptiSession,
};

#[cfg(any(feature = "nvtx-trace", feature = "nvml", feature = "cupti"))]
pub use atomr_accel_telemetry::{KernelInfo, KernelTrace, NoopKernelTrace};
