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

/// TensorRT integration (Phase 8). Re-exports the sibling
/// `atomr-accel-tensorrt` crate root so callers can `use
/// atomr_accel_cuda::prelude::tensorrt::*` to pick up `TrtActor` /
/// `TrtMsg` / `IBuilderConfig` alongside `DeviceActor`.
#[cfg(feature = "tensorrt")]
pub use atomr_accel_tensorrt as tensorrt;
