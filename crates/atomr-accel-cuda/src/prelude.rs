//! Common imports for users of `atomr-accel-cuda`.

pub use crate::completion::{CompletionStrategy, HostFnCompletion};
pub use crate::device::{
    ContextActor, ContextMsg, DeviceActor, DeviceConfig, DeviceLoad, DeviceMsg, DeviceState,
    EnabledLibraries, HostBuf, KernelChildren, SgemmRequest,
};
pub use crate::dispatcher::GpuDispatcher;
pub use crate::error::{decider, device_supervisor_strategy, DeviceSupervisor, GpuError};
pub use crate::gpu_ref::GpuRef;
#[cfg(feature = "cufft")]
pub use crate::graph::FftR2COp;
#[allow(deprecated)]
pub use crate::graph::GraphOpLegacy;
#[cfg(feature = "curand")]
pub use crate::graph::RngFillUniformOp;
pub use crate::graph::{
    GraphActor, GraphHandle, GraphMsg, GraphOp, GraphRecordCtx, MemcpyOp, SgemmOp,
};
pub use crate::host::{
    PinnedBuf, PinnedBufferPool, PinnedBufferPoolConfig, PinnedPoolMsg, PinnedPoolStats,
};
pub use crate::kernel::dispatch::{
    DevSliceArg, GemmDispatch, GemmDispatchCtx, NvrtcDispatchCtx, NvrtcLaunchDispatch, RngDispatch,
    ScalarArg,
};
#[cfg(feature = "cusparse")]
pub use crate::kernel::dispatch::{SparseDispatch, SparseDispatchCtx, SendSparseHandle, SparseOp};
#[cfg(feature = "cublaslt")]
pub use crate::kernel::dispatch::{BlasLtDispatch, BlasLtDispatchCtx};
#[cfg(feature = "cudnn")]
pub use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};
#[cfg(feature = "cufft")]
pub use crate::kernel::dispatch::{FftDispatch, FftDispatchCtx};
#[cfg(feature = "cutensor")]
pub use crate::kernel::dispatch::{TensorDispatch, TensorDispatchCtx};
#[cfg(feature = "nccl")]
pub use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx};
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
pub use crate::dtype::{CudaDtype, SolverSupported};
#[cfg(feature = "cusolver")]
pub use crate::kernel::{
    CholeskyRequest, GesvdjBatchedRequest, GetrfBatchedRequest, HegvdRequest, LuRequest,
    LuSolveRequest, PotrfBatchedRequest, QrRequest, SolverActor, SolverDispatch, SolverMsg,
    SvdRequest, SyevdRequest, SygvdRequest, Uplo,
};
#[cfg(all(feature = "cusolver", feature = "cusolver-sp"))]
pub use crate::kernel::{SparseCholeskyRequest, SparseLuRequest, SparseQrRequest};

#[cfg(feature = "cusparse")]
pub use crate::kernel::{CsrMatrix, SparseActor, SparseMsg};

#[cfg(feature = "cutensor")]
pub use crate::dtype::TensorSupported;
#[cfg(feature = "cutensor")]
pub use crate::kernel::{
    ComputeDesc, ContractRequest, ElementwiseBinaryRequest, ElementwiseTrinaryRequest,
    OperandSpec, PermutationRequest, ReductionRequest, TensorActor, TensorMsg, TensorSpec,
};

#[cfg(feature = "cublaslt")]
pub use crate::kernel::{
    Activation, BlasLtActor, BlasLtMsg, Epilogue, HeuristicCacheRef, MatmulRequest, ScaleSet,
    BlasLtWorkspacePool,
};

#[cfg(feature = "nvrtc")]
pub use crate::kernel::{KernelArg, KernelHandle, NvrtcActor, NvrtcMsg, NvrtcOpts};

#[cfg(feature = "nccl")]
pub use crate::kernel::{
    AllGatherRequest, AllReduceRequest, AllToAllRequest, AllToAllvRequest, BroadcastRequest,
    CollectiveActor, CollectiveMsg, GroupGuard, NcclCapabilities, NcclReduceSupported,
    PreMulSumOp, RecvRequest, ReduceOp, ReduceRequest, ReduceScatterRequest, SendRequest,
};
#[cfg(feature = "nccl")]
pub use crate::multi_device::{NcclWorldActor, NcclWorldConfig, NcclWorldMsg};
pub use crate::stream::{
    ActorHints, PerActorAllocator, PooledAllocator, Priority, SingleStreamAllocator,
    StreamAllocator, WorkloadKind,
};

#[cfg(feature = "cudnn")]
pub use crate::kernel::{
    ActivationFwdRequest, ActivationKind, ActivationRequest, AttentionMask, AttentionParams,
    BatchNormRequest, ConvBwdDataRequest, ConvBwdFilterRequest, ConvDescParams, ConvForwardRequest,
    ConvFwdRequest, ConvParams, CudnnActor, CudnnMsg, DropoutFwdRequest, EpilogueKind,
    GroupNormRequest, InstanceNormRequest, LayerNormRequest, LrnFwdRequest, LrnParams,
    MultiHeadAttnBwdRequest, MultiHeadAttnFwdRequest, NormBwdRequest, NormMode, NormPhase,
    PoolBwdRequest, PoolFwdRequest, PoolMode, PoolParams, RnnBwdRequest, RnnDirection,
    RnnFwdRequest, RnnMode, RnnParams, SoftmaxFwdRequest, SoftmaxMode, SoftmaxRequest,
    TensorLayout,
};

#[cfg(feature = "cufft")]
pub use crate::kernel::{
    FftActor, FftCallbackKind, FftDirection, FftKind, FftMsg, FftPlan, FftPlanMany,
    FftRequest, PlanKey,
};

#[cfg(feature = "curand")]
pub use crate::kernel::{Distribution, FillRequest, RngActor, RngGeneratorKind, RngMsg};

/// TensorRT integration (Phase 8).
#[cfg(feature = "tensorrt")]
pub use atomr_accel_tensorrt as tensorrt;

/// CUTLASS template-instantiation crate (Phase 6).
#[cfg(feature = "cutlass")]
pub use atomr_accel_cutlass as cutlass;

// Phase 9 — observability backends.
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
