//! Kernel-actor wrappers around CUDA library handles (§3.2).
//!
//! Each library actor follows a uniform shape:
//!
//! * a `Real { handle, stream, completion, state, … }` variant holding
//!   the cudarc handle plus the per-actor caches it needs;
//! * a `Mock` variant for GPU-free tests;
//! * a `props(stream, allocator, completion, state)` constructor that
//!   panics with `"ContextPoisoned: <Lib>::new failed: …"` if the
//!   handle can't be created, so the supervisor restarts;
//! * a `mock_props()` constructor that replies `Unrecoverable("…not
//!   supported in mock mode")` to every variant.
//!
//! The shared kernel-enqueue body lives in
//! [`envelope::run_kernel`] — every library actor calls it instead of
//! reimplementing the validate / enqueue / spawn-completion-await /
//! reply / drop-keep-alive sequence.
//!
//! F2 ships: `BlasActor`, `CudnnActor`, `FftActor`, `RngActor`.
//! F3 adds: `SolverActor`, `BlasLtActor`, `NvrtcActor`.
//! F4 adds: `CollectiveActor` (NCCL).

pub mod dispatch;
pub mod envelope;
pub mod record;

pub use dispatch::{
    CollectiveDispatch, CollectiveDispatchCtx, CudnnDispatch, CudnnDispatchCtx, DevSliceArg,
    GemmDispatch, GemmDispatchCtx, NvrtcDispatchCtx, NvrtcLaunchDispatch, RngDispatch, ScalarArg,
    SolverDispatch, SolverDispatchCtx, SparseDispatch, SparseDispatchCtx, TensorDispatch,
    TensorDispatchCtx,
};
#[cfg(feature = "cublaslt")]
pub use dispatch::{BlasLtDispatch, BlasLtDispatchCtx};
#[cfg(feature = "cufft")]
pub use dispatch::{FftDispatch, FftDispatchCtx};

mod blas;

pub use blas::{BlasActor, BlasMsg};

#[cfg(feature = "cudnn")]
mod cudnn_actor;
#[cfg(feature = "cudnn")]
pub use cudnn_actor::{
    ActivationKind, ActivationRequest, ConvForwardRequest, ConvParams, CudnnActor, CudnnMsg,
    SoftmaxRequest,
};

#[cfg(feature = "cufft")]
pub mod fft;
#[cfg(feature = "cufft")]
pub use fft::{
    FftActor, FftCallbackKind, FftDirection, FftKind, FftMsg, FftPlan, FftPlanMany, FftRequest,
    PlanKey,
};

#[cfg(feature = "curand")]
pub mod rng;
#[cfg(feature = "curand")]
pub use rng::{Distribution, FillRequest, RngActor, RngGeneratorKind, RngMsg};

#[cfg(feature = "cusolver")]
mod solver;
#[cfg(feature = "cusolver")]
pub use solver::{SolverActor, SolverMsg, Uplo};

#[cfg(feature = "cublaslt")]
pub mod blas_lt;
#[cfg(feature = "cublaslt")]
pub use blas_lt::{
    Activation, BlasLtActor, BlasLtMsg, Epilogue, HeuristicCacheRef, MatmulRequest, ScaleSet,
    WorkspacePool,
};

#[cfg(feature = "nvrtc")]
mod nvrtc;
#[cfg(feature = "nvrtc")]
pub use nvrtc::{KernelArg, KernelHandle, NvrtcActor, NvrtcMsg, NvrtcOpts};

#[cfg(feature = "nccl")]
mod collective;
#[cfg(feature = "nccl")]
pub use collective::{CollectiveActor, CollectiveMsg, ReduceOp};

#[cfg(feature = "cusparse")]
mod sparse;
#[cfg(feature = "cusparse")]
pub use sparse::{CsrMatrix, SparseActor, SparseMsg};

#[cfg(feature = "cutensor")]
mod tensor;
#[cfg(feature = "cutensor")]
pub use tensor::{TensorActor, TensorMsg, TensorSpec};
