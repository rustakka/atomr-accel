//! Boxed-dispatch traits for the per-actor message enums.
//!
//! Phase 0.3 introduces a hybrid "typed public API, type-erased mailbox"
//! pattern: each actor message variant carries a `Box<dyn FooDispatch>`
//! whose concrete `FooRequest<T>` is parameterized over the operand
//! dtype. The actor's mailbox stays mono-type
//! (`actor::Msg = BlasMsg`, etc.), but the dtype dimension travels
//! through the box.
//!
//! Each dispatch trait exposes:
//! - `dtype_name()` — stable string for tracing.
//! - `op_name()` — string identifier for the op (e.g. `"gemm"`,
//!   `"axpy"`, …). Used in error annotations and tests.
//! - `dispatch(self: Box<Self>, ctx: &BlasDispatchCtx)` — runs the op,
//!   replying via the request's internal `oneshot::Sender`.
//!
//! Only cuBLAS dispatchers (Phase 1 cuBLAS slice) live here in
//! concrete form. Stubs for cuBLASLt / cuDNN / cuFFT / cuRAND /
//! cuSOLVER / cuSPARSE / cuTENSOR / NCCL are intentionally **not**
//! defined here — those are owned by their respective parallel agents
//! (Phase 1 sub-tasks).

use std::sync::Arc;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;

/// Bundle of resources every cuBLAS dispatcher needs to run an op.
///
/// `cublas` is wrapped in an `Arc` so a dispatch closure can `clone()`
/// it into the kernel envelope's keep-alive without taking ownership.
/// `state` is held by the dispatcher only for parity with the existing
/// `BlasInner::Real` field — at present the dispatchers don't read
/// generation off the state directly (each `GpuRef::access` does that
/// individually), so it's `#[allow(dead_code)]` at the field level.
pub struct BlasDispatchCtx<'a> {
    pub cublas: &'a Arc<cudarc::cublas::CudaBlas>,
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub state: &'a Arc<DeviceState>,
}

/// Erased `GemmRequest<T>`. Implementors live in
/// [`crate::kernel::blas::gemm`]. `dispatch` consumes the box and
/// drives the op through `kernel::envelope::run_kernel`.
pub trait GemmDispatch: Send + 'static {
    fn dtype_name(&self) -> &'static str;
    fn op_name(&self) -> &'static str;
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>);
}

/// Erased `GemmStridedBatchedRequest<T>`.
pub trait GemmStridedBatchedDispatch: Send + 'static {
    fn dtype_name(&self) -> &'static str;
    fn op_name(&self) -> &'static str;
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>);
}

/// Erased L1 ops: axpy, dot, nrm2, scal, asum, iamax, iamin, copy,
/// swap, rot.
pub trait BlasL1Dispatch: Send + 'static {
    fn dtype_name(&self) -> &'static str;
    fn op_name(&self) -> &'static str;
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>);
}

/// Erased L2 ops: gemv, ger.
pub trait BlasL2Dispatch: Send + 'static {
    fn dtype_name(&self) -> &'static str;
    fn op_name(&self) -> &'static str;
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>);
}

/// Erased L3 ops other than gemm: geam, syrk, trsm.
pub trait BlasL3Dispatch: Send + 'static {
    fn dtype_name(&self) -> &'static str;
    fn op_name(&self) -> &'static str;
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>);
}
