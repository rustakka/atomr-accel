//! Per-actor `*Dispatch` traits for boxed dtype-generic dispatch.
//!
//! Phase 0.2 introduces the **Option C** hybrid: each actor's
//! `Msg` type stays a single non-generic enum (so `Actor::Msg` and
//! the `KernelChildren` typed actor refs continue to work), while one
//! variant per actor — the canonical "Exec" variant — carries a
//! `Box<dyn *Dispatch>`. The dispatch trait owns the typed
//! [`crate::gpu_ref::GpuRef<T>`] payload and erases the dtype at the
//! enum boundary.
//!
//! Migration order across actors lands one PR per actor; this file
//! grows by one trait per PR. The cuFFT slice ([`FftDispatch`]) is
//! the first wave — sibling trait stubs (cuBLAS `GemmDispatch`,
//! cuRAND `RngDispatch`, cuDNN `*Dispatch`, …) land with their
//! respective actor PRs and **must not** be added here speculatively.

#![cfg(feature = "cufft")]

use std::sync::Arc;

use crate::completion::CompletionStrategy;
use crate::dtype::DType;
use crate::kernel::fft::PlanKey;

/// Per-execution context bundle handed to every [`FftDispatch::dispatch`]
/// invocation. The actor packs its current stream + completion strategy
/// + plan handle (already resolved against the LRU cache) into this
/// bundle so individual dispatch impls can stay lean.
pub struct FftDispatchCtx<'a> {
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    /// Already-resolved cuFFT plan (`Arc<CudaFft>`). Type-erased to
    /// `dyn std::any::Any` to keep this trait import-light; the actor
    /// downcasts inside [`crate::kernel::fft`].
    pub plan: Arc<dyn std::any::Any + Send + Sync>,
}

/// Dispatch trait for typed cuFFT requests (`FftRequest<T>` for
/// `T: FftSupported`). `Send + 'static` because requests cross actor
/// mailboxes.
pub trait FftDispatch: Send + 'static {
    /// Reflect the dtype of the underlying request. Used by the actor
    /// to populate the [`crate::kernel::fft::PlanKey`] before the
    /// dispatch closure runs.
    fn dtype_kind(&self) -> DType;

    /// Plan-cache key the actor uses to resolve a `CudaFft` handle
    /// before invoking [`FftDispatch::dispatch`]. The resolved plan
    /// lands in [`FftDispatchCtx::plan`].
    fn plan_key(&self) -> PlanKey;

    /// Run the request: validate the typed `GpuRef`s, enqueue the
    /// kernel via [`crate::kernel::envelope::run_kernel`], and reply
    /// over the embedded oneshot.
    fn dispatch(self: Box<Self>, ctx: &FftDispatchCtx<'_>);
}
