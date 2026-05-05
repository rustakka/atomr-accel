//! Per-actor dispatch traits — typed public requests boxed-erased into
//! a single mailbox message variant (Phase 0.3).
//!
//! Pattern (from the plan):
//!
//! ```text
//! pub enum BlasLtMsg {
//!     Matmul(Box<dyn BlasLtDispatch>),
//!     // … legacy variants kept for back-compat …
//! }
//!
//! pub trait BlasLtDispatch: Send + 'static {
//!     fn dtype_kind(&self) -> DTypeKind;
//!     fn dispatch(self: Box<Self>, ctx: &BlasLtDispatchCtx<'_>);
//! }
//!
//! pub struct MatmulRequest<T: GemmSupported> { … }
//! impl<T: GemmSupported> BlasLtDispatch for MatmulRequest<T> { … }
//! ```
//!
//! Each library actor lands its own marker trait + `*DispatchCtx`
//! bundle in this module as it migrates. Other agents land their
//! traits (e.g. `BlasDispatch`, `CudnnDispatch`) here in parallel; see
//! the per-phase work-item file for ownership.

#[cfg(feature = "cublaslt")]
use crate::dtype::DTypeKind;

#[cfg(feature = "cublaslt")]
mod blaslt_dispatch_internal {
    //! Hidden helper module so the cuBLASLt context type can name
    //! cudarc/internal types without exposing them through the public
    //! `BlasLtDispatch` trait surface.
    use std::sync::Arc;

    use cudarc::cublaslt::CudaBlasLT;
    use tokio::sync::oneshot;

    use crate::completion::CompletionStrategy;
    use crate::error::GpuError;
    use crate::kernel::blas_lt::heuristic::HeuristicCacheRef;
    use crate::kernel::blas_lt::workspace::WorkspacePool;

    /// Per-call context handed to a `BlasLtDispatch::dispatch` impl.
    /// Holds shared, mutable-by-design state (workspace pool,
    /// heuristic cache) plus the runtime handles each typed
    /// `MatmulRequest<T>` needs to enqueue a kernel.
    pub struct BlasLtDispatchCtx<'a> {
        pub blas_lt: Arc<CudaBlasLT>,
        pub stream: &'a Arc<cudarc::driver::CudaStream>,
        pub completion: &'a Arc<dyn CompletionStrategy>,
        pub workspace: &'a WorkspacePool,
        pub heuristic: HeuristicCacheRef,
        pub sm_arch: u32,
    }

    /// Convenience helper: short-circuit a typed request whose target
    /// dtype isn't supported on the running cuBLASLt build by sending
    /// a typed error on the reply channel.
    pub fn reply_unsupported(
        reply: oneshot::Sender<Result<(), GpuError>>,
        dtype_name: &'static str,
    ) {
        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
            "BlasLtDispatch: dtype {dtype_name} unsupported in this build"
        ))));
    }
}

#[cfg(feature = "cublaslt")]
pub use blaslt_dispatch_internal::{reply_unsupported, BlasLtDispatchCtx};

/// Boxed-dispatch trait the cuBLASLt actor uses to call into a typed
/// `MatmulRequest<T>` after type-erasing it through the mailbox.
///
/// Implementors live in [`crate::kernel::blas_lt::matmul`].
#[cfg(feature = "cublaslt")]
pub trait BlasLtDispatch: Send + 'static {
    /// Stable cache-key tag for the input dtype.
    fn dtype_kind(&self) -> DTypeKind;

    /// Run the typed matmul body. Consumes `self` so the request's
    /// owned `GpuRef`s and reply channel can flow into the kernel
    /// envelope without an extra clone.
    fn dispatch(self: Box<Self>, ctx: &BlasLtDispatchCtx<'_>);
}
