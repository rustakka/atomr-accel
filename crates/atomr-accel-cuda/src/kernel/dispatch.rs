//! Per-library dispatch traits — the dynamic "boxed payload" each
//! kernel actor accepts via its `*Msg::Fill(...)` / `*Msg::Apply(...)`
//! variant. The trait abstracts over the dtype-specific generic so
//! the actor's mailbox stays a single `Send + 'static` enum.
//!
//! Today only [`RngDispatch`] is wired (Phase 1 cuRAND); cuBLAS,
//! cuDNN, etc. continue to use their concrete request structs.

use std::sync::Arc;

use crate::completion::CompletionStrategy;
use crate::error::GpuError;

/// Erased payload accepted by `RngActor` via `RngMsg::Fill`. The
/// concrete implementor (typically [`crate::kernel::FillRequest<T>`])
/// owns the typed [`crate::gpu_ref::GpuRef<T>`] destination plus the
/// distribution parameters.
///
/// `fill` runs *on the actor's pinned thread* — the actor has already
/// taken the cuRAND generator lock and is providing it via the
/// `generator` argument. The implementor is responsible for:
///
/// * re-locking / accessing its `GpuRef<T>`,
/// * calling `cudarc::curand::sys::curandGenerate*` (or the higher
///   safe wrapper) on `generator`,
/// * handing the keep-alive to [`crate::kernel::envelope::run_kernel`]
///   so the buffer outlives the kernel,
/// * sending the reply.
///
/// Returning `Err` from this fn means a *pre-launch* validation
/// failure (stale `GpuRef`, multi-writer alias, unsupported dtype,
/// etc.). A successful return means the kernel was enqueued and the
/// reply will arrive via the completion future.
pub trait RngDispatch: Send + 'static {
    fn fill(
        self: Box<Self>,
        generator: cudarc::curand::sys::curandGenerator_t,
        stream: &Arc<cudarc::driver::CudaStream>,
        completion: &Arc<dyn CompletionStrategy>,
    ) -> Result<(), GpuError>;
}
