//! Per-actor dispatch traits used by the dtype-generic message
//! pipeline (Phase 0.2 — Option C: typed public API, boxed-dispatch
//! internals).
//!
//! This file is the single source of truth for the boxed dispatcher
//! types each library actor consumes. Phase 2 NCCL agents own the
//! [`CollectiveDispatch`] trait and its [`CollectiveDispatchCtx`].
//! Other Dispatch traits (`GemmDispatch`, `CudnnDispatch`, etc.)
//! are owned by their respective Phase 1/2 agents and may be added
//! here later — leave their slots open.

#[cfg(feature = "nccl")]
use std::sync::Arc;

#[cfg(feature = "nccl")]
use crate::completion::CompletionStrategy;
#[cfg(feature = "nccl")]
use crate::device::DeviceState;

/// Static-tag describing the dtype carried by a boxed dispatch
/// payload. Mirrors the `DType` enum that lives in
/// `atomr-accel/src/dtype.rs` once Phase 0 lands; until then, this
/// local enum is the identifier callers can match on for diagnostics
/// and tracing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchDType {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
    F8E4m3,
    F8E5m2,
}

/// Boxed dispatch trait for collective (NCCL) operations. Every
/// dtype-generic NCCL request struct (`AllReduceRequest<T>`,
/// `AllGatherRequest<T>`, …) implements this trait so the
/// `CollectiveActor` can carry one boxed dispatcher per message
/// without monomorphising the actor itself over `T`.
#[cfg(feature = "nccl")]
pub trait CollectiveDispatch: Send + 'static {
    /// Reports the dtype of the payload. Used for tracing /
    /// diagnostics; not load-bearing for correctness.
    fn dtype_kind(&self) -> DispatchDType;

    /// Best-effort device id of the primary tensor on the request,
    /// for cross-device validation in `CollectiveActor`. `None`
    /// indicates the request carries no device-bound buffer (rare —
    /// p2p Recv with a target buffer always has one).
    fn device_id(&self) -> Option<u32>;

    /// Issue the NCCL call against the comm in `ctx`. Implementors
    /// are responsible for sending the reply on their internal
    /// `oneshot::Sender<Result<(), GpuError>>` exactly once.
    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>);
}

/// Bundle of the per-rank context the boxed dispatcher needs. Carries
/// an `Arc<cudarc::nccl::Comm>` (NB: cudarc's `Comm` is `!Sync` and
/// not internally `Arc`-able — `CollectiveActor` wraps it in a
/// `SendComm` newtype before exposing this borrow).
#[cfg(feature = "nccl")]
pub struct CollectiveDispatchCtx<'a> {
    pub comm: &'a cudarc::nccl::Comm,
    pub state: &'a Arc<DeviceState>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
}
