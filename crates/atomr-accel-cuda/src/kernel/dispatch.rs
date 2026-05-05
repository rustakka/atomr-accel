//! Per-actor dispatch traits — type-erased entry points so a single
//! `*Msg` enum can carry typed [`crate::dtype::CudaDtype`] payloads
//! through the actor mailbox without exploding the variant count
//! (Phase 0.2 of the CUDA-coverage roadmap).
//!
//! # Pattern
//!
//! Each library actor exposes
//!
//! ```ignore
//! pub enum FooMsg {
//!     Op(Box<dyn FooDispatch>),
//!     // legacy #[deprecated] variants ...
//! }
//! ```
//!
//! plus a `*DispatchCtx` bundle holding the actor's stream, completion
//! strategy, library handle, and whatever caches the op needs. The
//! per-op typed request struct (`ConvFwdRequest<T>`, etc.) lives next
//! to the actor and `impl`s the dispatch trait by walking the
//! envelope path with the right typed FFI calls.
//!
//! Adding a new dtype: write the `impl FooDispatch for FooRequest<T>`
//! body once, parameterised by `T: CudnnSupported` (or whichever
//! capability bound applies).
//!
//! This file currently exposes only the cuDNN trait; the NCCL,
//! cuTENSOR, and other Phase 2 traits live in their own dispatch
//! modules to keep parallel-agent diffs isolated.

#![allow(dead_code)]

#[cfg(feature = "cudnn")]
pub use cudnn_dispatch::{CudnnDispatch, CudnnDispatchCtx};

#[cfg(feature = "cudnn")]
mod cudnn_dispatch {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::completion::CompletionStrategy;

    /// Context handed to a [`CudnnDispatch::dispatch`] call. Holds the
    /// shared cuDNN handle, the actor's stream and completion
    /// strategy, and (Mutex-wrapped) references to the actor's plan
    /// cache and persistent workspace.
    pub struct CudnnDispatchCtx<'a> {
        pub handle: Arc<cudarc::cudnn::Cudnn>,
        pub stream: Arc<cudarc::driver::CudaStream>,
        pub completion: Arc<dyn CompletionStrategy>,
        pub plan_cache: &'a Mutex<crate::kernel::cudnn::graph::PlanCache>,
        pub workspace: &'a Mutex<Option<cudarc::driver::CudaSlice<u8>>>,
    }

    /// Dispatch trait for typed cuDNN ops. Each per-op request
    /// (`ConvFwdRequest<T>`, `LayerNormRequest<T>`, …) `impl`s this
    /// directly — the `T` parameter is fully bound on the impl side.
    ///
    /// `dtype_name` is for telemetry / error messages and returns the
    /// dtype's static [`crate::dtype::CudaDtype::NAME`].
    /// `op_kind` returns a static tag identifying the op family
    /// (`"conv_fwd"`, `"layernorm"`, …) for the same purpose plus
    /// plan-cache keying.
    pub trait CudnnDispatch: Send + 'static {
        fn dtype_name(&self) -> &'static str;
        fn op_kind(&self) -> &'static str;
        fn dispatch(self: Box<Self>, ctx: &CudnnDispatchCtx<'_>);
    }
}
