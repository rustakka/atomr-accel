//! Boxed-dispatch payloads for the Phase 0.4 generic alloc/copy
//! variants of [`DeviceMsg`] and [`ContextMsg`].
//!
//! The previous design carried one enum variant per dtype (`AllocateF32`,
//! `AllocateF64`, `AllocateI8`, …) which scaled poorly: each new dtype
//! doubled the alloc surface and tripled the copy surface. This module
//! replaces that fan-out with three boxed-trait-object dispatchers:
//!
//! - [`AllocDispatch`] — typed buffer allocation
//! - [`CopyToHostDispatch`] — D2H async copy
//! - [`CopyFromHostDispatch`] — H2D async copy
//!
//! Concrete request structs (`AllocReq<T>`, `CopyToHostReq<T>`,
//! `CopyFromHostReq<T>`) implement the matching trait and ride inside a
//! single `Box<dyn …>`. The DeviceActor's `handle` arm forwards them to
//! the ContextActor verbatim — the typed `T: CudaDtype` parameter is
//! preserved through the box, so `GpuRef<T>` keeps its static dtype on
//! the receiving side.
//!
//! See `device::device_actor` for the legacy `#[deprecated]` enum
//! variants kept for back-compat, and `device::context_actor` for the
//! receiving side that calls `.run(...)` on the boxed dispatcher.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::dtype::{CudaDtype, DType};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

use super::alloc_msg::HostBuf;
use super::state::DeviceState;

/// Trait object the DeviceActor stashes inside a single
/// [`crate::device::DeviceMsg::Alloc`] variant. The actual `T` is
/// erased at the box boundary; the receiving ContextActor calls
/// [`AllocDispatch::run`] which downcasts back into the concrete
/// `AllocReq<T>` and performs a typed `stream.alloc_zeros::<T>(len)`.
pub trait AllocDispatch: Send + 'static {
    /// Concrete dtype carried by this dispatcher.
    fn dtype(&self) -> DType;

    /// Element count being allocated.
    fn len(&self) -> usize;

    /// Execute the allocation against the given context's primary
    /// stream and reply on the embedded `oneshot` channel.
    ///
    /// `mock_mode == true` makes this match the legacy mock-mode
    /// behaviour of [`super::context_actor::ContextActor::alloc`] —
    /// it always replies with `GpuError::Unrecoverable("alloc not
    /// supported in mock mode")` so existing tests are unaffected.
    fn run(
        self: Box<Self>,
        stream: Option<&Arc<CudaStream>>,
        state: &Arc<DeviceState>,
        mock_mode: bool,
    );
}

/// Concrete typed allocation request. Held inside a
/// `Box<dyn AllocDispatch>` while in flight.
pub struct AllocReq<T: CudaDtype> {
    pub len: usize,
    pub reply: oneshot::Sender<Result<GpuRef<T>, GpuError>>,
}

impl<T: CudaDtype> AllocDispatch for AllocReq<T> {
    fn dtype(&self) -> DType {
        T::KIND
    }

    fn len(&self) -> usize {
        self.len
    }

    fn run(
        self: Box<Self>,
        stream: Option<&Arc<CudaStream>>,
        state: &Arc<DeviceState>,
        mock_mode: bool,
    ) {
        let AllocReq { len, reply } = *self;
        if mock_mode {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "alloc not supported in mock mode".into(),
            )));
            return;
        }
        let Some(stream) = stream else {
            let _ = reply.send(Err(GpuError::GpuRefStale("context not ready")));
            return;
        };
        match stream.alloc_zeros::<T>(len) {
            Ok(slice) => {
                let _ = reply.send(Ok(GpuRef::<T>::new(Arc::new(slice), state)));
            }
            Err(e) => {
                let _ = reply.send(Err(GpuError::OutOfMemory(format!("alloc {len}: {e}"))));
            }
        }
    }
}

/// Trait object behind [`crate::device::DeviceMsg::CopyToHost`].
/// Carries the typed source `GpuRef<T>` plus host destination buffer.
pub trait CopyToHostDispatch: Send + 'static {
    fn dtype(&self) -> DType;
    fn run(self: Box<Self>, stream: Arc<CudaStream>, completion: Arc<dyn CompletionStrategy>);
}

pub struct CopyToHostReq<T: CudaDtype> {
    pub src: GpuRef<T>,
    pub dst: HostBuf<T>,
    pub reply: oneshot::Sender<Result<HostBuf<T>, GpuError>>,
}

impl<T: CudaDtype> CopyToHostDispatch for CopyToHostReq<T> {
    fn dtype(&self) -> DType {
        T::KIND
    }

    fn run(self: Box<Self>, stream: Arc<CudaStream>, completion: Arc<dyn CompletionStrategy>) {
        let CopyToHostReq { src, dst, reply } = *self;
        super::context_actor::run_copy_to_host(src, dst, stream, completion, reply);
    }
}

/// Trait object behind [`crate::device::DeviceMsg::CopyFromHost`].
pub trait CopyFromHostDispatch: Send + 'static {
    fn dtype(&self) -> DType;
    fn run(self: Box<Self>, stream: Arc<CudaStream>, completion: Arc<dyn CompletionStrategy>);
}

pub struct CopyFromHostReq<T: CudaDtype> {
    pub src: HostBuf<T>,
    pub dst: GpuRef<T>,
    pub reply: oneshot::Sender<Result<HostBuf<T>, GpuError>>,
}

impl<T: CudaDtype> CopyFromHostDispatch for CopyFromHostReq<T> {
    fn dtype(&self) -> DType {
        T::KIND
    }

    fn run(self: Box<Self>, stream: Arc<CudaStream>, completion: Arc<dyn CompletionStrategy>) {
        let CopyFromHostReq { src, dst, reply } = *self;
        super::context_actor::run_copy_from_host(src, dst, stream, completion, reply);
    }
}
