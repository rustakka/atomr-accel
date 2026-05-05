//! Per-actor dispatch traits and context bundles.
//!
//! Each library actor that handles dtype-generic requests routes a
//! `Box<dyn *Dispatch>` through this module so the actor mailbox can
//! stay a single `Send + 'static` enum (`TensorMsg`, `BlasMsg`, …)
//! while the underlying request payloads remain typed (`Request<T>`).
//!
//! cuTENSOR (Phase 2) is the first call site: see
//! [`TensorDispatch`](TensorDispatch) and [`TensorDispatchCtx`].
//!
//! Other actors' Dispatch traits live in this same module but are
//! defined alongside their owning kernel actor; cuDNN, NCCL, cuBLAS
//! etc. each add a `*Dispatch` trait and `*DispatchCtx` here as they
//! land. Each addition is independent of the others — there is no
//! cross-trait coupling.

#[cfg(feature = "cutensor")]
pub use self::tensor::{TensorDispatch, TensorDispatchCtx, WorkspacePool};

#[cfg(feature = "cutensor")]
mod tensor {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::completion::CompletionStrategy;
    use crate::error::GpuError;
    use crate::kernel::tensor::plan_cache::PlanCache;
    use crate::kernel::tensor::SendHandle;

    /// Workspace pool keyed by power-of-2 byte sizes — one allocator
    /// per bucket. Mirrors the cuBLASLt `WorkspacePool` pattern: an
    /// op asks for `n` bytes, the pool rounds up to the next power of
    /// two, and a single `CudaSlice<u8>` per bucket is reused across
    /// invocations.
    pub struct WorkspacePool {
        stream: Arc<cudarc::driver::CudaStream>,
        buckets: Mutex<Vec<Bucket>>,
    }

    struct Bucket {
        size: usize,
        slice: cudarc::driver::CudaSlice<u8>,
    }

    impl WorkspacePool {
        pub fn new(stream: Arc<cudarc::driver::CudaStream>) -> Self {
            Self {
                stream,
                buckets: Mutex::new(Vec::new()),
            }
        }

        /// Round `n` up to the next power of two, allocating the
        /// bucket lazily if it doesn't already exist. Returns the
        /// rounded byte count so the caller can pass it to cuTENSOR's
        /// `workspace_size` argument.
        pub fn ensure(&self, n: usize) -> Result<usize, GpuError> {
            if n == 0 {
                return Ok(0);
            }
            let bucket_size = n.next_power_of_two();
            let mut g = self.buckets.lock();
            if g.iter().any(|b| b.size == bucket_size) {
                return Ok(bucket_size);
            }
            let slice = self
                .stream
                .alloc_zeros::<u8>(bucket_size)
                .map_err(|e| GpuError::OutOfMemory(format!("cutensor workspace: {e}")))?;
            g.push(Bucket {
                size: bucket_size,
                slice,
            });
            Ok(bucket_size)
        }

        /// Run `f` with a mutable reference to the matching bucket's
        /// slice. The closure may pass the slice's device pointer to
        /// cuTENSOR — the lock keeps any concurrent `ensure` calls
        /// from racing against the active op. Returns `None` if the
        /// requested size has not been ensured yet.
        pub fn with_bucket<F, R>(&self, n: usize, f: F) -> Option<R>
        where
            F: FnOnce(&mut cudarc::driver::CudaSlice<u8>) -> R,
        {
            if n == 0 {
                return None;
            }
            let bucket_size = n.next_power_of_two();
            let mut g = self.buckets.lock();
            let b = g.iter_mut().find(|b| b.size == bucket_size)?;
            Some(f(&mut b.slice))
        }
    }

    /// Resources every cuTENSOR dispatcher (`Box<dyn TensorDispatch>`)
    /// needs at execution time. Constructed once by `TensorActor::Real`
    /// and passed by reference into `TensorDispatch::dispatch`.
    pub struct TensorDispatchCtx {
        pub handle: Arc<Mutex<SendHandle>>,
        pub stream: Arc<cudarc::driver::CudaStream>,
        pub completion: Arc<dyn CompletionStrategy>,
        pub plan_cache: Arc<PlanCache>,
        pub workspace: Arc<WorkspacePool>,
    }

    /// Type-erased cuTENSOR request. Implementors carry their typed
    /// `Request<T>` payload internally; the dispatch site recovers the
    /// type by virtual call.
    pub trait TensorDispatch: Send + 'static {
        /// Stable string used in panic / log / cache messages.
        fn op_tag(&self) -> &'static str;
        /// Stable string for the scalar dtype.
        fn dtype_tag(&self) -> &'static str;
        /// Hand the typed request to its actor-side handler.
        fn dispatch(self: Box<Self>, ctx: &TensorDispatchCtx);
        /// Reply with `GpuError::Unrecoverable("TensorActor in mock
        /// mode")` when the actor is `Mock`.
        fn fail_mock(self: Box<Self>);
    }
}
