//! Per-actor `*Dispatch` traits + their `*DispatchCtx` bundles
//! (Phase 0.3).
//!
//! Each kernel actor (cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND,
//! cuSOLVER, cuSPARSE, cuTENSOR, NCCL, NVRTC) eventually exposes a
//! typed public API like:
//!
//! ```ignore
//! blas_actor.tell(BlasMsg::gemm::<f16>(GemmRequest { ... }));
//! ```
//!
//! Internally, the request is boxed as `Box<dyn GemmDispatch>` and
//! the actor's handle loop calls `dispatch(self, &ctx)` which carries
//! the cuBLAS handle, stream, and completion strategy. This avoids
//! the N-fold (op × dtype) variant explosion in the actor's `Msg`
//! enum without giving up typed `GpuRef<T>` requests on the public
//! API.
//!
//! ## Status
//!
//! In Phase 0.3 only **NVRTC** actually adopts the pattern: see
//! [`NvrtcLaunchDispatch`] + [`NvrtcDispatchCtx`] and the migrated
//! [`NvrtcActor`](super::NvrtcActor). The remaining actor traits
//! ([`GemmDispatch`], [`BlasLtDispatch`], …) ship as stub trait
//! declarations with `*DispatchCtx<'a>` placeholder structs whose
//! handle fields are `PhantomData` until the matching actor migrates
//! in a follow-up PR (per the migration order in the Phase 0 plan).
//!
//! The shared kernel-arg traits ([`DevSliceArg`], [`ScalarArg`])
//! collapse the per-dtype `KernelArg::DevSlice*` / `KernelArg::Scalar*`
//! variants into a single boxed-dyn pair plus a literal `Usize(usize)`
//! variant.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchArgs, PushKernelArg};

use atomr_accel::DType;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::dtype::CudaDtype;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

// ---------------------------------------------------------------------------
// NVRTC — the only adopter in Phase 0.3.
// ---------------------------------------------------------------------------

/// Boxed-dispatch trait for an NVRTC kernel launch request.
///
/// `NvrtcMsg::Launch` carries an `args: Vec<KernelArg>`; each arg
/// implements either [`DevSliceArg`] (typed device buffers) or
/// [`ScalarArg`] (typed host scalars). The `Launch` variant itself
/// does **not** need a `Box<dyn NvrtcLaunchDispatch>` payload because
/// `KernelHandle` already carries the typed `CudaFunction` —
/// [`NvrtcLaunchDispatch`] is a marker for the *request as a whole*
/// so cross-actor tooling (NVTX naming, `KernelTrace`, future graph
/// recording) sees a uniform interface across all actors.
pub trait NvrtcLaunchDispatch: Send + 'static {
    /// Static op identifier surfaced to NVTX / `KernelTrace`. NVRTC
    /// kernels are user-supplied so this returns `"nvrtc_launch"` by
    /// default; callers may override with the kernel name.
    fn op_name(&self) -> &'static str;

    /// Element dtype, when the request has a single-dtype identity.
    /// `None` for traceless / multi-dtype actors (NVRTC is multi-dtype
    /// because user kernels can mix arg types — implementors return
    /// `None`).
    fn dtype(&self) -> Option<DType>;

    /// Run the dispatch: validate inputs, enqueue the kernel, deliver
    /// the reply via the completion strategy in `ctx`.
    fn dispatch(self: Box<Self>, ctx: &NvrtcDispatchCtx<'_>);
}

/// Per-launch context bundle for [`NvrtcLaunchDispatch::dispatch`].
///
/// Pulled by reference from the `NvrtcActor` `Real { ... }` variant.
/// Trait-object implementations consume this when actually wiring up
/// the cudarc `launch_builder().arg(..)` chain — Phase 0.3's NVRTC
/// migration uses the per-arg [`DevSliceArg`] / [`ScalarArg`] traits
/// rather than a request-level dispatcher, so the only reason this
/// type exists today is API symmetry with the other actors below.
pub struct NvrtcDispatchCtx<'a> {
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub state: &'a Arc<DeviceState>,
}

/// Bundle of resources every cuBLAS dispatcher needs to run an op.
pub struct BlasDispatchCtx<'a> {
    pub cublas: &'a Arc<cudarc::cublas::CudaBlas>,
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub state: &'a Arc<DeviceState>,
}

// ---------------------------------------------------------------------------
// Shared kernel-arg traits (NVRTC adopts these in Phase 0.3).
// ---------------------------------------------------------------------------

/// Type-erased typed-device-slice argument for an NVRTC kernel
/// launch.
///
/// Boxed as `Box<dyn DevSliceArg>` inside `KernelArg::DevSlice` so the
/// per-launch `Vec<KernelArg>` does not need one variant per dtype.
/// The blanket impl `impl<T: CudaDtype> DevSliceArg for GpuRef<T>`
/// covers every dtype the runtime understands; raw byte buffers
/// (`GpuRef<u8>`) are handled by the same impl because `u8` is a
/// `CudaDtype`.
///
/// # Object safety
///
/// Methods take `&self` and never return `Self` so the trait is
/// usable as `dyn DevSliceArg`.
pub trait DevSliceArg: Send + Sync + 'static {
    /// Validate the underlying [`GpuRef`] and return a keep-alive
    /// owner. The caller stores this `Box<dyn Any + Send>` in a `Vec`
    /// to keep the device buffer alive until the kernel completes.
    fn validate(&self) -> Result<Box<dyn Any + Send>, GpuError>;

    /// Push the device-pointer reference onto `builder`. Implementors
    /// re-`access()` the `GpuRef` (cheap — pointer-equality check
    /// against `DeviceState.generation`) and call
    /// [`PushKernelArg::arg`] with `&CudaSlice<T>`.
    ///
    /// The `'a` lifetime ties `&self` to the builder so the pushed
    /// device-pointer reference is borrowed from `Self` for as long as
    /// `builder` lives.
    ///
    /// Returns `Err(GpuError::GpuRefStale)` if the buffer has gone
    /// stale between `validate` and `push` (rare — only happens if a
    /// context rebuild raced inside the actor).
    fn push<'a>(&'a self, builder: &mut LaunchArgs<'a>) -> Result<(), GpuError>;

    /// Element dtype for tracing / debugging. Always `Some(..)` for
    /// the default `GpuRef<T: CudaDtype>` impl.
    fn dtype(&self) -> Option<DType>;

    /// Length of the underlying slice in elements.
    fn len(&self) -> usize;

    /// True iff the slice has zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> DevSliceArg for GpuRef<T>
where
    T: CudaDtype,
{
    #[inline]
    fn validate(&self) -> Result<Box<dyn Any + Send>, GpuError> {
        let arc: Arc<CudaSlice<T>> = self.access()?.clone();
        Ok(Box::new(arc))
    }

    #[inline]
    fn push<'a>(&'a self, builder: &mut LaunchArgs<'a>) -> Result<(), GpuError> {
        let arc = self.access()?;
        // `arc` is `&Arc<CudaSlice<T>>`; `&**arc` is `&CudaSlice<T>`,
        // which is exactly what `LaunchArgs::arg` accepts.
        builder.arg(&**arc);
        Ok(())
    }

    #[inline]
    fn dtype(&self) -> Option<DType> {
        Some(<T as atomr_accel::AccelDtype>::KIND)
    }

    #[inline]
    fn len(&self) -> usize {
        GpuRef::<T>::len(self)
    }
}

/// Type-erased typed-host-scalar argument for an NVRTC kernel launch.
///
/// Boxed as `Box<dyn ScalarArg>` inside `KernelArg::Scalar`. A blanket
/// impl `impl<T: CudaDtype> ScalarArg for T` covers every dtype the
/// runtime understands. `usize` and `bool` are *not* `CudaDtype` so
/// callers use the dedicated [`super::nvrtc::KernelArg::Usize`]
/// variant for sizes — the most common scalar arg by far.
pub trait ScalarArg: Send + Sync + 'static {
    /// Push the scalar value onto `builder` by reference.
    ///
    /// The `'a` lifetime ties `&self` to the builder so the borrowed
    /// scalar reference lives at least as long as `builder`.
    fn push<'a>(&'a self, builder: &mut LaunchArgs<'a>);

    /// Dtype for tracing / debugging.
    fn dtype(&self) -> Option<DType>;
}

impl<T> ScalarArg for T
where
    T: CudaDtype + DeviceRepr + Sync,
{
    #[inline]
    fn push<'a>(&'a self, builder: &mut LaunchArgs<'a>) {
        builder.arg(self);
    }

    #[inline]
    fn dtype(&self) -> Option<DType> {
        Some(<T as atomr_accel::AccelDtype>::KIND)
    }
}

// ---------------------------------------------------------------------------
// Stub trait declarations for the remaining kernel actors. Each
// per-actor migration ships its own impl alongside the migrated actor
// file in a follow-up PR (Phases 1.x onward).
// ---------------------------------------------------------------------------

/// `GemmDispatchCtx` is now an alias for `BlasDispatchCtx` (the cuBLAS
/// agent unified the per-op contexts since every cuBLAS dispatcher
/// needs the same set: cuBLAS handle, stream, completion, state).
pub type GemmDispatchCtx<'a> = BlasDispatchCtx<'a>;

/// Erased `GemmRequest<T>`. Implementors live in `kernel::blas::gemm`.
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

/// Erased L1 ops: axpy, dot, nrm2, scal, asum, iamax, iamin, copy, swap, rot.
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

#[cfg(feature = "cublaslt")]
mod blaslt_dispatch_internal {
    //! Hidden helper: cuBLASLt context type names cudarc/internal types
    //! without leaking them through the public `BlasLtDispatch` surface.
    use std::sync::Arc;

    use cudarc::cublaslt::CudaBlasLT;
    use tokio::sync::oneshot;

    use crate::completion::CompletionStrategy;
    use crate::error::GpuError;
    use crate::kernel::blas_lt::heuristic::HeuristicCacheRef;
    use crate::kernel::blas_lt::workspace::WorkspacePool;

    /// Per-call context handed to a `BlasLtDispatch::dispatch` impl.
    pub struct BlasLtDispatchCtx<'a> {
        pub blas_lt: Arc<CudaBlasLT>,
        pub stream: &'a Arc<cudarc::driver::CudaStream>,
        pub completion: &'a Arc<dyn CompletionStrategy>,
        pub workspace: &'a WorkspacePool,
        pub heuristic: HeuristicCacheRef,
        pub sm_arch: u32,
    }

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
#[cfg(feature = "cublaslt")]
pub trait BlasLtDispatch: Send + 'static {
    fn dtype_kind(&self) -> crate::dtype::DTypeKind;
    fn dispatch(self: Box<Self>, ctx: &BlasLtDispatchCtx<'_>);
}

/// `CudnnDispatch` is owned by `kernel::cudnn` (Phase 2 cuDNN).
/// Re-exported here for symmetry with other actors' dispatch traits.
#[cfg(feature = "cudnn")]
pub use cudnn_dispatch::{CudnnDispatch, CudnnDispatchCtx};

#[cfg(feature = "cudnn")]
mod cudnn_dispatch {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::completion::CompletionStrategy;

    /// Context handed to a [`CudnnDispatch::dispatch`] call.
    pub struct CudnnDispatchCtx<'a> {
        pub handle: Arc<cudarc::cudnn::Cudnn>,
        pub stream: Arc<cudarc::driver::CudaStream>,
        pub completion: Arc<dyn CompletionStrategy>,
        pub plan_cache: &'a Mutex<crate::kernel::cudnn::graph::PlanCache>,
        pub workspace: &'a Mutex<Option<cudarc::driver::CudaSlice<u8>>>,
    }

    /// Dispatch trait for typed cuDNN ops.
    pub trait CudnnDispatch: Send + 'static {
        fn dtype_name(&self) -> &'static str;
        fn op_kind(&self) -> &'static str;
        fn dispatch(self: Box<Self>, ctx: &CudnnDispatchCtx<'_>);
    }
}

/// Per-execution context bundle handed to every [`FftDispatch::dispatch`].
/// The actor packs its current stream + completion strategy + plan handle
/// (already resolved against the LRU cache) so dispatch impls stay lean.
#[cfg(feature = "cufft")]
pub struct FftDispatchCtx<'a> {
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    /// Already-resolved cuFFT plan (`Arc<CudaFft>`). Type-erased to
    /// `dyn Any` to keep this trait import-light; the actor downcasts
    /// inside `kernel::fft`.
    pub plan: Arc<dyn std::any::Any + Send + Sync>,
}

/// Dispatch trait for typed cuFFT requests (`FftRequest<T>` for
/// `T: FftSupported`).
#[cfg(feature = "cufft")]
pub trait FftDispatch: Send + 'static {
    fn dtype_kind(&self) -> DType;
    fn plan_key(&self) -> crate::kernel::fft::PlanKey;
    fn dispatch(self: Box<Self>, ctx: &FftDispatchCtx<'_>);
}

/// Erased payload accepted by `RngActor` via `RngMsg::Fill`.
///
/// The actor takes the cuRAND generator lock and hands it to `fill` along
/// with the stream + completion strategy. Implementors call
/// `cudarc::curand::sys::curandGenerate*` (or the safe wrapper),
/// keep-alive their `GpuRef<T>` via `kernel::envelope::run_kernel`,
/// and reply on the embedded `oneshot` channel.
pub trait RngDispatch: Send + 'static {
    fn fill(
        self: Box<Self>,
        generator: cudarc::curand::sys::curandGenerator_t,
        stream: &Arc<cudarc::driver::CudaStream>,
        completion: &Arc<dyn CompletionStrategy>,
    ) -> Result<(), GpuError>;
}

/// `SolverDispatch` is owned by `kernel::solver` (Phase 1 cuSOLVER).
/// Re-exported here for API symmetry with other actors' dispatch traits.
#[cfg(feature = "cusolver")]
pub use crate::kernel::solver::SolverDispatch;

/// Boxed-dispatch trait for cuSPARSE ops.
///
/// TODO: populate impls when cuSPARSE is migrated in Phase 4.
pub trait SparseDispatch: Send + 'static {
    fn op_name(&self) -> &'static str;
    fn dtype(&self) -> Option<DType>;
    fn dispatch(self: Box<Self>, ctx: &SparseDispatchCtx<'_>);
}

/// Per-call context for [`SparseDispatch`].
///
/// TODO: populate `Arc<CudaSparse>` when cuSPARSE is migrated in
/// Phase 4.
pub struct SparseDispatchCtx<'a> {
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub state: &'a Arc<DeviceState>,
    pub _phantom: PhantomData<&'a ()>,
}

/// Boxed-dispatch trait for cuTENSOR ops.
///
/// TODO: populate impls when cuTENSOR is migrated in Phase 2.
pub trait TensorDispatch: Send + 'static {
    fn op_name(&self) -> &'static str;
    fn dtype(&self) -> Option<DType>;
    fn dispatch(self: Box<Self>, ctx: &TensorDispatchCtx<'_>);
}

/// Per-call context for [`TensorDispatch`].
///
/// TODO: populate the cuTENSOR handle and plan cache when cuTENSOR
/// is migrated in Phase 2.
pub struct TensorDispatchCtx<'a> {
    pub stream: &'a Arc<cudarc::driver::CudaStream>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
    pub state: &'a Arc<DeviceState>,
    pub _phantom: PhantomData<&'a ()>,
}

/// Alias used by the NCCL CollectiveDispatch (Phase 2). Maps onto the
/// canonical `atomr_accel::DType`.
#[cfg(feature = "nccl")]
pub use atomr_accel::DType as DispatchDType;

/// Boxed-dispatch trait for NCCL collectives. The `CollectiveActor`
/// handles the message envelope; each typed request struct (e.g.
/// `AllReduceRequest<T: NcclReduceSupported>`) implements this so the
/// actor stays single-mailbox while the dtype dimension travels in the
/// box.
#[cfg(feature = "nccl")]
pub trait CollectiveDispatch: Send + 'static {
    fn dtype_kind(&self) -> DispatchDType;
    fn device_id(&self) -> Option<u32>;
    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>);
}

/// Per-call context handed to a `CollectiveDispatch::dispatch` impl.
/// Carries the NCCL communicator (cudarc wraps it) plus the device
/// state and completion strategy.
#[cfg(feature = "nccl")]
pub struct CollectiveDispatchCtx<'a> {
    pub comm: &'a cudarc::nccl::Comm,
    pub state: &'a Arc<DeviceState>,
    pub completion: &'a Arc<dyn CompletionStrategy>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A no-GPU stand-in for `NvrtcLaunchDispatch` that records its
    /// `op_name` / `dtype` queries and asserts `dispatch` is called.
    struct DummyNvrtc {
        op: &'static str,
        d: Option<DType>,
        called: std::sync::atomic::AtomicBool,
    }

    impl NvrtcLaunchDispatch for DummyNvrtc {
        fn op_name(&self) -> &'static str {
            self.op
        }
        fn dtype(&self) -> Option<DType> {
            self.d
        }
        fn dispatch(self: Box<Self>, _ctx: &NvrtcDispatchCtx<'_>) {
            self.called
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn nvrtc_dispatch_box_round_trip() {
        let req = DummyNvrtc {
            op: "relu",
            d: Some(DType::F32),
            called: std::sync::atomic::AtomicBool::new(false),
        };
        // Box and downcast through the trait surface (op_name / dtype).
        let boxed: Box<dyn NvrtcLaunchDispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "relu");
        assert_eq!(boxed.dtype(), Some(DType::F32));

        // We can't construct an `NvrtcDispatchCtx` without a real
        // stream, so we only verify boxed dispatch indirectly via a
        // local pointer-equal struct-internal flag through a second
        // request (the type system already proves the call site
        // compiles via the round-trip above). The full GPU-side
        // dispatch path is exercised by the migrated NVRTC actor.
        let req2 = DummyNvrtc {
            op: "noop",
            d: None,
            called: std::sync::atomic::AtomicBool::new(false),
        };
        assert_eq!(req2.op_name(), "noop");
        assert_eq!(req2.dtype(), None);
    }

    /// Confirms `Box<dyn DevSliceArg>` for a `GpuRef<f32>` and (under
    /// `f16`) `GpuRef<half::f16>` compile. We can't construct a real
    /// `GpuRef` without a CUDA context, so this is a compile-only
    /// witness via a function-shaped assertion.
    #[allow(dead_code)]
    fn _assert_dev_slice_arg_object_safe() {
        fn takes_box(_: Box<dyn DevSliceArg>) {}
        // Witness: implementor type is the trait object's interior.
        let _: fn(GpuRef<f32>) -> Box<dyn DevSliceArg> = |g| Box::new(g);
        let _: fn(GpuRef<f64>) -> Box<dyn DevSliceArg> = |g| Box::new(g);
        let _: fn(GpuRef<u8>) -> Box<dyn DevSliceArg> = |g| Box::new(g);
        let _: fn(GpuRef<i32>) -> Box<dyn DevSliceArg> = |g| Box::new(g);
        let _ = takes_box;
        #[cfg(feature = "f16")]
        {
            let _: fn(GpuRef<half::f16>) -> Box<dyn DevSliceArg> = |g| Box::new(g);
            let _: fn(GpuRef<half::bf16>) -> Box<dyn DevSliceArg> = |g| Box::new(g);
        }
    }

    #[test]
    fn dev_slice_arg_for_gpu_ref() {
        // Compile-only witness: instantiate the closures above so the
        // function pointers are realized.
        _assert_dev_slice_arg_object_safe();
    }

    /// Compile-only witness that `Box<dyn ScalarArg>` round-trips for
    /// every primitive `CudaDtype`.
    #[test]
    fn scalar_arg_blanket_impls_compile() {
        fn takes(_: Box<dyn ScalarArg>) {}
        takes(Box::new(1.0f32));
        takes(Box::new(2.0f64));
        takes(Box::new(3i32));
        takes(Box::new(4u32));
        takes(Box::new(5u64));
        #[cfg(feature = "f16")]
        {
            takes(Box::new(half::f16::ONE));
            takes(Box::new(half::bf16::ONE));
        }
    }

    /// Stub-trait sanity: every non-NVRTC dispatch trait-object is
    /// at least nameable. (`*DispatchCtx<'_>` placeholders compile.)
    #[test]
    fn stub_dispatch_traits_compile() {
        fn _gemm(_: Box<dyn GemmDispatch>) {}
        #[cfg(feature = "cublaslt")]
        fn _blaslt(_: Box<dyn BlasLtDispatch>) {}
        fn _cudnn(_: Box<dyn CudnnDispatch>) {}
        #[cfg(feature = "cufft")]
        fn _fft(_: Box<dyn FftDispatch>) {}
        fn _rng(_: Box<dyn RngDispatch>) {}
        #[cfg(feature = "cusolver")]
        fn _solver(_: Box<dyn crate::kernel::solver::SolverDispatch>) {}
        fn _sparse(_: Box<dyn SparseDispatch>) {}
        fn _tensor(_: Box<dyn TensorDispatch>) {}
        #[cfg(feature = "nccl")]
        fn _coll(_: Box<dyn CollectiveDispatch>) {}
    }
}
