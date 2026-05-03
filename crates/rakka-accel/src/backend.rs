//! Backend identity traits.
//!
//! A `rakka-accel` backend (CUDA, ROCm, Metal, oneAPI, Vulkan compute,
//! …) is a coherent triple of *device handle*, *stream / queue*, and
//! *event / fence* types. These traits name them so portable code
//! can be parameterized over `B: AccelBackend` without committing to
//! a vendor SDK.
//!
//! Backends still expose richer concrete types directly — the
//! cuBLAS / cuDNN / cuFFT actors live in the `rakka-accel-cuda`
//! crate and are not part of this trait surface. The traits here
//! capture only the shape that every backend has to provide.

use std::fmt::Debug;
use std::sync::Arc;

use crate::error::AccelError;

/// Marker trait identifying a compute-acceleration backend.
///
/// Implemented by the `Backend` zero-sized type in each backend
/// crate. Use as a trait bound:
///
/// ```ignore
/// fn observe_load<B: AccelBackend>(d: &B::Device) -> u32 { ... }
/// ```
///
/// The associated types name the lifetime-bounded handles a
/// backend hands out. They're all `Send + Sync + 'static` so they
/// can travel across actor boundaries.
pub trait AccelBackend: Send + Sync + 'static {
    /// Display name, e.g. `"cuda"`, `"rocm"`, `"metal"`.
    const NAME: &'static str;

    /// Device handle (e.g. `cudarc::driver::CudaContext`,
    /// `hipDevice_t`, `MTLDevice`).
    type Device: AccelDevice<Backend = Self>;

    /// Per-actor stream / queue handle (e.g.
    /// `cudarc::driver::CudaStream`, `hipStream_t`,
    /// `MTLCommandQueue`).
    type Stream: AccelStream<Backend = Self>;

    /// Recordable / waitable synchronization primitive (e.g.
    /// `cudaEvent_t`, `hipEvent_t`, `MTLEvent`).
    type Event: Debug + Send + Sync + 'static;

    /// Backend-specific error variants supplement the core
    /// [`AccelError`] enum via the `LibraryError { lib, msg }`
    /// catch-all. Backends that need finer granularity wrap
    /// `AccelError` in their own type; the core itself is
    /// `#[non_exhaustive]` so adding variants is a minor bump,
    /// not a breaking change.
    type Error: std::error::Error + Send + Sync + From<AccelError> + 'static;
}

/// Device-handle contract: identification + a hook to observe
/// generation rebuilds (sticky-error recovery).
pub trait AccelDevice: Send + Sync + 'static {
    type Backend: AccelBackend;

    /// Stable, opaque device id. CUDA returns the ordinal; ROCm
    /// returns the hipDevice_t; Metal returns a hashed
    /// `MTLDevice.registryID`.
    fn device_id(&self) -> u32;

    /// Current generation counter. Bumped every time the underlying
    /// device context is torn down + rebuilt (e.g. cuda sticky-error
    /// recovery). `AccelRef`s minted against an older generation
    /// fail their next `access()`.
    fn generation(&self) -> u64;
}

/// Stream / queue contract: ordered submission of work, plus the
/// ability to record an event for cross-stream synchronization.
pub trait AccelStream: Send + Sync + 'static {
    type Backend: AccelBackend;

    /// Record an event into this stream. Other streams can wait on
    /// the resulting handle via [`AccelStream::wait_event`].
    fn record_event(&self) -> Result<<Self::Backend as AccelBackend>::Event, AccelError>;

    /// Wait on an event recorded into another stream before
    /// scheduling further work on this one. Backends without
    /// cross-queue events synthesize a host-side block.
    fn wait_event(&self, event: &<Self::Backend as AccelBackend>::Event) -> Result<(), AccelError>;
}

/// Convenience type alias for a shared device handle. Every backend
/// hands out devices through `Arc<B::Device>` so they survive context
/// rebuilds without invalidating the outer `ActorRef`.
pub type Device<B> = Arc<<B as AccelBackend>::Device>;

/// Convenience type alias for a shared stream handle.
pub type Stream<B> = Arc<<B as AccelBackend>::Stream>;
