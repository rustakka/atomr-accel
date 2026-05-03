//! `AccelRef<T, B>` — backend-agnostic typed device pointer.
//!
//! Each backend defines its own concrete buffer type
//! (`cudarc::driver::CudaSlice<T>` for CUDA, `hip-sys` slices for
//! ROCm, `MTLBuffer` for Metal, …). This module declares the
//! generation-validated wrapper contract every backend's concrete
//! `*Ref<T>` type satisfies.
//!
//! Backends that want to share more shape than this trait offers
//! are encouraged to ship a `pub type AccelRef<T> = MyConcreteRef<T>;`
//! re-export so application code can pattern-match on the concrete
//! type when needed.

use std::marker::PhantomData;

use crate::backend::AccelBackend;
use crate::error::AccelError;

/// Trait implemented by every backend's typed-pointer wrapper.
///
/// The generation token check is the contract: each backend's
/// `access()` returns `Err(AccelError::AccelRefStale)` if the
/// device generation has advanced past the one the ref was minted
/// against. Code that walks `AccelRef`s never has to know which
/// backend is underneath.
pub trait AccelRef<T, B: AccelBackend>: Clone + Send + Sync + 'static {
    /// Number of `T` elements in the buffer.
    fn len(&self) -> usize;

    /// Returns true if `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generation token captured at allocation time. Backends mint
    /// fresh refs against `device.generation()` and validate the
    /// match on every `access()`.
    fn generation(&self) -> u64;

    /// Originating device id. Used by multi-device routing to
    /// reject cross-device misuse (e.g. AllReduce input mismatch).
    fn device_id(&self) -> Option<u32>;

    /// Validate the ref is still usable. Returns `Err` if the
    /// device generation has moved or the device is shutting down.
    fn check(&self) -> Result<(), AccelError>;
}

/// Marker struct so portable code can reference an
/// "abstract `AccelRef<T>`" without committing to a backend.
/// Concrete backends usually expose their own typedef
/// (e.g. `rakka_accel_cuda::GpuRef<T>`).
pub struct AnyRef<T, B: AccelBackend> {
    _phantom: PhantomData<(T, B)>,
}
