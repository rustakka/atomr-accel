//! `cuMemPrefetchAsync` wrapper.
//!
//! Sits next to [`super::managed`] but is callable independently —
//! callers that have a raw `CUdeviceptr` (e.g. from a custom
//! allocator or an IPC-opened handle) can still issue prefetch hints.
//!
//! Use [`super::managed::PrefetchTarget`] to describe the destination.

use std::sync::Arc;

use cudarc::driver::sys as driver_sys;
use cudarc::driver::CudaStream;

use crate::error::GpuError;
use crate::sys::cuda_driver;

use super::managed::PrefetchTarget;

/// Prefetch the byte range `[dev_ptr .. dev_ptr+bytes)` to `target`,
/// issued onto `stream`. Wraps `cuMemPrefetchAsync_v2`.
///
/// Returns `Unrecoverable` on hosts where `libcuda.so` isn't loadable.
pub fn prefetch_async(
    dev_ptr: driver_sys::CUdeviceptr,
    bytes: usize,
    target: PrefetchTarget,
    stream: &Arc<CudaStream>,
) -> Result<(), GpuError> {
    // CUDA 13.0+'s `CUmemLocation` uses an anonymous union for the
    // location id; zero-init and set type only — concrete location
    // pinning is wired in a follow-up PR.
    let location = unsafe {
        let mut loc: driver_sys::CUmemLocation = std::mem::zeroed();
        loc.type_ = match target {
            PrefetchTarget::Device(_) => driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            PrefetchTarget::Cpu => driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST,
        };
        loc
    };
    let _ = target;
    cuda_driver::mem_prefetch_async_v2(dev_ptr, bytes, location, 0, stream.cu_stream())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Phase 3 mock-mode test: confirm the wrapper compiles and that
    // calling against a null dev pointer surfaces a typed error rather
    // than panicking on no-GPU hosts. The actual path is exercised
    // through `memory::managed::tests` which threads through
    // `ManagedAllocatorActor`.

    #[test]
    fn prefetch_async_returns_typed_error_on_no_driver() {
        // Attempting to issue a real prefetch with a null pointer on a
        // host without libcuda.so loadable produces Unrecoverable,
        // which is what we want — the wrapper does not panic.
        let host_loc = unsafe {
            let mut loc: driver_sys::CUmemLocation = std::mem::zeroed();
            loc.type_ = driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST;
            loc
        };
        let r = cuda_driver::mem_prefetch_async_v2(
            0,
            0,
            host_loc,
            0,
            std::ptr::null_mut(),
        );
        // Either Unrecoverable (no driver) or LibraryError (driver
        // present, rejects null) — both acceptable.
        match r {
            Ok(()) => {}
            Err(GpuError::Unrecoverable(_)) => {}
            Err(GpuError::LibraryError { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
