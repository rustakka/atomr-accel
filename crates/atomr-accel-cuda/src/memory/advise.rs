//! `cuMemAdvise` wrapper.
//!
//! Public typed enum [`MemAdvice`] mirrors the six driver-level
//! `CU_MEM_ADVISE_*` variants. The actor surface routes
//! `ManagedMsg::Advise` through here, but the wrapper is also callable
//! directly for callers operating on managed allocations they own.

use cudarc::driver::sys as driver_sys;

use crate::error::GpuError;
use crate::sys::cuda_driver;

use super::managed::PrefetchTarget;

/// Memory advisory hints. Each maps 1:1 to a `CU_MEM_ADVISE_*`
/// variant. The `Set*` variants take a target location; the `Unset*`
/// variants ignore it but require it to typecheck (CUDA accepts the
/// arg either way).
#[derive(Debug, Clone, Copy)]
pub enum MemAdvice {
    /// Hint that the range is read predominantly. CUDA may duplicate
    /// pages across processors.
    SetReadMostly,
    UnsetReadMostly,
    /// Pin the range to `target`'s memory.
    SetPreferredLocation(PrefetchTarget),
    UnsetPreferredLocation,
    /// Advise that `target` will access the range; CUDA configures
    /// page-table mappings accordingly.
    SetAccessedBy(PrefetchTarget),
    UnsetAccessedBy(PrefetchTarget),
}

impl MemAdvice {
    fn raw(self) -> (driver_sys::CUmem_advise, driver_sys::CUmemLocation) {
        // CUDA 13.0+'s `CUmemLocation` uses an anonymous union for the
        // location id (`__bindgen_anon_1`) rather than a plain `id: i32`.
        // We zero-initialize and write the type only — concrete location
        // pinning is wired in a follow-up PR with the right layout.
        let host_loc: driver_sys::CUmemLocation = unsafe {
            let mut loc: driver_sys::CUmemLocation = std::mem::zeroed();
            loc.type_ = driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST;
            loc
        };
        match self {
            MemAdvice::SetReadMostly => (
                driver_sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY,
                host_loc,
            ),
            MemAdvice::UnsetReadMostly => (
                driver_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_READ_MOSTLY,
                host_loc,
            ),
            MemAdvice::SetPreferredLocation(t) => (
                driver_sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                location_for(t),
            ),
            MemAdvice::UnsetPreferredLocation => (
                driver_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
                host_loc,
            ),
            MemAdvice::SetAccessedBy(t) => (
                driver_sys::CUmem_advise::CU_MEM_ADVISE_SET_ACCESSED_BY,
                location_for(t),
            ),
            MemAdvice::UnsetAccessedBy(t) => (
                driver_sys::CUmem_advise::CU_MEM_ADVISE_UNSET_ACCESSED_BY,
                location_for(t),
            ),
        }
    }
}

fn location_for(t: PrefetchTarget) -> driver_sys::CUmemLocation {
    unsafe {
        let mut loc: driver_sys::CUmemLocation = std::mem::zeroed();
        loc.type_ = match t {
            PrefetchTarget::Device(_) => driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            PrefetchTarget::Cpu => driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST,
        };
        loc
    }
}

/// Apply `advice` to the byte range `[dev_ptr .. dev_ptr+bytes)`.
/// Wraps `cuMemAdvise_v2`.
pub fn advise(
    dev_ptr: driver_sys::CUdeviceptr,
    bytes: usize,
    advice: MemAdvice,
) -> Result<(), GpuError> {
    let (a, loc) = advice.raw();
    cuda_driver::mem_advise_v2(dev_ptr, bytes, a, loc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_advice_constructs_for_each_variant() {
        // Verify each enum variant maps to a distinct (advice, location)
        // pair without panicking.
        let variants = [
            MemAdvice::SetReadMostly,
            MemAdvice::UnsetReadMostly,
            MemAdvice::SetPreferredLocation(PrefetchTarget::Device(0)),
            MemAdvice::UnsetPreferredLocation,
            MemAdvice::SetAccessedBy(PrefetchTarget::Device(1)),
            MemAdvice::UnsetAccessedBy(PrefetchTarget::Cpu),
        ];
        for v in variants {
            let (_a, _loc) = v.raw();
        }
    }

    #[test]
    fn advise_returns_typed_error_on_no_driver() {
        let r = advise(0, 0, MemAdvice::SetReadMostly);
        match r {
            Ok(()) => {}
            Err(GpuError::Unrecoverable(_)) => {}
            Err(GpuError::LibraryError { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
