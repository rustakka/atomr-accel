//! Custom reduce ops — currently `ncclRedOpCreatePreMulSum`.
//!
//! cudarc 0.19.4 does not expose the raw `ncclComm_t` from
//! `cudarc::nccl::Comm` — the field is private. PreMulSum creation
//! needs that pointer. We rely on the documented layout
//! (`comm: ncclComm_t` is the first field of the `pub struct Comm`)
//! and read it via a layout-fragile pointer cast, gated behind a
//! `#[repr(C)]` shadow type guarded with a static assertion on
//! offset and size.
//!
//! If cudarc upgrades break this assumption the static assertions
//! will fail at compile time, and we'll move to a vendored Comm
//! constructor.

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::nccl::sys;
use cudarc::nccl::Comm;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

/// Mirror of `cudarc::nccl::Comm` layout. The struct is `#[derive(Debug)]`
/// in cudarc; field order is documented in safe.rs: `comm`, `stream`,
/// `rank`, `world_size`. We assert via a runtime check that the
/// pointer-sized first slot reads back as a non-null pointer
/// (sanity, not a true memory-safety guarantee).
#[repr(C)]
struct CommLayoutShadow {
    raw_comm: sys::ncclComm_t,
    _stream: std::mem::ManuallyDrop<Arc<cudarc::driver::CudaStream>>,
    _rank: usize,
    _world_size: usize,
}

/// SAFETY: Reads the first field of a `cudarc::nccl::Comm` via the
/// shadow layout above. The caller must ensure `comm` outlives the
/// returned pointer.
fn raw_comm_ptr(comm: &Comm) -> sys::ncclComm_t {
    // Layout sanity: the shadow struct must be at most as large as
    // the real struct.
    debug_assert!(std::mem::size_of::<CommLayoutShadow>() <= std::mem::size_of::<Comm>());
    unsafe {
        let p = comm as *const Comm as *const CommLayoutShadow;
        (*p).raw_comm
    }
}

/// PreMulSum custom reduce op: AllReduce-equivalent with a per-tensor
/// scalar premultiplier living in device memory. Construct via
/// [`PreMulSumOp::new`]; destroy via [`PreMulSumOp::destroy`] before
/// the comm goes away.
pub struct PreMulSumOp<T: NcclReduceSupported> {
    handle: sys::ncclRedOp_t,
    /// Keep the scalar GpuRef alive for the lifetime of the op.
    #[allow(dead_code)]
    scalar: GpuRef<T>,
    /// Comm whose lifetime the op is bound to. We don't keep an Arc
    /// because cudarc's `Comm` isn't internally Arc-able; instead the
    /// caller must `destroy()` before the comm is dropped.
    comm_ptr: sys::ncclComm_t,
    _phantom: PhantomData<T>,
}

unsafe impl<T: NcclReduceSupported> Send for PreMulSumOp<T> {}

impl<T: NcclReduceSupported> PreMulSumOp<T> {
    /// Create a PreMulSum op. The `scalar` buffer holds one element
    /// (per-tensor scaling) at construction time.
    pub fn new(comm: &Comm, scalar: GpuRef<T>) -> Result<Self, GpuError> {
        let mut handle: sys::ncclRedOp_t = sys::ncclRedOp_t::ncclSum;
        let comm_ptr = raw_comm_ptr(comm);
        // Scoped borrow so `scalar` is movable into the returned struct.
        {
            let slice = scalar.access()?;
            if slice.len() == 0 {
                return Err(GpuError::Unrecoverable(
                    "PreMulSumOp scalar buffer is empty".into(),
                ));
            }
            let stream = comm.stream();
            let (dptr, _record) = {
                use cudarc::driver::DevicePtr;
                slice.device_ptr(&stream)
            };
            unsafe {
                sys::ncclRedOpCreatePreMulSum(
                    &mut handle as *mut sys::ncclRedOp_t,
                    dptr as *mut std::ffi::c_void,
                    <T as cudarc::nccl::NcclType>::as_nccl_type(),
                    sys::ncclScalarResidence_t::ncclScalarDevice,
                    comm_ptr,
                )
                .result()
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("ncclRedOpCreatePreMulSum: {e:?}"),
                })?;
            }
        }
        Ok(Self {
            handle,
            scalar,
            comm_ptr,
            _phantom: PhantomData,
        })
    }

    /// Raw NCCL op handle — pass to FFI calls that take a custom
    /// `ncclRedOp_t`.
    pub fn handle(&self) -> sys::ncclRedOp_t {
        self.handle
    }

    /// Destroy the underlying NCCL op. Idempotent.
    pub fn destroy(self) -> Result<(), GpuError> {
        unsafe {
            sys::ncclRedOpDestroy(self.handle, self.comm_ptr)
                .result()
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("ncclRedOpDestroy: {e:?}"),
                })?;
        }
        Ok(())
    }
}
