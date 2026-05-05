//! `cuIpcGetMemHandle` / `cuIpcOpenMemHandle` / `cuIpcCloseMemHandle`
//! wrappers (gated `cuda-ipc`).
//!
//! IPC mem handles let one CUDA process expose a device allocation to
//! another for zero-copy sharing. The 64-byte payload is opaque; ship
//! it via your application's IPC channel (Unix socket, shared file,
//! gRPC, etc.) and reopen it on the destination.
//!
//! Lifecycle notes:
//! - Both processes must use the same CUDA driver and a peer-capable
//!   GPU pair.
//! - `OpenedMem` owns the lifetime of the imported pointer; `Drop`
//!   calls `cuIpcCloseMemHandle`.
//! - The exporter must outlive the importer's use of the handle —
//!   freeing the source allocation while the importer still holds an
//!   open handle is undefined.

#![cfg(feature = "cuda-ipc")]

use cudarc::driver::sys as driver_sys;

use crate::error::GpuError;
use crate::sys::cuda_driver;

/// Cross-process IPC handle for a memory range. 64 bytes of opaque
/// payload — interpret only by re-opening on the destination.
#[derive(Clone, Copy)]
pub struct IpcMemHandle {
    pub(crate) raw: driver_sys::CUipcMemHandle,
}

impl IpcMemHandle {
    pub fn as_bytes(&self) -> [u8; 64] {
        // SAFETY: `[c_char; 64]` and `[u8; 64]` share layout.
        unsafe { std::mem::transmute::<[std::ffi::c_char; 64], [u8; 64]>(self.raw.reserved) }
    }

    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        let raw = driver_sys::CUipcMemHandle_st {
            // SAFETY: layout-compatible.
            reserved: unsafe {
                std::mem::transmute::<[u8; 64], [std::ffi::c_char; 64]>(bytes)
            },
        };
        Self { raw }
    }
}

impl std::fmt::Debug for IpcMemHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcMemHandle").finish()
    }
}

unsafe impl Send for IpcMemHandle {}
unsafe impl Sync for IpcMemHandle {}

/// Imported memory handle. `Drop` releases the mapping via
/// `cuIpcCloseMemHandle`. Cloning is not supported — only one
/// `OpenedMem` may exist per import to keep the `Drop` semantics
/// straightforward.
#[derive(Debug)]
pub struct OpenedMem {
    dev_ptr: driver_sys::CUdeviceptr,
    bytes: usize,
}

impl OpenedMem {
    pub fn dev_ptr(&self) -> driver_sys::CUdeviceptr {
        self.dev_ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for OpenedMem {
    fn drop(&mut self) {
        if self.dev_ptr != 0 {
            let _ = cuda_driver::ipc_close_mem_handle(self.dev_ptr);
        }
    }
}

unsafe impl Send for OpenedMem {}
unsafe impl Sync for OpenedMem {}

/// Export an IPC handle for a device allocation.
pub fn get_mem_handle(dev_ptr: driver_sys::CUdeviceptr) -> Result<IpcMemHandle, GpuError> {
    cuda_driver::ipc_get_mem_handle(dev_ptr).map(|raw| IpcMemHandle { raw })
}

/// Open a previously-exported IPC handle.
///
/// `bytes` is the original allocation size; we pass it through to
/// `OpenedMem` so callers can build a typed slice on top.
pub fn open_mem_handle(handle: IpcMemHandle, bytes: usize) -> Result<OpenedMem, GpuError> {
    let dev_ptr = cuda_driver::ipc_open_mem_handle_v2(
        handle.raw,
        // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
        1,
    )?;
    Ok(OpenedMem { dev_ptr, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_round_trip() {
        let bytes: [u8; 64] = std::array::from_fn(|i| (i * 3) as u8 ^ 0x55);
        let h = IpcMemHandle::from_bytes(bytes);
        let round = h.as_bytes();
        assert_eq!(round, bytes);
        // Type-level send/sync sanity.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<IpcMemHandle>();
        assert_send_sync::<OpenedMem>();
    }

    #[test]
    fn open_returns_typed_error_on_no_driver() {
        let h = IpcMemHandle::from_bytes([0u8; 64]);
        let r = open_mem_handle(h, 0);
        match r {
            Ok(_) => {}
            Err(GpuError::Unrecoverable(_)) => {}
            Err(GpuError::LibraryError { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
