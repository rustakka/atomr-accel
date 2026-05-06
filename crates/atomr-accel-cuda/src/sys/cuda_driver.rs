//! Thin, panic-safe wrappers around CUDA driver-API entry points
//! that cudarc 0.19 only exposes at the `sys` level.
//!
//! Every function in this module wraps the raw `unsafe extern "C"`
//! call in `std::panic::catch_unwind`, because cudarc's
//! dynamic-loader path panics if `libcuda.so` isn't present at
//! runtime. The wrappers convert "library not loadable" panics into
//! [`crate::error::GpuError::Unrecoverable`] so kernel actors stay
//! alive on no-GPU hosts.
//!
//! All pointer / handle arguments are forwarded as-is — the caller
//! is responsible for validity.

use cudarc::driver::sys as driver_sys;
use cudarc::runtime::sys as runtime_sys;

use crate::error::GpuError;

const LIB_DRIVER: &str = "driver";
const LIB_RUNTIME: &str = "runtime";

fn driver_check(s: driver_sys::CUresult, op: &str) -> Result<(), GpuError> {
    if s == driver_sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB_DRIVER,
            msg: format!("{op}: {s:?}"),
        })
    }
}

fn runtime_check(s: runtime_sys::cudaError_t, op: &str) -> Result<(), GpuError> {
    if s == runtime_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB_RUNTIME,
            msg: format!("{op}: {s:?}"),
        })
    }
}

/// Invoke `f`, mapping any panic from the cudarc dynamic loader into
/// `Unrecoverable`. Used when `libcuda.so` may not be loadable on the
/// host (CI, dev laptops without an NVIDIA driver).
fn guarded<F, R>(op: &'static str, f: F) -> Result<R, GpuError>
where
    F: FnOnce() -> Result<R, GpuError>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(GpuError::Unrecoverable(format!(
            "{op}: CUDA driver not loadable"
        ))),
    }
}

// ---------------------------------------------------------------------------
// cuMemPrefetchAsync (driver-API, used by `memory::prefetch`).
// ---------------------------------------------------------------------------

/// Prefetch `[dev_ptr .. dev_ptr+count)` to a target memory location
/// on `stream`. Wraps `cuMemPrefetchAsync_v2` (the v2 shape that
/// takes a `CUmemLocation`).
pub fn mem_prefetch_async_v2(
    dev_ptr: driver_sys::CUdeviceptr,
    count: usize,
    location: driver_sys::CUmemLocation,
    flags: u32,
    stream: driver_sys::CUstream,
) -> Result<(), GpuError> {
    guarded("cuMemPrefetchAsync_v2", || {
        // SAFETY: pointer + length validity is the caller's contract.
        let s =
            unsafe { driver_sys::cuMemPrefetchAsync_v2(dev_ptr, count, location, flags, stream) };
        driver_check(s, "cuMemPrefetchAsync_v2")
    })
}

// ---------------------------------------------------------------------------
// cuMemAdvise (driver-API, used by `memory::advise`).
// ---------------------------------------------------------------------------

/// Apply a memory advisory hint to a managed-memory range. Wraps
/// `cuMemAdvise_v2` (the v2 shape that takes a `CUmemLocation`).
pub fn mem_advise_v2(
    dev_ptr: driver_sys::CUdeviceptr,
    count: usize,
    advice: driver_sys::CUmem_advise,
    location: driver_sys::CUmemLocation,
) -> Result<(), GpuError> {
    guarded("cuMemAdvise_v2", || {
        // SAFETY: caller-supplied pointer and length must be valid.
        let s = unsafe { driver_sys::cuMemAdvise_v2(dev_ptr, count, advice, location) };
        driver_check(s, "cuMemAdvise_v2")
    })
}

// ---------------------------------------------------------------------------
// cuIpc* (driver-API, used by `memory::ipc` and `event::ipc`).
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-ipc")]
pub fn ipc_get_mem_handle(
    dev_ptr: driver_sys::CUdeviceptr,
) -> Result<driver_sys::CUipcMemHandle, GpuError> {
    guarded("cuIpcGetMemHandle", || {
        let mut handle = driver_sys::CUipcMemHandle_st {
            reserved: [0; 64usize],
        };
        // SAFETY: out-pointer + caller-provided dev_ptr.
        let s = unsafe { driver_sys::cuIpcGetMemHandle(&mut handle as *mut _, dev_ptr) };
        driver_check(s, "cuIpcGetMemHandle")?;
        Ok(handle)
    })
}

#[cfg(feature = "cuda-ipc")]
pub fn ipc_open_mem_handle_v2(
    handle: driver_sys::CUipcMemHandle,
    flags: u32,
) -> Result<driver_sys::CUdeviceptr, GpuError> {
    guarded("cuIpcOpenMemHandle_v2", || {
        let mut dptr: driver_sys::CUdeviceptr = 0;
        // SAFETY: out-pointer + caller-supplied handle.
        let s = unsafe { driver_sys::cuIpcOpenMemHandle_v2(&mut dptr as *mut _, handle, flags) };
        driver_check(s, "cuIpcOpenMemHandle_v2")?;
        Ok(dptr)
    })
}

#[cfg(feature = "cuda-ipc")]
pub fn ipc_close_mem_handle(dev_ptr: driver_sys::CUdeviceptr) -> Result<(), GpuError> {
    guarded("cuIpcCloseMemHandle", || {
        // SAFETY: dev_ptr returned by a prior cuIpcOpenMemHandle_v2.
        let s = unsafe { driver_sys::cuIpcCloseMemHandle(dev_ptr) };
        driver_check(s, "cuIpcCloseMemHandle")
    })
}

#[cfg(feature = "cuda-ipc")]
pub fn ipc_get_event_handle(
    event: driver_sys::CUevent,
) -> Result<driver_sys::CUipcEventHandle, GpuError> {
    guarded("cuIpcGetEventHandle", || {
        let mut handle = driver_sys::CUipcEventHandle_st {
            reserved: [0; 64usize],
        };
        // SAFETY: out-pointer + caller-supplied event handle.
        let s = unsafe { driver_sys::cuIpcGetEventHandle(&mut handle as *mut _, event) };
        driver_check(s, "cuIpcGetEventHandle")?;
        Ok(handle)
    })
}

#[cfg(feature = "cuda-ipc")]
pub fn ipc_open_event_handle(
    handle: driver_sys::CUipcEventHandle,
) -> Result<driver_sys::CUevent, GpuError> {
    guarded("cuIpcOpenEventHandle", || {
        let mut event: driver_sys::CUevent = std::ptr::null_mut();
        // SAFETY: out-pointer + caller-supplied handle bytes.
        let s = unsafe { driver_sys::cuIpcOpenEventHandle(&mut event as *mut _, handle) };
        driver_check(s, "cuIpcOpenEventHandle")?;
        Ok(event)
    })
}

// ---------------------------------------------------------------------------
// cuModule* (driver-API, used by `module`).
// ---------------------------------------------------------------------------

/// Load a cubin/fatbin/PTX image from a memory buffer. The buffer must
/// outlive the returned `CUmodule` for the duration of any pending
/// kernel launch — the driver may keep references to embedded
/// strings.
pub fn module_load_data(image: *const std::ffi::c_void) -> Result<driver_sys::CUmodule, GpuError> {
    guarded("cuModuleLoadData", || {
        let mut m: driver_sys::CUmodule = std::ptr::null_mut();
        // SAFETY: out-pointer; image is a caller-owned slice of bytes.
        let s = unsafe { driver_sys::cuModuleLoadData(&mut m as *mut _, image) };
        driver_check(s, "cuModuleLoadData")?;
        Ok(m)
    })
}

pub fn module_unload(m: driver_sys::CUmodule) -> Result<(), GpuError> {
    guarded("cuModuleUnload", || {
        // SAFETY: m was returned by a prior `cuModuleLoad*`.
        let s = unsafe { driver_sys::cuModuleUnload(m) };
        driver_check(s, "cuModuleUnload")
    })
}

pub fn module_get_function(
    m: driver_sys::CUmodule,
    name: &std::ffi::CStr,
) -> Result<driver_sys::CUfunction, GpuError> {
    guarded("cuModuleGetFunction", || {
        let mut f: driver_sys::CUfunction = std::ptr::null_mut();
        // SAFETY: out-pointer; name is a caller-owned C string.
        let s = unsafe { driver_sys::cuModuleGetFunction(&mut f as *mut _, m, name.as_ptr()) };
        driver_check(s, "cuModuleGetFunction")?;
        Ok(f)
    })
}

// ---------------------------------------------------------------------------
// cuLaunchKernel / cuLaunchCooperativeKernel (driver-API, used by `module`).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn launch_kernel(
    f: driver_sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    stream: driver_sys::CUstream,
    kernel_params: *mut *mut std::ffi::c_void,
) -> Result<(), GpuError> {
    guarded("cuLaunchKernel", || {
        // SAFETY: the kernel-params array's lifetime is the caller's
        // responsibility; the driver consumes it synchronously.
        let s = unsafe {
            driver_sys::cuLaunchKernel(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                stream,
                kernel_params,
                std::ptr::null_mut(),
            )
        };
        driver_check(s, "cuLaunchKernel")
    })
}

#[allow(clippy::too_many_arguments)]
pub fn launch_cooperative_kernel(
    f: driver_sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    stream: driver_sys::CUstream,
    kernel_params: *mut *mut std::ffi::c_void,
) -> Result<(), GpuError> {
    guarded("cuLaunchCooperativeKernel", || {
        // SAFETY: see `launch_kernel`. Cooperative launches additionally
        // require the kernel to fit on the device's SM count.
        let s = unsafe {
            driver_sys::cuLaunchCooperativeKernel(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                stream,
                kernel_params,
            )
        };
        driver_check(s, "cuLaunchCooperativeKernel")
    })
}

// ---------------------------------------------------------------------------
// runtime-API IPC (matches `cudaIpc*`, used as an alternative path on
// systems where the driver-API bindings aren't available — the v2 shape
// only ships on CUDA 12+).
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-ipc")]
pub fn runtime_ipc_get_mem_handle(
    dev_ptr: *mut std::ffi::c_void,
) -> Result<runtime_sys::cudaIpcMemHandle_t, GpuError> {
    guarded("cudaIpcGetMemHandle", || {
        let mut handle = runtime_sys::cudaIpcMemHandle_st {
            reserved: [0; 64usize],
        };
        // SAFETY: out-pointer; dev_ptr is the caller's contract.
        let s = unsafe { runtime_sys::cudaIpcGetMemHandle(&mut handle as *mut _, dev_ptr) };
        runtime_check(s, "cudaIpcGetMemHandle")?;
        Ok(handle)
    })
}
