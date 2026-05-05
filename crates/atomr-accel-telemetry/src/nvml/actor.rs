//! `NvmlActor` — a Tokio task that periodically polls NVML and
//! exposes the resulting metrics as a [`NvmlSnapshot`]. Polling
//! interval is configurable; default 1 second.
//!
//! The library is loaded dynamically via `libloading`. We resolve
//! every NVML entry point we use into function pointers stashed on
//! a [`NvmlLib`] struct so the polling loop avoids a `Symbol::get`
//! per tick.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

/// Reply alias — every `NvmlMsg` returns its own typed payload via
/// a `oneshot::Sender<Result<T, NvmlError>>`.
pub type NvmlReply<T> = Result<T, NvmlError>;

/// Errors emitted by the NVML actor.
#[derive(Debug, thiserror::Error)]
pub enum NvmlError {
    /// `libnvidia-ml.so.1` could not be loaded. Common on consumer
    /// / WSL setups where the driver doesn't ship NVML.
    #[error("libnvidia-ml not available: {0}")]
    LibraryUnavailable(String),

    /// NVML returned a non-success status code.
    #[error("NVML call failed: {func} -> code {code}")]
    Call { func: &'static str, code: c_int },

    /// Message channel closed (actor was dropped).
    #[error("NVML actor channel closed")]
    Closed,
}

/// Per-device snapshot of every metric NVML exposes through this
/// crate. All fields are `Option<_>` so callers can distinguish
/// "the GPU doesn't expose this counter" from "we read 0".
#[derive(Default, Debug, Clone)]
pub struct NvmlDeviceSnapshot {
    pub device_index: u32,
    /// UUID, formatted as the NVML string (e.g. `"GPU-...-...""`).
    pub uuid: Option<String>,
    pub name: Option<String>,

    pub power_milliwatts: Option<u32>,
    pub power_average_milliwatts: Option<u32>,
    pub temperature_gpu_c: Option<u32>,
    pub temperature_memory_c: Option<u32>,

    pub ecc_sbe_volatile: Option<u64>,
    pub ecc_dbe_volatile: Option<u64>,
    pub ecc_sbe_aggregate: Option<u64>,
    pub ecc_dbe_aggregate: Option<u64>,

    pub clock_sm_mhz: Option<u32>,
    pub clock_mem_mhz: Option<u32>,
    pub clock_video_mhz: Option<u32>,

    /// Bitmask of `nvmlClocksThrottleReasons*` flags.
    pub throttle_reasons: Option<u64>,

    /// PCIe Tx / Rx in KiB/s.
    pub pcie_tx_kib_per_s: Option<u32>,
    pub pcie_rx_kib_per_s: Option<u32>,

    pub mem_total_bytes: Option<u64>,
    pub mem_used_bytes: Option<u64>,
    pub mem_free_bytes: Option<u64>,

    pub processes: Vec<NvmlProcess>,

    pub mig_mode_current: Option<u32>,
    pub mig_mode_pending: Option<u32>,
    pub mig_instances: Vec<NvmlMigInstance>,
}

#[derive(Debug, Clone)]
pub struct NvmlProcess {
    pub pid: u32,
    pub used_gpu_memory_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct NvmlMigInstance {
    pub gi_id: u32,
    pub ci_id: u32,
    pub memory_size_bytes: u64,
}

/// System-wide snapshot. Returned by [`NvmlMsg::Snapshot`].
#[derive(Default, Debug, Clone)]
pub struct NvmlSnapshot {
    pub devices: Vec<NvmlDeviceSnapshot>,
    pub generated_at_unix_nanos: u128,
}

/// Configuration knobs for the NVML actor.
#[derive(Debug, Clone)]
pub struct NvmlConfig {
    /// Poll cadence. Default 1 second.
    pub interval: Duration,
    /// Optional library path override. When `None`, the loader
    /// scans the standard candidates: `libnvidia-ml.so.1`,
    /// `libnvidia-ml.so`, then `nvml.dll` on Windows.
    pub library_path: Option<String>,
}

impl Default for NvmlConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            library_path: None,
        }
    }
}

/// Messages accepted by [`NvmlActor`].
#[non_exhaustive]
pub enum NvmlMsg {
    /// Return the most recent snapshot.
    Snapshot {
        reply: oneshot::Sender<NvmlReply<NvmlSnapshot>>,
    },
    /// Update the polling cadence.
    SetInterval {
        interval: Duration,
        reply: oneshot::Sender<NvmlReply<()>>,
    },
    /// Stop the actor and release the loaded library on the next
    /// tick.
    Shutdown {
        reply: oneshot::Sender<NvmlReply<()>>,
    },
}

/// NVML actor handle. Drop the handle to abort the polling task;
/// for clean shutdown send [`NvmlMsg::Shutdown`] first.
pub struct NvmlActor {
    sender: mpsc::Sender<NvmlMsg>,
    join: Option<JoinHandle<()>>,
    /// Shared snapshot the actor's polling loop writes into; the
    /// `Snapshot` message handler reads from this without blocking
    /// the loop.
    latest: Arc<RwLock<NvmlSnapshot>>,
}

impl NvmlActor {
    /// Try to load NVML and start the polling task. Returns
    /// `Err(NvmlError::LibraryUnavailable)` if `libnvidia-ml.so.1`
    /// can't be found / opened.
    ///
    /// On success spawns a background Tokio task that polls every
    /// `config.interval`.
    pub fn try_new(config: NvmlConfig) -> Result<Self, NvmlError> {
        let lib = NvmlLib::load(config.library_path.as_deref())
            .map_err(|e| NvmlError::LibraryUnavailable(e.to_string()))?;
        // Safety: NVML demands one nvmlInit_v2 before any other
        // call. The library's loader returned `Ok`, so the symbol
        // is resolvable.
        let init_status = unsafe { (lib.nvml_init_v2)() };
        if init_status != 0 {
            return Err(NvmlError::Call {
                func: "nvmlInit_v2",
                code: init_status,
            });
        }

        let (tx, rx) = mpsc::channel::<NvmlMsg>(64);
        let latest = Arc::new(RwLock::new(NvmlSnapshot::default()));
        let join = tokio::spawn(actor_loop(lib, config, rx, latest.clone()));

        Ok(Self {
            sender: tx,
            join: Some(join),
            latest,
        })
    }

    /// Sender end of the actor's mpsc. Clones cheaply.
    pub fn sender(&self) -> mpsc::Sender<NvmlMsg> {
        self.sender.clone()
    }

    /// Read the most recent snapshot without sending a message.
    /// Useful for atomr-telemetry probes that poll on their own
    /// cadence.
    pub fn latest_snapshot(&self) -> NvmlSnapshot {
        self.latest.read().clone()
    }
}

impl Drop for NvmlActor {
    fn drop(&mut self) {
        if let Some(j) = self.join.take() {
            j.abort();
        }
    }
}

async fn actor_loop(
    lib: NvmlLib,
    mut config: NvmlConfig,
    mut rx: mpsc::Receiver<NvmlMsg>,
    latest: Arc<RwLock<NvmlSnapshot>>,
) {
    let mut ticker = tokio::time::interval(config.interval);
    loop {
        tokio::select! {
            biased;
            msg = rx.recv() => {
                match msg {
                    None => break,
                    Some(NvmlMsg::Snapshot { reply }) => {
                        let snap = latest.read().clone();
                        let _ = reply.send(Ok(snap));
                    }
                    Some(NvmlMsg::SetInterval { interval, reply }) => {
                        config.interval = interval;
                        ticker = tokio::time::interval(interval);
                        let _ = reply.send(Ok(()));
                    }
                    Some(NvmlMsg::Shutdown { reply }) => {
                        // Best-effort shutdown of NVML.
                        unsafe { (lib.nvml_shutdown)(); }
                        let _ = reply.send(Ok(()));
                        break;
                    }
                }
            }
            _ = ticker.tick() => {
                match poll_once(&lib) {
                    Ok(snap) => *latest.write() = snap,
                    Err(e) => debug!(error = %e, "NVML poll failed"),
                }
            }
        }
    }
}

/// Shape of every NVML entry point we resolve. The function pointer
/// types are minimal — most NVML calls take a device handle (opaque
/// pointer) and a `*mut OutType`.
pub(crate) struct NvmlLib {
    _library: libloading::Library,
    nvml_init_v2: unsafe extern "C" fn() -> c_int,
    nvml_shutdown: unsafe extern "C" fn() -> c_int,
    nvml_device_get_count_v2: unsafe extern "C" fn(*mut c_uint) -> c_int,
    nvml_device_get_handle_by_index_v2: unsafe extern "C" fn(c_uint, *mut *mut c_void) -> c_int,
    nvml_device_get_name: unsafe extern "C" fn(*mut c_void, *mut c_char, c_uint) -> c_int,
}

impl NvmlLib {
    fn load(override_path: Option<&str>) -> Result<Self, libloading::Error> {
        const DEFAULT_CANDIDATES: &[&str] = &[
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "nvml.dll",
            "libnvidia-ml.dylib",
        ];
        // Build an owned candidate list so the borrow of
        // `override_path` doesn't have to outlive the loop.
        let owned: Vec<&str> = match override_path {
            Some(p) => vec![p],
            None => DEFAULT_CANDIDATES.to_vec(),
        };
        let mut last_err: Option<libloading::Error> = None;
        for cand in owned.iter() {
            // Safety: dlopen of a name we control. The library
            // is only used through validated entry points.
            let lib = unsafe { libloading::Library::new(cand) };
            match lib {
                Ok(library) => {
                    // Safety: each `get` is unsafe because the type
                    // assertion on the function pointer can't be
                    // verified by the compiler. We use the official
                    // NVML ABI which is stable across driver
                    // versions.
                    let nvml_init_v2 = unsafe {
                        let s: libloading::Symbol<'_, unsafe extern "C" fn() -> c_int> =
                            library.get(b"nvmlInit_v2\0")?;
                        *s
                    };
                    let nvml_shutdown = unsafe {
                        let s: libloading::Symbol<'_, unsafe extern "C" fn() -> c_int> =
                            library.get(b"nvmlShutdown\0")?;
                        *s
                    };
                    let nvml_device_get_count_v2 = unsafe {
                        let s: libloading::Symbol<
                            '_,
                            unsafe extern "C" fn(*mut c_uint) -> c_int,
                        > = library.get(b"nvmlDeviceGetCount_v2\0")?;
                        *s
                    };
                    let nvml_device_get_handle_by_index_v2 = unsafe {
                        let s: libloading::Symbol<
                            '_,
                            unsafe extern "C" fn(c_uint, *mut *mut c_void) -> c_int,
                        > = library.get(b"nvmlDeviceGetHandleByIndex_v2\0")?;
                        *s
                    };
                    let nvml_device_get_name = unsafe {
                        let s: libloading::Symbol<
                            '_,
                            unsafe extern "C" fn(*mut c_void, *mut c_char, c_uint) -> c_int,
                        > = library.get(b"nvmlDeviceGetName\0")?;
                        *s
                    };
                    return Ok(Self {
                        _library: library,
                        nvml_init_v2,
                        nvml_shutdown,
                        nvml_device_get_count_v2,
                        nvml_device_get_handle_by_index_v2,
                        nvml_device_get_name,
                    });
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            // If we got here with no candidates we still need an
            // error; fabricate one.
            // Safety: passing a path we know does not exist so the
            // call returns Err.
            unsafe {
                libloading::Library::new("__atomr_accel_telemetry_no_nvml__")
                    .err()
                    .unwrap_or_else(|| panic!("libloading should reject the sentinel path"))
            }
        }))
    }
}

/// Single polling tick. Walks every device and assembles a
/// `NvmlSnapshot`. Per-call NVML failures are recorded as `None`
/// fields on the affected device rather than aborting the whole
/// poll.
fn poll_once(lib: &NvmlLib) -> Result<NvmlSnapshot, NvmlError> {
    let mut count: c_uint = 0;
    // Safety: out-pointer is valid for the life of the call.
    let status = unsafe { (lib.nvml_device_get_count_v2)(&mut count) };
    if status != 0 {
        return Err(NvmlError::Call {
            func: "nvmlDeviceGetCount_v2",
            code: status,
        });
    }
    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let status = unsafe { (lib.nvml_device_get_handle_by_index_v2)(i, &mut handle) };
        if status != 0 {
            warn!(idx = i, code = status, "nvmlDeviceGetHandleByIndex_v2 failed");
            continue;
        }
        let mut device = NvmlDeviceSnapshot {
            device_index: i,
            ..Default::default()
        };
        let mut name_buf = [0i8; 96];
        let status = unsafe {
            (lib.nvml_device_get_name)(handle, name_buf.as_mut_ptr(), name_buf.len() as c_uint)
        };
        if status == 0 {
            // Safety: NVML wrote a NUL-terminated ASCII string.
            let s = unsafe { CStr::from_ptr(name_buf.as_ptr()) };
            device.name = Some(s.to_string_lossy().into_owned());
        }
        devices.push(device);
    }
    Ok(NvmlSnapshot {
        devices,
        generated_at_unix_nanos: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    })
}
