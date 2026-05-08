//! `Fft` — Python handle wrapping `ActorRef<FftMsg>`.
//!
//! Obtained via `Device.fft()` (only when the `cufft` feature is
//! compiled in *and* the device's `EnabledLibraries::CUFFT` flag is
//! set).
//!
//! # Phase 1.5++ — cuFFT depth (Path B: host-driven 1-shot FFTs)
//!
//! The four legacy 1-shot variants (`Forward1dR2C`, `Inverse1dC2R`,
//! `Exec1dC2C`, `Forward2dR2C`) take pre-built `GpuRef<f32>` /
//! `GpuRef<cufft_sys::float2>` buffers. Exposing those buffers as
//! Python `GpuBufferC64` classes would require `cufft_sys::float2` to
//! impl the crate's `CudaDtype` (which subsumes `AccelDtype` /
//! `DeviceRepr` / `ValidAsZeroBits`) — none of which is a thing we
//! can add from the Python crate (it'd require touching
//! `atomr-accel-cuda`).
//!
//! Instead we go through the canonical typed `FftMsg::Exec(
//! Box<dyn FftDispatch>)` path with a `FftRequest<f32>` that carries
//! `GpuRef<u8>` raw byte buffers. The wrapper internally:
//!
//! 1. Allocates u8 device buffers sized for the input + output.
//! 2. Uploads the numpy host array as raw bytes
//!    (`HostBuf::Owned<u8>`).  Complex (`numpy.complex64`,
//!    `numpy::Complex32`) is `#[repr(C)] (re, im)` and laid out
//!    identically to `cufft_sys::float2` `{ x, y }`, so this is just
//!    a transmute on the boundary.
//! 3. Sends `FftMsg::Exec(FftRequest::<f32>::new(plan_key, direction,
//!    src_u8, dst_u8, reply))` to the `FftActor`.
//! 4. Downloads the output bytes and reinterprets them as `f32` or
//!    `Complex32` for return.
//!
//! Inverse C2R is **not** normalized by 1/N (cuFFT contract); the
//! Python wrapper documents this and leaves normalization to the
//! caller (typically a downstream kernel or `numpy` divide).
//!
//! TODO Phase 1.5++ followups:
//! * Typed `GpuBufferC64` / `GpuBufferC128` — requires
//!   `CudaDtype for cufft_sys::float2/double2` upstream.
//! * f64 / 2-D / 3-D / plan-many / callback-mode coverage via a
//!   single `exec_typed` method (deferred until typed complex buffers
//!   land).
//! * Plan-cache stats / explicit plan handles surfaced to Python.
//! * RTC + multi-GPU FFT.

#![cfg(feature = "cufft")]

use std::time::Duration;

use numpy::{Complex32, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::device::{DeviceMsg, HostBuf};
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::{FftDirection, FftKind, FftMsg, FftRequest, PlanKey};
use atomr_core::actor::ActorRef;

use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Fft", module = "atomr_accel._native")]
pub struct PyFft {
    #[allow(dead_code)]
    actor_ref: ActorRef<FftMsg>,
    /// Device actor — used to allocate the temporary `u8` byte
    /// buffers each 1-shot FFT method needs as scratch input/output.
    device_ref: ActorRef<DeviceMsg>,
}

impl PyFft {
    pub fn new(actor_ref: ActorRef<FftMsg>, device_ref: ActorRef<DeviceMsg>) -> Self {
        Self {
            actor_ref,
            device_ref,
        }
    }
}

/// Map a direction string ('forward' / 'inverse') into the typed
/// `FftDirection`. Case-insensitive.
fn direction_from_str(s: &str) -> PyResult<FftDirection> {
    match s.to_ascii_lowercase().as_str() {
        "forward" | "fwd" | "f" => Ok(FftDirection::Forward),
        "inverse" | "inv" | "backward" | "i" => Ok(FftDirection::Inverse),
        other => Err(errors::map_str(format!(
            "direction must be 'forward' or 'inverse' (got {other:?})"
        ))),
    }
}

/// Allocate `len` `u8` bytes on-device through the shared runtime,
/// blocking on the actor reply.
fn alloc_u8_blocking(
    py: Python<'_>,
    device: &ActorRef<DeviceMsg>,
    len: usize,
    timeout_secs: f64,
) -> PyResult<GpuRef<u8>> {
    let actor = device.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::alloc::<u8>(len, tx));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(g))) => Ok(g),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("fft alloc timed out")),
            }
        })
    })
}

/// Upload a `Vec<u8>` into a previously-allocated `GpuRef<u8>`.
fn upload_bytes_blocking(
    py: Python<'_>,
    device: &ActorRef<DeviceMsg>,
    dst: GpuRef<u8>,
    bytes: Vec<u8>,
    timeout_secs: f64,
) -> PyResult<GpuRef<u8>> {
    let actor = device.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            let dst_clone = dst.clone();
            actor.tell(DeviceMsg::copy_from_host::<u8>(
                HostBuf::Owned(bytes),
                dst,
                tx,
            ));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(_))) => Ok(dst_clone),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("fft upload timed out")),
            }
        })
    })
}

/// Download `len` bytes from a `GpuRef<u8>` into a fresh `Vec<u8>`.
fn download_bytes_blocking(
    py: Python<'_>,
    device: &ActorRef<DeviceMsg>,
    src: GpuRef<u8>,
    len: usize,
    timeout_secs: f64,
) -> PyResult<Vec<u8>> {
    let actor = device.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::copy_to_host::<u8>(
                src,
                HostBuf::Owned(vec![0u8; len]),
                tx,
            ));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(HostBuf::Owned(v)))) => Ok(v),
                Ok(Ok(Ok(_))) => Err(errors::map_str("unexpected pinned reply")),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("fft download timed out")),
            }
        })
    })
}

/// Execute an FFT through the typed `FftMsg::Exec` path. The caller
/// has already populated `src_u8` with the input bytes; on return
/// `dst_u8` holds the transform output bytes (which the caller
/// downloads + reinterprets).
fn exec_fft_blocking(
    py: Python<'_>,
    fft: &ActorRef<FftMsg>,
    plan_key: PlanKey,
    direction: FftDirection,
    src_u8: GpuRef<u8>,
    dst_u8: GpuRef<u8>,
    timeout_secs: f64,
) -> PyResult<()> {
    let actor = fft.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            let req = FftRequest::<f32>::new(plan_key, direction, src_u8, dst_u8, tx);
            actor.tell(FftMsg::Exec(Box::new(req)));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("fft dropped reply")),
                Err(_) => Err(errors::map_str("fft exec timed out")),
            }
        })
    })
}

/// Reinterpret a `Vec<u8>` of length `len * 4` as a `Vec<f32>`. The
/// length is asserted; misalignment isn't possible because we own
/// the bytes (an owned `Vec` always satisfies the alignment of any
/// type whose size divides the allocator's chunk size — but to be
/// safe against future allocator changes we copy element-wise via
/// `from_le_bytes`).
fn bytes_to_f32_vec(bytes: Vec<u8>) -> Vec<f32> {
    debug_assert!(bytes.len() % 4 == 0);
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Reinterpret a `Vec<u8>` of length `len * 8` as a `Vec<Complex32>`.
fn bytes_to_complex32_vec(bytes: Vec<u8>) -> Vec<Complex32> {
    debug_assert!(bytes.len() % 8 == 0);
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(8) {
        let re = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let im = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        out.push(Complex32::new(re, im));
    }
    out
}

/// Reinterpret a `&[f32]` as bytes (little-endian, host-native — but
/// CUDA uses host-native, so the round-trip is fine on x86_64).
fn f32_slice_to_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for v in src {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

/// Reinterpret a `&[Complex32]` as bytes.
fn complex32_slice_to_bytes(src: &[Complex32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 8);
    for c in src {
        out.extend_from_slice(&c.re.to_ne_bytes());
        out.extend_from_slice(&c.im.to_ne_bytes());
    }
    out
}

#[pymethods]
impl PyFft {
    fn __repr__(&self) -> &'static str {
        "Fft(handle)"
    }

    /// 1-D real → complex forward FFT (f32 in, complex64 out).
    ///
    /// Args:
    ///     src: `numpy.float32` array of length `n * batch`. Layout
    ///         is interleaved per batch: `[batch0_n_samples,
    ///         batch1_n_samples, ...]`.
    ///     n: transform length (samples per batch).
    ///     batch: number of independent transforms (default 1).
    ///
    /// Returns:
    ///     `numpy.complex64` array of length `(n // 2 + 1) * batch`.
    ///     cuFFT R2C produces a non-redundant Hermitian half-spectrum.
    #[pyo3(signature = (src, n=None, batch=1, timeout_secs=10.0))]
    fn forward_1d_r2c_f32<'py>(
        &self,
        py: Python<'py>,
        src: PyReadonlyArray1<'_, f32>,
        n: Option<i32>,
        batch: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let host_src: &[f32] = src.as_slice().map_err(errors::map_str)?;
        let total_in = host_src.len();
        if batch <= 0 {
            return Err(errors::map_str("batch must be >= 1"));
        }
        let n: i32 = match n {
            Some(v) => v,
            None => {
                if total_in % batch as usize != 0 {
                    return Err(errors::map_str(format!(
                        "src length {} not divisible by batch {}",
                        total_in, batch
                    )));
                }
                (total_in / batch as usize) as i32
            }
        };
        if n <= 0 {
            return Err(errors::map_str("n must be >= 1"));
        }
        let n_per_batch = n as usize;
        let expected_in = n_per_batch * batch as usize;
        if total_in != expected_in {
            return Err(errors::map_str(format!(
                "src length {} != n * batch ({} * {} = {})",
                total_in, n, batch, expected_in
            )));
        }
        let n_out_per_batch = (n_per_batch / 2) + 1;
        let total_out = n_out_per_batch * batch as usize;

        // Bytes: f32 input is `total_in * 4`; complex64 output is
        // `total_out * 8`.
        let in_bytes_len = total_in * 4;
        let out_bytes_len = total_out * 8;

        let in_bytes = f32_slice_to_bytes(host_src);
        let plan_key = PlanKey::plan_1d(n, FftKind::R2C, batch);

        let src_u8 = alloc_u8_blocking(py, &self.device_ref, in_bytes_len, timeout_secs)?;
        let dst_u8 = alloc_u8_blocking(py, &self.device_ref, out_bytes_len, timeout_secs)?;
        let src_u8 = upload_bytes_blocking(py, &self.device_ref, src_u8, in_bytes, timeout_secs)?;
        exec_fft_blocking(
            py,
            &self.actor_ref,
            plan_key,
            FftDirection::Forward,
            src_u8,
            dst_u8.clone(),
            timeout_secs,
        )?;
        let out_bytes =
            download_bytes_blocking(py, &self.device_ref, dst_u8, out_bytes_len, timeout_secs)?;
        let out_vec = bytes_to_complex32_vec(out_bytes);
        Ok(PyArray1::from_vec_bound(py, out_vec))
    }

    /// 1-D complex → real inverse FFT (complex64 in, f32 out).
    ///
    /// Caller is responsible for 1/N normalization (cuFFT does NOT
    /// normalize inverse transforms).
    ///
    /// Args:
    ///     src: `numpy.complex64` array of length `(n // 2 + 1) *
    ///         batch`. The Hermitian half-spectrum.
    ///     n: real-domain transform length (samples per batch).
    ///         Required — cannot be inferred from `src` alone since
    ///         R2C drops the redundant half.
    ///     batch: number of independent transforms (default 1).
    ///
    /// Returns:
    ///     `numpy.float32` array of length `n * batch`.
    #[pyo3(signature = (src, n, batch=1, timeout_secs=10.0))]
    fn inverse_1d_c2r_f32<'py>(
        &self,
        py: Python<'py>,
        src: PyReadonlyArray1<'_, Complex32>,
        n: i32,
        batch: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if n <= 0 {
            return Err(errors::map_str("n must be >= 1"));
        }
        if batch <= 0 {
            return Err(errors::map_str("batch must be >= 1"));
        }
        let host_src: &[Complex32] = src.as_slice().map_err(errors::map_str)?;
        let n_per_batch = n as usize;
        let n_in_per_batch = (n_per_batch / 2) + 1;
        let expected_in = n_in_per_batch * batch as usize;
        if host_src.len() != expected_in {
            return Err(errors::map_str(format!(
                "src length {} != (n/2+1) * batch ({} * {} = {})",
                host_src.len(),
                n_in_per_batch,
                batch,
                expected_in
            )));
        }
        let total_out = n_per_batch * batch as usize;
        let in_bytes_len = expected_in * 8;
        let out_bytes_len = total_out * 4;

        let in_bytes = complex32_slice_to_bytes(host_src);
        let plan_key = PlanKey::plan_1d(n, FftKind::C2R, batch);

        let src_u8 = alloc_u8_blocking(py, &self.device_ref, in_bytes_len, timeout_secs)?;
        let dst_u8 = alloc_u8_blocking(py, &self.device_ref, out_bytes_len, timeout_secs)?;
        let src_u8 = upload_bytes_blocking(py, &self.device_ref, src_u8, in_bytes, timeout_secs)?;
        exec_fft_blocking(
            py,
            &self.actor_ref,
            plan_key,
            FftDirection::Inverse,
            src_u8,
            dst_u8.clone(),
            timeout_secs,
        )?;
        let out_bytes =
            download_bytes_blocking(py, &self.device_ref, dst_u8, out_bytes_len, timeout_secs)?;
        let out_vec = bytes_to_f32_vec(out_bytes);
        Ok(PyArray1::from_vec_bound(py, out_vec))
    }

    /// 1-D complex ↔ complex FFT (complex64 in/out).
    ///
    /// Args:
    ///     src: `numpy.complex64` array of length `n * batch`.
    ///     direction: `'forward'` or `'inverse'`.
    ///     n: transform length per batch (default: inferred from
    ///         `src.len() // batch`).
    ///     batch: number of independent transforms (default 1).
    ///
    /// Returns:
    ///     `numpy.complex64` array of length `n * batch`. Inverse is
    ///     NOT normalized by 1/N.
    #[pyo3(signature = (src, direction="forward", n=None, batch=1, timeout_secs=10.0))]
    fn exec_1d_c2c_f32<'py>(
        &self,
        py: Python<'py>,
        src: PyReadonlyArray1<'_, Complex32>,
        direction: &str,
        n: Option<i32>,
        batch: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let dir = direction_from_str(direction)?;
        if batch <= 0 {
            return Err(errors::map_str("batch must be >= 1"));
        }
        let host_src: &[Complex32] = src.as_slice().map_err(errors::map_str)?;
        let total = host_src.len();
        let n: i32 = match n {
            Some(v) => v,
            None => {
                if total % batch as usize != 0 {
                    return Err(errors::map_str(format!(
                        "src length {} not divisible by batch {}",
                        total, batch
                    )));
                }
                (total / batch as usize) as i32
            }
        };
        if n <= 0 {
            return Err(errors::map_str("n must be >= 1"));
        }
        let expected = n as usize * batch as usize;
        if total != expected {
            return Err(errors::map_str(format!(
                "src length {} != n * batch ({} * {} = {})",
                total, n, batch, expected
            )));
        }
        let bytes_len = total * 8;

        let in_bytes = complex32_slice_to_bytes(host_src);
        let plan_key = PlanKey::plan_1d(n, FftKind::C2C, batch);

        let src_u8 = alloc_u8_blocking(py, &self.device_ref, bytes_len, timeout_secs)?;
        let dst_u8 = alloc_u8_blocking(py, &self.device_ref, bytes_len, timeout_secs)?;
        let src_u8 = upload_bytes_blocking(py, &self.device_ref, src_u8, in_bytes, timeout_secs)?;
        exec_fft_blocking(
            py,
            &self.actor_ref,
            plan_key,
            dir,
            src_u8,
            dst_u8.clone(),
            timeout_secs,
        )?;
        let out_bytes =
            download_bytes_blocking(py, &self.device_ref, dst_u8, bytes_len, timeout_secs)?;
        let out_vec = bytes_to_complex32_vec(out_bytes);
        Ok(PyArray1::from_vec_bound(py, out_vec))
    }

    /// 2-D real → complex forward FFT (f32 in, complex64 out).
    ///
    /// Args:
    ///     src: `numpy.float32` array of length `nx * ny`, row-major.
    ///     nx: number of rows (slow dim).
    ///     ny: number of columns (fast dim).
    ///
    /// Returns:
    ///     `numpy.complex64` array of length `nx * (ny // 2 + 1)`,
    ///     row-major. cuFFT 2-D R2C produces a Hermitian
    ///     half-spectrum along the *fast* dim.
    #[pyo3(signature = (src, nx, ny, timeout_secs=10.0))]
    fn forward_2d_r2c_f32<'py>(
        &self,
        py: Python<'py>,
        src: PyReadonlyArray1<'_, f32>,
        nx: i32,
        ny: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        if nx <= 0 || ny <= 0 {
            return Err(errors::map_str("nx and ny must be >= 1"));
        }
        let host_src: &[f32] = src.as_slice().map_err(errors::map_str)?;
        let total_in = (nx as usize) * (ny as usize);
        if host_src.len() != total_in {
            return Err(errors::map_str(format!(
                "src length {} != nx * ny ({} * {} = {})",
                host_src.len(),
                nx,
                ny,
                total_in
            )));
        }
        let total_out = (nx as usize) * ((ny as usize) / 2 + 1);
        let in_bytes_len = total_in * 4;
        let out_bytes_len = total_out * 8;

        let in_bytes = f32_slice_to_bytes(host_src);
        let plan_key = PlanKey::plan_2d(nx, ny, FftKind::R2C);

        let src_u8 = alloc_u8_blocking(py, &self.device_ref, in_bytes_len, timeout_secs)?;
        let dst_u8 = alloc_u8_blocking(py, &self.device_ref, out_bytes_len, timeout_secs)?;
        let src_u8 = upload_bytes_blocking(py, &self.device_ref, src_u8, in_bytes, timeout_secs)?;
        exec_fft_blocking(
            py,
            &self.actor_ref,
            plan_key,
            FftDirection::Forward,
            src_u8,
            dst_u8.clone(),
            timeout_secs,
        )?;
        let out_bytes =
            download_bytes_blocking(py, &self.device_ref, dst_u8, out_bytes_len, timeout_secs)?;
        let out_vec = bytes_to_complex32_vec(out_bytes);
        Ok(PyArray1::from_vec_bound(py, out_vec))
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFft>()?;
    Ok(())
}
