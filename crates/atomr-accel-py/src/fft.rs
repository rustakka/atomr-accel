//! `Fft` — Python handle wrapping `ActorRef<FftMsg>`.
//!
//! Obtained via `Device.fft()` (only when the `cufft` feature is
//! compiled in *and* the device's `EnabledLibraries::CUFFT` flag is
//! set).
//!
//! # Phase 1.5++ — cuFFT depth, two paths
//!
//! ## Path B — host-driven 1-shot FFTs (kept for ergonomics)
//!
//! The four 1-shot methods (`forward_1d_r2c_f32`, `inverse_1d_c2r_f32`,
//! `exec_1d_c2c_f32`, `forward_2d_r2c_f32`) take a `numpy` host array,
//! allocate scratch device byte buffers, upload, dispatch via
//! `FftMsg::Exec(FftRequest::<f32, u8, u8>::new(...))`, then download.
//! Convenient for one-off transforms; less efficient than reusing
//! pinned GPU buffers across calls.
//!
//! ## Path A — typed Complex GPU buffers
//!
//! `exec_typed_f32` (and the f64 counterpart) take pre-allocated
//! typed buffers (`GpuBufferF32` for the real lane, `GpuBufferC64`
//! for the complex lane; `GpuBufferF64` / `GpuBufferC128` for the f64
//! lane). The dispatch goes through
//! `FftMsg::Exec(FftRequest::<T, I, O>::new(...))` directly — no
//! host-side scratch alloc, no byte marshalling. Reuse the same
//! buffers across many calls (and pair with `Device.allocate_*` to
//! keep VRAM steady-state).
//!
//! Inverse C2R is **not** normalized by 1/N (cuFFT contract); the
//! Python wrapper documents this and leaves normalization to the
//! caller (typically a downstream kernel or `numpy` divide).
//!
//! TODO Phase 1.5++ followups:
//! * Plan-cache stats / explicit plan handles surfaced to Python.
//! * RTC + multi-GPU FFT.
//! * 2-D / 3-D + plan-many for `exec_typed_*` (the plan-key API
//!   already supports them; we just need richer Python signatures).

#![cfg(feature = "cufft")]

use std::time::Duration;

use numpy::{Complex32, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::device::{DeviceMsg, HostBuf};
use atomr_accel_cuda::dtype::{C32, C64};
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::{FftDirection, FftKind, FftMsg, FftRequest, PlanKey};
use atomr_core::actor::ActorRef;

use crate::buffer::{
    PyGpuBufferC128, PyGpuBufferC64, PyGpuBufferF32, PyGpuBufferF64,
};
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

/// Map a transform-kind string ('r2c' / 'c2r' / 'c2c' / 'd2z' / 'z2d'
/// / 'z2z') into [`FftKind`]. Case-insensitive.
fn kind_from_str(s: &str) -> PyResult<FftKind> {
    match s.to_ascii_lowercase().as_str() {
        "r2c" => Ok(FftKind::R2C),
        "c2r" => Ok(FftKind::C2R),
        "c2c" => Ok(FftKind::C2C),
        "d2z" => Ok(FftKind::D2Z),
        "z2d" => Ok(FftKind::Z2D),
        "z2z" => Ok(FftKind::Z2Z),
        other => Err(errors::map_str(format!(
            "kind must be one of r2c|c2r|c2c|d2z|z2d|z2z (got {other:?})"
        ))),
    }
}

/// Build a [`PlanKey`] from rank + dims + kind + batch. `rank` is
/// inferred from how many of `nx`/`ny`/`nz` are positive.
fn plan_key_from_dims(
    nx: i32,
    ny: Option<i32>,
    nz: Option<i32>,
    kind: FftKind,
    batch: i32,
) -> PyResult<PlanKey> {
    if nx <= 0 {
        return Err(errors::map_str("nx must be >= 1"));
    }
    if batch <= 0 {
        return Err(errors::map_str("batch must be >= 1"));
    }
    Ok(match (ny, nz) {
        (None, _) => PlanKey::plan_1d(nx, kind, batch),
        (Some(ny), None) => {
            if ny <= 0 {
                return Err(errors::map_str("ny must be >= 1"));
            }
            PlanKey::plan_2d(nx, ny, kind)
        }
        (Some(ny), Some(nz)) => {
            if ny <= 0 || nz <= 0 {
                return Err(errors::map_str("ny and nz must be >= 1"));
            }
            PlanKey::plan_3d(nx, ny, nz, kind)
        }
    })
}

/// Generic typed-buffer FFT dispatch — Path A. Both buffers are
/// already-typed `GpuRef`s on-device; no host scratch is allocated.
/// `T` is the scalar lane (`f32` or `f64`), `I`/`O` are the per-side
/// element types.
fn exec_typed_blocking<T, I, O>(
    py: Python<'_>,
    fft: &ActorRef<FftMsg>,
    plan_key: PlanKey,
    direction: FftDirection,
    input: GpuRef<I>,
    output: GpuRef<O>,
    timeout_secs: f64,
) -> PyResult<()>
where
    T: atomr_accel_cuda::dtype::FftSupported,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    let actor = fft.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            let req = FftRequest::<T, I, O>::new(plan_key, direction, input, output, tx);
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

    /// Phase 1.5++ Path A — typed-buffer cuFFT dispatch on the f32
    /// scalar lane (R2C / C2R / C2C). Takes pre-allocated typed
    /// `GpuBufferF32` (real lane) and `GpuBufferC64` (complex lane)
    /// instead of host arrays — no scratch alloc, no byte marshalling.
    ///
    /// Args:
    ///     kind: one of ``"r2c"``, ``"c2r"``, ``"c2c"``.
    ///     real_buf: `GpuBufferF32`. Source for ``r2c``, destination
    ///         for ``c2r``. Pass `None` for ``c2c`` (the complex
    ///         source/dest pair drives that path).
    ///     complex_buf: `GpuBufferC64`. Destination for ``r2c``,
    ///         source for ``c2r``, both for ``c2c`` if
    ///         ``complex_buf_out`` is omitted (in-place).
    ///     complex_buf_out: optional second `GpuBufferC64`. Only used
    ///         by ``c2c`` to override the destination (otherwise
    ///         ``complex_buf`` is the in-place input + output).
    ///     direction: ``"forward"`` (default) or ``"inverse"``. Only
    ///         meaningful for ``c2c``.
    ///     nx, ny, nz: transform dimensions. ``ny`` / ``nz`` are
    ///         optional — `None` means a 1-D / 2-D plan respectively.
    ///     batch: number of independent transforms (default 1, only
    ///         honored on 1-D plans).
    ///
    /// Returns:
    ///     `None`. The output buffer is mutated in-place.
    ///
    /// Caller is responsible for sizing the buffers correctly:
    /// R2C input length is `n_real * batch`; output is
    /// `(n_real // 2 + 1) * batch`. C2R is the reverse. C2C is
    /// `n_real * batch` on both sides. cuFFT does **not** normalize
    /// inverse transforms by 1/N.
    #[pyo3(signature = (
        kind,
        real_buf=None,
        complex_buf=None,
        complex_buf_out=None,
        direction="forward",
        nx=None,
        ny=None,
        nz=None,
        batch=1,
        timeout_secs=10.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn exec_typed_f32(
        &self,
        py: Python<'_>,
        kind: &str,
        real_buf: Option<Py<PyGpuBufferF32>>,
        complex_buf: Option<Py<PyGpuBufferC64>>,
        complex_buf_out: Option<Py<PyGpuBufferC64>>,
        direction: &str,
        nx: Option<i32>,
        ny: Option<i32>,
        nz: Option<i32>,
        batch: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let kind = kind_from_str(kind)?;
        let dir = direction_from_str(direction)?;
        // The f32 lane only covers R2C / C2R / C2C — D2Z / Z2D / Z2Z
        // belong to `exec_typed_f64`.
        match kind {
            FftKind::R2C | FftKind::C2R | FftKind::C2C => {}
            other => {
                return Err(errors::map_str(format!(
                    "exec_typed_f32: kind {:?} is not on the f32 lane (use exec_typed_f64)",
                    other
                )));
            }
        }
        let nx = nx.ok_or_else(|| errors::map_str("nx is required"))?;
        let plan_key = plan_key_from_dims(nx, ny, nz, kind, batch)?;

        match kind {
            FftKind::R2C => {
                let r = real_buf
                    .ok_or_else(|| errors::map_str("r2c requires real_buf (input)"))?;
                let c = complex_buf
                    .ok_or_else(|| errors::map_str("r2c requires complex_buf (output)"))?;
                let r_ref = r
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("real_buf consumed"))?;
                let c_ref = c
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("complex_buf consumed"))?;
                exec_typed_blocking::<f32, f32, C32>(
                    py,
                    &self.actor_ref,
                    plan_key,
                    dir,
                    r_ref,
                    c_ref,
                    timeout_secs,
                )
            }
            FftKind::C2R => {
                let c = complex_buf
                    .ok_or_else(|| errors::map_str("c2r requires complex_buf (input)"))?;
                let r = real_buf
                    .ok_or_else(|| errors::map_str("c2r requires real_buf (output)"))?;
                let c_ref = c
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("complex_buf consumed"))?;
                let r_ref = r
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("real_buf consumed"))?;
                exec_typed_blocking::<f32, C32, f32>(
                    py,
                    &self.actor_ref,
                    plan_key,
                    dir,
                    c_ref,
                    r_ref,
                    timeout_secs,
                )
            }
            FftKind::C2C => {
                let src = complex_buf
                    .ok_or_else(|| errors::map_str("c2c requires complex_buf (input)"))?;
                let src_ref = src
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("complex_buf consumed"))?;
                let dst_ref = match complex_buf_out {
                    Some(d) => d
                        .borrow(py)
                        .clone_ref()
                        .ok_or_else(|| errors::map_str("complex_buf_out consumed"))?,
                    // In-place: alias the source buffer.
                    None => src_ref.clone(),
                };
                exec_typed_blocking::<f32, C32, C32>(
                    py,
                    &self.actor_ref,
                    plan_key,
                    dir,
                    src_ref,
                    dst_ref,
                    timeout_secs,
                )
            }
            _ => unreachable!(),
        }
    }

    /// Phase 1.5++ Path A — typed-buffer cuFFT dispatch on the f64
    /// scalar lane (D2Z / Z2D / Z2Z). Mirrors [`Self::exec_typed_f32`]
    /// with `GpuBufferF64` / `GpuBufferC128` buffers.
    ///
    /// Args mirror `exec_typed_f32`; ``kind`` is one of
    /// ``"d2z"`` (real → complex), ``"z2d"`` (complex → real),
    /// ``"z2z"`` (complex ↔ complex).
    #[pyo3(signature = (
        kind,
        real_buf=None,
        complex_buf=None,
        complex_buf_out=None,
        direction="forward",
        nx=None,
        ny=None,
        nz=None,
        batch=1,
        timeout_secs=10.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn exec_typed_f64(
        &self,
        py: Python<'_>,
        kind: &str,
        real_buf: Option<Py<PyGpuBufferF64>>,
        complex_buf: Option<Py<PyGpuBufferC128>>,
        complex_buf_out: Option<Py<PyGpuBufferC128>>,
        direction: &str,
        nx: Option<i32>,
        ny: Option<i32>,
        nz: Option<i32>,
        batch: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let kind = kind_from_str(kind)?;
        let dir = direction_from_str(direction)?;
        match kind {
            FftKind::D2Z | FftKind::Z2D | FftKind::Z2Z => {}
            other => {
                return Err(errors::map_str(format!(
                    "exec_typed_f64: kind {:?} is not on the f64 lane (use exec_typed_f32)",
                    other
                )));
            }
        }
        let nx = nx.ok_or_else(|| errors::map_str("nx is required"))?;
        let plan_key = plan_key_from_dims(nx, ny, nz, kind, batch)?;

        match kind {
            FftKind::D2Z => {
                let r = real_buf
                    .ok_or_else(|| errors::map_str("d2z requires real_buf (input)"))?;
                let c = complex_buf
                    .ok_or_else(|| errors::map_str("d2z requires complex_buf (output)"))?;
                let r_ref = r
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("real_buf consumed"))?;
                let c_ref = c
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("complex_buf consumed"))?;
                exec_typed_blocking::<f64, f64, C64>(
                    py,
                    &self.actor_ref,
                    plan_key,
                    dir,
                    r_ref,
                    c_ref,
                    timeout_secs,
                )
            }
            FftKind::Z2D => {
                let c = complex_buf
                    .ok_or_else(|| errors::map_str("z2d requires complex_buf (input)"))?;
                let r = real_buf
                    .ok_or_else(|| errors::map_str("z2d requires real_buf (output)"))?;
                let c_ref = c
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("complex_buf consumed"))?;
                let r_ref = r
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("real_buf consumed"))?;
                exec_typed_blocking::<f64, C64, f64>(
                    py,
                    &self.actor_ref,
                    plan_key,
                    dir,
                    c_ref,
                    r_ref,
                    timeout_secs,
                )
            }
            FftKind::Z2Z => {
                let src = complex_buf
                    .ok_or_else(|| errors::map_str("z2z requires complex_buf (input)"))?;
                let src_ref = src
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("complex_buf consumed"))?;
                let dst_ref = match complex_buf_out {
                    Some(d) => d
                        .borrow(py)
                        .clone_ref()
                        .ok_or_else(|| errors::map_str("complex_buf_out consumed"))?,
                    None => src_ref.clone(),
                };
                exec_typed_blocking::<f64, C64, C64>(
                    py,
                    &self.actor_ref,
                    plan_key,
                    dir,
                    src_ref,
                    dst_ref,
                    timeout_secs,
                )
            }
            _ => unreachable!(),
        }
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
