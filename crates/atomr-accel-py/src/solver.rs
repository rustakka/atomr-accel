//! `Solver` — Python handle wrapping `ActorRef<SolverMsg>`.
//!
//! Obtained via `Device.solver()` (only when the `cusolver` feature is
//! compiled in *and* the device's `EnabledLibraries::CUSOLVER` flag
//! is set). Phase 1.5++ exposes the dense cuSOLVER op surface:
//!
//! - LU factorize / solve (`getrf` / `getrs`) — `lu_{f32,f64}` and
//!   `lu_solve_{f32,f64}`.
//! - Cholesky (`potrf`) — `cholesky_{f32,f64}`.
//! - QR (`geqrf`) — `qr_{f32,f64}`.
//! - SVD (`gesvd`) — `svd_{f32,f64}`.
//! - Symmetric eigendecomposition (`syevd`) — `eigh_{f32,f64}`.
//!
//! Each method awaits the actor reply through the shared tokio runtime
//! with the GIL released, mapping `GpuError` to the typed exception
//! hierarchy. The mock-mode `SolverActor` replies `Unrecoverable` for
//! every dispatch, so test_solver.py exercises the routing without
//! touching real hardware.

#![cfg(feature = "cusolver")]

use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{
    CholeskyRequest, LuRequest, LuSolveRequest, QrRequest, SolverMsg, SvdRequest, SyevdRequest,
    Uplo,
};
use atomr_core::actor::ActorRef;

use crate::buffer::{PyGpuBufferF32, PyGpuBufferF64, PyGpuBufferI32};
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Solver", module = "atomr_accel._native")]
pub struct PySolver {
    actor_ref: ActorRef<SolverMsg>,
}

impl PySolver {
    pub fn new(actor_ref: ActorRef<SolverMsg>) -> Self {
        Self { actor_ref }
    }
}

/// Map a user-facing string (case-insensitive) to a [`Uplo`] storage
/// triangle. Mirrors `fill_from_str` in `blas.rs`. Accepts the short
/// form (`"U"`/`"L"`) and the long form (`"UPPER"`/`"LOWER"`).
fn uplo_from_str(s: &str) -> PyResult<Uplo> {
    match s.to_ascii_uppercase().as_str() {
        "U" | "UPPER" => Ok(Uplo::Upper),
        "L" | "LOWER" => Ok(Uplo::Lower),
        _ => Err(errors::map_str(format!(
            "uplo must be 'U'/'UPPER' or 'L'/'LOWER' (got {s:?})"
        ))),
    }
}

#[pymethods]
impl PySolver {
    fn __repr__(&self) -> &'static str {
        "Solver(handle)"
    }

    // ─── LU factorize (getrf) ───────────────────────────────────────

    /// f32 LU factorization in place: `a = P · L · U`. Pivots are
    /// written to `ipiv` (length `min(m, n)`). `lda` defaults to `m`.
    #[pyo3(signature = (a, ipiv, m, n, lda=None, timeout_secs=60.0))]
    fn lu_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        ipiv: Py<PyGpuBufferI32>,
        m: i32,
        n: i32,
        lda: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda; // dense::run_lu hard-codes lda=m today; reserved for future use.
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let ipiv = ipiv
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("ipiv consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(LuRequest::<f32> {
                    a,
                    m,
                    n,
                    ipiv,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("lu_f32 timed out")),
                }
            })
        })
    }

    /// f64 LU factorization. Same semantics as `lu_f32`.
    #[pyo3(signature = (a, ipiv, m, n, lda=None, timeout_secs=60.0))]
    fn lu_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        ipiv: Py<PyGpuBufferI32>,
        m: i32,
        n: i32,
        lda: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let ipiv = ipiv
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("ipiv consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(LuRequest::<f64> {
                    a,
                    m,
                    n,
                    ipiv,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("lu_f64 timed out")),
                }
            })
        })
    }

    // ─── LU solve (getrs) ───────────────────────────────────────────

    /// f32 LU solve: `b ← op(LU)⁻¹ · b`. Pass `trans=True` for
    /// transposed solves.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (lu, ipiv, b, n, nrhs, ldb=None, trans=false, timeout_secs=60.0))]
    fn lu_solve_f32(
        &self,
        py: Python<'_>,
        lu: Py<PyGpuBufferF32>,
        ipiv: Py<PyGpuBufferI32>,
        b: Py<PyGpuBufferF32>,
        n: i32,
        nrhs: i32,
        ldb: Option<i32>,
        trans: bool,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = ldb;
        let lu = lu
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("lu consumed"))?;
        let ipiv = ipiv
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("ipiv consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(LuSolveRequest::<f32> {
                    lu,
                    ipiv,
                    b,
                    n,
                    nrhs,
                    trans,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("lu_solve_f32 timed out")),
                }
            })
        })
    }

    /// f64 LU solve. Same semantics as `lu_solve_f32`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (lu, ipiv, b, n, nrhs, ldb=None, trans=false, timeout_secs=60.0))]
    fn lu_solve_f64(
        &self,
        py: Python<'_>,
        lu: Py<PyGpuBufferF64>,
        ipiv: Py<PyGpuBufferI32>,
        b: Py<PyGpuBufferF64>,
        n: i32,
        nrhs: i32,
        ldb: Option<i32>,
        trans: bool,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = ldb;
        let lu = lu
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("lu consumed"))?;
        let ipiv = ipiv
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("ipiv consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(LuSolveRequest::<f64> {
                    lu,
                    ipiv,
                    b,
                    n,
                    nrhs,
                    trans,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("lu_solve_f64 timed out")),
                }
            })
        })
    }

    // ─── Cholesky (potrf) ───────────────────────────────────────────

    /// f32 Cholesky: `a ← chol(a)`. Reads/writes the triangle selected
    /// by `uplo` (`'U'` or `'L'`, default `'U'`).
    #[pyo3(signature = (a, n, uplo="U", lda=None, timeout_secs=60.0))]
    fn cholesky_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        n: i32,
        uplo: &str,
        lda: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let uplo = uplo_from_str(uplo)?;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(CholeskyRequest::<f32> {
                    a,
                    n,
                    uplo,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("cholesky_f32 timed out")),
                }
            })
        })
    }

    /// f64 Cholesky. Same semantics as `cholesky_f32`.
    #[pyo3(signature = (a, n, uplo="U", lda=None, timeout_secs=60.0))]
    fn cholesky_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        n: i32,
        uplo: &str,
        lda: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let uplo = uplo_from_str(uplo)?;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(CholeskyRequest::<f64> {
                    a,
                    n,
                    uplo,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("cholesky_f64 timed out")),
                }
            })
        })
    }

    // ─── QR (geqrf) ────────────────────────────────────────────────

    /// f32 QR factorization: writes the upper-triangular `R` into the
    /// upper triangle of `a` and the elementary reflectors into the
    /// strict lower triangle, with the scalar factors in `tau`
    /// (length `min(m, n)`).
    #[pyo3(signature = (a, tau, m, n, lda=None, timeout_secs=60.0))]
    fn qr_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        tau: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        lda: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let tau = tau
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("tau consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(QrRequest::<f32> {
                    a,
                    m,
                    n,
                    tau,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("qr_f32 timed out")),
                }
            })
        })
    }

    /// f64 QR factorization. Same semantics as `qr_f32`.
    #[pyo3(signature = (a, tau, m, n, lda=None, timeout_secs=60.0))]
    fn qr_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        tau: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        lda: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let tau = tau
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("tau consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(QrRequest::<f64> {
                    a,
                    m,
                    n,
                    tau,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("qr_f64 timed out")),
                }
            })
        })
    }

    // ─── SVD (gesvd) ────────────────────────────────────────────────

    /// f32 SVD: `a = U · diag(s) · Vᵀ`. `s` is `min(m, n)` singular
    /// values. `u` and `vt` are optional — pass `None` to skip
    /// computing the corresponding factor (equivalent to `jobu='N'` /
    /// `jobvt='N'`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, s, m, n, u=None, vt=None, timeout_secs=60.0))]
    fn svd_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        s: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        u: Option<Py<PyGpuBufferF32>>,
        vt: Option<Py<PyGpuBufferF32>>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let s = s
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("s consumed"))?;
        let u = match u {
            Some(b) => Some(
                b.borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("u consumed"))?,
            ),
            None => None,
        };
        let vt = match vt {
            Some(b) => Some(
                b.borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("vt consumed"))?,
            ),
            None => None,
        };
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(SvdRequest::<f32> {
                    a,
                    m,
                    n,
                    s,
                    u,
                    vt,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("svd_f32 timed out")),
                }
            })
        })
    }

    /// f64 SVD. Same semantics as `svd_f32`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, s, m, n, u=None, vt=None, timeout_secs=60.0))]
    fn svd_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        s: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        u: Option<Py<PyGpuBufferF64>>,
        vt: Option<Py<PyGpuBufferF64>>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let s = s
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("s consumed"))?;
        let u = match u {
            Some(b) => Some(
                b.borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("u consumed"))?,
            ),
            None => None,
        };
        let vt = match vt {
            Some(b) => Some(
                b.borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("vt consumed"))?,
            ),
            None => None,
        };
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(SvdRequest::<f64> {
                    a,
                    m,
                    n,
                    s,
                    u,
                    vt,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("svd_f64 timed out")),
                }
            })
        })
    }

    // ─── Symmetric eigendecomposition (syevd) ───────────────────────

    /// f32 symmetric eigendecomposition: `a · v = v · diag(w)` for
    /// symmetric `a`. Eigenvalues land in `w` (length `n`); when
    /// `compute_vectors=True` the eigenvectors overwrite `a`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, w, n, uplo="L", lda=None, compute_vectors=true, timeout_secs=60.0))]
    fn eigh_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        w: Py<PyGpuBufferF32>,
        n: i32,
        uplo: &str,
        lda: Option<i32>,
        compute_vectors: bool,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let uplo = uplo_from_str(uplo)?;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let w = w
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("w consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(SyevdRequest::<f32> {
                    a,
                    n,
                    uplo,
                    w,
                    compute_vectors,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("eigh_f32 timed out")),
                }
            })
        })
    }

    /// f64 symmetric eigendecomposition. Same semantics as `eigh_f32`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, w, n, uplo="L", lda=None, compute_vectors=true, timeout_secs=60.0))]
    fn eigh_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        w: Py<PyGpuBufferF64>,
        n: i32,
        uplo: &str,
        lda: Option<i32>,
        compute_vectors: bool,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let _ = lda;
        let uplo = uplo_from_str(uplo)?;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let w = w
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("w consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SolverMsg::Op(Box::new(SyevdRequest::<f64> {
                    a,
                    n,
                    uplo,
                    w,
                    compute_vectors,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("solver dropped reply")),
                    Err(_) => Err(errors::map_str("eigh_f64 timed out")),
                }
            })
        })
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolver>()?;
    Ok(())
}
