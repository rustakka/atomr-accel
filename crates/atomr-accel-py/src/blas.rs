//! `Blas` — Python handle wrapping `ActorRef<BlasMsg>`.
//!
//! Obtained via `Device.blas()`. Every method awaits the actor reply
//! through the shared tokio runtime with the GIL released, mapping
//! `GpuError` into the typed exception hierarchy.
//!
//! Phase 1 surface — `gemm_f32`, `gemm_f64`, and `axpy_f32` exercise
//! the typed dispatch path end-to-end (mock-mode replies surface as
//! `Unrecoverable`). Strided-batched gemm, the rest of L1 (dot, nrm2,
//! scal, asum, …), L2 (gemv, ger), and L3 (geam, syrk, trsm) follow
//! in the Phase 1.5 tracking issue.

use std::time::Duration;

use cudarc::cublas::sys::cublasOperation_t;
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{AxpyRequest, BlasMsg, GemmRequest};
use atomr_core::actor::ActorRef;

use crate::buffer::{PyGpuBufferF32, PyGpuBufferF64};
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Blas", module = "atomr_accel._native")]
pub struct PyBlas {
    actor_ref: ActorRef<BlasMsg>,
}

impl PyBlas {
    pub fn new(actor_ref: ActorRef<BlasMsg>) -> Self {
        Self { actor_ref }
    }
}

fn op_from_str(s: &str) -> PyResult<cublasOperation_t> {
    match s.to_ascii_uppercase().as_str() {
        "N" => Ok(cublasOperation_t::CUBLAS_OP_N),
        "T" => Ok(cublasOperation_t::CUBLAS_OP_T),
        "C" => Ok(cublasOperation_t::CUBLAS_OP_C),
        _ => Err(errors::map_str(format!(
            "trans must be 'N', 'T', or 'C' (got {s:?})"
        ))),
    }
}

#[pymethods]
impl PyBlas {
    /// f32 GEMM: `c = alpha * op(a) · op(b) + beta * c`. `a` is
    /// `m × k` (or `k × m` if `trans_a='T'`), `b` is `k × n`, `c` is
    /// `m × n`. Strides default to natural column-major leading
    /// dimensions; pass explicit `lda`/`ldb`/`ldc` for slicing.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        trans_a: &str,
        trans_b: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let trans_a = op_from_str(trans_a)?;
        let trans_b = op_from_str(trans_b)?;
        let lda = lda.unwrap_or(if trans_a == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            k
        });
        let ldb = ldb.unwrap_or(if trans_b == cublasOperation_t::CUBLAS_OP_N {
            k
        } else {
            n
        });
        let ldc = ldc.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(BlasMsg::gemm::<f32>(GemmRequest::<f32> {
                    a,
                    b,
                    c,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                    lda,
                    ldb,
                    ldc,
                    reply: tx,
                }));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("gemm_f32 timed out")),
                }
            })
        })
    }

    /// f64 GEMM. Same semantics as `gemm_f32` but operating on `f64`
    /// buffers.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        beta: f64,
        trans_a: &str,
        trans_b: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let trans_a = op_from_str(trans_a)?;
        let trans_b = op_from_str(trans_b)?;
        let lda = lda.unwrap_or(if trans_a == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            k
        });
        let ldb = ldb.unwrap_or(if trans_b == cublasOperation_t::CUBLAS_OP_N {
            k
        } else {
            n
        });
        let ldc = ldc.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(BlasMsg::gemm::<f64>(GemmRequest::<f64> {
                    a,
                    b,
                    c,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                    lda,
                    ldb,
                    ldc,
                    reply: tx,
                }));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("gemm_f64 timed out")),
                }
            })
        })
    }

    /// f32 AXPY: `y = alpha * x + y`. Demonstrates the L1 dispatch
    /// path; full L1/L2/L3 coverage is tracked as a Phase 1.5 issue.
    #[pyo3(signature = (alpha, x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn axpy_f32(
        &self,
        py: Python<'_>,
        alpha: f32,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = AxpyRequest::<f32> {
                    n,
                    alpha,
                    x,
                    incx,
                    y,
                    incy,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("axpy_f32 timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "Blas(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlas>()?;
    Ok(())
}
