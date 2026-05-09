//! `Blas` — Python handle wrapping `ActorRef<BlasMsg>`.
//!
//! Obtained via `Device.blas()`. Every method awaits the actor reply
//! through the shared tokio runtime with the GIL released, mapping
//! `GpuError` into the typed exception hierarchy.
//!
//! Phase 1.5 surface — full method-level cuBLAS coverage. Beyond
//! `gemm_f32` / `gemm_f64` / `axpy_f32`, the wrapper now exposes
//! strided-batched gemm (`gemm_strided_batched_{f32,f64}`) plus the
//! rest of L1 (`axpy_f64`, `dot_*`, `nrm2_*`, `scal_*`, `asum_*`,
//! `iamax_*`, `iamin_*`, `copy_*`, `swap_*`, `rot_*`), L2
//! (`gemv_*`, `ger_*`), and L3 (`geam_*`, `syrk_*`, `trsm_*`). All
//! variants follow the same pattern: borrow the buffers, build the
//! typed request, dispatch through the matching `BlasMsg::*` variant,
//! await on the shared runtime with the GIL released.

use std::time::Duration;

use cudarc::cublas::sys::{
    cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t,
};
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{
    AsumRequest, AxpyRequest, BlasMsg, CopyRequest, DotRequest, GeamRequest, GemmRequest,
    GemmStridedBatchedRequest, GemvRequest, GerRequest, IamaxRequest, IaminRequest, Nrm2Request,
    RotRequest, ScalRequest, SwapRequest, SyrkRequest, TrsmRequest,
};
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

fn fill_from_str(s: &str) -> PyResult<cublasFillMode_t> {
    match s.to_ascii_uppercase().as_str() {
        "L" | "LOWER" => Ok(cublasFillMode_t::CUBLAS_FILL_MODE_LOWER),
        "U" | "UPPER" => Ok(cublasFillMode_t::CUBLAS_FILL_MODE_UPPER),
        _ => Err(errors::map_str(format!(
            "uplo must be 'L'/'LOWER' or 'U'/'UPPER' (got {s:?})"
        ))),
    }
}

fn side_from_str(s: &str) -> PyResult<cublasSideMode_t> {
    match s.to_ascii_uppercase().as_str() {
        "L" | "LEFT" => Ok(cublasSideMode_t::CUBLAS_SIDE_LEFT),
        "R" | "RIGHT" => Ok(cublasSideMode_t::CUBLAS_SIDE_RIGHT),
        _ => Err(errors::map_str(format!(
            "side must be 'L'/'LEFT' or 'R'/'RIGHT' (got {s:?})"
        ))),
    }
}

fn diag_from_str(s: &str) -> PyResult<cublasDiagType_t> {
    match s.to_ascii_uppercase().as_str() {
        "N" | "NON_UNIT" | "NONUNIT" => Ok(cublasDiagType_t::CUBLAS_DIAG_NON_UNIT),
        "U" | "UNIT" => Ok(cublasDiagType_t::CUBLAS_DIAG_UNIT),
        _ => Err(errors::map_str(format!(
            "diag must be 'N'/'NON_UNIT' or 'U'/'UNIT' (got {s:?})"
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

    /// f32 strided-batched GEMM. Per-batch strides describe the element
    /// offset between consecutive batch entries inside a single
    /// allocation; see cuBLAS docs for `cublasSgemmStridedBatched`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        stride_a, stride_b, stride_c, batch_count,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_strided_batched_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        k: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_count: i32,
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
                actor.tell(BlasMsg::gemm_strided_batched::<f32>(
                    GemmStridedBatchedRequest::<f32> {
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
                        stride_a,
                        stride_b,
                        stride_c,
                        batch_size: batch_count,
                        reply: tx,
                    },
                ));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("gemm_strided_batched_f32 timed out")),
                }
            })
        })
    }

    /// f64 strided-batched GEMM. See `gemm_strided_batched_f32`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        stride_a, stride_b, stride_c, batch_count,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_strided_batched_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        k: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_count: i32,
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
                actor.tell(BlasMsg::gemm_strided_batched::<f64>(
                    GemmStridedBatchedRequest::<f64> {
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
                        stride_a,
                        stride_b,
                        stride_c,
                        batch_size: batch_count,
                        reply: tx,
                    },
                ));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("gemm_strided_batched_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: AXPY ──────────────────────────────

    /// f32 AXPY: `y = alpha * x + y`.
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

    /// f64 AXPY: `y = alpha * x + y`.
    #[pyo3(signature = (alpha, x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn axpy_f64(
        &self,
        py: Python<'_>,
        alpha: f64,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
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
                let req = AxpyRequest::<f64> {
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
                    Err(_) => Err(errors::map_str("axpy_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: DOT ───────────────────────────────

    /// f32 DOT: returns `sum_i x[i] * y[i]`.
    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn dot_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<f32> {
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
                let req = DotRequest::<f32> {
                    n,
                    x,
                    incx,
                    y,
                    incy,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("dot_f32 timed out")),
                }
            })
        })
    }

    /// f64 DOT.
    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn dot_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<f64> {
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
                let req = DotRequest::<f64> {
                    n,
                    x,
                    incx,
                    y,
                    incy,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("dot_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: NRM2 ──────────────────────────────

    /// f32 NRM2: returns `||x||_2`.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn nrm2_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<f32> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = Nrm2Request::<f32> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("nrm2_f32 timed out")),
                }
            })
        })
    }

    /// f64 NRM2.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn nrm2_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<f64> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = Nrm2Request::<f64> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("nrm2_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: SCAL ──────────────────────────────

    /// f32 SCAL: `x = alpha * x` (in-place).
    #[pyo3(signature = (alpha, x, n=None, incx=1, timeout_secs=10.0))]
    fn scal_f32(
        &self,
        py: Python<'_>,
        alpha: f32,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ScalRequest::<f32> {
                    n,
                    alpha,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("scal_f32 timed out")),
                }
            })
        })
    }

    /// f64 SCAL.
    #[pyo3(signature = (alpha, x, n=None, incx=1, timeout_secs=10.0))]
    fn scal_f64(
        &self,
        py: Python<'_>,
        alpha: f64,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ScalRequest::<f64> {
                    n,
                    alpha,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("scal_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: ASUM ──────────────────────────────

    /// f32 ASUM: returns `sum_i |x[i]|`.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn asum_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<f32> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = AsumRequest::<f32> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("asum_f32 timed out")),
                }
            })
        })
    }

    /// f64 ASUM.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn asum_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<f64> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = AsumRequest::<f64> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("asum_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: IAMAX / IAMIN ─────────────────────

    /// f32 IAMAX: returns the index of the largest-magnitude element.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamax_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<i32> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = IamaxRequest::<f32> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("iamax_f32 timed out")),
                }
            })
        })
    }

    /// f64 IAMAX.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamax_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<i32> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = IamaxRequest::<f64> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("iamax_f64 timed out")),
                }
            })
        })
    }

    /// f32 IAMIN: returns the index of the smallest-magnitude element.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamin_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<i32> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = IaminRequest::<f32> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("iamin_f32 timed out")),
                }
            })
        })
    }

    /// f64 IAMIN.
    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamin_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<i32> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = IaminRequest::<f64> {
                    n,
                    x,
                    incx,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("iamin_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: COPY ──────────────────────────────

    /// f32 COPY: `y <- x`.
    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn copy_f32(
        &self,
        py: Python<'_>,
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
                let req = CopyRequest::<f32> {
                    n,
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
                    Err(_) => Err(errors::map_str("copy_f32 timed out")),
                }
            })
        })
    }

    /// f64 COPY.
    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn copy_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
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
                let req = CopyRequest::<f64> {
                    n,
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
                    Err(_) => Err(errors::map_str("copy_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: SWAP ──────────────────────────────

    /// f32 SWAP: swap x and y in-place.
    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn swap_f32(
        &self,
        py: Python<'_>,
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
                let req = SwapRequest::<f32> {
                    n,
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
                    Err(_) => Err(errors::map_str("swap_f32 timed out")),
                }
            })
        })
    }

    /// f64 SWAP.
    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn swap_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
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
                let req = SwapRequest::<f64> {
                    n,
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
                    Err(_) => Err(errors::map_str("swap_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L1: ROT ───────────────────────────────

    /// f32 ROT: Givens rotation `(x_i, y_i) := (c·x_i + s·y_i, -s·x_i + c·y_i)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, y, c, s, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn rot_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        c: f32,
        s: f32,
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
                let req = RotRequest::<f32> {
                    n,
                    x,
                    incx,
                    y,
                    incy,
                    c,
                    s,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("rot_f32 timed out")),
                }
            })
        })
    }

    /// f64 ROT.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, y, c, s, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn rot_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        c: f64,
        s: f64,
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
                let req = RotRequest::<f64> {
                    n,
                    x,
                    incx,
                    y,
                    incy,
                    c,
                    s,
                    reply: tx,
                };
                actor.tell(BlasMsg::L1(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("rot_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L2: GEMV ──────────────────────────────

    /// f32 GEMV: `y = alpha * op(a) · x + beta * y`. `a` is `m × n` in
    /// column-major storage; `lda` defaults to `m`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, x, y, m, n,
        alpha=1.0, beta=0.0,
        trans="N", lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn gemv_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        beta: f32,
        trans: &str,
        lda: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GemvRequest::<f32> {
                    trans,
                    m,
                    n,
                    alpha,
                    beta,
                    a,
                    lda,
                    x,
                    incx,
                    y,
                    incy,
                    reply: tx,
                };
                actor.tell(BlasMsg::L2(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("gemv_f32 timed out")),
                }
            })
        })
    }

    /// f64 GEMV.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, x, y, m, n,
        alpha=1.0, beta=0.0,
        trans="N", lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn gemv_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        beta: f64,
        trans: &str,
        lda: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GemvRequest::<f64> {
                    trans,
                    m,
                    n,
                    alpha,
                    beta,
                    a,
                    lda,
                    x,
                    incx,
                    y,
                    incy,
                    reply: tx,
                };
                actor.tell(BlasMsg::L2(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("gemv_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L2: GER ───────────────────────────────

    /// f32 GER (rank-1 update): `a += alpha * x · y^T`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, a, m, n,
        alpha=1.0, lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn ger_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        a: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        lda: Option<i32>,
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
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GerRequest::<f32> {
                    m,
                    n,
                    alpha,
                    x,
                    incx,
                    y,
                    incy,
                    a,
                    lda,
                    reply: tx,
                };
                actor.tell(BlasMsg::L2(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("ger_f32 timed out")),
                }
            })
        })
    }

    /// f64 GER.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, a, m, n,
        alpha=1.0, lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn ger_f64(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        a: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        lda: Option<i32>,
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
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GerRequest::<f64> {
                    m,
                    n,
                    alpha,
                    x,
                    incx,
                    y,
                    incy,
                    a,
                    lda,
                    reply: tx,
                };
                actor.tell(BlasMsg::L2(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("ger_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L3: GEAM ──────────────────────────────

    /// f32 GEAM: `c = alpha * op(a) + beta * op(b)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n,
        alpha=1.0, beta=1.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn geam_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
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
            n
        });
        let ldb = ldb.unwrap_or(if trans_b == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            n
        });
        let ldc = ldc.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GeamRequest::<f32> {
                    trans_a,
                    trans_b,
                    m,
                    n,
                    alpha,
                    a,
                    lda,
                    beta,
                    b,
                    ldb,
                    c,
                    ldc,
                    reply: tx,
                };
                actor.tell(BlasMsg::L3(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("geam_f32 timed out")),
                }
            })
        })
    }

    /// f64 GEAM.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n,
        alpha=1.0, beta=1.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn geam_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
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
            n
        });
        let ldb = ldb.unwrap_or(if trans_b == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            n
        });
        let ldc = ldc.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GeamRequest::<f64> {
                    trans_a,
                    trans_b,
                    m,
                    n,
                    alpha,
                    a,
                    lda,
                    beta,
                    b,
                    ldb,
                    c,
                    ldc,
                    reply: tx,
                };
                actor.tell(BlasMsg::L3(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("geam_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L3: SYRK ──────────────────────────────

    /// f32 SYRK: `c = alpha * op(a) · op(a)^T + beta * c`. `uplo` is
    /// either `'L'`/`'LOWER'` or `'U'`/`'UPPER'`; `trans` is `'N'` or
    /// `'T'`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, c, n, k,
        alpha=1.0, beta=0.0,
        uplo="L", trans="N",
        lda=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn syrk_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        uplo: &str,
        trans: &str,
        lda: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(if trans == cublasOperation_t::CUBLAS_OP_N {
            n
        } else {
            k
        });
        let ldc = ldc.unwrap_or(n);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = SyrkRequest::<f32> {
                    uplo,
                    trans,
                    n,
                    k,
                    alpha,
                    a,
                    lda,
                    beta,
                    c,
                    ldc,
                    reply: tx,
                };
                actor.tell(BlasMsg::L3(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("syrk_f32 timed out")),
                }
            })
        })
    }

    /// f64 SYRK.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, c, n, k,
        alpha=1.0, beta=0.0,
        uplo="L", trans="N",
        lda=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn syrk_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        n: i32,
        k: i32,
        alpha: f64,
        beta: f64,
        uplo: &str,
        trans: &str,
        lda: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(if trans == cublasOperation_t::CUBLAS_OP_N {
            n
        } else {
            k
        });
        let ldc = ldc.unwrap_or(n);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = SyrkRequest::<f64> {
                    uplo,
                    trans,
                    n,
                    k,
                    alpha,
                    a,
                    lda,
                    beta,
                    c,
                    ldc,
                    reply: tx,
                };
                actor.tell(BlasMsg::L3(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("syrk_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── L3: TRSM ──────────────────────────────

    /// f32 TRSM: triangular solve `op(a) · X = alpha * b` (or
    /// `X · op(a) = alpha * b`). The solution is written in-place over
    /// `b`. `side` is `'L'`/`'R'`, `uplo` is `'L'`/`'U'`, `trans` is
    /// `'N'`/`'T'`/`'C'`, `diag` is `'N'`/`'U'`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, m, n,
        alpha=1.0,
        side="L", uplo="L", trans="N", diag="N",
        lda=None, ldb=None,
        timeout_secs=30.0,
    ))]
    fn trsm_f32(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        side: &str,
        uplo: &str,
        trans: &str,
        diag: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
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
        let side = side_from_str(side)?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let diag = diag_from_str(diag)?;
        let lda = lda.unwrap_or(if side == cublasSideMode_t::CUBLAS_SIDE_LEFT {
            m
        } else {
            n
        });
        let ldb = ldb.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = TrsmRequest::<f32> {
                    side,
                    uplo,
                    trans,
                    diag,
                    m,
                    n,
                    alpha,
                    a,
                    lda,
                    b,
                    ldb,
                    reply: tx,
                };
                actor.tell(BlasMsg::L3(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("trsm_f32 timed out")),
                }
            })
        })
    }

    /// f64 TRSM.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, m, n,
        alpha=1.0,
        side="L", uplo="L", trans="N", diag="N",
        lda=None, ldb=None,
        timeout_secs=30.0,
    ))]
    fn trsm_f64(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        side: &str,
        uplo: &str,
        trans: &str,
        diag: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
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
        let side = side_from_str(side)?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let diag = diag_from_str(diag)?;
        let lda = lda.unwrap_or(if side == cublasSideMode_t::CUBLAS_SIDE_LEFT {
            m
        } else {
            n
        });
        let ldb = ldb.unwrap_or(m);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = TrsmRequest::<f64> {
                    side,
                    uplo,
                    trans,
                    diag,
                    m,
                    n,
                    alpha,
                    a,
                    lda,
                    b,
                    ldb,
                    reply: tx,
                };
                actor.tell(BlasMsg::L3(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                    Err(_) => Err(errors::map_str("trsm_f64 timed out")),
                }
            })
        })
    }

    // ─────────────────────── Async (asyncio) variants ─────────────
    //
    // Each `_async` method mirrors its blocking counterpart; the
    // synchronous setup (cloning ActorRef + extracting `GpuRef<T>`
    // from the Python buffer wrapper) happens before entering the
    // async block, so the actor pipeline runs without the GIL.
    // Bodies are intentionally duplicated rather than refactored —
    // the per-method enum dispatch / strided-batched fields make a
    // shared helper mostly mechanical, and keeping the request
    // construction close to the method aids readability.

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_f32_async<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_f64_async<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        stride_a, stride_b, stride_c, batch_count,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_strided_batched_f32_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        k: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_count: i32,
        alpha: f32,
        beta: f32,
        trans_a: &str,
        trans_b: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::gemm_strided_batched::<f32>(
                GemmStridedBatchedRequest::<f32> {
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
                    stride_a,
                    stride_b,
                    stride_c,
                    batch_size: batch_count,
                    reply: tx,
                },
            ));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("gemm_strided_batched_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n, k,
        stride_a, stride_b, stride_c, batch_count,
        alpha=1.0, beta=0.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=60.0,
    ))]
    fn gemm_strided_batched_f64_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        k: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_count: i32,
        alpha: f64,
        beta: f64,
        trans_a: &str,
        trans_b: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::gemm_strided_batched::<f64>(
                GemmStridedBatchedRequest::<f64> {
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
                    stride_a,
                    stride_b,
                    stride_c,
                    batch_size: batch_count,
                    reply: tx,
                },
            ));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("gemm_strided_batched_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (alpha, x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn axpy_f32_async<'py>(
        &self,
        py: Python<'py>,
        alpha: f32,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(AxpyRequest::<f32> {
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("axpy_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (alpha, x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn axpy_f64_async<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(AxpyRequest::<f64> {
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("axpy_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn dot_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(DotRequest::<f32> {
                n,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("dot_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn dot_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(DotRequest::<f64> {
                n,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("dot_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn nrm2_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(Nrm2Request::<f32> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("nrm2_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn nrm2_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(Nrm2Request::<f64> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("nrm2_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (alpha, x, n=None, incx=1, timeout_secs=10.0))]
    fn scal_f32_async<'py>(
        &self,
        py: Python<'py>,
        alpha: f32,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(ScalRequest::<f32> {
                n,
                alpha,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("scal_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (alpha, x, n=None, incx=1, timeout_secs=10.0))]
    fn scal_f64_async<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(ScalRequest::<f64> {
                n,
                alpha,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("scal_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn asum_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(AsumRequest::<f32> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("asum_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn asum_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(AsumRequest::<f64> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("asum_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamax_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(IamaxRequest::<f32> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("iamax_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamax_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(IamaxRequest::<f64> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("iamax_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamin_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(IaminRequest::<f32> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("iamin_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, n=None, incx=1, timeout_secs=10.0))]
    fn iamin_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let n = n.unwrap_or_else(|| x.len() as i32);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(IaminRequest::<f64> {
                n,
                x,
                incx,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(v))) => Ok(v),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("iamin_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn copy_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(CopyRequest::<f32> {
                n,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("copy_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn copy_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(CopyRequest::<f64> {
                n,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("copy_f64 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn swap_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(SwapRequest::<f32> {
                n,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("swap_f32 timed out")),
            }
        })
    }

    #[pyo3(signature = (x, y, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn swap_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(SwapRequest::<f64> {
                n,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("swap_f64 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, y, c, s, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn rot_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        c: f32,
        s: f32,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(RotRequest::<f32> {
                n,
                x,
                incx,
                y,
                incy,
                c,
                s,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("rot_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, y, c, s, n=None, incx=1, incy=1, timeout_secs=10.0))]
    fn rot_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        c: f64,
        s: f64,
        n: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L1(Box::new(RotRequest::<f64> {
                n,
                x,
                incx,
                y,
                incy,
                c,
                s,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("rot_f64 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, x, y, m, n,
        alpha=1.0, beta=0.0,
        trans="N", lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn gemv_f32_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF32>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        beta: f32,
        trans: &str,
        lda: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L2(Box::new(GemvRequest::<f32> {
                trans,
                m,
                n,
                alpha,
                beta,
                a,
                lda,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("gemv_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, x, y, m, n,
        alpha=1.0, beta=0.0,
        trans="N", lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn gemv_f64_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF64>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        beta: f64,
        trans: &str,
        lda: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L2(Box::new(GemvRequest::<f64> {
                trans,
                m,
                n,
                alpha,
                beta,
                a,
                lda,
                x,
                incx,
                y,
                incy,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("gemv_f64 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, a, m, n,
        alpha=1.0, lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn ger_f32_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        a: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        lda: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L2(Box::new(GerRequest::<f32> {
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                a,
                lda,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("ger_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, a, m, n,
        alpha=1.0, lda=None, incx=1, incy=1,
        timeout_secs=30.0,
    ))]
    fn ger_f64_async<'py>(
        &self,
        py: Python<'py>,
        x: Py<PyGpuBufferF64>,
        y: Py<PyGpuBufferF64>,
        a: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        lda: Option<i32>,
        incx: i32,
        incy: i32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let lda = lda.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L2(Box::new(GerRequest::<f64> {
                m,
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                a,
                lda,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("ger_f64 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n,
        alpha=1.0, beta=1.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn geam_f32_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        beta: f32,
        trans_a: &str,
        trans_b: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
            n
        });
        let ldb = ldb.unwrap_or(if trans_b == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            n
        });
        let ldc = ldc.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L3(Box::new(GeamRequest::<f32> {
                trans_a,
                trans_b,
                m,
                n,
                alpha,
                a,
                lda,
                beta,
                b,
                ldb,
                c,
                ldc,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("geam_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, c, m, n,
        alpha=1.0, beta=1.0,
        trans_a="N", trans_b="N",
        lda=None, ldb=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn geam_f64_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        beta: f64,
        trans_a: &str,
        trans_b: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
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
            n
        });
        let ldb = ldb.unwrap_or(if trans_b == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            n
        });
        let ldc = ldc.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L3(Box::new(GeamRequest::<f64> {
                trans_a,
                trans_b,
                m,
                n,
                alpha,
                a,
                lda,
                beta,
                b,
                ldb,
                c,
                ldc,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("geam_f64 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, c, n, k,
        alpha=1.0, beta=0.0,
        uplo="L", trans="N",
        lda=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn syrk_f32_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        uplo: &str,
        trans: &str,
        lda: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(if trans == cublasOperation_t::CUBLAS_OP_N {
            n
        } else {
            k
        });
        let ldc = ldc.unwrap_or(n);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L3(Box::new(SyrkRequest::<f32> {
                uplo,
                trans,
                n,
                k,
                alpha,
                a,
                lda,
                beta,
                c,
                ldc,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("syrk_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, c, n, k,
        alpha=1.0, beta=0.0,
        uplo="L", trans="N",
        lda=None, ldc=None,
        timeout_secs=30.0,
    ))]
    fn syrk_f64_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF64>,
        c: Py<PyGpuBufferF64>,
        n: i32,
        k: i32,
        alpha: f64,
        beta: f64,
        uplo: &str,
        trans: &str,
        lda: Option<i32>,
        ldc: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let lda = lda.unwrap_or(if trans == cublasOperation_t::CUBLAS_OP_N {
            n
        } else {
            k
        });
        let ldc = ldc.unwrap_or(n);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L3(Box::new(SyrkRequest::<f64> {
                uplo,
                trans,
                n,
                k,
                alpha,
                a,
                lda,
                beta,
                c,
                ldc,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("syrk_f64 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, m, n,
        alpha=1.0,
        side="L", uplo="L", trans="N", diag="N",
        lda=None, ldb=None,
        timeout_secs=30.0,
    ))]
    fn trsm_f32_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        alpha: f32,
        side: &str,
        uplo: &str,
        trans: &str,
        diag: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let side = side_from_str(side)?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let diag = diag_from_str(diag)?;
        let lda = lda.unwrap_or(if side == cublasSideMode_t::CUBLAS_SIDE_LEFT {
            m
        } else {
            n
        });
        let ldb = ldb.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L3(Box::new(TrsmRequest::<f32> {
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha,
                a,
                lda,
                b,
                ldb,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("trsm_f32 timed out")),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        a, b, m, n,
        alpha=1.0,
        side="L", uplo="L", trans="N", diag="N",
        lda=None, ldb=None,
        timeout_secs=30.0,
    ))]
    fn trsm_f64_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF64>,
        b: Py<PyGpuBufferF64>,
        m: i32,
        n: i32,
        alpha: f64,
        side: &str,
        uplo: &str,
        trans: &str,
        diag: &str,
        lda: Option<i32>,
        ldb: Option<i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let side = side_from_str(side)?;
        let uplo = fill_from_str(uplo)?;
        let trans = op_from_str(trans)?;
        let diag = diag_from_str(diag)?;
        let lda = lda.unwrap_or(if side == cublasSideMode_t::CUBLAS_SIDE_LEFT {
            m
        } else {
            n
        });
        let ldb = ldb.unwrap_or(m);
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(BlasMsg::L3(Box::new(TrsmRequest::<f64> {
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha,
                a,
                lda,
                b,
                ldb,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("blas dropped reply")),
                Err(_) => Err(errors::map_str("trsm_f64 timed out")),
            }
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
