//! Capture-mode contract.
//!
//! Library actors that participate in [`crate::pipeline`] or
//! [`crate::graph`] implement [`RecordMode`] so they can be driven
//! without registering host-callback completion. The capture caller
//! (a `PipelineStage` adapter or `GraphActor`) feeds an operation,
//! the library actor enqueues it onto the supplied stream, and the
//! caller manages cross-stream synchronization itself via
//! `CudaEvent`s (pipeline) or graph instantiation (graph).
//!
//! **Capture-safe vs. unsafe:** anything calling host functions
//! (`HostFnCompletion`'s `cuLaunchHostFunc`, `tokio::spawn`,
//! synchronous memcpy) is *not* capture-safe. RecordMode impls must
//! enqueue purely on the stream and return synchronously.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

#[cfg(feature = "curand")]
use cudarc::curand::CudaRng;

#[cfg(feature = "cufft")]
use cudarc::cufft::CudaFft;

/// Library-actor opt-in to capture-mode enqueue.
///
/// Implementations enqueue `op` onto `stream` synchronously and return
/// without awaiting completion. The caller is responsible for any
/// downstream `CudaEvent` recording / wait.
pub trait RecordMode {
    /// The operation type, typically a thin enum mirroring the actor's
    /// public `Msg` enum but stripped of `oneshot::Sender` reply
    /// channels.
    type Op;

    fn enqueue_record(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        op: Self::Op,
    ) -> Result<(), GpuError>;
}

/// Record-mode op for cuBLAS SGEMM. Mirrors `BlasMsg::Sgemm`'s
/// payload minus the reply channel.
pub struct BlasSgemmOp {
    pub a: GpuRef<f32>,
    pub b: GpuRef<f32>,
    pub c: GpuRef<f32>,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: f32,
    pub beta: f32,
}

/// Memcpy op (host-side `cudaMemcpyAsync` device-to-device on the
/// captured stream). Capture-safe.
pub struct MemcpyOp {
    pub src: GpuRef<f32>,
    pub dst: GpuRef<f32>,
}

/// Uniform RNG fill op. Capture-safe in cuRAND when no host counter
/// is consulted; the actor records an in-place uniform fill against
/// the supplied buffer.
#[cfg(feature = "curand")]
pub struct RngFillUniformOp {
    pub dst: GpuRef<f32>,
}

/// Capture-mode wrapper around a `CudaBlas` handle. Used by
/// [`crate::graph::GraphActor`] when it needs to record an SGEMM
/// inside a stream-capture region.
pub struct BlasRecorder<'a> {
    pub handle: &'a CudaBlas,
}

/// Capture-mode wrapper for in-stream device-to-device memcpy.
pub struct MemcpyRecorder;

impl RecordMode for MemcpyRecorder {
    type Op = MemcpyOp;
    fn enqueue_record(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        op: Self::Op,
    ) -> Result<(), GpuError> {
        let MemcpyOp { src, dst } = op;
        let src_slice = src.access()?.clone();
        let dst_slice = dst.access()?.clone();
        let mut dst_owned = Arc::try_unwrap(dst_slice).map_err(|_| {
            GpuError::Unrecoverable("MemcpyRecorder: dst has multiple refs".into())
        })?;
        stream
            .memcpy_dtod(&*src_slice, &mut dst_owned)
            .map_err(|e| GpuError::LibraryError {
                lib: "driver",
                msg: format!("record memcpy_dtod: {e}"),
            })?;
        dst.record_write(stream);
        let _ = (src_slice, dst_owned);
        Ok(())
    }
}

#[cfg(feature = "curand")]
pub struct RngRecorder<'a> {
    pub rng: &'a CudaRng,
}

#[cfg(feature = "cufft")]
pub struct FftR2COp {
    pub src: GpuRef<f32>,
    pub dst: GpuRef<cudarc::cufft::sys::float2>,
}

#[cfg(feature = "cufft")]
pub struct FftRecorder<'a> {
    pub plan: &'a CudaFft,
}

#[cfg(feature = "cufft")]
impl<'a> RecordMode for FftRecorder<'a> {
    type Op = FftR2COp;
    fn enqueue_record(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        op: Self::Op,
    ) -> Result<(), GpuError> {
        let FftR2COp { src, dst } = op;
        let src_slice = src.access()?.clone();
        let dst_slice = dst.access()?.clone();
        let mut dst_owned = Arc::try_unwrap(dst_slice).map_err(|_| {
            GpuError::Unrecoverable("FftRecorder: dst has multiple refs".into())
        })?;
        self.plan
            .exec_r2c(&*src_slice, &mut dst_owned)
            .map_err(|e| GpuError::LibraryError {
                lib: "cufft",
                msg: format!("record exec_r2c: {e}"),
            })?;
        dst.record_write(stream);
        let _ = (src_slice, dst_owned);
        Ok(())
    }
}

#[cfg(feature = "curand")]
impl<'a> RecordMode for RngRecorder<'a> {
    type Op = RngFillUniformOp;
    fn enqueue_record(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        op: Self::Op,
    ) -> Result<(), GpuError> {
        let RngFillUniformOp { dst } = op;
        let dst_slice = dst.access()?.clone();
        let mut owned = Arc::try_unwrap(dst_slice).map_err(|_| {
            GpuError::Unrecoverable("RngRecorder: dst has multiple refs".into())
        })?;
        self.rng
            .fill_with_uniform(&mut owned)
            .map_err(|e| GpuError::LibraryError {
                lib: "curand",
                msg: format!("record fill_uniform: {e:?}"),
            })?;
        dst.record_write(stream);
        let _ = owned;
        Ok(())
    }
}

impl<'a> RecordMode for BlasRecorder<'a> {
    type Op = BlasSgemmOp;

    fn enqueue_record(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        op: Self::Op,
    ) -> Result<(), GpuError> {
        let BlasSgemmOp { a, b, c, m, n, k, alpha, beta } = op;
        let a_slice = a.access()?.clone();
        let b_slice = b.access()?.clone();
        let c_slice = c.access()?.clone();
        let mut c_owned = Arc::try_unwrap(c_slice).map_err(|_| {
            GpuError::Unrecoverable(
                "BlasRecorder: C has multiple live references".into(),
            )
        })?;

        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m, n, k,
            alpha,
            lda: m, ldb: k, beta, ldc: m,
        };
        // SAFETY: m/n/k validity is the caller's contract.
        unsafe {
            self.handle
                .gemm(cfg, &*a_slice, &*b_slice, &mut c_owned)
                .map_err(|e| GpuError::LibraryError {
                    lib: "cublas",
                    msg: format!("record gemm: {e}"),
                })?;
        }
        c.record_write(stream);
        // Slices must outlive the stream operation; in capture mode
        // the graph-exec object holds them. We leak ownership back
        // into the GpuRef by re-Arcing — only safe because the
        // graph machinery keeps the buffers alive until the graph
        // is destroyed.
        let _ = (a_slice, b_slice, c_owned);
        Ok(())
    }
}
