//! `cub::DeviceHistogram` — fixed-bin / range histograms.
//!
//! Phase 5.1 fixes the bin count at 256 (matches `u8` histograms);
//! the dispatcher allocates the output via the caller and the kernel
//! atomically merges per-block shared-memory accumulators into the
//! global output.

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::{AccelDtype, CudaDtype};
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::nvrtc::SmArch;
use atomr_accel_cuda::kernel::{KernelArg, NvrtcMsg};
use atomr_core::actor::ActorRef;

use crate::dispatch::{compile_or_get_handle, launch, launch_config_for};
use crate::kernels::emit_histogram_source;
use crate::{reply_err, CubDispatchBase, CubDispatchCtx, KernelSourceCache};

pub struct HistogramRequest<T: CudaDtype> {
    pub input: GpuRef<T>,
    /// Output bin counts (`u32` per CUB convention).
    pub bins: GpuRef<u32>,
    pub num_bins: u32,
    /// Lower / upper sample bound (inclusive lower, exclusive upper)
    /// passed as f32 since CUB's `HistogramRange` takes a ref to
    /// host-side bin boundaries. For fixed-bin variants, store
    /// `[min, max]`.
    pub lower_level: f32,
    pub upper_level: f32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> HistogramRequest<T> {
    pub fn new(
        input: GpuRef<T>,
        bins: GpuRef<u32>,
        num_bins: u32,
        lower_level: f32,
        upper_level: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            input,
            bins,
            num_bins,
            lower_level,
            upper_level,
            reply,
            _phantom: PhantomData,
        }
    }
}

pub trait CubHistogramDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for HistogramRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        "histogram_even"
    }
    fn dtype_name(&self) -> &'static str {
        <T as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubHistogramDispatch for HistogramRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>) {
        let nvrtc = match ctx.nvrtc {
            Some(n) => n.clone(),
            None => {
                reply_err(
                    self.reply,
                    GpuError::Unrecoverable(
                        "atomr-accel-cub::CubHistogram: NvrtcActor not wired into CubActor".into(),
                    ),
                );
                return;
            }
        };
        let cache = ctx.kernel_cache.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(run_histogram::<T>(me, nvrtc, cache, arch));
    }
}

async fn run_histogram<T: CudaDtype>(
    req: HistogramRequest<T>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) {
    let HistogramRequest {
        input,
        bins,
        num_bins: _num_bins,
        lower_level,
        upper_level,
        reply,
        ..
    } = req;
    let result =
        compile_and_launch::<T>(input, bins, lower_level, upper_level, nvrtc, cache, arch).await;
    let _ = reply.send(result);
}

async fn compile_and_launch<T: CudaDtype>(
    input: GpuRef<T>,
    bins: GpuRef<u32>,
    lower_level: f32,
    upper_level: f32,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let dtype = <T as AccelDtype>::NAME.to_string();
    let (src, kname) = emit_histogram_source::<T>();
    let handle = compile_or_get_handle(
        nvrtc.clone(),
        cache,
        "histogram_even".into(),
        dtype,
        src,
        kname,
        arch,
    )
    .await?;

    let n = input.len();
    let args = vec![
        KernelArg::DevSlice(Box::new(input)),
        KernelArg::DevSlice(Box::new(bins)),
        KernelArg::Usize(n),
        KernelArg::Scalar(Box::new(lower_level)),
        KernelArg::Scalar(Box::new(upper_level)),
    ];
    launch(&nvrtc, handle, args, launch_config_for(n)).await
}

#[cfg(test)]
mod tests {
    #[test]
    fn histogram_op_name_stable() {
        // op_name string is read by the kernel-cache key so it must
        // remain stable across releases.
        let dtypes = ["f32", "f64", "i32", "u32"];
        for dt in dtypes {
            let k = crate::kernel_key("histogram_even", dt);
            assert!(k.starts_with("cub_histogram_even_"));
        }
    }
}
