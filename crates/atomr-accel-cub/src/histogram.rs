//! `cub::DeviceHistogram` — fixed-bin / range histograms.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;

use crate::{reply_err, CubDispatchBase, CubDispatchCtx};

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
        <T as atomr_accel_cuda::dtype::AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubHistogramDispatch for HistogramRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubHistogram::histogram_even<{}> — kernel compile path lands in Phase 5.1",
                <T as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
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
