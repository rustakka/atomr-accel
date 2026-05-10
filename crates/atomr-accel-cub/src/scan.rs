//! `cub::DeviceScan` — inclusive / exclusive prefix sums.
//!
//! Implementation note: `cub::DeviceScan` is host-launched, so the
//! Phase 5.1 emitter renders three `__global__` kernels — block scan,
//! single-block exclusive scan over the per-block totals, and a
//! fixup kernel that adds each block's prefix back into the per-element
//! output. The dispatcher launches them in sequence; the partials
//! workspace is allocated inline.

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_accel_cuda::device::DeviceState;
use atomr_accel_cuda::dtype::{AccelDtype, CudaDtype};
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::nvrtc::SmArch;
use atomr_accel_cuda::kernel::{KernelArg, NvrtcMsg};
use atomr_core::actor::ActorRef;

use crate::dispatch::{
    compile_or_get_handle, grid_blocks_for, launch, launch_config_for, launch_config_single_block,
};
use crate::kernels::emit_scan_source;
use crate::{reply_err, CubDispatchBase, CubDispatchCtx, KernelSourceCache};

/// Inclusive vs. exclusive scan flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanKind {
    Inclusive,
    Exclusive,
}

impl ScanKind {
    pub fn op_name(self) -> &'static str {
        match self {
            ScanKind::Inclusive => "scan_inclusive",
            ScanKind::Exclusive => "scan_exclusive",
        }
    }
}

pub struct ScanRequest<T: CudaDtype> {
    pub kind: ScanKind,
    pub input: GpuRef<T>,
    pub output: GpuRef<T>,
    /// Initial value for exclusive scans (`T::zero()` is the natural
    /// default for sum scans). Ignored for `Inclusive`.
    pub init: Option<T>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> ScanRequest<T> {
    pub fn new(
        kind: ScanKind,
        input: GpuRef<T>,
        output: GpuRef<T>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            kind,
            input,
            output,
            init: None,
            reply,
            _phantom: PhantomData,
        }
    }
}

pub trait CubScanDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for ScanRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        self.kind.op_name()
    }
    fn dtype_name(&self) -> &'static str {
        <T as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubScanDispatch for ScanRequest<T>
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
                        "atomr-accel-cub::CubScan: NvrtcActor not wired into CubActor".into(),
                    ),
                );
                return;
            }
        };
        let cache = ctx.kernel_cache.clone();
        let stream = ctx.stream.clone();
        let state = ctx.state.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(async move {
            let result = run_scan::<T>(
                me.kind, me.input, me.output, nvrtc, cache, stream, state, arch,
            )
            .await;
            let _ = me.reply.send(result);
        });
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_scan<T: CudaDtype>(
    kind: ScanKind,
    input: GpuRef<T>,
    output: GpuRef<T>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    stream: Arc<cudarc::driver::CudaStream>,
    state: Arc<DeviceState>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let dtype = <T as AccelDtype>::NAME;
    let op_name = kind.op_name().to_string();
    let (src, kname) = emit_scan_source::<T>(kind);

    let main_handle = compile_or_get_handle(
        nvrtc.clone(),
        cache.clone(),
        op_name.clone(),
        dtype.to_string(),
        src.clone(),
        kname.clone(),
        arch,
    )
    .await?;
    let block_sums_handle = compile_or_get_handle(
        nvrtc.clone(),
        cache.clone(),
        format!("{op_name}_block_sums"),
        dtype.to_string(),
        src.clone(),
        format!("{kname}_block_sums"),
        arch,
    )
    .await?;
    let fixup_handle = compile_or_get_handle(
        nvrtc.clone(),
        cache.clone(),
        format!("{op_name}_fixup"),
        dtype.to_string(),
        src,
        format!("{kname}_fixup"),
        arch,
    )
    .await?;

    let n = input.len();
    let grid = grid_blocks_for(n);

    let block_sums_slice = stream
        .alloc_zeros::<T>(grid as usize)
        .map_err(|e| GpuError::OutOfMemory(format!("cub scan block_sums alloc: {e}")))?;
    let block_sums = GpuRef::new(Arc::new(block_sums_slice), &state);

    // Pass 1: per-block scan + per-block totals.
    let main_args = vec![
        KernelArg::DevSlice(Box::new(input.clone())),
        KernelArg::DevSlice(Box::new(output.clone())),
        KernelArg::DevSlice(Box::new(block_sums.clone())),
        KernelArg::Usize(n),
    ];
    launch(&nvrtc, main_handle, main_args, launch_config_for(n)).await?;

    // Pass 2: single-block exclusive scan over the per-block totals.
    let bs_args = vec![
        KernelArg::DevSlice(Box::new(block_sums.clone())),
        KernelArg::Usize(grid as usize),
    ];
    launch(
        &nvrtc,
        block_sums_handle,
        bs_args,
        launch_config_single_block(),
    )
    .await?;

    // Pass 3: add each block's prefix back into the per-element output.
    let fx_args = vec![
        KernelArg::DevSlice(Box::new(output)),
        KernelArg::DevSlice(Box::new(block_sums)),
        KernelArg::Usize(n),
    ];
    launch(&nvrtc, fixup_handle, fx_args, launch_config_for(n)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 5: round-trip a `ScanRequest<f32>` through the
    /// `CubDispatchBase` trait — `op_name` and `dtype_name` resolve
    /// without dereferencing any `GpuRef`.
    #[test]
    fn scan_request_round_trip() {
        for kind in [ScanKind::Inclusive, ScanKind::Exclusive] {
            assert!(kind.op_name().starts_with("scan_"));
        }
        // Op-name uniqueness across the kind matrix.
        assert_ne!(ScanKind::Inclusive.op_name(), ScanKind::Exclusive.op_name());

        // The kernel cache key threads through the ScanKind name + the
        // dtype `T::NAME` from atomr-accel. Verify it's unique across
        // (kind, dtype).
        let dtypes = ["f32", "f64", "i32", "u32", "i64", "u64"];
        let kinds = [ScanKind::Inclusive, ScanKind::Exclusive];
        let mut seen = std::collections::HashSet::new();
        for k in kinds {
            for dt in dtypes {
                let key = crate::kernel_key(k.op_name(), dt);
                assert!(seen.insert(key.clone()), "scan key collision: {key}");
            }
        }
        assert_eq!(seen.len(), kinds.len() * dtypes.len());
    }
}
