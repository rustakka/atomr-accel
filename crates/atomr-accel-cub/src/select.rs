//! `cub::DeviceSelect` — Flagged / If / Unique selection +
//! `cub::DevicePartition` — partition by predicate.
//!
//! Phase 5.1 ships single-tile select / partition only (`n ≤
//! crate::kernels::TILE_ELEMENTS`); larger inputs return a structured
//! `Unrecoverable` error from the dispatcher with a Phase 5.2 hint.

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

use crate::dispatch::{compile_or_get_handle, launch, launch_config_single_block};
use crate::kernels::{emit_partition_source, emit_select_source, TILE_ELEMENTS};
use crate::{reply_err, CubDispatchBase, CubDispatchCtx, KernelSourceCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectMode {
    /// `DeviceSelect::Flagged` — keep entries whose `flags[i] != 0`.
    Flagged,
    /// `DeviceSelect::Unique` — drop adjacent duplicates.
    Unique,
}

pub struct SelectRequest<T: CudaDtype> {
    pub mode: SelectMode,
    pub input: GpuRef<T>,
    pub output: GpuRef<T>,
    /// Single-element output for the selected count.
    pub num_selected: GpuRef<u32>,
    /// Optional per-element `u8` flag buffer for `SelectMode::Flagged`.
    pub flags: Option<GpuRef<u8>>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> SelectRequest<T> {
    pub fn new(
        mode: SelectMode,
        input: GpuRef<T>,
        output: GpuRef<T>,
        num_selected: GpuRef<u32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            mode,
            input,
            output,
            num_selected,
            flags: None,
            reply,
            _phantom: PhantomData,
        }
    }

    pub fn with_flags(mut self, flags: GpuRef<u8>) -> Self {
        self.flags = Some(flags);
        self
    }
}

pub trait CubSelectDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for SelectRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        match self.mode {
            SelectMode::Flagged => "select_flagged",
            SelectMode::Unique => "select_unique",
        }
    }
    fn dtype_name(&self) -> &'static str {
        <T as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubSelectDispatch for SelectRequest<T>
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
                        "atomr-accel-cub::CubSelect: NvrtcActor not wired into CubActor".into(),
                    ),
                );
                return;
            }
        };
        let n = self.input.len();
        if n > TILE_ELEMENTS as usize {
            reply_err(
                self.reply,
                GpuError::Unrecoverable(format!(
                    "atomr-accel-cub::CubSelect: n={n} exceeds single-tile limit ({}); \
                     multi-block lands in Phase 5.2",
                    TILE_ELEMENTS
                )),
            );
            return;
        }
        if matches!(self.mode, SelectMode::Flagged) && self.flags.is_none() {
            reply_err(
                self.reply,
                GpuError::Unrecoverable(
                    "atomr-accel-cub::CubSelect: SelectMode::Flagged requires a flags buffer"
                        .into(),
                ),
            );
            return;
        }
        let cache = ctx.kernel_cache.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(run_select::<T>(me, nvrtc, cache, arch));
    }
}

async fn run_select<T: CudaDtype>(
    req: SelectRequest<T>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) {
    let SelectRequest {
        mode,
        input,
        output,
        num_selected,
        flags,
        reply,
        ..
    } = req;
    let result = compile_and_launch_select::<T>(
        mode,
        input,
        output,
        num_selected,
        flags,
        nvrtc,
        cache,
        arch,
    )
    .await;
    let _ = reply.send(result);
}

#[allow(clippy::too_many_arguments)]
async fn compile_and_launch_select<T: CudaDtype>(
    mode: SelectMode,
    input: GpuRef<T>,
    output: GpuRef<T>,
    num_selected: GpuRef<u32>,
    flags: Option<GpuRef<u8>>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let dtype = <T as AccelDtype>::NAME.to_string();
    let op = match mode {
        SelectMode::Flagged => "select_flagged",
        SelectMode::Unique => "select_unique",
    };
    let (src, kname) = emit_select_source::<T>(mode);
    let handle = compile_or_get_handle(
        nvrtc.clone(),
        cache,
        op.into(),
        dtype,
        src,
        kname,
        arch,
    )
    .await?;

    let n = input.len() as u32;
    let args = match mode {
        SelectMode::Flagged => vec![
            KernelArg::DevSlice(Box::new(input)),
            KernelArg::DevSlice(Box::new(flags.unwrap())),
            KernelArg::DevSlice(Box::new(output)),
            KernelArg::DevSlice(Box::new(num_selected)),
            KernelArg::Scalar(Box::new(n)),
        ],
        SelectMode::Unique => vec![
            KernelArg::DevSlice(Box::new(input)),
            KernelArg::DevSlice(Box::new(output)),
            KernelArg::DevSlice(Box::new(num_selected)),
            KernelArg::Scalar(Box::new(n)),
        ],
    };

    launch(&nvrtc, handle, args, launch_config_single_block()).await
}

/// `DevicePartition::Flagged` — partition input into selected / rejected.
pub struct PartitionRequest<T: CudaDtype> {
    pub input: GpuRef<T>,
    pub output: GpuRef<T>,
    pub flags: GpuRef<u8>,
    pub num_selected: GpuRef<u32>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> PartitionRequest<T> {
    pub fn new(
        input: GpuRef<T>,
        output: GpuRef<T>,
        flags: GpuRef<u8>,
        num_selected: GpuRef<u32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            input,
            output,
            flags,
            num_selected,
            reply,
            _phantom: PhantomData,
        }
    }
}

pub trait CubPartitionDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for PartitionRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        "partition_flagged"
    }
    fn dtype_name(&self) -> &'static str {
        <T as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubPartitionDispatch for PartitionRequest<T>
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
                        "atomr-accel-cub::CubPartition: NvrtcActor not wired".into(),
                    ),
                );
                return;
            }
        };
        let n = self.input.len();
        if n > TILE_ELEMENTS as usize {
            reply_err(
                self.reply,
                GpuError::Unrecoverable(format!(
                    "atomr-accel-cub::CubPartition: n={n} exceeds single-tile limit ({}); \
                     multi-block lands in Phase 5.2",
                    TILE_ELEMENTS
                )),
            );
            return;
        }
        let cache = ctx.kernel_cache.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(run_partition::<T>(me, nvrtc, cache, arch));
    }
}

async fn run_partition<T: CudaDtype>(
    req: PartitionRequest<T>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) {
    let PartitionRequest {
        input,
        output,
        flags,
        num_selected,
        reply,
        ..
    } = req;
    let result =
        compile_and_launch_partition::<T>(input, output, flags, num_selected, nvrtc, cache, arch)
            .await;
    let _ = reply.send(result);
}

async fn compile_and_launch_partition<T: CudaDtype>(
    input: GpuRef<T>,
    output: GpuRef<T>,
    flags: GpuRef<u8>,
    num_selected: GpuRef<u32>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let dtype = <T as AccelDtype>::NAME.to_string();
    let (src, kname) = emit_partition_source::<T>();
    let handle = compile_or_get_handle(
        nvrtc.clone(),
        cache,
        "partition_flagged".into(),
        dtype,
        src,
        kname,
        arch,
    )
    .await?;

    let n = input.len() as u32;
    let args = vec![
        KernelArg::DevSlice(Box::new(input)),
        KernelArg::DevSlice(Box::new(flags)),
        KernelArg::DevSlice(Box::new(output)),
        KernelArg::DevSlice(Box::new(num_selected)),
        KernelArg::Scalar(Box::new(n)),
    ];

    launch(&nvrtc, handle, args, launch_config_single_block()).await
}
