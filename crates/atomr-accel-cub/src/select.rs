//! `cub::DeviceSelect` — Flagged / If / Unique selection +
//! `cub::DevicePartition` — partition by predicate.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;

use crate::{reply_err, CubDispatchBase, CubDispatchCtx};

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
        <T as atomr_accel_cuda::dtype::AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubSelectDispatch for SelectRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        let op = match self.mode {
            SelectMode::Flagged => "select_flagged",
            SelectMode::Unique => "select_unique",
        };
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubSelect::{}<{}> — kernel compile path lands in Phase 5.1",
                op,
                <T as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
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
        <T as atomr_accel_cuda::dtype::AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubPartitionDispatch for PartitionRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubPartition::partition_flagged<{}> — kernel compile path lands in Phase 5.1",
                <T as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
}
