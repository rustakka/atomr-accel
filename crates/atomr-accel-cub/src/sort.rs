//! `cub::DeviceRadixSort` — keys / pairs, ascending or descending.
//!
//! Phase 5.1 ships single-tile sort only (`n ≤
//! crate::kernels::TILE_ELEMENTS`); the dispatcher returns
//! `GpuError::Unrecoverable("…multi-block in Phase 5.2")` for
//! larger inputs.

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
use crate::kernels::{emit_sort_source, TILE_ELEMENTS};
use crate::{reply_err, CubDispatchBase, CubDispatchCtx, KernelSourceCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

impl SortDirection {
    pub fn op_suffix(self) -> &'static str {
        match self {
            SortDirection::Ascending => "asc",
            SortDirection::Descending => "desc",
        }
    }
}

pub struct SortRequest<K: CudaDtype, V: CudaDtype = u32> {
    pub direction: SortDirection,
    pub keys_in: GpuRef<K>,
    pub keys_out: GpuRef<K>,
    /// Optional value buffer for key-value sort. When `None`, the
    /// dispatch invokes `DeviceRadixSort::SortKeys`; when `Some(...)`,
    /// `SortPairs`.
    pub values_in: Option<GpuRef<V>>,
    pub values_out: Option<GpuRef<V>>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<(K, V)>,
}

impl<K: CudaDtype> SortRequest<K, u32> {
    pub fn keys_only(
        direction: SortDirection,
        keys_in: GpuRef<K>,
        keys_out: GpuRef<K>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            direction,
            keys_in,
            keys_out,
            values_in: None,
            values_out: None,
            reply,
            _phantom: PhantomData,
        }
    }
}

impl<K: CudaDtype, V: CudaDtype> SortRequest<K, V> {
    pub fn pairs(
        direction: SortDirection,
        keys_in: GpuRef<K>,
        keys_out: GpuRef<K>,
        values_in: GpuRef<V>,
        values_out: GpuRef<V>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            direction,
            keys_in,
            keys_out,
            values_in: Some(values_in),
            values_out: Some(values_out),
            reply,
            _phantom: PhantomData,
        }
    }
}

pub trait CubSortDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<K, V> CubDispatchBase for SortRequest<K, V>
where
    K: CudaDtype,
    V: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        match self.direction {
            SortDirection::Ascending => "sort_asc",
            SortDirection::Descending => "sort_desc",
        }
    }
    fn dtype_name(&self) -> &'static str {
        <K as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<K, V> CubSortDispatch for SortRequest<K, V>
where
    K: CudaDtype,
    V: CudaDtype,
{
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>) {
        let nvrtc = match ctx.nvrtc {
            Some(n) => n.clone(),
            None => {
                reply_err(
                    self.reply,
                    GpuError::Unrecoverable(
                        "atomr-accel-cub::CubSort: NvrtcActor not wired into CubActor".into(),
                    ),
                );
                return;
            }
        };
        let n = self.keys_in.len();
        if n > TILE_ELEMENTS as usize {
            reply_err(
                self.reply,
                GpuError::Unrecoverable(format!(
                    "atomr-accel-cub::CubSort: n={n} exceeds single-block limit ({}); \
                     multi-block radix sort lands in Phase 5.2",
                    TILE_ELEMENTS
                )),
            );
            return;
        }

        let cache = ctx.kernel_cache.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(run_sort::<K, V>(me, nvrtc, cache, arch));
    }
}

async fn run_sort<K: CudaDtype, V: CudaDtype>(
    req: SortRequest<K, V>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) {
    let SortRequest {
        direction,
        keys_in,
        keys_out,
        values_in,
        values_out,
        reply,
        ..
    } = req;
    let paired = values_in.is_some() && values_out.is_some();
    let result = compile_and_launch::<K, V>(
        direction,
        paired,
        keys_in,
        keys_out,
        values_in,
        values_out,
        nvrtc,
        cache,
        arch,
    )
    .await;
    let _ = reply.send(result);
}

#[allow(clippy::too_many_arguments)]
async fn compile_and_launch<K: CudaDtype, V: CudaDtype>(
    direction: SortDirection,
    paired: bool,
    keys_in: GpuRef<K>,
    keys_out: GpuRef<K>,
    values_in: Option<GpuRef<V>>,
    values_out: Option<GpuRef<V>>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let kdt = <K as AccelDtype>::NAME;
    let vdt = <V as AccelDtype>::NAME;
    let (src, kname) = emit_sort_source::<K, V>(direction, paired);
    // Encode paired-ness in the cache key so keys-only and paired
    // sorts of the same K dtype don't collide.
    let op = format!(
        "sort_{}_{}",
        direction.op_suffix(),
        if paired { "pairs" } else { "keys" }
    );
    let dtype = if paired {
        format!("{kdt}_{vdt}")
    } else {
        kdt.to_string()
    };

    let handle = compile_or_get_handle(nvrtc.clone(), cache, op, dtype, src, kname, arch).await?;

    let n = keys_in.len();
    let args = if paired {
        vec![
            KernelArg::DevSlice(Box::new(keys_in)),
            KernelArg::DevSlice(Box::new(keys_out)),
            KernelArg::DevSlice(Box::new(values_in.unwrap())),
            KernelArg::DevSlice(Box::new(values_out.unwrap())),
            KernelArg::Usize(n),
        ]
    } else {
        vec![
            KernelArg::DevSlice(Box::new(keys_in)),
            KernelArg::DevSlice(Box::new(keys_out)),
            KernelArg::Usize(n),
        ]
    };

    launch(&nvrtc, handle, args, launch_config_single_block()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 5: round-trip a sort request through the trait surface.
    #[test]
    fn sort_request_round_trip() {
        // op_suffix uniqueness.
        let suffixes: std::collections::HashSet<&str> = [
            SortDirection::Ascending.op_suffix(),
            SortDirection::Descending.op_suffix(),
        ]
        .into_iter()
        .collect();
        assert_eq!(suffixes.len(), 2);

        // Cache-key matrix is unique across (direction, dtype).
        let dtypes = ["f32", "f64", "i32", "u32", "i64", "u64"];
        let mut seen = std::collections::HashSet::new();
        for d in [SortDirection::Ascending, SortDirection::Descending] {
            for dt in dtypes {
                let op = match d {
                    SortDirection::Ascending => "sort_asc",
                    SortDirection::Descending => "sort_desc",
                };
                let k = crate::kernel_key(op, dt);
                assert!(seen.insert(k.clone()), "sort key collision: {k}");
            }
        }
        assert_eq!(seen.len(), 2 * dtypes.len());
    }
}
