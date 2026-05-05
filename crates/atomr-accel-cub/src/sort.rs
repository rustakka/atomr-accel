//! `cub::DeviceRadixSort` — keys / pairs, ascending or descending.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;

use crate::{reply_err, CubDispatchBase, CubDispatchCtx};

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
        <K as atomr_accel_cuda::dtype::AccelDtype>::NAME
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
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        let op = match self.direction {
            SortDirection::Ascending => "sort_asc",
            SortDirection::Descending => "sort_desc",
        };
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubSort::{}<{}> — kernel compile path lands in Phase 5.1",
                op,
                <K as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
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
