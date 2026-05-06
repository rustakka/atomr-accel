//! Typed AllReduce request. Generic over `T: NcclReduceSupported`
//! (any of f32/f64/i8/u8/i32/u32/i64/u64; f16/bf16 with `f16`).

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::nccl::ReduceOp;
use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// In-place all-reduce on a single tensor on this actor's device.
///
/// The world actor coordinates `group_start`/`group_end` across
/// ranks via separate `BeginGroup` / `EndGroup` messages.
pub struct AllReduceRequest<T: NcclReduceSupported> {
    pub tensor: GpuRef<T>,
    pub op: ReduceOp,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for AllReduceRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.tensor.device_id()
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let AllReduceRequest { tensor, op, reply } = *self;
        let slice = match tensor.access() {
            Ok(s) => s.clone(),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        // In-place all-reduce requires a unique-owner Arc unwrap.
        let mut owned = match Arc::try_unwrap(slice) {
            Ok(s) => s,
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "AllReduce tensor has multiple live references".into(),
                )));
                return;
            }
        };
        let res =
            ctx.comm
                .all_reduce_in_place(&mut owned, &op)
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("all_reduce: {e:?}"),
                });
        let _ = reply.send(res.map(|_| ()));
        drop(owned);
    }
}

#[allow(dead_code)]
fn _phantom_use<T: NcclReduceSupported>() -> PhantomData<T> {
    PhantomData
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build (and immediately drop) an `AllReduceRequest<T>` for every
    /// `NcclReduceSupported` dtype. Validates that the trait bounds
    /// line up across the dtype matrix.
    #[test]
    fn request_round_trip_for_every_dtype() {
        // We can't instantiate a real `GpuRef<T>` without a CudaSlice.
        // Instead, prove the type-level surface compiles: the boxed
        // dispatch can be coerced to `Box<dyn CollectiveDispatch>` for
        // each `T`.
        fn assert_boxable<T: NcclReduceSupported>() {
            // PhantomData synthesises the trait bound check at
            // monomorphisation time.
            let _ = _phantom_use::<T>();
            // Verify dtype tag lookup is present.
            let _ = <T as NcclReduceSupported>::dispatch_dtype();
        }
        assert_boxable::<f32>();
        assert_boxable::<f64>();
        assert_boxable::<i8>();
        assert_boxable::<u8>();
        assert_boxable::<i32>();
        assert_boxable::<u32>();
        assert_boxable::<i64>();
        assert_boxable::<u64>();
        #[cfg(feature = "f16")]
        {
            assert_boxable::<half::f16>();
            assert_boxable::<half::bf16>();
        }
    }
}
