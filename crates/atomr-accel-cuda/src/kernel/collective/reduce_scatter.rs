//! Typed ReduceScatter request — generic over `T: NcclReduceSupported`.

use std::sync::Arc;

use cudarc::nccl::ReduceOp;
use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// ReduceScatter: each rank contributes `send` (length `N *
/// world_size`) and receives a reduced shard of length `N` into
/// `recv`.
pub struct ReduceScatterRequest<T: NcclReduceSupported> {
    pub send: GpuRef<T>,
    pub recv: GpuRef<T>,
    pub op: ReduceOp,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for ReduceScatterRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.send.device_id().or_else(|| self.recv.device_id())
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let ReduceScatterRequest {
            send,
            recv,
            op,
            reply,
        } = *self;
        let send_slice = match send.access() {
            Ok(s) => s.clone(),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let recv_slice = match recv.access() {
            Ok(s) => s.clone(),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let mut recv_owned = match Arc::try_unwrap(recv_slice) {
            Ok(s) => s,
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "ReduceScatter recv buffer has multiple live references".into(),
                )));
                return;
            }
        };
        let res = ctx
            .comm
            .reduce_scatter(&*send_slice, &mut recv_owned, &op)
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("reduce_scatter: {e:?}"),
            });
        let _ = reply.send(res.map(|_| ()));
        drop(recv_owned);
        drop(send_slice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_scatter_request_round_trip() {
        fn assert_supported<T: NcclReduceSupported>() {
            let _ = <T as NcclReduceSupported>::dispatch_dtype();
        }
        assert_supported::<f32>();
        assert_supported::<f64>();
        assert_supported::<i8>();
        assert_supported::<u8>();
        assert_supported::<i32>();
        assert_supported::<u32>();
        assert_supported::<i64>();
        assert_supported::<u64>();
        #[cfg(feature = "f16")]
        {
            assert_supported::<half::f16>();
            assert_supported::<half::bf16>();
        }
    }
}
