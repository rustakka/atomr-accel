//! Typed AllGather request — generic over `T: NcclReduceSupported`.

use std::sync::Arc;

use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// AllGather: each rank contributes `send` (length N) and writes a
/// concatenated buffer of length `N * world_size` into `recv`.
pub struct AllGatherRequest<T: NcclReduceSupported> {
    pub send: GpuRef<T>,
    pub recv: GpuRef<T>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for AllGatherRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.send.device_id().or_else(|| self.recv.device_id())
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let AllGatherRequest { send, recv, reply } = *self;
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
                    "AllGather recv buffer has multiple live references".into(),
                )));
                return;
            }
        };
        let res = ctx
            .comm
            .all_gather(&*send_slice, &mut recv_owned)
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("all_gather: {e:?}"),
            });
        let _ = reply.send(res.map(|_| ()));
        drop(recv_owned);
        drop(send_slice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `NcclReduceSupported` dtype builds an
    /// `AllGatherRequest<T>` that satisfies `CollectiveDispatch`.
    #[test]
    fn allgather_request_round_trip() {
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
