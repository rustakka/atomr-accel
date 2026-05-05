//! Typed point-to-point Send / Recv requests.

use std::sync::Arc;

use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// Send `data` to rank `peer`. Per NCCL, must be paired with a
/// matching `RecvRequest` on the peer inside the same group call.
pub struct SendRequest<T: NcclReduceSupported> {
    pub data: GpuRef<T>,
    pub peer: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for SendRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.data.device_id()
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let SendRequest { data, peer, reply } = *self;
        let slice = match data.access() {
            Ok(s) => s.clone(),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let res = ctx
            .comm
            .send(&*slice, peer)
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("send: {e:?}"),
            });
        let _ = reply.send(res);
        drop(slice);
    }
}

/// Receive a buffer from rank `peer` into `data`.
pub struct RecvRequest<T: NcclReduceSupported> {
    pub data: GpuRef<T>,
    pub peer: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for RecvRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.data.device_id()
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let RecvRequest { data, peer, reply } = *self;
        let slice = match data.access() {
            Ok(s) => s.clone(),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let mut owned = match Arc::try_unwrap(slice) {
            Ok(s) => s,
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "Recv buffer has multiple live references".into(),
                )));
                return;
            }
        };
        let res = ctx
            .comm
            .recv(&mut owned, peer)
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("recv: {e:?}"),
            });
        let _ = reply.send(res.map(|_| ()));
        drop(owned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both `SendRequest<T>` and `RecvRequest<T>` impl
    /// `CollectiveDispatch` for the full reduce-supported dtype set.
    #[test]
    fn send_recv_request_round_trip() {
        fn assert_send_recv<T: NcclReduceSupported>() {
            let _ = <T as NcclReduceSupported>::dispatch_dtype();
        }
        assert_send_recv::<f32>();
        assert_send_recv::<f64>();
        assert_send_recv::<i8>();
        assert_send_recv::<u8>();
        assert_send_recv::<i32>();
        assert_send_recv::<u32>();
        assert_send_recv::<i64>();
        assert_send_recv::<u64>();
        #[cfg(feature = "f16")]
        {
            assert_send_recv::<half::f16>();
            assert_send_recv::<half::bf16>();
        }
    }
}
