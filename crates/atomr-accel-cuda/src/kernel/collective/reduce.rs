//! Typed Reduce request: reduce-to-root variant of AllReduce.

use std::sync::Arc;

use cudarc::nccl::ReduceOp;
use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// Reduce: each rank contributes `send`; the result lands in `recv`
/// on `root`. On non-root ranks, `recv` may be `None`.
pub struct ReduceRequest<T: NcclReduceSupported> {
    pub send: GpuRef<T>,
    pub recv: Option<GpuRef<T>>,
    pub op: ReduceOp,
    pub root: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for ReduceRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.send
            .device_id()
            .or_else(|| self.recv.as_ref().and_then(|r| r.device_id()))
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let ReduceRequest {
            send,
            recv,
            op,
            root,
            reply,
        } = *self;
        let send_slice = match send.access() {
            Ok(s) => s.clone(),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        // On the root rank we must have a recv buffer; on other
        // ranks recv is allowed to be None.
        let recv_owned: Option<cudarc::driver::CudaSlice<T>> = match recv {
            Some(r) => match r.access() {
                Ok(s) => match Arc::try_unwrap(s.clone()) {
                    Ok(o) => Some(o),
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "Reduce recv buffer has multiple live references".into(),
                        )));
                        return;
                    }
                },
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            },
            None => None,
        };

        let res = match recv_owned {
            Some(mut owned) => ctx
                .comm
                .reduce(&*send_slice, Some(&mut owned), &op, root)
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("reduce: {e:?}"),
                })
                .map(|_| {
                    drop(owned);
                }),
            None => ctx
                .comm
                .reduce::<_, cudarc::driver::CudaSlice<T>, T>(&*send_slice, None, &op, root)
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("reduce: {e:?}"),
                })
                .map(|_| ()),
        };
        let _ = reply.send(res);
        drop(send_slice);
    }
}
