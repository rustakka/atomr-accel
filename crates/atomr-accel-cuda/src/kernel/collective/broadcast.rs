//! Typed Broadcast request — generic over `T: NcclReduceSupported`.

use std::sync::Arc;

use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// In-place broadcast: every rank's `data` buffer is overwritten with
/// the contents of `data` on the rank whose `comm.rank() == root`.
pub struct BroadcastRequest<T: NcclReduceSupported> {
    pub data: GpuRef<T>,
    pub root: usize,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for BroadcastRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.data.device_id()
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let BroadcastRequest { data, root, reply } = *self;
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
                    "Broadcast data has multiple live references".into(),
                )));
                return;
            }
        };
        let root_i32 = match i32::try_from(root) {
            Ok(r) => r,
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                    "Broadcast: root {root} does not fit in i32"
                ))));
                return;
            }
        };
        let res = ctx
            .comm
            .broadcast_in_place(&mut owned, root_i32)
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("broadcast_in_place: {e:?}"),
            });
        let _ = reply.send(res.map(|_| ()));
        drop(owned);
    }
}
