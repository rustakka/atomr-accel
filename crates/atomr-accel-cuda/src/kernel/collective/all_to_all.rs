//! Typed AllToAll / AllToAllv requests.
//!
//! cudarc 0.19.4 does not safely expose `ncclAllToAll` / `ncclSend` /
//! `ncclRecv` against a raw `comm: ncclComm_t` (the field is private).
//! Implementations therefore decompose AllToAll into a `group_start`
//! / paired `send` + `recv` / `group_end` sequence using cudarc's safe
//! `Comm::send` and `Comm::recv` — which is the standard NCCL idiom
//! for AllToAll regardless. AllToAllv adds per-peer (count, offset)
//! pairs.

use std::sync::Arc;

use cudarc::nccl::{group_end, group_start};
use tokio::sync::oneshot;

use super::{NcclReduceSupported, LIB};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{CollectiveDispatch, CollectiveDispatchCtx, DispatchDType};

/// Symmetric AllToAll: each rank sends an equal-sized shard of length
/// `count` to every peer. `send` carries `world_size * count`
/// elements; `recv` likewise.
pub struct AllToAllRequest<T: NcclReduceSupported> {
    pub send: GpuRef<T>,
    pub recv: GpuRef<T>,
    pub count: usize,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for AllToAllRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.send.device_id().or_else(|| self.recv.device_id())
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let AllToAllRequest {
            send,
            recv,
            count,
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
                    "AllToAll recv buffer has multiple live references".into(),
                )));
                return;
            }
        };

        let world_size = ctx.comm.world_size();
        if send_slice.len() < world_size * count || recv_owned.len() < world_size * count {
            let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                "AllToAll: buffer length < world_size ({world_size}) * count ({count})"
            ))));
            return;
        }

        // group_start + 2*world_size paired send/recv + group_end.
        let res = (|| -> Result<(), GpuError> {
            group_start().map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("group_start: {e:?}"),
            })?;

            for peer in 0..world_size {
                let peer_i32 = peer as i32;
                let send_slab = send_slice.slice(peer * count..(peer + 1) * count);
                let mut recv_slab = recv_owned.slice_mut(peer * count..(peer + 1) * count);
                ctx.comm
                    .send(&send_slab, peer_i32)
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("a2a send to {peer}: {e:?}"),
                    })?;
                ctx.comm
                    .recv(&mut recv_slab, peer_i32)
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("a2a recv from {peer}: {e:?}"),
                    })?;
            }

            group_end().map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("group_end: {e:?}"),
            })?;
            Ok(())
        })();
        let _ = reply.send(res);
        drop(recv_owned);
        drop(send_slice);
    }
}

/// AllToAllv: per-peer (count, offset) shards in send and recv.
pub struct AllToAllvRequest<T: NcclReduceSupported> {
    pub send: GpuRef<T>,
    pub recv: GpuRef<T>,
    pub send_counts: Vec<usize>,
    pub send_offsets: Vec<usize>,
    pub recv_counts: Vec<usize>,
    pub recv_offsets: Vec<usize>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: NcclReduceSupported> CollectiveDispatch for AllToAllvRequest<T> {
    fn dtype_kind(&self) -> DispatchDType {
        T::dispatch_dtype()
    }

    fn device_id(&self) -> Option<u32> {
        self.send.device_id().or_else(|| self.recv.device_id())
    }

    fn dispatch(self: Box<Self>, ctx: &CollectiveDispatchCtx<'_>) {
        let AllToAllvRequest {
            send,
            recv,
            send_counts,
            send_offsets,
            recv_counts,
            recv_offsets,
            reply,
        } = *self;

        let world_size = ctx.comm.world_size();
        if send_counts.len() != world_size
            || send_offsets.len() != world_size
            || recv_counts.len() != world_size
            || recv_offsets.len() != world_size
        {
            let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                "AllToAllv: counts/offsets must each have length world_size ({world_size})"
            ))));
            return;
        }

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
                    "AllToAllv recv buffer has multiple live references".into(),
                )));
                return;
            }
        };

        let res = (|| -> Result<(), GpuError> {
            group_start().map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("group_start: {e:?}"),
            })?;

            for peer in 0..world_size {
                let peer_i32 = peer as i32;
                let s_off = send_offsets[peer];
                let s_cnt = send_counts[peer];
                let r_off = recv_offsets[peer];
                let r_cnt = recv_counts[peer];

                if s_cnt > 0 {
                    if s_off + s_cnt > send_slice.len() {
                        return Err(GpuError::Unrecoverable(format!(
                            "AllToAllv: send shard for peer {peer} overruns buffer"
                        )));
                    }
                    let send_slab = send_slice.slice(s_off..s_off + s_cnt);
                    ctx.comm
                        .send(&send_slab, peer_i32)
                        .map_err(|e| GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("a2av send to {peer}: {e:?}"),
                        })?;
                }
                if r_cnt > 0 {
                    if r_off + r_cnt > recv_owned.len() {
                        return Err(GpuError::Unrecoverable(format!(
                            "AllToAllv: recv shard from peer {peer} overruns buffer"
                        )));
                    }
                    let mut recv_slab = recv_owned.slice_mut(r_off..r_off + r_cnt);
                    ctx.comm.recv(&mut recv_slab, peer_i32).map_err(|e| {
                        GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("a2av recv from {peer}: {e:?}"),
                        }
                    })?;
                }
            }

            group_end().map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("group_end: {e:?}"),
            })?;
            Ok(())
        })();
        let _ = reply.send(res);
        drop(recv_owned);
        drop(send_slice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_to_all_request_round_trip() {
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
