//! `CollectiveActor` — wraps an [`cudarc::nccl::Comm`] for one rank
//! within an `NcclWorldActor` group.
//!
//! Each `CollectiveActor` is bound to one specific
//! [`crate::device::DeviceState`] (one rank in the NCCL world). The
//! parent `NcclWorldActor` spawns N of these (one per device) and
//! routes messages to all of them in a `group_start/group_end`
//! pair where appropriate.

use std::sync::Arc;

use async_trait::async_trait;
pub use cudarc::nccl::ReduceOp;
use cudarc::nccl::{group_end, group_start, Comm};
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

const LIB: &str = "nccl";

pub enum CollectiveMsg {
    /// In-place all-reduce on a single tensor on this actor's device.
    /// The world actor coordinates `group_start`/`group_end` across
    /// ranks via separate `BeginGroup` / `EndGroup` messages.
    AllReduceF32 {
        tensor: GpuRef<f32>,
        op: ReduceOp,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Broadcast on the local rank: if `is_root`, send `data`; else
    /// receive into `data`.
    BroadcastF32 {
        data: GpuRef<f32>,
        root: usize,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    BeginGroup {
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    EndGroup {
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct CollectiveActor {
    inner: CollectiveInner,
}

struct SendComm(Comm);
unsafe impl Send for SendComm {}
unsafe impl Sync for SendComm {}

#[allow(dead_code)]
enum CollectiveInner {
    Real {
        comm: SendComm,
        state: Arc<DeviceState>,
        completion: Arc<dyn CompletionStrategy>,
    },
    Mock,
}

impl CollectiveActor {
    /// Build a `Props<CollectiveActor>` capturing a single-rank Comm.
    /// Each call constructs a single-shot factory (the comm cannot
    /// be cloned). The returned Props panics on second
    /// instantiation — supervisor restart loops therefore become
    /// fatal for NCCL world actors. NcclWorldActor handles this by
    /// orchestrating world rebuilds explicitly.
    pub fn props_for_rank(
        comm: Comm,
        state: Arc<DeviceState>,
        completion: Arc<dyn CompletionStrategy>,
    ) -> Props<Self> {
        use parking_lot::Mutex;
        let comm_slot = Arc::new(Mutex::new(Some(SendComm(comm))));
        Props::create(move || {
            let comm = comm_slot
                .lock()
                .take()
                .expect("Unrecoverable: CollectiveActor restart with consumed Comm — NcclWorldActor must rebuild the world");
            CollectiveActor {
                inner: CollectiveInner::Real {
                    comm,
                    state: state.clone(),
                    completion: completion.clone(),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| CollectiveActor {
            inner: CollectiveInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for CollectiveActor {
    type Msg = CollectiveMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: CollectiveMsg) {
        match &self.inner {
            CollectiveInner::Mock => mock_reply(msg),
            CollectiveInner::Real { comm, state, .. } => {
                if let Some(dev) = msg_device_id(&msg) {
                    if dev != state.device_id() {
                        let _ = msg_reply_err(
                            msg,
                            GpuError::Unrecoverable(format!(
                                "collective tensor on wrong device: expected {}, got {}",
                                state.device_id(),
                                dev
                            )),
                        );
                        return;
                    }
                }
                handle_real(comm, msg);
            }
        }
    }
}

fn msg_device_id(msg: &CollectiveMsg) -> Option<u32> {
    match msg {
        CollectiveMsg::AllReduceF32 { tensor, .. } => tensor.device_id(),
        CollectiveMsg::BroadcastF32 { data, .. } => data.device_id(),
        _ => None,
    }
}

fn mock_reply(msg: CollectiveMsg) {
    let err = || GpuError::Unrecoverable("CollectiveActor in mock mode".into());
    msg_reply_err(msg, err());
}

fn msg_reply_err(msg: CollectiveMsg, e: GpuError) {
    match msg {
        CollectiveMsg::AllReduceF32 { reply, .. } => {
            let _ = reply.send(Err(e));
        }
        CollectiveMsg::BroadcastF32 { reply, .. } => {
            let _ = reply.send(Err(e));
        }
        CollectiveMsg::BeginGroup { reply } => {
            let _ = reply.send(Err(e));
        }
        CollectiveMsg::EndGroup { reply } => {
            let _ = reply.send(Err(e));
        }
    }
}

fn handle_real(comm: &SendComm, msg: CollectiveMsg) {
    match msg {
        CollectiveMsg::AllReduceF32 { tensor, op, reply } => {
            let slice = match tensor.access() {
                Ok(s) => s.clone(),
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            };
            // In-place all-reduce avoids the unwrap-Arc dance.
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
                comm.0
                    .all_reduce_in_place(&mut owned, &op)
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("all_reduce: {e:?}"),
                    });
            // We don't await completion here — NcclWorldActor's
            // EndGroup awaits stream completion at the world level.
            let _ = reply.send(res.map(|_| ()));
            // owned drops at end of scope; the stream copy dependency
            // means the GPU operation will outlive the host owner
            // only if we synchronize in EndGroup. Caller
            // contract.
            drop(owned);
        }
        CollectiveMsg::BroadcastF32 { data, root, reply } => {
            let slice = match data.access() {
                Ok(s) => s.clone(),
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            };
            let owned = match Arc::try_unwrap(slice) {
                Ok(s) => s,
                Err(_) => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "Broadcast data has multiple live references".into(),
                    )));
                    return;
                }
            };
            // F4 skeleton: cudarc's broadcast takes &S send + &mut R
            // recv as separate buffers; supporting in-place requires
            // a small temp on root or a separate recv buffer per
            // rank. Defer until DataParallelTrainer wires this in.
            let _ = root;
            let _ = comm;
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: "BroadcastF32: F4 skeleton; needs separate send/recv buffers".into(),
            }));
            drop(owned);
        }
        CollectiveMsg::BeginGroup { reply } => {
            let res = group_start()
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("group_start: {e:?}"),
                })
                .map(|_| ());
            let _ = reply.send(res);
        }
        CollectiveMsg::EndGroup { reply } => {
            let res = group_end()
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("group_end: {e:?}"),
                })
                .map(|_| ());
            let _ = reply.send(res);
        }
    }
}
