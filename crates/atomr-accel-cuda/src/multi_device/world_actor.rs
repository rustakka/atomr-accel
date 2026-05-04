//! `NcclWorldActor` — supervises a multi-GPU NCCL world.
//!
//! Spawns N `DeviceActor`s (one per device id in the world config),
//! waits for each to report `ContextReady`, snapshots their CUDA
//! contexts to mint per-rank streams, calls
//! `Comm::from_devices(streams)` to build the NCCL group, and spawns
//! one `CollectiveActor` per rank.
//!
//! Routing model: when the world receives an `AllReduceF32` with
//! `Vec<GpuRef<f32>>`, it cross-validates each `GpuRef`'s device-id
//! and dispatches to the matching `CollectiveActor`. Replies arrive
//! via per-rank oneshots; the world joins them and reports a single
//! result.

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::nccl::Comm;
use atomr_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::completion::{CompletionStrategy, HostFnCompletion};
use crate::device::{DeviceActor, DeviceConfig, DeviceMsg};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::{CollectiveActor, CollectiveMsg, ReduceOp};

#[derive(Debug, Clone)]
pub struct NcclWorldConfig {
    pub device_ids: Vec<u32>,
    pub root: usize,
}

impl NcclWorldConfig {
    pub fn new(device_ids: Vec<u32>) -> Self {
        Self {
            device_ids,
            root: 0,
        }
    }
}

pub enum NcclWorldMsg {
    AllReduceF32 {
        tensors: Vec<GpuRef<f32>>,
        op: ReduceOp,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Internal: child reports it's ready.
    ChildReady {
        device_idx: usize,
        device_ref: ActorRef<DeviceMsg>,
    },
    /// Internal: a per-device generation watch fired, meaning a
    /// `ContextActor` rebuilt its CUDA context and the existing NCCL
    /// communicators are now invalid. The world tears the
    /// collectives down and rebuilds.
    DeviceContextChanged {
        device_idx: usize,
        new_generation: u64,
    },
}

pub struct NcclWorldActor {
    config: NcclWorldConfig,
    devices: Vec<Option<ActorRef<DeviceMsg>>>,
    collectives: Vec<Option<ActorRef<CollectiveMsg>>>,
    /// Set true once `try_build_world` has run successfully.
    built: bool,
    /// Per-device generation last seen. When a device reports a
    /// generation change above this value, the world is rebuilt.
    last_generation: Vec<u64>,
    #[allow(dead_code)]
    completion: Arc<dyn CompletionStrategy>,
}

impl NcclWorldActor {
    pub fn props(config: NcclWorldConfig) -> Props<Self> {
        Props::create(move || {
            let n = config.device_ids.len();
            NcclWorldActor {
                config: config.clone(),
                devices: (0..n).map(|_| None).collect(),
                collectives: (0..n).map(|_| None).collect(),
                built: false,
                last_generation: vec![0; n],
                completion: Arc::new(HostFnCompletion::new()),
            }
        })
    }

    async fn try_build_world(&mut self, ctx: &mut Context<Self>) {
        if self.built {
            return;
        }
        if self.devices.iter().any(|d| d.is_none()) {
            return;
        }

        // Snapshot each device's CudaContext.
        let mut snaps = Vec::with_capacity(self.devices.len());
        for d in &self.devices {
            let dref = d.as_ref().unwrap();
            let (tx, rx) = oneshot::channel();
            dref.tell(DeviceMsg::SnapshotContext { reply: tx });
            match rx.await {
                Ok(Some(c)) => snaps.push(c),
                _ => {
                    warn!("NcclWorldActor: a device reported no context; aborting world-build");
                    return;
                }
            }
        }

        // Mint a fresh stream per device for the comm.
        let mut streams = Vec::with_capacity(snaps.len());
        for c in &snaps {
            match c.new_stream() {
                Ok(s) => streams.push(s),
                Err(e) => {
                    warn!(error = %e, "NcclWorldActor: new_stream failed");
                    return;
                }
            }
        }

        // Build the NCCL world. This can panic on no-driver hosts;
        // catch_unwind preserves the actor.
        let comms_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Comm::from_devices(streams.clone())
        }));
        let comms = match comms_res {
            Ok(Ok(cs)) => cs,
            Ok(Err(e)) => {
                warn!(error = ?e, "NcclWorldActor: Comm::from_devices failed");
                return;
            }
            Err(_) => {
                warn!("NcclWorldActor: NCCL not loadable on this host");
                return;
            }
        };

        // Spawn one CollectiveActor per rank as a child of this
        // world. Each takes its `Comm` by move.
        for (i, comm) in comms.into_iter().enumerate() {
            // We need the per-device DeviceState to drive the
            // CollectiveActor's cross-device validation. We don't
            // have a public `state` accessor on DeviceActor; for
            // F4.x we pass a fresh DeviceState per rank — the
            // device-id matches the request's device-id which is
            // what cross-validation needs.
            let state = Arc::new(crate::device::DeviceState::new(self.config.device_ids[i]));
            let comp: Arc<dyn CompletionStrategy> = Arc::new(HostFnCompletion::new());
            let props = CollectiveActor::props_for_rank(comm, state, comp);
            match ctx.spawn::<CollectiveActor>(props, &format!("nccl-{i}")) {
                Ok(r) => self.collectives[i] = Some(r),
                Err(e) => {
                    warn!(error = %e, "spawn CollectiveActor[{i}] failed");
                    return;
                }
            }
        }
        self.built = true;
        info!(devices = self.devices.len(), "NcclWorldActor: world built");
    }

    fn dispatch_all_reduce_f32(
        &self,
        tensors: Vec<GpuRef<f32>>,
        op: ReduceOp,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) {
        if !self.built {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "NcclWorldActor: world not built yet".into(),
            )));
            return;
        }
        if tensors.len() != self.config.device_ids.len() {
            let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                "AllReduce: expected {} tensors, got {}",
                self.config.device_ids.len(),
                tensors.len()
            ))));
            return;
        }
        for (i, t) in tensors.iter().enumerate() {
            if let Some(d) = t.device_id() {
                if d != self.config.device_ids[i] {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "AllReduce: tensor[{i}] on device {d}, expected {}",
                        self.config.device_ids[i]
                    ))));
                    return;
                }
            }
        }
        // Per-rank dispatch: send AllReduceF32 to each collective and
        // await all replies, then post a single combined reply on
        // the user's channel. NCCL requires
        // group_start/group_end framing only when issuing multiple
        // ops in a row; a single AllReduce per rank is fine
        // standalone.
        let collectives: Vec<_> = self
            .collectives
            .iter()
            .map(|c| c.as_ref().unwrap().clone())
            .collect();
        tokio::spawn(async move {
            let mut rxs = Vec::with_capacity(tensors.len());
            for (c, t) in collectives.into_iter().zip(tensors) {
                let (tx, rx) = oneshot::channel();
                let op_clone = match op {
                    ReduceOp::Sum => ReduceOp::Sum,
                    ReduceOp::Prod => ReduceOp::Prod,
                    ReduceOp::Max => ReduceOp::Max,
                    ReduceOp::Min => ReduceOp::Min,
                    ReduceOp::Avg => ReduceOp::Avg,
                };
                c.tell(CollectiveMsg::AllReduceF32 {
                    tensor: t,
                    op: op_clone,
                    reply: tx,
                });
                rxs.push(rx);
            }
            let mut combined = Ok(());
            for rx in rxs {
                match rx.await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        combined = Err(e);
                        break;
                    }
                    Err(_) => {
                        combined = Err(GpuError::Unrecoverable(
                            "AllReduce: a collective actor dropped its reply".into(),
                        ));
                        break;
                    }
                }
            }
            let _ = reply.send(combined);
        });
    }
}

#[async_trait]
impl Actor for NcclWorldActor {
    type Msg = NcclWorldMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        let world_ref = ctx.self_ref().clone();
        for (i, &ord) in self.config.device_ids.iter().enumerate() {
            let cfg = DeviceConfig::new(ord);
            match ctx.spawn::<DeviceActor>(DeviceActor::props(cfg), &format!("dev-{i}")) {
                Ok(r) => {
                    self.devices[i] = Some(r.clone());
                    let world = world_ref.clone();
                    let dr = r.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        world.tell(NcclWorldMsg::ChildReady {
                            device_idx: i,
                            device_ref: dr,
                        });
                    });
                }
                Err(e) => panic!("Unrecoverable: spawn DeviceActor[{i}]: {e}"),
            }
        }
    }

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: NcclWorldMsg) {
        match msg {
            NcclWorldMsg::ChildReady {
                device_idx,
                device_ref,
            } => {
                self.devices[device_idx] = Some(device_ref.clone());

                // Subscribe to this device's generation watch and
                // bridge changes into `DeviceContextChanged` events
                // on our own mailbox.
                let world_ref = ctx.self_ref().clone();
                let dr = device_ref.clone();
                tokio::spawn(async move {
                    let watch_rx_res = dr
                        .ask_with(
                            move |tx| DeviceMsg::WatchGeneration { reply: tx },
                            std::time::Duration::from_secs(5),
                        )
                        .await;
                    let mut rx = match watch_rx_res {
                        Ok(rx) => rx,
                        Err(_) => return,
                    };
                    let mut last = *rx.borrow();
                    while rx.changed().await.is_ok() {
                        let gen = *rx.borrow();
                        if gen != last {
                            last = gen;
                            world_ref.tell(NcclWorldMsg::DeviceContextChanged {
                                device_idx,
                                new_generation: gen,
                            });
                        }
                    }
                });

                self.try_build_world(ctx).await;
            }
            NcclWorldMsg::DeviceContextChanged {
                device_idx,
                new_generation,
            } => {
                let prev = self.last_generation.get(device_idx).copied().unwrap_or(0);
                if new_generation <= prev {
                    return;
                }
                self.last_generation[device_idx] = new_generation;
                if !self.built {
                    return;
                }
                tracing::warn!(
                    device_idx,
                    new_generation,
                    "NcclWorldActor: device context rebuilt — tearing down NCCL world"
                );
                // Tear down all collective actors. They were spawned
                // as children so stop them via their refs.
                for c in self.collectives.iter_mut() {
                    if let Some(c) = c.take() {
                        c.stop();
                    }
                }
                self.built = false;
                // Try to rebuild now that the device fleet is back.
                self.try_build_world(ctx).await;
            }
            NcclWorldMsg::AllReduceF32 { tensors, op, reply } => {
                self.dispatch_all_reduce_f32(tensors, op, reply);
            }
        }
    }
}
