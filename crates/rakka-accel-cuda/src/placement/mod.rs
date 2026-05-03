//! `PlacementActor` — picks the best-fit `DeviceActor` for each
//! request based on a configurable [`PlacementPolicy`].
//!
//! Polls each device's `DeviceMsg::Stats` periodically (default 250
//! ms) to maintain a load snapshot. Callers send `Pick` to receive
//! a `DeviceChoice` with the selected `ActorRef<DeviceMsg>`.

#[cfg(feature = "cluster")]
pub mod sharded;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use crate::device::{DeviceLoad, DeviceMsg};
use crate::error::GpuError;
use crate::stream::Priority;

#[derive(Debug, Clone, Copy, Default)]
pub struct PlacementHints {
    pub min_free_bytes: usize,
    pub min_compute_cap: Option<(i32, i32)>,
    pub priority: Option<Priority>,
}

pub struct DeviceChoice {
    pub device_id: u32,
    pub device: ActorRef<DeviceMsg>,
    pub load: DeviceLoad,
}

pub trait PlacementPolicy: Send + Sync + 'static {
    fn choose(&self, hints: &PlacementHints, candidates: &[(u32, &DeviceLoad)]) -> Option<u32>;
}

/// Round-robin policy. Ignores `hints`.
pub struct RoundRobinPolicy {
    cursor: Mutex<usize>,
}

impl Default for RoundRobinPolicy {
    fn default() -> Self {
        Self {
            cursor: Mutex::new(0),
        }
    }
}

impl PlacementPolicy for RoundRobinPolicy {
    fn choose(&self, _hints: &PlacementHints, candidates: &[(u32, &DeviceLoad)]) -> Option<u32> {
        if candidates.is_empty() {
            return None;
        }
        let mut c = self.cursor.lock();
        let idx = *c % candidates.len();
        *c = c.wrapping_add(1);
        Some(candidates[idx].0)
    }
}

/// Least-loaded by `queue_depth + active_streams` heuristic. Filters
/// out devices below `min_free_bytes` / `min_compute_cap`.
pub struct LeastLoadedPolicy;

impl PlacementPolicy for LeastLoadedPolicy {
    fn choose(&self, hints: &PlacementHints, candidates: &[(u32, &DeviceLoad)]) -> Option<u32> {
        let mut best: Option<(u32, u64)> = None;
        for (id, load) in candidates {
            if load.free_bytes < hints.min_free_bytes {
                continue;
            }
            if let Some((mj, mn)) = hints.min_compute_cap {
                if load.compute_cap.0 < mj || (load.compute_cap.0 == mj && load.compute_cap.1 < mn)
                {
                    continue;
                }
            }
            let score = load.queue_depth as u64 + load.active_streams as u64;
            match best {
                None => best = Some((*id, score)),
                Some((_, s)) if score < s => best = Some((*id, score)),
                _ => {}
            }
        }
        best.map(|(id, _)| id)
    }
}

pub enum PlacementMsg {
    Pick {
        hints: PlacementHints,
        reply: oneshot::Sender<Result<DeviceChoice, GpuError>>,
    },
    /// Internal: timer fires the per-device stats poll.
    PollStats,
    /// Internal: a single device's stats reply arrived, update the
    /// cached load snapshot.
    StatsUpdate { slot: usize, load: DeviceLoad },
}

pub struct PlacementActor {
    devices: Vec<(u32, ActorRef<DeviceMsg>)>,
    loads: Vec<DeviceLoad>,
    policy: Arc<dyn PlacementPolicy>,
    poll_interval: Duration,
}

impl PlacementActor {
    pub fn props(
        devices: Vec<(u32, ActorRef<DeviceMsg>)>,
        policy: Arc<dyn PlacementPolicy>,
    ) -> Props<Self> {
        let n = devices.len();
        Props::create(move || PlacementActor {
            devices: devices.clone(),
            loads: (0..n)
                .map(|_| DeviceLoad {
                    free_bytes: 0,
                    total_bytes: 0,
                    active_streams: 0,
                    queue_depth: 0,
                    compute_cap: (0, 0),
                })
                .collect(),
            policy: policy.clone(),
            poll_interval: Duration::from_millis(250),
        })
    }

    fn schedule_poll(&self, ctx: &Context<Self>) {
        let self_ref = ctx.self_ref().clone();
        let interval = self.poll_interval;
        tokio::spawn(async move {
            tokio::time::sleep(interval).await;
            self_ref.tell(PlacementMsg::PollStats);
        });
    }
}

#[async_trait]
impl Actor for PlacementActor {
    type Msg = PlacementMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        self.schedule_poll(ctx);
    }

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: PlacementMsg) {
        match msg {
            PlacementMsg::Pick { hints, reply } => {
                let candidates: Vec<(u32, &DeviceLoad)> = self
                    .devices
                    .iter()
                    .zip(self.loads.iter())
                    .map(|((id, _), load)| (*id, load))
                    .collect();
                match self.policy.choose(&hints, &candidates) {
                    None => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "placement: no eligible device".into(),
                        )));
                    }
                    Some(id) => {
                        let pos = self.devices.iter().position(|(d, _)| *d == id).unwrap();
                        let _ = reply.send(Ok(DeviceChoice {
                            device_id: id,
                            device: self.devices[pos].1.clone(),
                            load: self.loads[pos],
                        }));
                    }
                }
            }
            PlacementMsg::PollStats => {
                // Fire one Stats request per device. Each reply
                // arrives via a tokio task that posts a
                // `StatsUpdate { slot, load }` back to this actor's
                // mailbox — closing the feedback loop without
                // needing &mut self inside the async block.
                let self_ref = ctx.self_ref().clone();
                for (i, (_, dev)) in self.devices.iter().enumerate() {
                    let (tx, rx) = oneshot::channel();
                    dev.tell(DeviceMsg::Stats { reply: tx });
                    let self_ref2 = self_ref.clone();
                    tokio::spawn(async move {
                        if let Ok(load) = rx.await {
                            self_ref2.tell(PlacementMsg::StatsUpdate { slot: i, load });
                        }
                    });
                }
                self.schedule_poll(ctx);
            }
            PlacementMsg::StatsUpdate { slot, load } => {
                if let Some(s) = self.loads.get_mut(slot) {
                    *s = load;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;

    use crate::device::DeviceActor;
    use crate::device::DeviceConfig;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn round_robin_picks_alternates() {
        let sys = ActorSystem::create("placement-rr", Config::empty())
            .await
            .unwrap();
        let d0 = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "d0")
            .unwrap();
        let d1 = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(1)), "d1")
            .unwrap();
        let actor = sys
            .actor_of(
                PlacementActor::props(
                    vec![(0, d0), (1, d1)],
                    Arc::new(RoundRobinPolicy::default()),
                ),
                "placement",
            )
            .unwrap();

        let mut picks = Vec::new();
        for _ in 0..4 {
            let (tx, rx) = oneshot::channel();
            actor.tell(PlacementMsg::Pick {
                hints: PlacementHints::default(),
                reply: tx,
            });
            let c = tokio::time::timeout(Duration::from_secs(2), rx)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            picks.push(c.device_id);
        }
        // Round-robin alternates: 0,1,0,1.
        assert_eq!(picks, vec![0, 1, 0, 1]);

        sys.terminate().await;
    }
}
