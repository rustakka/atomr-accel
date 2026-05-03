//! `ModelReplicaPool<Req, Resp>` — N replicas of a model routed via
//! a configurable [`RoutingPolicy`].
//!
//! Each replica is itself an actor with a `Send + 'static` typed
//! mailbox; the pool forwards each request to one replica chosen by
//! the policy.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPolicy {
    RoundRobin,
    /// Least-loaded by replica's outstanding-request counter (host
    /// side estimate). Falls back to round-robin if all replicas tie.
    LeastLoaded,
}

pub trait ReplicaMessage: Send + 'static {
    /// Build the per-replica `Submit` variant from a request +
    /// reply channel.
    type Req: Send + 'static;
    type Resp: Send + 'static;
    fn make_submit(
        req: Self::Req,
        reply: oneshot::Sender<Result<Self::Resp, GpuError>>,
    ) -> Self;
}

pub struct ReplicaPoolConfig<Msg: ReplicaMessage> {
    pub replicas: Vec<ActorRef<Msg>>,
    pub policy: RoutingPolicy,
}

impl<Msg: ReplicaMessage> Clone for ReplicaPoolConfig<Msg> {
    fn clone(&self) -> Self {
        Self {
            replicas: self.replicas.clone(),
            policy: self.policy,
        }
    }
}

pub enum ReplicaPoolMsg<Msg: ReplicaMessage> {
    Submit {
        req: Msg::Req,
        reply: oneshot::Sender<Result<Msg::Resp, GpuError>>,
    },
}

pub struct ModelReplicaPool<Msg: ReplicaMessage> {
    config: ReplicaPoolConfig<Msg>,
    cursor: Mutex<usize>,
    /// Outstanding request count per replica (used by LeastLoaded).
    /// Decremented when a reply arrives — F4 skeleton: we count
    /// dispatches only, not completions; F4.x adds a feedback hook.
    counters: Vec<Arc<parking_lot::Mutex<u32>>>,
}

impl<Msg: ReplicaMessage> ModelReplicaPool<Msg> {
    pub fn props(config: ReplicaPoolConfig<Msg>) -> Props<Self> {
        let n = config.replicas.len();
        let counters: Vec<_> = (0..n).map(|_| Arc::new(parking_lot::Mutex::new(0u32))).collect();
        Props::create(move || ModelReplicaPool {
            config: config.clone(),
            cursor: Mutex::new(0),
            counters: counters.clone(),
        })
    }

    fn pick_replica(&self) -> Option<usize> {
        let n = self.config.replicas.len();
        if n == 0 {
            return None;
        }
        match self.config.policy {
            RoutingPolicy::RoundRobin => {
                let mut c = self.cursor.lock();
                let idx = *c % n;
                *c = c.wrapping_add(1);
                Some(idx)
            }
            RoutingPolicy::LeastLoaded => {
                let mut min = u32::MAX;
                let mut min_idx = 0usize;
                for (i, c) in self.counters.iter().enumerate() {
                    let v = *c.lock();
                    if v < min {
                        min = v;
                        min_idx = i;
                    }
                }
                Some(min_idx)
            }
        }
    }
}

#[async_trait]
impl<Msg: ReplicaMessage> Actor for ModelReplicaPool<Msg> {
    type Msg = ReplicaPoolMsg<Msg>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ReplicaPoolMsg<Msg>) {
        match msg {
            ReplicaPoolMsg::Submit { req, reply } => {
                let Some(idx) = self.pick_replica() else {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "ModelReplicaPool: zero replicas".into(),
                    )));
                    return;
                };
                *self.counters[idx].lock() += 1;
                self.config.replicas[idx].tell(Msg::make_submit(req, reply));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{GpuMockActor, GpuMockMsg, MockSgemm};
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    impl ReplicaMessage for GpuMockMsg {
        type Req = MockSgemmReq;
        type Resp = Vec<f32>;
        fn make_submit(
            req: MockSgemmReq,
            reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
        ) -> Self {
            GpuMockMsg::Sgemm(Box::new(MockSgemm {
                a: req.a,
                b: req.b,
                m: req.m,
                n: req.n,
                k: req.k,
                reply,
            }))
        }
    }

    pub struct MockSgemmReq {
        pub a: Vec<f32>,
        pub b: Vec<f32>,
        pub m: usize,
        pub n: usize,
        pub k: usize,
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn round_robin_routes_across_replicas() {
        let sys = ActorSystem::create("replica-test", Config::empty()).await.unwrap();
        let r1 = sys.actor_of(GpuMockActor::props(), "r1").unwrap();
        let r2 = sys.actor_of(GpuMockActor::props(), "r2").unwrap();
        let pool = sys
            .actor_of(
                ModelReplicaPool::<GpuMockMsg>::props(ReplicaPoolConfig {
                    replicas: vec![r1, r2],
                    policy: RoutingPolicy::RoundRobin,
                }),
                "pool",
            )
            .unwrap();

        for _ in 0..4 {
            let (tx, rx) = oneshot::channel();
            pool.tell(ReplicaPoolMsg::Submit {
                req: MockSgemmReq {
                    a: vec![1.0, 0.0, 0.0, 1.0],
                    b: vec![1.0, 2.0, 3.0, 4.0],
                    m: 2, n: 2, k: 2,
                },
                reply: tx,
            });
            let v = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
            assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
        }

        sys.terminate().await;
    }
}
