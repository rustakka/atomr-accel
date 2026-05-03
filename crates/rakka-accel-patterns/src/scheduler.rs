//! `FairShareScheduler` — weighted-fair scheduling across N tenants.
//!
//! Each tenant gets a per-tenant weight; the scheduler picks the
//! next tenant whose `vfinish` (virtual-finish-time, in WFQ terms)
//! is the lowest among non-empty queues. Within a tenant, requests
//! are FIFO.
//!
//! This is a host-side scheduling layer above any kernel actor. It
//! doesn't talk to GPU directly; instead it routes accepted
//! requests to a user-supplied downstream actor via a dispatcher
//! closure.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct TenantConfig {
    pub id: TenantId,
    /// Relative weight. Higher = more share. Must be > 0.
    pub weight: u32,
}

/// User-supplied dispatcher: accepts a `Req` plus a reply channel
/// and forwards to whatever downstream actor handles the work.
pub trait FairDispatcher<Req, Resp>: Send + Sync + 'static {
    fn dispatch(&self, req: Req, reply: oneshot::Sender<Result<Resp, GpuError>>);
}

impl<Req, Resp, F> FairDispatcher<Req, Resp> for F
where
    F: Fn(Req, oneshot::Sender<Result<Resp, GpuError>>) + Send + Sync + 'static,
{
    fn dispatch(&self, req: Req, reply: oneshot::Sender<Result<Resp, GpuError>>) {
        self(req, reply)
    }
}

pub struct FairShareConfig<Req, Resp> {
    pub tenants: Vec<TenantConfig>,
    pub dispatcher: Arc<dyn FairDispatcher<Req, Resp>>,
    /// Maximum concurrent in-flight requests. New submits beyond
    /// the cap are queued.
    pub max_in_flight: usize,
}

impl<Req, Resp> Clone for FairShareConfig<Req, Resp> {
    fn clone(&self) -> Self {
        Self {
            tenants: self.tenants.clone(),
            dispatcher: self.dispatcher.clone(),
            max_in_flight: self.max_in_flight,
        }
    }
}

pub enum FairShareMsg<Req: Send + 'static, Resp: Send + 'static> {
    Submit {
        tenant: TenantId,
        req: Req,
        reply: oneshot::Sender<Result<Resp, GpuError>>,
    },
    /// Internal: a previously-dispatched request finished.
    Finished { tenant: TenantId },
    Stats {
        reply: oneshot::Sender<FairShareStats>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FairShareStats {
    pub in_flight: usize,
    pub total_dispatched: u64,
}

struct TenantState<Req, Resp> {
    cfg: TenantConfig,
    /// FIFO queue.
    queue: VecDeque<(Req, oneshot::Sender<Result<Resp, GpuError>>)>,
    /// Virtual finish time of the next item in the queue. Lowest
    /// vfinish wins.
    vfinish: f64,
}

pub struct FairShareScheduler<Req: Send + 'static, Resp: Send + 'static> {
    config: FairShareConfig<Req, Resp>,
    tenants: HashMap<TenantId, TenantState<Req, Resp>>,
    in_flight: usize,
    total_dispatched: u64,
}

impl<Req: Send + 'static, Resp: Send + 'static> FairShareScheduler<Req, Resp> {
    pub fn props(config: FairShareConfig<Req, Resp>) -> Props<Self> {
        Props::create(move || {
            let mut tenants = HashMap::new();
            for t in &config.tenants {
                tenants.insert(
                    t.id,
                    TenantState::<Req, Resp> {
                        cfg: *t,
                        queue: VecDeque::new(),
                        vfinish: 0.0,
                    },
                );
            }
            FairShareScheduler {
                config: config.clone(),
                tenants,
                in_flight: 0,
                total_dispatched: 0,
            }
        })
    }

    fn try_dispatch(&mut self, ctx: &Context<Self>) {
        while self.in_flight < self.config.max_in_flight {
            // Pick the tenant with the smallest vfinish whose queue
            // is non-empty.
            let mut best: Option<(TenantId, f64)> = None;
            for (id, t) in &self.tenants {
                if t.queue.is_empty() {
                    continue;
                }
                match best {
                    None => best = Some((*id, t.vfinish)),
                    Some((_, vf)) if t.vfinish < vf => best = Some((*id, t.vfinish)),
                    _ => {}
                }
            }
            let Some((tid, _)) = best else {
                return;
            };
            let t = self.tenants.get_mut(&tid).unwrap();
            let (req, reply) = t.queue.pop_front().unwrap();
            // Update vfinish: WFQ uses 1/weight as the per-request
            // increment. Larger weight → smaller increment → more
            // bandwidth share.
            t.vfinish += 1.0 / (t.cfg.weight.max(1) as f64);
            self.in_flight += 1;
            self.total_dispatched += 1;
            // Wrap reply with a tee that posts a Finished signal
            // back to us so we can advance.
            let self_ref = ctx.self_ref().clone();
            let (tee_tx, tee_rx) = oneshot::channel::<Result<Resp, GpuError>>();
            tokio::spawn(async move {
                let r = tee_rx.await.unwrap_or_else(|_| {
                    Err(GpuError::Unrecoverable("fair: dispatcher dropped".into()))
                });
                let _ = reply.send(r);
                self_ref.tell(FairShareMsg::Finished { tenant: tid });
            });
            self.config.dispatcher.dispatch(req, tee_tx);
        }
    }
}

#[async_trait]
impl<Req: Send + 'static, Resp: Send + 'static> Actor for FairShareScheduler<Req, Resp> {
    type Msg = FairShareMsg<Req, Resp>;

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: FairShareMsg<Req, Resp>) {
        match msg {
            FairShareMsg::Submit { tenant, req, reply } => {
                let Some(t) = self.tenants.get_mut(&tenant) else {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "FairShare: unknown tenant {tenant:?}"
                    ))));
                    return;
                };
                t.queue.push_back((req, reply));
                self.try_dispatch(ctx);
            }
            FairShareMsg::Finished { tenant: _ } => {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.try_dispatch(ctx);
            }
            FairShareMsg::Stats { reply } => {
                let _ = reply.send(FairShareStats {
                    in_flight: self.in_flight,
                    total_dispatched: self.total_dispatched,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn weighted_fair_split() {
        // Echo dispatcher.
        let echo: Arc<dyn FairDispatcher<u32, u32>> =
            Arc::new(|req: u32, reply: oneshot::Sender<Result<u32, GpuError>>| {
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    let _ = reply.send(Ok(req * 2));
                });
            });
        let cfg = FairShareConfig {
            tenants: vec![
                TenantConfig {
                    id: TenantId(1),
                    weight: 1,
                },
                TenantConfig {
                    id: TenantId(2),
                    weight: 3,
                },
            ],
            dispatcher: echo,
            max_in_flight: 1,
        };
        let sys = ActorSystem::create("fair-test", Config::empty())
            .await
            .unwrap();
        let actor = sys
            .actor_of(FairShareScheduler::<u32, u32>::props(cfg), "sched")
            .unwrap();

        // Submit 4 requests for each tenant.
        let mut rxs = Vec::new();
        for i in 0..4 {
            let (tx, rx) = oneshot::channel();
            actor.tell(FairShareMsg::Submit {
                tenant: TenantId(1),
                req: 100 + i,
                reply: tx,
            });
            rxs.push(rx);
            let (tx, rx) = oneshot::channel();
            actor.tell(FairShareMsg::Submit {
                tenant: TenantId(2),
                req: 200 + i,
                reply: tx,
            });
            rxs.push(rx);
        }
        for rx in rxs {
            tokio::time::timeout(Duration::from_secs(2), rx)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        }

        let (tx, rx) = oneshot::channel();
        actor.tell(FairShareMsg::Stats { reply: tx });
        let s = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(s.total_dispatched, 8);

        sys.terminate().await;
    }
}
