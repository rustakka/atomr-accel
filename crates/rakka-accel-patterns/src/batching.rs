//! `DynamicBatchingServer<Req, Resp>` — accumulates requests and
//! dispatches them in batches.
//!
//! Two flush triggers:
//! 1. Batch size reaches `max_batch`.
//! 2. `max_wait` elapses since the first item in the current batch.
//!
//! On flush, the actor calls a user-supplied [`BatchFn`] that turns
//! `Vec<Req>` into a single GPU call (typically a stacked `Sgemm` /
//! `Conv`) and produces a `Vec<Result<Resp, GpuError>>` of equal
//! length matching item order.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;
use tracing::{debug, warn};

use rakka_accel_cuda::error::GpuError;

/// Strategy for handling overflow when the in-flight batch is at
/// capacity but more requests arrive.
#[derive(Debug, Clone, Copy)]
pub enum BatchOverflow {
    /// Reply `Err(Unrecoverable)` immediately.
    Reject,
    /// Drop the oldest items in the current batch to make room.
    DropOldest,
}

/// User-supplied batch dispatcher. Receives a `Vec<Req>`; must
/// produce a same-length `Vec<Result<Resp, GpuError>>` where index `i`
/// holds the response for `Req` at index `i`.
pub trait BatchFn<Req, Resp>: Send + Sync + 'static {
    fn call(&self, batch: Vec<Req>) -> Vec<Result<Resp, GpuError>>;
}

impl<Req, Resp, F> BatchFn<Req, Resp> for F
where
    F: Fn(Vec<Req>) -> Vec<Result<Resp, GpuError>> + Send + Sync + 'static,
{
    fn call(&self, batch: Vec<Req>) -> Vec<Result<Resp, GpuError>> {
        self(batch)
    }
}

pub struct BatchingConfig<Req, Resp> {
    pub max_batch: usize,
    pub max_wait: Duration,
    pub batch_fn: Arc<dyn BatchFn<Req, Resp>>,
    pub overflow: BatchOverflow,
}

impl<Req, Resp> Clone for BatchingConfig<Req, Resp> {
    fn clone(&self) -> Self {
        Self {
            max_batch: self.max_batch,
            max_wait: self.max_wait,
            batch_fn: self.batch_fn.clone(),
            overflow: self.overflow,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchingStats {
    pub flushes: u64,
    pub items_processed: u64,
    pub items_dropped: u64,
}

pub enum BatchingMsg<Req: Send + 'static, Resp: Send + 'static> {
    Submit {
        req: Req,
        reply: oneshot::Sender<Result<Resp, GpuError>>,
    },
    /// Internal: timer fired, flush the current batch if non-empty.
    TimerFlush,
    FlushNow,
    Stats {
        reply: oneshot::Sender<BatchingStats>,
    },
}

pub struct DynamicBatchingServer<Req: Send + 'static, Resp: Send + 'static> {
    config: BatchingConfig<Req, Resp>,
    pending: Vec<(Req, oneshot::Sender<Result<Resp, GpuError>>)>,
    /// Instant when the current batch's first item arrived. None if
    /// the batch is empty.
    batch_started_at: Option<Instant>,
    stats: BatchingStats,
}

impl<Req: Send + 'static, Resp: Send + 'static> DynamicBatchingServer<Req, Resp> {
    pub fn props(config: BatchingConfig<Req, Resp>) -> Props<Self> {
        Props::create(move || DynamicBatchingServer {
            config: config.clone(),
            pending: Vec::with_capacity(config.max_batch),
            batch_started_at: None,
            stats: BatchingStats::default(),
        })
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let n = self.pending.len();
        let batch = std::mem::take(&mut self.pending);
        self.batch_started_at = None;
        let (reqs, replies): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
        let results = self.config.batch_fn.call(reqs);
        if results.len() != n {
            // Provider violated contract — fail every reply
            warn!(
                expected = n,
                got = results.len(),
                "BatchFn returned wrong number of results; failing batch"
            );
            for r in replies {
                let _ = r.send(Err(GpuError::Unrecoverable(
                    "BatchFn returned wrong number of results".into(),
                )));
            }
            return;
        }
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result);
        }
        self.stats.flushes += 1;
        self.stats.items_processed += n as u64;
        debug!(n, "batch flushed");
    }

    fn schedule_timer(&self, ctx: &Context<Self>) {
        let self_ref = ctx.self_ref().clone();
        let wait = self.config.max_wait;
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            self_ref.tell(BatchingMsg::TimerFlush);
        });
    }
}

#[async_trait]
impl<Req: Send + 'static, Resp: Send + 'static> Actor for DynamicBatchingServer<Req, Resp> {
    type Msg = BatchingMsg<Req, Resp>;

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: BatchingMsg<Req, Resp>) {
        match msg {
            BatchingMsg::Submit { req, reply } => {
                if self.pending.len() >= self.config.max_batch {
                    match self.config.overflow {
                        BatchOverflow::Reject => {
                            let _ = reply.send(Err(GpuError::Unrecoverable("batch full".into())));
                            self.stats.items_dropped += 1;
                            return;
                        }
                        BatchOverflow::DropOldest => {
                            // Drop oldest, reply with Err.
                            let (_, oldest_reply) = self.pending.remove(0);
                            let _ = oldest_reply.send(Err(GpuError::Unrecoverable(
                                "dropped by batching overflow policy".into(),
                            )));
                            self.stats.items_dropped += 1;
                        }
                    }
                }
                let was_empty = self.pending.is_empty();
                self.pending.push((req, reply));
                if was_empty {
                    self.batch_started_at = Some(Instant::now());
                    self.schedule_timer(ctx);
                }
                if self.pending.len() >= self.config.max_batch {
                    self.flush();
                }
            }
            BatchingMsg::TimerFlush => {
                if let Some(start) = self.batch_started_at {
                    if start.elapsed() >= self.config.max_wait {
                        self.flush();
                    } else {
                        // Spurious / early — reschedule.
                        self.schedule_timer(ctx);
                    }
                }
            }
            BatchingMsg::FlushNow => self.flush(),
            BatchingMsg::Stats { reply } => {
                let _ = reply.send(self.stats);
            }
        }
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        // Drain replies with an error so callers don't hang.
        for (_, reply) in std::mem::take(&mut self.pending) {
            let _ = reply.send(Err(GpuError::GpuRefStale("batching server stopped")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flushes_at_max_batch() {
        let echo: Arc<dyn BatchFn<u32, u32>> =
            Arc::new(|reqs: Vec<u32>| reqs.into_iter().map(|x| Ok(x * 2)).collect());
        let cfg = BatchingConfig {
            max_batch: 3,
            max_wait: Duration::from_secs(60), // long enough not to fire
            batch_fn: echo,
            overflow: BatchOverflow::Reject,
        };
        let sys = ActorSystem::create("batching-test", Config::empty())
            .await
            .unwrap();
        let actor = sys
            .actor_of(DynamicBatchingServer::<u32, u32>::props(cfg), "batcher")
            .unwrap();

        let r1: oneshot::Receiver<_> = {
            let (tx, rx) = oneshot::channel();
            actor.tell(BatchingMsg::Submit { req: 1, reply: tx });
            rx
        };
        let r2 = {
            let (tx, rx) = oneshot::channel();
            actor.tell(BatchingMsg::Submit { req: 2, reply: tx });
            rx
        };
        let r3 = {
            let (tx, rx) = oneshot::channel();
            actor.tell(BatchingMsg::Submit { req: 3, reply: tx });
            rx
        };

        let v1 = tokio::time::timeout(Duration::from_secs(2), r1)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let v2 = tokio::time::timeout(Duration::from_secs(2), r2)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let v3 = tokio::time::timeout(Duration::from_secs(2), r3)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!((v1, v2, v3), (2, 4, 6));

        sys.terminate().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flushes_on_timer() {
        let echo: Arc<dyn BatchFn<u32, u32>> =
            Arc::new(|reqs: Vec<u32>| reqs.into_iter().map(|x| Ok(x + 100)).collect());
        let cfg = BatchingConfig {
            max_batch: 100,
            max_wait: Duration::from_millis(50),
            batch_fn: echo,
            overflow: BatchOverflow::Reject,
        };
        let sys = ActorSystem::create("batching-timer-test", Config::empty())
            .await
            .unwrap();
        let actor = sys
            .actor_of(DynamicBatchingServer::<u32, u32>::props(cfg), "batcher")
            .unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(BatchingMsg::Submit { req: 7, reply: tx });
        let v = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(v, 107);

        sys.terminate().await;
    }
}
