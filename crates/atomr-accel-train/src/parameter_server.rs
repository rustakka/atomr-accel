//! `AsyncParameterServer` — central parameter store with async
//! gradient pushes and async weight pulls.
//!
//! Workers push gradients (via `PushGradient`); the server applies
//! them with the configured optimizer and a staleness window.
//! Workers pull the latest weights (via `PullWeights`) on their
//! own schedule. Bounded-staleness training tolerates a few steps
//! of drift in exchange for higher worker utilization.

use std::collections::VecDeque;
use std::time::Instant;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;

use crate::optimizer::OptimizerKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerId(pub u32);

#[derive(Debug, Clone, Copy, Default)]
pub struct ParameterServerStats {
    pub version: u64,
    pub gradients_applied: u64,
    pub weights_pulled: u64,
    pub avg_staleness: f32,
}

pub enum ParameterServerMsg {
    /// Worker pushes a gradient vector. `worker_version` is the
    /// version the worker had when it computed this gradient.
    PushGradient {
        worker: WorkerId,
        worker_version: u64,
        gradient: Vec<f32>,
        reply: oneshot::Sender<Result<u64, GpuError>>,
    },
    /// Worker pulls the latest weights + version.
    PullWeights {
        worker: WorkerId,
        reply: oneshot::Sender<(Vec<f32>, u64)>,
    },
    Stats {
        reply: oneshot::Sender<ParameterServerStats>,
    },
}

pub struct AsyncParameterServer {
    weights: Vec<f32>,
    version: u64,
    optimizer: OptimizerKind,
    /// Maximum allowed staleness — gradients computed against
    /// versions older than `version - max_staleness` are rejected.
    max_staleness: u64,
    gradients_applied: u64,
    weights_pulled: u64,
    /// Sliding window of (version - worker_version) for the last N
    /// applied gradients.
    staleness_window: VecDeque<u64>,
    started: Instant,
}

impl AsyncParameterServer {
    pub fn props(
        initial_weights: Vec<f32>,
        optimizer: OptimizerKind,
        max_staleness: u64,
    ) -> Props<Self> {
        Props::create(move || AsyncParameterServer {
            weights: initial_weights.clone(),
            version: 0,
            optimizer,
            max_staleness,
            gradients_applied: 0,
            weights_pulled: 0,
            staleness_window: VecDeque::with_capacity(128),
            started: Instant::now(),
        })
    }

    fn apply_gradient(&mut self, grad: &[f32]) {
        let lr = self.optimizer.lr();
        // SGD-style: w <- w - lr * grad.
        let n = self.weights.len().min(grad.len());
        for (w, g) in self.weights.iter_mut().zip(grad.iter()).take(n) {
            *w -= lr * g;
        }
        self.version += 1;
        self.gradients_applied += 1;
    }
}

#[async_trait]
impl Actor for AsyncParameterServer {
    type Msg = ParameterServerMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ParameterServerMsg) {
        match msg {
            ParameterServerMsg::PushGradient {
                worker: _,
                worker_version,
                gradient,
                reply,
            } => {
                let staleness = self.version.saturating_sub(worker_version);
                if staleness > self.max_staleness {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "parameter server: staleness {staleness} > max {}",
                        self.max_staleness
                    ))));
                    return;
                }
                self.apply_gradient(&gradient);
                self.staleness_window.push_back(staleness);
                if self.staleness_window.len() > 128 {
                    self.staleness_window.pop_front();
                }
                let _ = reply.send(Ok(self.version));
            }
            ParameterServerMsg::PullWeights { worker: _, reply } => {
                self.weights_pulled += 1;
                let _ = reply.send((self.weights.clone(), self.version));
            }
            ParameterServerMsg::Stats { reply } => {
                let avg_stale = if self.staleness_window.is_empty() {
                    0.0
                } else {
                    let sum: u64 = self.staleness_window.iter().sum();
                    sum as f32 / self.staleness_window.len() as f32
                };
                let _ = reply.send(ParameterServerStats {
                    version: self.version,
                    gradients_applied: self.gradients_applied,
                    weights_pulled: self.weights_pulled,
                    avg_staleness: avg_stale,
                });
            }
        }
        let _ = self.started; // suppress unused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn push_gradient_advances_version() {
        let sys = ActorSystem::create("ps-test", Config::empty())
            .await
            .unwrap();
        let ps = sys
            .actor_of(
                AsyncParameterServer::props(
                    vec![10.0, 20.0],
                    OptimizerKind::Sgd {
                        lr: 0.1,
                        momentum: 0.0,
                        weight_decay: 0.0,
                    },
                    /* max_staleness */ 4,
                ),
                "ps",
            )
            .unwrap();

        let (tx, rx) = oneshot::channel();
        ps.tell(ParameterServerMsg::PushGradient {
            worker: WorkerId(1),
            worker_version: 0,
            gradient: vec![1.0, 2.0],
            reply: tx,
        });
        let v = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(v, 1);

        let (tx, rx) = oneshot::channel();
        ps.tell(ParameterServerMsg::PullWeights {
            worker: WorkerId(1),
            reply: tx,
        });
        let (w, version) = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(version, 1);
        // w[0] = 10 - 0.1 * 1 = 9.9; w[1] = 20 - 0.1 * 2 = 19.8.
        assert!((w[0] - 9.9).abs() < 1e-5);
        assert!((w[1] - 19.8).abs() < 1e-5);

        sys.terminate().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_gradient_is_rejected() {
        let sys = ActorSystem::create("ps-stale", Config::empty())
            .await
            .unwrap();
        let ps = sys
            .actor_of(
                AsyncParameterServer::props(
                    vec![1.0],
                    OptimizerKind::Sgd {
                        lr: 0.1,
                        momentum: 0.0,
                        weight_decay: 0.0,
                    },
                    /* max_staleness */ 1,
                ),
                "ps",
            )
            .unwrap();

        // Advance to version 3.
        for _ in 0..3 {
            let (tx, rx) = oneshot::channel();
            ps.tell(ParameterServerMsg::PushGradient {
                worker: WorkerId(1),
                worker_version: 0,
                gradient: vec![0.1],
                reply: tx,
            });
            // Some pushes will be rejected once staleness exceeds 1;
            // we ignore individual results here.
            let _ = tokio::time::timeout(Duration::from_secs(2), rx)
                .await
                .unwrap()
                .unwrap();
        }
        // Now push with worker_version=0 against a much-newer
        // server; should be rejected.
        let (tx, rx) = oneshot::channel();
        ps.tell(ParameterServerMsg::PushGradient {
            worker: WorkerId(1),
            worker_version: 0,
            gradient: vec![0.1],
            reply: tx,
        });
        let r = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

        sys.terminate().await;
    }
}
