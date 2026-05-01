//! `SharedGpuStateCoordinator` — issues write tokens to N agents
//! sharing a `ManagedRef<f32>` world-state buffer.
//!
//! Agents acquire a `WriteToken` before mutating the shared state.
//! Only one token is outstanding at a time. Tokens are valid until
//! returned via `ReleaseWrite`. The shared state pointer is
//! distributed via `Snapshot`; readers can pull a clone of the
//! `ManagedRef<f32>` whenever they like (concurrent reads are
//! safe under the WriteToken protocol).

use std::collections::VecDeque;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::memory::ManagedRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteToken(pub u64);

pub enum SharedStateMsg {
    AcquireWrite {
        agent_id: u32,
        reply: oneshot::Sender<WriteToken>,
    },
    ReleaseWrite {
        token: WriteToken,
    },
    /// Pull a clone of the underlying `ManagedRef<f32>`. Returns
    /// `None` if no shared state was attached.
    Snapshot {
        reply: oneshot::Sender<Option<ManagedRef<f32>>>,
    },
    /// Attach (or replace) the underlying shared buffer. Used at
    /// startup to install the `ManagedRef<f32>` from
    /// `ManagedAllocatorActor`.
    AttachState {
        state: ManagedRef<f32>,
        reply: oneshot::Sender<()>,
    },
    Stats {
        reply: oneshot::Sender<SharedStateStats>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SharedStateStats {
    pub current_holder: Option<u32>,
    pub waiting: usize,
    pub tokens_issued: u64,
}

pub struct SharedGpuStateCoordinator {
    counter: u64,
    /// `(agent_id, reply_sender)` for waiters when a token is held.
    waiters: VecDeque<(u32, oneshot::Sender<WriteToken>)>,
    /// `(holder_agent_id, current_token)` while held.
    held: Option<(u32, WriteToken)>,
    /// Optional shared buffer. `None` until `AttachState` arrives.
    state: Option<ManagedRef<f32>>,
}

impl SharedGpuStateCoordinator {
    pub fn props() -> Props<Self> {
        Props::create(|| SharedGpuStateCoordinator {
            counter: 0,
            waiters: VecDeque::new(),
            held: None,
            state: None,
        })
    }

    fn issue_token(&mut self, agent_id: u32, reply: oneshot::Sender<WriteToken>) {
        self.counter += 1;
        let token = WriteToken(self.counter);
        self.held = Some((agent_id, token));
        let _ = reply.send(token);
    }
}

#[async_trait]
impl Actor for SharedGpuStateCoordinator {
    type Msg = SharedStateMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SharedStateMsg) {
        match msg {
            SharedStateMsg::AcquireWrite { agent_id, reply } => {
                if self.held.is_none() {
                    self.issue_token(agent_id, reply);
                } else {
                    self.waiters.push_back((agent_id, reply));
                }
            }
            SharedStateMsg::ReleaseWrite { token } => {
                if let Some((_, t)) = &self.held {
                    if *t == token {
                        self.held = None;
                        if let Some((next_id, next_reply)) = self.waiters.pop_front() {
                            self.issue_token(next_id, next_reply);
                        }
                    }
                }
            }
            SharedStateMsg::Snapshot { reply } => {
                let _ = reply.send(self.state.clone());
            }
            SharedStateMsg::AttachState { state, reply } => {
                self.state = Some(state);
                let _ = reply.send(());
            }
            SharedStateMsg::Stats { reply } => {
                let _ = reply.send(SharedStateStats {
                    current_holder: self.held.map(|(id, _)| id),
                    waiting: self.waiters.len(),
                    tokens_issued: self.counter,
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
    async fn fifo_token_handoff() {
        let sys = ActorSystem::create("shared-state-test", Config::empty()).await.unwrap();
        let actor = sys
            .actor_of(SharedGpuStateCoordinator::props(), "coord")
            .unwrap();

        // Agent 1 acquires.
        let (tx1, rx1) = oneshot::channel();
        actor.tell(SharedStateMsg::AcquireWrite { agent_id: 1, reply: tx1 });
        let t1 = tokio::time::timeout(Duration::from_secs(2), rx1).await.unwrap().unwrap();

        // Agent 2 queues.
        let (tx2, rx2) = oneshot::channel();
        actor.tell(SharedStateMsg::AcquireWrite { agent_id: 2, reply: tx2 });
        // Stats: holder=1, waiting=1.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (sx, sr) = oneshot::channel();
        actor.tell(SharedStateMsg::Stats { reply: sx });
        let stats = tokio::time::timeout(Duration::from_secs(2), sr).await.unwrap().unwrap();
        assert_eq!(stats.current_holder, Some(1));
        assert_eq!(stats.waiting, 1);

        // Agent 1 releases. Agent 2 should receive its token.
        actor.tell(SharedStateMsg::ReleaseWrite { token: t1 });
        let t2 = tokio::time::timeout(Duration::from_secs(2), rx2).await.unwrap().unwrap();
        assert_ne!(t1, t2);

        sys.terminate().await;
    }
}
