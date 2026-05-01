//! `ModelHotSwapServer` — live model replacement without dropping
//! in-flight requests.
//!
//! Lifecycle:
//! 1. Construct with an initial backend `ActorRef<M>`.
//! 2. Send `Predict { req, reply }` — forwards to the current
//!    backend.
//! 3. Send `SwapIn { new_backend }` — atomically replaces the
//!    backend ref. Subsequent `Predict`s route to the new backend.
//!    In-flight requests against the old backend are unaffected
//!    (the old backend keeps its own mailbox and replies normally).
//!
//! This is intentionally simple — it does not drain in-flight
//! requests before the swap (which would require explicit
//! coordination with the backend). For training-grade hot-swap with
//! drain semantics, layer the user-facing protocol on top.

use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

pub trait BackendProtocol: Send + 'static {
    type Msg: Send + 'static;
    type Req: Send + 'static;
    type Resp: Send + 'static;
    fn make_predict(
        req: Self::Req,
        reply: oneshot::Sender<Result<Self::Resp, GpuError>>,
    ) -> Self::Msg;
}

pub enum HotSwapMsg<P: BackendProtocol> {
    Predict {
        req: P::Req,
        reply: oneshot::Sender<Result<P::Resp, GpuError>>,
    },
    SwapIn {
        new_backend: ActorRef<P::Msg>,
        reply: oneshot::Sender<HotSwapStats>,
    },
    Stats {
        reply: oneshot::Sender<HotSwapStats>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HotSwapStats {
    pub generation: u64,
    pub predicted: u64,
    pub swaps: u64,
}

pub struct ModelHotSwapServer<P: BackendProtocol> {
    current: Arc<parking_lot::Mutex<ActorRef<P::Msg>>>,
    generation: u64,
    predicted: u64,
    swaps: u64,
}

impl<P: BackendProtocol> ModelHotSwapServer<P> {
    pub fn props(initial: ActorRef<P::Msg>) -> Props<Self> {
        Props::create(move || ModelHotSwapServer {
            current: Arc::new(parking_lot::Mutex::new(initial.clone())),
            generation: 0,
            predicted: 0,
            swaps: 0,
        })
    }
}

#[async_trait]
impl<P: BackendProtocol> Actor for ModelHotSwapServer<P> {
    type Msg = HotSwapMsg<P>;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: HotSwapMsg<P>) {
        match msg {
            HotSwapMsg::Predict { req, reply } => {
                let backend = self.current.lock().clone();
                self.predicted += 1;
                backend.tell(P::make_predict(req, reply));
            }
            HotSwapMsg::SwapIn { new_backend, reply } => {
                *self.current.lock() = new_backend;
                self.generation += 1;
                self.swaps += 1;
                let _ = reply.send(HotSwapStats {
                    generation: self.generation,
                    predicted: self.predicted,
                    swaps: self.swaps,
                });
            }
            HotSwapMsg::Stats { reply } => {
                let _ = reply.send(HotSwapStats {
                    generation: self.generation,
                    predicted: self.predicted,
                    swaps: self.swaps,
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

    enum BackendMsg {
        Predict {
            req: u32,
            reply: oneshot::Sender<Result<u32, GpuError>>,
        },
    }

    struct V1;
    #[async_trait]
    impl Actor for V1 {
        type Msg = BackendMsg;
        async fn handle(&mut self, _ctx: &mut Context<Self>, msg: BackendMsg) {
            match msg {
                BackendMsg::Predict { req, reply } => {
                    let _ = reply.send(Ok(req + 1));
                }
            }
        }
    }
    struct V2;
    #[async_trait]
    impl Actor for V2 {
        type Msg = BackendMsg;
        async fn handle(&mut self, _ctx: &mut Context<Self>, msg: BackendMsg) {
            match msg {
                BackendMsg::Predict { req, reply } => {
                    let _ = reply.send(Ok(req + 100));
                }
            }
        }
    }

    struct EchoProto;
    impl BackendProtocol for EchoProto {
        type Msg = BackendMsg;
        type Req = u32;
        type Resp = u32;
        fn make_predict(
            req: u32,
            reply: oneshot::Sender<Result<u32, GpuError>>,
        ) -> BackendMsg {
            BackendMsg::Predict { req, reply }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn swap_routes_to_new_backend() {
        let sys = ActorSystem::create("hot-swap", Config::empty()).await.unwrap();
        let v1 = sys.actor_of(rakka_core::actor::Props::create(|| V1), "v1").unwrap();
        let v2 = sys.actor_of(rakka_core::actor::Props::create(|| V2), "v2").unwrap();
        let server = sys
            .actor_of(ModelHotSwapServer::<EchoProto>::props(v1), "server")
            .unwrap();

        // Predict before swap.
        let (tx, rx) = oneshot::channel();
        server.tell(HotSwapMsg::Predict { req: 5, reply: tx });
        let r = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(r, 6);

        // Swap.
        let (tx, rx) = oneshot::channel();
        server.tell(HotSwapMsg::SwapIn { new_backend: v2, reply: tx });
        let s = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert_eq!(s.swaps, 1);

        // Predict after swap.
        let (tx, rx) = oneshot::channel();
        server.tell(HotSwapMsg::Predict { req: 5, reply: tx });
        let r = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(r, 105);

        sys.terminate().await;
    }
}
