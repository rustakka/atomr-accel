//! `VideoEffectsGraph` — a per-frame DAG of effect nodes.
//!
//! Each frame flows through a sequence of effect functions. F7
//! ships a linear-chain reference; F8+ adds branched DAG execution
//! for effects with multiple inputs (e.g. blend two filtered
//! variants of the same frame).

use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

pub trait Effect: Send + Sync + 'static {
    fn apply(&self, frame: &[u8]) -> Result<Vec<u8>, GpuError>;
}

impl<F> Effect for F
where
    F: Fn(&[u8]) -> Result<Vec<u8>, GpuError> + Send + Sync + 'static,
{
    fn apply(&self, frame: &[u8]) -> Result<Vec<u8>, GpuError> {
        self(frame)
    }
}

pub struct VideoEffectsConfig {
    pub effects: Vec<Arc<dyn Effect>>,
}

impl Clone for VideoEffectsConfig {
    fn clone(&self) -> Self {
        Self { effects: self.effects.clone() }
    }
}

pub enum VideoEffectsMsg {
    ProcessFrame {
        frame: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, GpuError>>,
    },
    /// Replace the active effect chain.
    UpdateEffects {
        effects: Vec<Arc<dyn Effect>>,
        reply: oneshot::Sender<()>,
    },
}

pub struct VideoEffectsGraph {
    config: VideoEffectsConfig,
}

impl VideoEffectsGraph {
    pub fn props(config: VideoEffectsConfig) -> Props<Self> {
        Props::create(move || VideoEffectsGraph { config: config.clone() })
    }
}

#[async_trait]
impl Actor for VideoEffectsGraph {
    type Msg = VideoEffectsMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: VideoEffectsMsg) {
        match msg {
            VideoEffectsMsg::ProcessFrame { frame, reply } => {
                let mut current = frame;
                for e in &self.config.effects {
                    match e.apply(&current) {
                        Ok(next) => current = next,
                        Err(err) => {
                            let _ = reply.send(Err(err));
                            return;
                        }
                    }
                }
                let _ = reply.send(Ok(current));
            }
            VideoEffectsMsg::UpdateEffects { effects, reply } => {
                self.config.effects = effects;
                let _ = reply.send(());
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
    async fn chain_applies_each_effect() {
        let invert: Arc<dyn Effect> = Arc::new(|f: &[u8]| Ok(f.iter().map(|x| 255 - *x).collect()));
        let half: Arc<dyn Effect> = Arc::new(|f: &[u8]| Ok(f.iter().map(|x| x / 2).collect()));
        let cfg = VideoEffectsConfig { effects: vec![invert, half] };

        let sys = ActorSystem::create("vfx-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(VideoEffectsGraph::props(cfg), "vfx").unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(VideoEffectsMsg::ProcessFrame {
            frame: vec![100, 200],
            reply: tx,
        });
        let out = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        // (255-100)/2 = 77, (255-200)/2 = 27.
        assert_eq!(out, vec![77, 27]);

        sys.terminate().await;
    }
}
