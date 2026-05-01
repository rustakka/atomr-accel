//! `MultiPassAnalysisActor` — runs N user-supplied analysis passes
//! over a single input frame and returns a `Vec<f32>` of per-pass
//! results.
//!
//! Each pass is a `Box<dyn Fn(&[f32]) -> Result<f32, GpuError>>`.
//! Passes run sequentially — for true parallelism over passes,
//! wrap each in a tokio task at the call site.

use std::sync::Arc;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

pub trait Pass: Send + Sync + 'static {
    fn run(&self, frame: &[f32]) -> Result<f32, GpuError>;
}

impl<F> Pass for F
where
    F: Fn(&[f32]) -> Result<f32, GpuError> + Send + Sync + 'static,
{
    fn run(&self, frame: &[f32]) -> Result<f32, GpuError> {
        self(frame)
    }
}

pub struct MultiPassConfig {
    pub passes: Vec<Arc<dyn Pass>>,
}

impl Clone for MultiPassConfig {
    fn clone(&self) -> Self {
        Self { passes: self.passes.clone() }
    }
}

pub enum MultiPassMsg {
    Analyze {
        frame: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
}

pub struct MultiPassAnalysisActor {
    config: MultiPassConfig,
}

impl MultiPassAnalysisActor {
    pub fn props(config: MultiPassConfig) -> Props<Self> {
        Props::create(move || MultiPassAnalysisActor { config: config.clone() })
    }
}

#[async_trait]
impl Actor for MultiPassAnalysisActor {
    type Msg = MultiPassMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: MultiPassMsg) {
        match msg {
            MultiPassMsg::Analyze { frame, reply } => {
                let mut out = Vec::with_capacity(self.config.passes.len());
                for p in &self.config.passes {
                    match p.run(&frame) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            return;
                        }
                    }
                }
                let _ = reply.send(Ok(out));
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
    async fn multi_pass_runs_each_pass() {
        let p1: Arc<dyn Pass> = Arc::new(|f: &[f32]| Ok(f.iter().sum::<f32>()));
        let p2: Arc<dyn Pass> = Arc::new(|f: &[f32]| Ok(f.len() as f32));
        let cfg = MultiPassConfig { passes: vec![p1, p2] };

        let sys = ActorSystem::create("mp-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(MultiPassAnalysisActor::props(cfg), "mp").unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(MultiPassMsg::Analyze {
            frame: vec![1.0, 2.0, 3.0, 4.0],
            reply: tx,
        });
        let v = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(v, vec![10.0, 4.0]);

        sys.terminate().await;
    }
}
