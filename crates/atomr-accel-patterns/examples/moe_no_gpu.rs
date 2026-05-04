//! Demonstrates `MoeRouter` with three expert actors and a softmax
//! gate. Run with:
//! `cargo run -p atomr-accel-patterns --example moe_no_gpu`

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr_config::Config;
use atomr_core::actor::{Actor, ActorSystem, Context, Props};
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;
use atomr_accel_patterns::moe::{ExpertProtocol, GateFn, MoeConfig, MoeMsg, MoeRouter};

enum ExpertMsg {
    Run {
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
}
struct Expert {
    bias: f32,
    name: &'static str,
}
#[async_trait]
impl Actor for Expert {
    type Msg = ExpertMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ExpertMsg) {
        match msg {
            ExpertMsg::Run { input, reply } => {
                println!("expert '{}' running on {} inputs", self.name, input.len());
                let v = input.iter().map(|x| x + self.bias).collect();
                let _ = reply.send(Ok(v));
            }
        }
    }
}
struct Proto;
impl ExpertProtocol for Proto {
    type Msg = ExpertMsg;
    fn make_run(input: Vec<f32>, reply: oneshot::Sender<Result<Vec<f32>, GpuError>>) -> ExpertMsg {
        ExpertMsg::Run { input, reply }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let sys = ActorSystem::create("moe-demo", Config::empty()).await?;
    let e0 = sys.actor_of(
        Props::create(|| Expert {
            bias: 1.0,
            name: "e0",
        }),
        "e0",
    )?;
    let e1 = sys.actor_of(
        Props::create(|| Expert {
            bias: 10.0,
            name: "e1",
        }),
        "e1",
    )?;
    let e2 = sys.actor_of(
        Props::create(|| Expert {
            bias: 100.0,
            name: "e2",
        }),
        "e2",
    )?;

    // Gate: based on input mean, prefer larger-bias experts for
    // larger inputs.
    let gate: Arc<dyn GateFn> = Arc::new(|input: &[f32]| {
        let mean = input.iter().copied().sum::<f32>() / input.len().max(1) as f32;
        // Three experts: scores ramp up with the input mean.
        Ok(vec![1.0 / (1.0 + mean), 0.5 + mean * 0.1, mean])
    });
    let cfg = MoeConfig::<Proto> {
        experts: vec![e0, e1, e2],
        gate,
        top_k: 2,
    };
    let router = sys.actor_of(MoeRouter::<Proto>::props(cfg), "router")?;

    for input in [vec![0.5; 4], vec![5.0; 4]] {
        let (tx, rx) = oneshot::channel();
        router.tell(MoeMsg::Run {
            input: input.clone(),
            reply: tx,
        });
        let v = tokio::time::timeout(Duration::from_secs(2), rx)
            .await??
            .map_err(|e: GpuError| -> Box<dyn std::error::Error> { Box::new(e) })?;
        println!("input={input:?} → blended {v:?}");
    }

    sys.terminate().await;
    Ok(())
}
