//! Demonstrates `InferenceCascade` with three CPU-side stages.
//!
//! Run with: `cargo run -p atomr-accel-patterns --example cascade_no_gpu`

use std::sync::Arc;
use std::time::Duration;

use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use tokio::sync::oneshot;

use atomr_accel_patterns::cascade::{
    CascadeConfig, CascadeMsg, CascadeStage, CascadeStageEntry, InferenceCascade,
};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    // Stage 0: cheap heuristic — even numbers get a low-confidence guess.
    let s0: Arc<dyn CascadeStage<i32, &'static str>> = Arc::new(|x: &i32| {
        if *x % 2 == 0 {
            Ok(("even", 0.4))
        } else {
            Ok(("odd", 0.4))
        }
    });
    // Stage 1: fancier — high confidence on small inputs.
    let s1: Arc<dyn CascadeStage<i32, &'static str>> = Arc::new(|x: &i32| {
        if x.abs() <= 5 {
            Ok(("small", 0.95))
        } else {
            Ok(("large-ish", 0.6))
        }
    });
    // Stage 2: catch-all.
    let s2: Arc<dyn CascadeStage<i32, &'static str>> = Arc::new(|_: &i32| Ok(("fallback", 1.0)));

    let cfg = CascadeConfig {
        stages: vec![
            CascadeStageEntry {
                stage: s0,
                confidence_threshold: 0.5,
            },
            CascadeStageEntry {
                stage: s1,
                confidence_threshold: 0.9,
            },
            CascadeStageEntry {
                stage: s2,
                confidence_threshold: 0.0,
            },
        ],
    };

    let sys = ActorSystem::create("cascade-demo", Config::empty()).await?;
    let cascade = sys.actor_of(InferenceCascade::<i32, &'static str>::props(cfg), "cascade")?;

    for input in [3, 4, 100, -7] {
        let (tx, rx) = oneshot::channel();
        cascade.tell(CascadeMsg::Predict {
            req: input,
            reply: tx,
        });
        let r = tokio::time::timeout(Duration::from_secs(2), rx).await??;
        let r = r?;
        println!(
            "{input} → {} (stage {}, conf {})",
            r.response, r.stage_index, r.confidence
        );
    }

    sys.terminate().await;
    Ok(())
}
