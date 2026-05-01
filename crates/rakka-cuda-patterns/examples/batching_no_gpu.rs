//! Demonstrates `DynamicBatchingServer` with a CPU-side echo
//! batch-fn. Runs entirely without a GPU — the batched call is
//! `Vec<u32> -> Vec<Result<u32>>` doing trivial arithmetic.
//!
//! Run with: `cargo run -p rakka-cuda-patterns --example batching_no_gpu`

use std::sync::Arc;
use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::sync::oneshot;

use rakka_cuda_patterns::batching::{
    BatchFn, BatchOverflow, BatchingConfig, BatchingMsg, DynamicBatchingServer,
};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Square each request.
    let square: Arc<dyn BatchFn<u32, u32>> =
        Arc::new(|reqs: Vec<u32>| reqs.into_iter().map(|x| Ok(x * x)).collect());

    let cfg = BatchingConfig {
        max_batch: 4,
        max_wait: Duration::from_millis(20),
        batch_fn: square,
        overflow: BatchOverflow::Reject,
    };

    let sys = ActorSystem::create("batching-demo", Config::empty()).await?;
    let server = sys.actor_of(DynamicBatchingServer::<u32, u32>::props(cfg), "server")?;

    // Submit 10 requests; pairs of 4 + 4 + 2 will flush in batches.
    let mut rxs = Vec::new();
    for i in 0..10u32 {
        let (tx, rx) = oneshot::channel();
        server.tell(BatchingMsg::Submit { req: i, reply: tx });
        rxs.push((i, rx));
    }

    for (i, rx) in rxs {
        let v = tokio::time::timeout(Duration::from_secs(2), rx).await??;
        println!("{i}^2 = {}", v?);
    }

    let (tx, rx) = oneshot::channel();
    server.tell(BatchingMsg::Stats { reply: tx });
    let stats = tokio::time::timeout(Duration::from_secs(2), rx).await??;
    println!("stats: flushes={} processed={}", stats.flushes, stats.items_processed);

    sys.terminate().await;
    Ok(())
}
