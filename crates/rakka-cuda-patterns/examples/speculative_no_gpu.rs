//! Demonstrates `SpeculativeDecoder` with toy draft/verifier
//! closures. Run with:
//! `cargo run -p rakka-cuda-patterns --example speculative_no_gpu`

use std::sync::Arc;
use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;
use rakka_cuda_patterns::speculative::{
    DraftFn, SpecMsg, SpeculativeConfig, SpeculativeDecoder, VerifierFn,
};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    // Draft proposes 4 consecutive tokens.
    let draft: Arc<dyn DraftFn> = Arc::new(|prefix: &[u32], k: usize| {
        let last = prefix.last().copied().unwrap_or(0);
        Ok((1..=k as u32).map(|i| last + i).collect())
    });
    // Verifier accepts the first 3 of every batch, then injects 99
    // as the replacement for the 4th.
    let verifier: Arc<dyn VerifierFn> = Arc::new(|_prefix: &[u32], cands: &[u32]| {
        let acc = cands.len().min(3);
        Ok((acc, if acc < cands.len() { Some(99) } else { None }))
    });
    let cfg = SpeculativeConfig {
        draft,
        verifier,
        k: 4,
        max_total_tokens: 12,
    };

    let sys = ActorSystem::create("spec-demo", Config::empty()).await?;
    let dec = sys.actor_of(SpeculativeDecoder::props(cfg), "dec")?;

    let (tx, rx) = oneshot::channel();
    dec.tell(SpecMsg::Decode {
        prefix: vec![0],
        reply: tx,
    });
    let (tokens, stats) = tokio::time::timeout(Duration::from_secs(2), rx).await??.map_err(|e: GpuError| -> Box<dyn std::error::Error> { Box::new(e) })?;
    println!("tokens: {tokens:?}");
    println!(
        "iters={} draft={} accepted={} final_len={}",
        stats.iterations, stats.draft_tokens, stats.accepted_tokens, stats.final_len
    );

    sys.terminate().await;
    Ok(())
}
