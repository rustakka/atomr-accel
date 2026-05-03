//! Demonstrates `FairShareScheduler` weighting tenant 2 at 3× tenant
//! 1's bandwidth.
//!
//! Run with: `cargo run -p rakka-accel-patterns --example fair_share_no_gpu`

use std::sync::Arc;
use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;
use rakka_accel_patterns::scheduler::{
    FairDispatcher, FairShareConfig, FairShareMsg, FairShareScheduler, TenantConfig, TenantId,
};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    // Echo dispatcher: replies after a short sleep.
    let echo: Arc<dyn FairDispatcher<u32, u32>> =
        Arc::new(|req: u32, reply: oneshot::Sender<Result<u32, GpuError>>| {
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(5)).await;
                let _ = reply.send(Ok(req));
            });
        });

    let cfg = FairShareConfig {
        tenants: vec![
            TenantConfig { id: TenantId(1), weight: 1 },
            TenantConfig { id: TenantId(2), weight: 3 },
        ],
        dispatcher: echo,
        max_in_flight: 1,
    };

    let sys = ActorSystem::create("fair-demo", Config::empty()).await?;
    let sched =
        sys.actor_of(FairShareScheduler::<u32, u32>::props(cfg), "sched")?;

    // Submit 6 requests for tenant 1 and 6 for tenant 2 simultaneously.
    let mut rxs = Vec::new();
    for i in 0..6u32 {
        let (tx, rx) = oneshot::channel();
        sched.tell(FairShareMsg::Submit { tenant: TenantId(1), req: 100 + i, reply: tx });
        rxs.push((TenantId(1), 100 + i, rx));
        let (tx, rx) = oneshot::channel();
        sched.tell(FairShareMsg::Submit { tenant: TenantId(2), req: 200 + i, reply: tx });
        rxs.push((TenantId(2), 200 + i, rx));
    }
    for (tid, req, rx) in rxs {
        let v = tokio::time::timeout(Duration::from_secs(5), rx).await??;
        println!("tenant {tid:?} req {req} → {}", v?);
    }

    let (tx, rx) = oneshot::channel();
    sched.tell(FairShareMsg::Stats { reply: tx });
    let s = tokio::time::timeout(Duration::from_secs(2), rx).await??;
    println!("dispatched={}", s.total_dispatched);

    sys.terminate().await;
    Ok(())
}
