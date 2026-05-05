//! `cupti_trace` — minimal end-to-end driver for [`CuptiSession`].
//!
//! Spawns the session, starts a kernel-launch + memcpy trace,
//! waits briefly, and drains records. On a host without CUPTI the
//! bootstrap returns `Err(CuptiError::LibraryUnavailable)` and the
//! example logs + exits 0 — so it still exercises the build /
//! type-check path in CI.

use atomr_accel_telemetry::cupti::{
    ActivityCategory, CuptiBootstrap, CuptiMsg, CuptiSession,
};
use tokio::sync::oneshot;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CUPTI must be loaded BEFORE cuInit. We log + continue on
    // hosts where libcupti is missing so this example doubles as a
    // CI smoke test.
    let _bootstrap = match CuptiBootstrap::install() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("(cupti_trace) bootstrap unavailable: {e}");
            None
        }
    };

    let session = CuptiSession::spawn();
    let tx = session.sender();

    let (rtx, rrx) = oneshot::channel();
    tx.send(CuptiMsg::Start {
        categories: vec![
            ActivityCategory::KernelLaunch,
            ActivityCategory::Memcpy,
            ActivityCategory::DriverApi,
            ActivityCategory::RuntimeApi,
        ],
        reply: rtx,
    })
    .await?;
    rrx.await??;

    // In a real session you'd run kernels here. We just sleep for a
    // beat so the actor's mpsc has a chance to drain.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let (rtx, rrx) = oneshot::channel();
    tx.send(CuptiMsg::Drain { reply: rtx }).await?;
    let activities = rrx.await??;
    eprintln!("(cupti_trace) drained {} activities", activities.len());

    let (rtx, rrx) = oneshot::channel();
    tx.send(CuptiMsg::Stop { reply: rtx }).await?;
    rrx.await??;

    Ok(())
}
