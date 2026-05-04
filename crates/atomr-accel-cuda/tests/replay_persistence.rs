//! Roundtrip Record → persistence Journal → LoadFromJournal for the
//! `replay` feature. Uses `InMemoryJournal` so runs without a real
//! database.

#![cfg(feature = "replay")]

use std::sync::Arc;
use std::time::Duration;

use atomr_accel_cuda::replay::{JournalEntry, ReplayHarness, ReplayMode, ReplayMsg};
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use atomr_persistence::InMemoryJournal;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_roundtrips_through_persistent_journal() {
    let sys = ActorSystem::create("replay-persistence", Config::empty())
        .await
        .unwrap();
    let journal: Arc<dyn atomr_persistence::Journal> = InMemoryJournal::new();
    let pid = "atomr-accel-cuda/replay/test";

    // First incarnation: record three events through the persisted
    // journal.
    let recorder = sys
        .actor_of(
            ReplayHarness::with_journal(ReplayMode::Record, journal.clone(), pid),
            "recorder",
        )
        .unwrap();
    for seed in 1..=3u64 {
        recorder.tell(ReplayMsg::Record(JournalEntry::RngSeed {
            actor_path: format!("rng-{seed}"),
            seed,
        }));
    }
    // Allow the asynchronous journal writes to complete.
    tokio::time::sleep(Duration::from_millis(100)).await;
    recorder.stop();

    // Second incarnation: spin up a fresh harness in Replay mode and
    // pull history from the journal.
    let replayer = sys
        .actor_of(
            ReplayHarness::with_journal(ReplayMode::Replay, journal.clone(), pid),
            "replayer",
        )
        .unwrap();
    let (tx, rx) = oneshot::channel();
    replayer.tell(ReplayMsg::LoadFromJournal {
        from_sequence_nr: 1,
        max: 100,
        reply: tx,
    });
    let n = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .expect("LoadFromJournal timeout")
        .expect("oneshot dropped")
        .expect("LoadFromJournal failed");
    assert_eq!(n, 3, "expected 3 entries from the journal");

    // Snapshot the loaded journal back out as JournalEntry instances.
    let (tx, rx) = oneshot::channel();
    replayer.tell(ReplayMsg::Snapshot { reply: tx });
    let entries = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .expect("Snapshot timeout")
        .expect("oneshot dropped");
    assert_eq!(entries.len(), 3);
    for (i, e) in entries.iter().enumerate() {
        let expected_seed = (i + 1) as u64;
        match e {
            JournalEntry::RngSeed { seed, .. } => assert_eq!(*seed, expected_seed),
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    sys.terminate().await;
}
