//! Smoke test the pinned-host buffer pool actor.
//!
//! The pool exercises raw `cuMemHostAlloc` / `cuMemFreeHost` via
//! cudarc — these only succeed when the CUDA driver is loadable on
//! the host. cudarc's `fallback-dynamic-loading` makes the build
//! work on no-GPU machines, but the actual alloc would fail there.
//!
//! Therefore: the test only verifies the actor's _construction +
//! Stats reply_ path (no allocation), which exercises the pre_start
//! mpsc-pump wiring without touching the CUDA driver.

use std::time::Duration;
use tokio::sync::oneshot;

use atomr_accel_cuda::prelude::*;
use atomr_testkit::TestKit;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_pool_replies_to_stats() {
    let cfg = PinnedBufferPoolConfig {
        initial_buffers: 0,
        max_buffers: 4,
        buffer_capacity_bytes: 1024,
        allow_oversize: false,
    };
    let kit = TestKit::new("pool-test").await;
    let pool = kit
        .system
        .actor_of(PinnedBufferPool::props(cfg), "pool")
        .unwrap();

    // Give pre_start a moment to wire the mpsc pump.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let (tx, rx) = oneshot::channel();
    pool.tell(PinnedPoolMsg::Stats { reply: tx });
    let stats = tokio::time::timeout(kit.default_timeout, rx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stats.in_use, 0);
    assert_eq!(stats.free, 0);
    assert_eq!(stats.total, 0);

    kit.shutdown().await;
}
