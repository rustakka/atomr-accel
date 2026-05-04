//! Mock-mode tests for the typed-allocate / typed-copy `DeviceMsg`
//! variants. Verifies that:
//! - All typed-allocate variants reply with `Unrecoverable` in mock
//!   mode (since the mock ContextActor doesn't have a stream).
//! - The legacy `Allocate` alias still works.
//! - `Stats` and `SnapshotContext` reply meaningful values.

use std::time::Duration;
use tokio::sync::oneshot;

use atomr_accel_cuda::prelude::*;
use atomr_testkit::TestKit;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_allocate_typed_variants() {
    let kit = TestKit::new("alloc-test").await;
    let sys = &kit.system;
    let dev = sys
        .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev0")
        .unwrap();

    // F32
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 { len: 16, reply: tx });
    let r = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

    // F64
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF64 { len: 8, reply: tx });
    let r = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

    // I32
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateI32 { len: 4, reply: tx });
    let r = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

    // U8
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateU8 {
        len: 1024,
        reply: tx,
    });
    let r = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

    kit.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_allocate_alias_still_works() {
    let kit = TestKit::new("alloc-legacy").await;
    let sys = &kit.system;
    let dev = sys
        .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev0")
        .unwrap();

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::Allocate { len: 16, reply: tx });
    let r = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

    kit.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stats_returns_load_snapshot() {
    let kit = TestKit::new("stats-test").await;
    let sys = &kit.system;
    let dev = sys
        .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev0")
        .unwrap();

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::Stats { reply: tx });
    let load = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    // Mock-mode load values are zeros except queue_depth.
    assert_eq!(load.compute_cap, (0, 0));
    assert_eq!(load.active_streams, 0);

    kit.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_context_returns_none_in_mock_mode() {
    let kit = TestKit::new("snapshot-test").await;
    let sys = &kit.system;
    let dev = sys
        .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "dev0")
        .unwrap();

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::SnapshotContext { reply: tx });
    let r = tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .unwrap()
        .unwrap();
    // Mock mode never installs a real CudaContext; SnapshotContext
    // returns whatever DeviceState has, which is None for mock.
    assert!(r.is_none());

    kit.shutdown().await;
}
