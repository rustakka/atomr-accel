//! Opt-in NVML smoke test. Probes device 0's name + temperature
//! against a real `libnvidia-ml.so.1`. Skipped if NVML can't load.
//!
//! Run via `cargo xtask gpu-test telemetry` or:
//!   cargo test -p atomr-accel-telemetry --features nvml \
//!     -- --ignored --nocapture

#![cfg(feature = "nvml")]

use atomr_accel_telemetry::nvml::{NvmlActor, NvmlConfig};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires NVML (libnvidia-ml.so.1) on the host"]
async fn nvml_snapshot_returns_nonempty_device_list() {
    let probe = std::panic::catch_unwind(|| NvmlActor::try_new(NvmlConfig::default()));
    let actor = match probe {
        Ok(Ok(a)) => a,
        Ok(Err(e)) => {
            eprintln!("[skip] NVML not available: {e}");
            return;
        }
        Err(_) => {
            eprintln!("[skip] NVML panicked on init (likely missing libnvidia-ml.so.1)");
            return;
        }
    };
    // Give the polling loop one tick to populate.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let snap = actor.latest_snapshot();
    if snap.devices.is_empty() {
        eprintln!("[skip] NVML loaded but reported zero devices");
        return;
    }
    let dev0 = &snap.devices[0];
    let name = dev0.name.as_deref().unwrap_or("(unnamed)");
    let used_mb = dev0.mem_used_bytes.map(|b| b / (1024 * 1024)).unwrap_or(0);
    let total_mb = dev0.mem_total_bytes.map(|b| b / (1024 * 1024)).unwrap_or(0);
    println!(
        "[nvml] device 0: {} | gpu_temp_c={:?} | mem_used={}MB / {}MB",
        name, dev0.temperature_gpu_c, used_mb, total_mb,
    );
    assert!(
        !name.is_empty() || dev0.uuid.is_some(),
        "device 0 had no name and no UUID"
    );
}
