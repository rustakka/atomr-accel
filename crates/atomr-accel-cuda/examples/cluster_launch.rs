//! `cluster_launch` — Phase 5 demo: build a [`LaunchSpec`] with a
//! non-trivial cluster dimension, validate it against the portable
//! 8-block cap, and print the resulting cluster geometry.
//!
//! Build / run (Hopper-only):
//!     cargo run -p atomr-accel-cuda --example cluster_launch \
//!         --features cuda-runtime-tests,hopper
//!
//! This example does *not* launch a real kernel — the cluster-launch
//! FFI surface lands once an sm_90a NVRTC source ships alongside this
//! crate. The example demonstrates the host-side validation path.

use atomr_accel_cuda::hopper::cluster::{ClusterDim, LaunchSpec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = LaunchSpec::new((128, 1, 1), (256, 1, 1))
        .with_cluster(ClusterDim::new(2, 2, 2))?
        .with_shared_bytes(48 * 1024);

    println!("LaunchSpec:");
    println!("  grid_dim   = {:?}", spec.grid_dim);
    println!("  block_dim  = {:?}", spec.block_dim);
    println!(
        "  cluster    = {:?} (= {} blocks per cluster)",
        spec.cluster_dim,
        spec.cluster_dim.block_count()
    );
    println!("  shared     = {} bytes", spec.shared_bytes);
    println!("  has_cluster= {}", spec.has_cluster());

    Ok(())
}
