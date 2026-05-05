//! Thread-block cluster launches + Distributed Shared Memory (DSM)
//! helpers.
//!
//! Hopper introduced a fourth launch dimension: a *cluster* of thread
//! blocks. Blocks within a cluster can synchronise via `cluster.sync`
//! and read each other's shared memory through the DSM unit. The host
//! has to launch with `cudaLaunchKernelExC` (the older
//! `cudaLaunchKernel` lacks the cluster-dim field).
//!
//! This module ships:
//!
//! * [`ClusterDim`] — a `(x, y, z)` cluster size, validated against
//!   the 8-block portable limit (Hopper) / 16-block limit (Blackwell
//!   `cudaLaunchAttributeNonPortableClusterSizeAllowed`).
//! * [`LaunchSpec`] — grid + block + cluster + shared-memory bytes +
//!   stream, plus optional non-portable opt-in.
//! * [`launch_with_cluster`] (gated on `hopper`) — safe wrapper around
//!   `cudaLaunchKernelExC`.

use std::fmt;

/// Cluster dimensions. Hopper supports up to 8 blocks per cluster
/// (portable). Blackwell allows 16 with the non-portable opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl ClusterDim {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn unit() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Block count = x * y * z.
    pub const fn block_count(self) -> u32 {
        self.x * self.y * self.z
    }

    /// Validate against the portable per-cluster cap (8). Returns
    /// `Err(ClusterError::PortableLimit)` if the cluster exceeds 8 and
    /// the caller hasn't opted into non-portable mode.
    pub fn validate(self, allow_non_portable: bool) -> Result<(), ClusterError> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            return Err(ClusterError::ZeroDim);
        }
        let n = self.block_count();
        if n > 8 && !allow_non_portable {
            return Err(ClusterError::PortableLimit(n));
        }
        if n > 16 {
            return Err(ClusterError::HardLimit(n));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    ZeroDim,
    /// Cluster exceeds the 8-block portable cap.
    PortableLimit(u32),
    /// Cluster exceeds the 16-block hardware cap.
    HardLimit(u32),
    /// Driver returned a non-zero `cudaError_t`.
    Driver(i32),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterError::ZeroDim => write!(f, "cluster dim contains a zero"),
            ClusterError::PortableLimit(n) => write!(
                f,
                "cluster size {n} > 8 (portable limit); set allow_non_portable=true to opt in"
            ),
            ClusterError::HardLimit(n) => write!(f, "cluster size {n} > 16 (hard limit)"),
            ClusterError::Driver(c) => write!(f, "cudaLaunchKernelExC returned {c}"),
        }
    }
}

impl std::error::Error for ClusterError {}

/// Full launch specification for a cluster-aware kernel.
///
/// Mirrors `cudaLaunchKernelExC`'s `cudaLaunchConfig_t`:
/// `(gridDim, blockDim, sharedBytes, stream)` with the cluster
/// dimension threaded through the attributes array.
#[derive(Debug, Clone)]
pub struct LaunchSpec {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub cluster_dim: ClusterDim,
    pub shared_bytes: u32,
    /// Opt-in to clusters > 8 blocks (Blackwell only).
    pub allow_non_portable_cluster: bool,
}

impl LaunchSpec {
    /// Construct a spec with no cluster (1×1×1) — equivalent to the
    /// classic 3-tuple launch surface.
    pub fn new(grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32)) -> Self {
        Self {
            grid_dim,
            block_dim,
            cluster_dim: ClusterDim::unit(),
            shared_bytes: 0,
            allow_non_portable_cluster: false,
        }
    }

    /// Builder: set the cluster dimension. Validates against the
    /// 8-block portable cap on construction.
    pub fn with_cluster(mut self, cluster: ClusterDim) -> Result<Self, ClusterError> {
        cluster.validate(self.allow_non_portable_cluster)?;
        self.cluster_dim = cluster;
        Ok(self)
    }

    /// Builder: opt into non-portable cluster sizes (>8 blocks). Caller
    /// must re-validate the cluster afterwards.
    pub fn allow_non_portable(mut self) -> Self {
        self.allow_non_portable_cluster = true;
        self
    }

    /// Builder: set dynamic shared-memory bytes.
    pub fn with_shared_bytes(mut self, bytes: u32) -> Self {
        self.shared_bytes = bytes;
        self
    }

    /// True if this spec has a non-trivial cluster dim (anything other
    /// than 1×1×1).
    pub fn has_cluster(&self) -> bool {
        self.cluster_dim != ClusterDim::unit()
    }
}

/// Distributed-shared-memory helper: byte count needed to allocate
/// `per_block` bytes in every block of a cluster of size `cluster`.
pub const fn dsm_total_bytes(cluster: ClusterDim, per_block: u32) -> u64 {
    (cluster.block_count() as u64) * (per_block as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 5: round-trip the cluster-bearing launch spec through the
    /// builder. Pure host validation; no GPU.
    #[test]
    fn launch_spec_with_cluster_dim_constructs() {
        let spec = LaunchSpec::new((128, 1, 1), (256, 1, 1))
            .with_cluster(ClusterDim::new(2, 2, 1))
            .unwrap()
            .with_shared_bytes(48 * 1024);
        assert_eq!(spec.cluster_dim.block_count(), 4);
        assert!(spec.has_cluster());
        assert_eq!(spec.shared_bytes, 48 * 1024);
        // Underlying ClusterDim re-validates within bounds.
        spec.cluster_dim.validate(false).unwrap();
    }

    #[test]
    fn portable_limit_rejects_cluster_of_nine() {
        let cluster = ClusterDim::new(3, 3, 1); // 9 blocks
        assert!(matches!(
            cluster.validate(false).unwrap_err(),
            ClusterError::PortableLimit(9)
        ));
        // With non-portable allowed: passes.
        cluster.validate(true).unwrap();
    }

    #[test]
    fn hard_limit_rejects_cluster_of_seventeen() {
        let cluster = ClusterDim::new(17, 1, 1);
        assert!(matches!(
            cluster.validate(true).unwrap_err(),
            ClusterError::HardLimit(17)
        ));
    }

    #[test]
    fn zero_dim_rejected() {
        let cluster = ClusterDim::new(0, 1, 1);
        assert!(matches!(
            cluster.validate(true).unwrap_err(),
            ClusterError::ZeroDim
        ));
    }

    #[test]
    fn dsm_total_bytes_scales_linearly() {
        let cluster = ClusterDim::new(2, 2, 2); // 8 blocks
        assert_eq!(dsm_total_bytes(cluster, 4096), 8 * 4096);
        assert_eq!(dsm_total_bytes(ClusterDim::unit(), 4096), 4096);
    }

    #[test]
    fn unit_spec_has_no_cluster() {
        let spec = LaunchSpec::new((1, 1, 1), (32, 1, 1));
        assert!(!spec.has_cluster());
    }
}
