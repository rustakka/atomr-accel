//! `atomr-cluster-sharding` adapter for [`super::PlacementActor`].
//!
//! Bridges the GPU-fleet placement layer to atomr's typed sharding
//! primitives. Callers wrap their [`crate::device::DeviceMsg`] with a
//! [`RoutedDeviceMsg { entity_id, msg }`] envelope, and the adapter
//! exposes an [`EntityRef<DeviceExtractor>`] whose `tell` routes the
//! message to the device owning that entity id.
//!
//! This enables:
//! - Cluster-wide shard handoff if/when a remote forwarder is wired in
//!   via [`ShardRegion::set_remote_forwarder`].
//! - Consistent placement: identical `entity_id`s always land on the
//!   same device, even across restarts of the calling code.
//!
//! The current implementation uses a simple FxHash-mod-N consistent
//! routing policy (no live-load awareness). A follow-up can install a
//! custom [`ShardCoordinator`]-driven allocation strategy that polls
//! the underlying [`super::PlacementActor`]'s load snapshot.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use atomr_cluster_sharding::{EntityRef, MessageExtractor, ShardCoordinator, ShardRegion};
use atomr_core::actor::ActorRef;

use crate::device::DeviceMsg;

/// Envelope used by the sharding adapter. Wraps an underlying
/// `DeviceMsg` with the entity id the caller wants the message to be
/// routed by. The adapter's `MessageExtractor` reads `entity_id` to
/// pick the destination shard / device.
pub struct RoutedDeviceMsg {
    pub entity_id: String,
    pub msg: DeviceMsg,
}

/// `MessageExtractor` impl for [`RoutedDeviceMsg`]. Hashes `entity_id`
/// into one of `shard_count` shards.
pub struct DeviceExtractor {
    shard_count: usize,
}

impl DeviceExtractor {
    pub fn new(shard_count: usize) -> Self {
        Self {
            shard_count: shard_count.max(1),
        }
    }
}

impl MessageExtractor for DeviceExtractor {
    type Message = RoutedDeviceMsg;

    fn entity_id(&self, message: &Self::Message) -> String {
        message.entity_id.clone()
    }

    fn shard_id(&self, message: &Self::Message) -> String {
        let mut h = DefaultHasher::new();
        message.entity_id.hash(&mut h);
        let n = h.finish() as usize % self.shard_count;
        format!("shard-{n}")
    }
}

/// Adapter that publishes a [`ShardRegion<DeviceExtractor>`] backed by
/// a fixed pool of pre-spawned [`DeviceActor`](crate::device::DeviceActor)
/// refs. Each shard maps to one device via `shard_index % devices.len()`,
/// so identical `entity_id`s always reach the same device.
pub struct PlacementShardingAdapter {
    region: Arc<ShardRegion<DeviceExtractor>>,
}

impl PlacementShardingAdapter {
    /// Build the adapter from a fleet of device refs. `region_id` is
    /// the cluster-visible name of this region (mirrors akka.net's
    /// type-name parameter to `ClusterSharding.Start(...)`).
    ///
    /// `shard_count` controls the routing granularity; larger values
    /// distribute consecutive entity ids more evenly. Defaults to the
    /// number of devices when zero.
    pub fn start(
        region_id: impl Into<String>,
        devices: Vec<ActorRef<DeviceMsg>>,
        shard_count: usize,
    ) -> Self {
        let n_devices = devices.len().max(1);
        let n_shards = if shard_count == 0 {
            n_devices
        } else {
            shard_count
        };
        let extractor = Arc::new(DeviceExtractor::new(n_shards));
        let coord = Arc::new(ShardCoordinator::new());
        // Devices captured in a shared Arc so the handler closures
        // (one per shard) all see the same fleet.
        let devices = Arc::new(devices);
        let devices_for_factory = devices.clone();
        let region = ShardRegion::new(
            region_id,
            extractor,
            coord,
            Arc::new(move || {
                // Each shard creates its own EntityHandler; capture
                // the shared device pool so all handlers route into
                // the same fleet. Dispatch is consistent-hash by
                // entity_id mod n_devices.
                let devices = devices_for_factory.clone();
                Box::new(move |entity_id: &str, msg: RoutedDeviceMsg| {
                    if devices.is_empty() {
                        return;
                    }
                    let mut h = DefaultHasher::new();
                    entity_id.hash(&mut h);
                    let idx = (h.finish() as usize) % devices.len();
                    devices[idx].tell(msg.msg);
                })
            }),
        );
        Self { region }
    }

    /// Build a typed handle to a particular entity.
    pub fn entity(&self, entity_id: impl Into<String>) -> EntityRef<DeviceExtractor> {
        EntityRef::new(self.region.clone(), entity_id.into())
    }

    /// Direct access to the underlying [`ShardRegion`] — useful for
    /// installing a remote forwarder or inspecting shard counts.
    pub fn region(&self) -> Arc<ShardRegion<DeviceExtractor>> {
        self.region.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceActor, DeviceConfig};
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;
    use std::time::Duration;
    use tokio::sync::oneshot;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn entity_ref_routes_to_one_of_the_devices() {
        let sys = ActorSystem::create("sharding-adapter", Config::empty())
            .await
            .unwrap();
        let d0 = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(0)), "d0")
            .unwrap();
        let d1 = sys
            .actor_of(DeviceActor::props(DeviceConfig::mock(1)), "d1")
            .unwrap();
        let adapter = PlacementShardingAdapter::start("gpu", vec![d0, d1], 16);

        // Same entity_id should hash to the same device deterministically.
        let entity = adapter.entity("user-42");
        let (tx, rx) = oneshot::channel();
        entity.tell(RoutedDeviceMsg {
            entity_id: "user-42".into(),
            msg: DeviceMsg::Allocate { len: 16, reply: tx },
        });
        // We don't verify which device served it — only that the route
        // delivered (the reply arrives, even if it's an
        // Unrecoverable from mock mode).
        let _ = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .expect("Allocate reply should arrive within timeout");

        sys.terminate().await;
    }
}
