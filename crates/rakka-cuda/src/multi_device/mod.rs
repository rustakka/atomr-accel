//! Top-level multi-device actors that span multiple `DeviceActor`s.
//!
//! F4 ships:
//! - [`world_actor::NcclWorldActor`] — supervises N `DeviceActor`s
//!   and N `CollectiveActor`s for a single NCCL group.

#[cfg(feature = "nccl")]
mod world_actor;

#[cfg(feature = "nccl")]
pub use world_actor::{NcclWorldActor, NcclWorldConfig, NcclWorldMsg};
