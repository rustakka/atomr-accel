//! `DeviceActor` (outer tier) + `ContextActor` (inner tier) — §5.11.

mod alloc_dispatch;
mod alloc_msg;
mod context_actor;
pub mod device_actor;
mod state;

pub use alloc_dispatch::{
    AllocDispatch, AllocReq, CopyFromHostDispatch, CopyFromHostReq, CopyToHostDispatch,
    CopyToHostReq,
};
pub use alloc_msg::{DeviceLoad, HostBuf};
pub use context_actor::{ContextActor, ContextMsg};
pub use device_actor::{
    DeviceActor, DeviceConfig, DeviceMsg, EnabledLibraries, KernelChildren, SgemmRequest,
    WorkRequest,
};
pub use state::DeviceState;
