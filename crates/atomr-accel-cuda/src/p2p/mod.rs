//! P2P (peer-to-peer) topology + cross-device async memcpy.
//!
//! cudarc 0.19 exposes peer access only at the `sys` layer:
//! `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`,
//! `cuMemcpyPeerAsync`. This module wraps those with explicit
//! `unsafe` blocks behind an actor surface.
//!
//! Lifecycle:
//! 1. Construct with N `ActorRef<DeviceMsg>` siblings.
//! 2. Send `EnableAll` — actor snapshots each device's
//!    `Arc<CudaContext>`, probes `cuDeviceCanAccessPeer` for every
//!    pair, calls `cuCtxEnablePeerAccess` on directions that
//!    succeed, and replies with the resulting [`P2pGraph`].
//! 3. Send `CopyF32 { src, src_device, dst, dst_device }` — actor
//!    issues `cuMemcpyPeerAsync` on a fresh destination-side stream
//!    and replies after `cudaStreamSynchronize`.

#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context, Props};
use cudarc::driver::sys as driver_sys;
use cudarc::driver::CudaContext;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use parking_lot::Mutex;
use tokio::sync::oneshot;
use tracing::info;

use crate::device::DeviceMsg;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;

#[derive(Debug, Clone)]
pub struct P2pGraph {
    pub edges: Vec<Vec<bool>>,
    pub device_count: u32,
}

impl P2pGraph {
    pub fn new(n: u32) -> Self {
        Self {
            edges: (0..n).map(|_| vec![false; n as usize]).collect(),
            device_count: n,
        }
    }

    pub fn can_pair(&self, a: u32, b: u32) -> bool {
        self.edges[a as usize][b as usize]
    }

    /// Connected components (NVLink islands).
    pub fn islands(&self) -> Vec<HashSet<u32>> {
        let n = self.device_count as usize;
        let mut visited = vec![false; n];
        let mut out = Vec::new();
        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut stack = vec![i];
            let mut island = HashSet::new();
            while let Some(j) = stack.pop() {
                if visited[j] {
                    continue;
                }
                visited[j] = true;
                island.insert(j as u32);
                for k in 0..n {
                    if !visited[k] && (self.edges[j][k] || self.edges[k][j]) {
                        stack.push(k);
                    }
                }
            }
            out.push(island);
        }
        out
    }
}

pub enum P2pMsg {
    EnableAll {
        reply: oneshot::Sender<Result<P2pGraph, GpuError>>,
    },
    CanAccess {
        from: u32,
        to: u32,
        reply: oneshot::Sender<bool>,
    },
    /// Async peer copy from `src` (on `src_device`) to `dst`
    /// (on `dst_device`). Both `GpuRef`s must be valid; copy size
    /// is `min(src.len, dst.len)` × sizeof(f32).
    CopyF32 {
        src: GpuRef<f32>,
        src_device: u32,
        dst: GpuRef<f32>,
        dst_device: u32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Topology {
        reply: oneshot::Sender<P2pGraph>,
    },
    /// Internal: a device's generation_watch fired, meaning its
    /// `ContextActor` rebuilt the underlying `CudaContext`. The
    /// cached `Arc<CudaContext>` in `self.contexts` is now stale —
    /// re-snapshot it. Peer-access mappings persist across rebuilds
    /// only if the new context lives in the same primary slot, so
    /// the topology marks itself disabled and a follow-up
    /// `EnableAll` is required to refresh `graph` + re-issue
    /// `cuCtxEnablePeerAccess`.
    RefreshDevice {
        device_idx: u32,
        new_generation: u64,
    },
}

struct SendCtx(Arc<CudaContext>);
unsafe impl Send for SendCtx {}
unsafe impl Sync for SendCtx {}

pub struct P2pTopology {
    devices: Vec<ActorRef<DeviceMsg>>,
    contexts: Mutex<Vec<Option<SendCtx>>>,
    graph: P2pGraph,
    enabled: bool,
}

impl P2pTopology {
    pub fn props(devices: Vec<ActorRef<DeviceMsg>>) -> Props<Self> {
        let n = devices.len() as u32;
        Props::create(move || P2pTopology {
            devices: devices.clone(),
            contexts: Mutex::new((0..n).map(|_| None).collect()),
            graph: P2pGraph::new(n),
            enabled: false,
        })
    }
}

#[async_trait]
impl Actor for P2pTopology {
    type Msg = P2pMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        // Subscribe to each sibling DeviceActor's generation_watch so a
        // ContextActor rebuild invalidates this topology's cached
        // contexts. The bridge spawns one task per device that asks
        // the device for its watch::Receiver and forwards every change
        // back to us as a RefreshDevice message.
        let self_ref = ctx.self_ref().clone();
        for (idx, dev) in self.devices.iter().enumerate() {
            let topo_ref = self_ref.clone();
            let dev_ref = dev.clone();
            tokio::spawn(async move {
                let watch_rx_res = dev_ref
                    .ask_with(
                        move |tx| DeviceMsg::WatchGeneration { reply: tx },
                        std::time::Duration::from_secs(5),
                    )
                    .await;
                let mut rx = match watch_rx_res {
                    Ok(rx) => rx,
                    Err(_) => return,
                };
                let mut last = *rx.borrow();
                while rx.changed().await.is_ok() {
                    let gen = *rx.borrow();
                    if gen != last {
                        last = gen;
                        topo_ref.tell(P2pMsg::RefreshDevice {
                            device_idx: idx as u32,
                            new_generation: gen,
                        });
                    }
                }
            });
        }
    }

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: P2pMsg) {
        match msg {
            P2pMsg::EnableAll { reply } => {
                let n = self.devices.len();
                // Snapshot each device's context. Returns None until
                // ContextActor::Init completes.
                let mut snaps: Vec<Option<Arc<CudaContext>>> = Vec::with_capacity(n);
                for d in &self.devices {
                    let (tx, rx) = oneshot::channel();
                    d.tell(DeviceMsg::SnapshotContext { reply: tx });
                    match rx.await {
                        Ok(c) => snaps.push(c),
                        Err(_) => snaps.push(None),
                    }
                }
                {
                    let mut g = self.contexts.lock();
                    for (i, s) in snaps.iter().enumerate() {
                        g[i] = s.clone().map(SendCtx);
                    }
                }

                let mut graph = P2pGraph::new(n as u32);
                let any_unloadable = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for i in 0..n {
                        let Some(ctx_a) = snaps[i].as_ref() else {
                            continue;
                        };
                        for j in 0..n {
                            if i == j {
                                graph.edges[i][j] = true;
                                continue;
                            }
                            let Some(_) = snaps[j].as_ref() else { continue };
                            let mut can = 0i32;
                            // cuDeviceCanAccessPeer takes ordinals.
                            let s = unsafe {
                                driver_sys::cuDeviceCanAccessPeer(
                                    &mut can as *mut _,
                                    ctx_a.cu_device(),
                                    snaps[j].as_ref().unwrap().cu_device(),
                                )
                            };
                            if s == driver_sys::cudaError_enum::CUDA_SUCCESS && can == 1 {
                                graph.edges[i][j] = true;
                            }
                        }
                    }
                    Ok::<(), GpuError>(())
                }));
                if any_unloadable.is_err() {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "P2pTopology::EnableAll: CUDA driver not loadable".into(),
                    )));
                    return;
                }

                // Enable peer access in each direction where probe
                // succeeded. cuCtxEnablePeerAccess must be called from
                // the source context (set current) targeting the peer.
                let enable_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for i in 0..n {
                        let Some(ctx_a) = snaps[i].as_ref() else {
                            continue;
                        };
                        let _ = ctx_a.bind_to_thread();
                        for j in 0..n {
                            if i == j || !graph.edges[i][j] {
                                continue;
                            }
                            let peer = snaps[j].as_ref().unwrap();
                            let s = unsafe { driver_sys::cuCtxEnablePeerAccess(peer.cu_ctx(), 0) };
                            // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED is fine.
                            if s != driver_sys::cudaError_enum::CUDA_SUCCESS
                                && s
                                    != driver_sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                            {
                                graph.edges[i][j] = false;
                            }
                        }
                    }
                }));
                let _ = enable_res; // partial enables are best-effort

                self.graph = graph.clone();
                self.enabled = true;
                info!(devices = n, "P2pTopology::EnableAll done");
                let _ = reply.send(Ok(graph));
            }
            P2pMsg::CanAccess { from, to, reply } => {
                let v = if from == to {
                    true
                } else {
                    self.graph
                        .edges
                        .get(from as usize)
                        .and_then(|row| row.get(to as usize).copied())
                        .unwrap_or(false)
                };
                let _ = reply.send(v);
            }
            P2pMsg::CopyF32 {
                src,
                src_device,
                dst,
                dst_device,
                reply,
            } => {
                if !self.enabled {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "P2pTopology: call EnableAll first".into(),
                    )));
                    return;
                }
                if !self.graph.can_pair(src_device, dst_device) {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "P2pTopology: device {src_device} cannot peer-access {dst_device}"
                    ))));
                    return;
                }
                let ctxs = self.contexts.lock();
                let src_ctx = match ctxs.get(src_device as usize).and_then(|c| c.as_ref()) {
                    Some(c) => c.0.clone(),
                    None => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                            "P2pTopology: src device {src_device} context not available"
                        ))));
                        return;
                    }
                };
                let dst_ctx = match ctxs.get(dst_device as usize).and_then(|c| c.as_ref()) {
                    Some(c) => c.0.clone(),
                    None => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                            "P2pTopology: dst device {dst_device} context not available"
                        ))));
                        return;
                    }
                };
                drop(ctxs);

                let src_slice = match src.access() {
                    Ok(s) => s.clone(),
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let dst_slice = match dst.access() {
                    Ok(s) => s.clone(),
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let mut dst_owned = match Arc::try_unwrap(dst_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "P2pCopy: dst has multiple live references".into(),
                        )));
                        return;
                    }
                };

                let len = std::cmp::min(src_slice.len(), dst_owned.len());
                let bytes = len * std::mem::size_of::<f32>();
                // Mint a destination-side stream for the copy.
                let dst_stream = match dst_ctx.new_stream() {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = reply.send(Err(GpuError::LibraryError {
                            lib: "driver",
                            msg: format!("dst new_stream: {e}"),
                        }));
                        return;
                    }
                };

                // F9.2: if the source `GpuRef` carries a recorded
                // last-write stream (set by upstream BlasActor /
                // CudnnActor / etc.), inject a cross-stream event
                // wait so the peer copy doesn't race with in-flight
                // writes — and we don't have to host-synchronize.
                let last_write_src = src.last_write_stream();
                let copy_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Some(src_stream) = last_write_src.as_ref() {
                        let ev =
                            src_stream
                                .record_event(None)
                                .map_err(|e| GpuError::LibraryError {
                                    lib: "driver",
                                    msg: format!("p2p: src record_event: {e}"),
                                })?;
                        // Wait on the destination side. Cross-context
                        // event waits work because cuCtxEnablePeerAccess
                        // was already called in EnableAll.
                        dst_stream.wait(&ev).map_err(|e| GpuError::LibraryError {
                            lib: "driver",
                            msg: format!("p2p: dst wait: {e}"),
                        })?;
                    }
                    let (src_ptr, _g1) = src_slice.device_ptr(&dst_stream);
                    let (dst_ptr, _g2) = dst_owned.device_ptr_mut(&dst_stream);
                    let s = unsafe {
                        driver_sys::cuMemcpyPeerAsync(
                            dst_ptr,
                            dst_ctx.cu_ctx(),
                            src_ptr,
                            src_ctx.cu_ctx(),
                            bytes,
                            dst_stream.cu_stream(),
                        )
                    };
                    drop((_g1, _g2));
                    if s != driver_sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(GpuError::LibraryError {
                            lib: "driver",
                            msg: format!("cuMemcpyPeerAsync: {s:?}"),
                        });
                    }
                    // Synchronize as the host-visible barrier. A
                    // future improvement (F10) would replace this
                    // with a HostFnCompletion-style callback so the
                    // actor never blocks the OS thread.
                    dst_stream
                        .synchronize()
                        .map_err(|e| GpuError::LibraryError {
                            lib: "driver",
                            msg: format!("cudaStreamSynchronize: {e}"),
                        })?;
                    Ok(())
                }));
                let result = match copy_res {
                    Ok(r) => r,
                    Err(_) => Err(GpuError::Unrecoverable(
                        "P2pCopy: CUDA driver not loadable".into(),
                    )),
                };
                dst.record_write(&dst_stream);
                let _ = reply.send(result);
                drop(dst_owned);
            }
            P2pMsg::Topology { reply } => {
                let _ = reply.send(self.graph.clone());
            }
            P2pMsg::RefreshDevice {
                device_idx,
                new_generation,
            } => {
                info!(
                    device_idx,
                    new_generation,
                    "P2pTopology: device context rebuilt — invalidating cached snapshot"
                );
                // Re-snapshot just the affected device. Other devices'
                // cached contexts remain valid until they themselves
                // bump their generation.
                let dev = match self.devices.get(device_idx as usize) {
                    Some(d) => d.clone(),
                    None => return,
                };
                let (tx, rx) = oneshot::channel();
                dev.tell(DeviceMsg::SnapshotContext { reply: tx });
                let new_ctx = rx.await.unwrap_or_default();
                {
                    let mut g = self.contexts.lock();
                    if let Some(slot) = g.get_mut(device_idx as usize) {
                        *slot = new_ctx.map(SendCtx);
                    }
                }
                // Peer-access on the rebuilt context isn't enabled
                // automatically. Force callers to re-issue EnableAll
                // before the next CopyF32 — otherwise the copy would
                // surface a CUDA_ERROR_INVALID_VALUE.
                self.enabled = false;
            }
        }
    }
}
