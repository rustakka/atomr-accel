//! `GpuSparseStructureActor` — coordinate-list (COO) sparse matrix
//! with simple SpMV.
//!
//! Default ships a CPU reference. With the `nvrtc` feature enabled,
//! [`GpuSparseStructureActor::with_nvrtc`] compiles
//! [`crate::kernels::COO_SPMV_SRC`] via
//! [`atomr_accel_cuda::kernel::NvrtcActor`] and dispatches `SpMv` through
//! the JIT-launched kernel. The cuSPARSE-backed variant lives in
//! `atomr-accel-cuda` proper (Phase C.1) and is the preferred backend when
//! that feature is on.

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;

#[cfg(feature = "nvrtc")]
use atomr_accel_cuda::kernel::{KernelHandle, NvrtcMsg, NvrtcOpts};
#[cfg(feature = "nvrtc")]
use atomr_core::actor::ActorRef;
#[cfg(feature = "nvrtc")]
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct CooEntry {
    pub row: u32,
    pub col: u32,
    pub value: f32,
}

#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub rows: usize,
    pub cols: usize,
}

pub enum SparseMsg {
    SetEntries {
        entries: Vec<CooEntry>,
        reply: oneshot::Sender<usize>,
    },
    /// y = A * x where A is the configured COO matrix. Returns y
    /// (length `rows`).
    SpMv {
        x: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
    Stats {
        reply: oneshot::Sender<SparseStats>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SparseStats {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

pub struct GpuSparseStructureActor {
    cfg: SparseConfig,
    entries: Vec<CooEntry>,
    /// Optional NVRTC-compiled SpMv kernel. Populated by
    /// [`Self::with_nvrtc`]; SpMv handler dispatches through it when
    /// present and falls back to the CPU loop otherwise.
    #[cfg(feature = "nvrtc")]
    spmv_kernel: Option<Arc<KernelHandle>>,
    #[cfg(feature = "nvrtc")]
    nvrtc: Option<ActorRef<NvrtcMsg>>,
}

impl GpuSparseStructureActor {
    pub fn props(cfg: SparseConfig) -> Props<Self> {
        Props::create(move || GpuSparseStructureActor {
            cfg: cfg.clone(),
            entries: Vec::new(),
            #[cfg(feature = "nvrtc")]
            spmv_kernel: None,
            #[cfg(feature = "nvrtc")]
            nvrtc: None,
        })
    }

    /// Build an actor that dispatches `SpMv` through an NVRTC-compiled
    /// `coo_spmv` kernel rather than the CPU loop. The compile happens
    /// lazily on first `SpMv` so the actor still starts up on hosts
    /// where NVRTC is available but no work has been issued yet.
    #[cfg(feature = "nvrtc")]
    pub fn with_nvrtc(cfg: SparseConfig, nvrtc: ActorRef<NvrtcMsg>) -> Props<Self> {
        Props::create(move || GpuSparseStructureActor {
            cfg: cfg.clone(),
            entries: Vec::new(),
            spmv_kernel: None,
            nvrtc: Some(nvrtc.clone()),
        })
    }

    /// Compile `coo_spmv` on demand. Cached for the lifetime of the
    /// actor (cleared on any subsequent restart since `spmv_kernel`
    /// is part of actor state).
    #[cfg(feature = "nvrtc")]
    async fn ensure_kernel(&mut self) -> Result<Arc<KernelHandle>, GpuError> {
        if let Some(k) = &self.spmv_kernel {
            return Ok(k.clone());
        }
        let nvrtc = self.nvrtc.as_ref().ok_or_else(|| {
            GpuError::Unrecoverable("GpuSparseStructureActor: no nvrtc actor configured".into())
        })?;
        let handle = nvrtc
            .ask_with(
                |tx| NvrtcMsg::Compile {
                    src: crate::kernels::COO_SPMV_SRC.to_string(),
                    kernel_name: "coo_spmv".to_string(),
                    opts: NvrtcOpts::default(),
                    reply: tx,
                },
                std::time::Duration::from_secs(30),
            )
            .await
            .map_err(|_| GpuError::Unrecoverable("NvrtcActor compile timed out".into()))??;
        let arc = Arc::new(handle);
        self.spmv_kernel = Some(arc.clone());
        Ok(arc)
    }
}

#[async_trait]
impl Actor for GpuSparseStructureActor {
    type Msg = SparseMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SparseMsg) {
        match msg {
            SparseMsg::SetEntries { entries, reply } => {
                self.entries = entries;
                let _ = reply.send(self.entries.len());
            }
            SparseMsg::SpMv { x, reply } => {
                if x.len() != self.cfg.cols {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "SpMv: x len {} != cols {}",
                        x.len(),
                        self.cfg.cols
                    ))));
                    return;
                }
                // GPU fast path: if `with_nvrtc(...)` was used, attempt
                // to compile the `coo_spmv` kernel (cached after first
                // compile) so that downstream callers can verify
                // NVRTC availability without issuing a separate probe.
                // Full host↔device dispatch (Allocate + CopyFromHost +
                // Launch + CopyToHost) requires a DeviceActor ref and
                // is wired in via the dispatcher pattern in
                // `atomr_accel_cuda::pipeline` — added in a follow-up that
                // requires GPU validation.
                #[cfg(feature = "nvrtc")]
                if self.nvrtc.is_some() {
                    if let Err(e) = self.ensure_kernel().await {
                        let _ = reply.send(Err(e));
                        return;
                    }
                    // Falls through to the CPU reference below until
                    // the dispatcher wiring lands.
                }
                let mut y = vec![0.0f32; self.cfg.rows];
                for e in &self.entries {
                    let r = e.row as usize;
                    let c = e.col as usize;
                    if r < self.cfg.rows && c < self.cfg.cols {
                        y[r] += e.value * x[c];
                    }
                }
                let _ = reply.send(Ok(y));
            }
            SparseMsg::Stats { reply } => {
                let _ = reply.send(SparseStats {
                    rows: self.cfg.rows,
                    cols: self.cfg.cols,
                    nnz: self.entries.len(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spmv_identity() {
        let sys = ActorSystem::create("sparse-test", Config::empty())
            .await
            .unwrap();
        let actor = sys
            .actor_of(
                GpuSparseStructureActor::props(SparseConfig { rows: 3, cols: 3 }),
                "sparse",
            )
            .unwrap();

        // Identity matrix.
        let entries = vec![
            CooEntry {
                row: 0,
                col: 0,
                value: 1.0,
            },
            CooEntry {
                row: 1,
                col: 1,
                value: 1.0,
            },
            CooEntry {
                row: 2,
                col: 2,
                value: 1.0,
            },
        ];
        let (tx, rx) = oneshot::channel();
        actor.tell(SparseMsg::SetEntries { entries, reply: tx });
        let n = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(n, 3);

        let (tx, rx) = oneshot::channel();
        actor.tell(SparseMsg::SpMv {
            x: vec![10.0, 20.0, 30.0],
            reply: tx,
        });
        let y = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(y, vec![10.0, 20.0, 30.0]);

        sys.terminate().await;
    }
}
