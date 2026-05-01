//! `GpuSparseStructureActor` — coordinate-list (COO) sparse matrix
//! with simple SpMV.
//!
//! F8 ships a CPU reference. F9+ swaps the SpMV loop for cuSPARSE
//! once cudarc has a safe wrapper.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

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
}

impl GpuSparseStructureActor {
    pub fn props(cfg: SparseConfig) -> Props<Self> {
        Props::create(move || GpuSparseStructureActor {
            cfg: cfg.clone(),
            entries: Vec::new(),
        })
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
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spmv_identity() {
        let sys = ActorSystem::create("sparse-test", Config::empty()).await.unwrap();
        let actor = sys
            .actor_of(
                GpuSparseStructureActor::props(SparseConfig { rows: 3, cols: 3 }),
                "sparse",
            )
            .unwrap();

        // Identity matrix.
        let entries = vec![
            CooEntry { row: 0, col: 0, value: 1.0 },
            CooEntry { row: 1, col: 1, value: 1.0 },
            CooEntry { row: 2, col: 2, value: 1.0 },
        ];
        let (tx, rx) = oneshot::channel();
        actor.tell(SparseMsg::SetEntries { entries, reply: tx });
        let n = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert_eq!(n, 3);

        let (tx, rx) = oneshot::channel();
        actor.tell(SparseMsg::SpMv { x: vec![10.0, 20.0, 30.0], reply: tx });
        let y = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(y, vec![10.0, 20.0, 30.0]);

        sys.terminate().await;
    }
}
