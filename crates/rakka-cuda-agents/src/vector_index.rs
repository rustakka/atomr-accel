//! `GpuVectorIndex` — similarity-search index abstraction.
//!
//! F4 ships a CPU fallback implementation (linear scan with cosine
//! similarity) so RAG/agent code can build against the trait. F5
//! adds a GPU-accelerated implementation once a Rust FAISS-equivalent
//! is available or NVRTC custom kernels for FAISS-like indices land.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

pub struct VectorEntry {
    pub id: u64,
    pub embedding: Vec<f32>,
}

pub enum VectorIndexMsg {
    Insert {
        entry: VectorEntry,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Search {
        query: Vec<f32>,
        top_k: usize,
        reply: oneshot::Sender<Result<Vec<(u64, f32)>, GpuError>>,
    },
    Stats {
        reply: oneshot::Sender<usize>,
    },
}

/// CPU-resident linear-scan vector index.
pub struct CpuVectorIndex {
    dim: usize,
    entries: Vec<VectorEntry>,
}

impl CpuVectorIndex {
    pub fn props(dim: usize) -> Props<Self> {
        Props::create(move || CpuVectorIndex { dim, entries: Vec::new() })
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

#[async_trait]
impl Actor for CpuVectorIndex {
    type Msg = VectorIndexMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: VectorIndexMsg) {
        match msg {
            VectorIndexMsg::Insert { entry, reply } => {
                if entry.embedding.len() != self.dim {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "vector dim {} != index dim {}",
                        entry.embedding.len(),
                        self.dim
                    ))));
                    return;
                }
                self.entries.push(entry);
                let _ = reply.send(Ok(()));
            }
            VectorIndexMsg::Search { query, top_k, reply } => {
                if query.len() != self.dim {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "query dim {} != index dim {}",
                        query.len(),
                        self.dim
                    ))));
                    return;
                }
                let mut scored: Vec<(u64, f32)> = self
                    .entries
                    .iter()
                    .map(|e| (e.id, Self::cosine(&e.embedding, &query)))
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(top_k);
                let _ = reply.send(Ok(scored));
            }
            VectorIndexMsg::Stats { reply } => {
                let _ = reply.send(self.entries.len());
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
    async fn cpu_index_topk() {
        let sys = ActorSystem::create("vec-test", Config::empty()).await.unwrap();
        let idx = sys.actor_of(CpuVectorIndex::props(3), "idx").unwrap();

        for (id, e) in [
            (1u64, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.7, 0.7, 0.0]),
        ]
        .into_iter()
        {
            let (tx, rx) = oneshot::channel();
            idx.tell(VectorIndexMsg::Insert {
                entry: VectorEntry { id, embedding: e },
                reply: tx,
            });
            tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        }

        let (tx, rx) = oneshot::channel();
        idx.tell(VectorIndexMsg::Search {
            query: vec![1.0, 0.0, 0.0],
            top_k: 2,
            reply: tx,
        });
        let res = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        // ID 1 is best match (cosine = 1.0).
        assert_eq!(res[0].0, 1);
        assert!((res[0].1 - 1.0).abs() < 1e-5);

        sys.terminate().await;
    }
}
