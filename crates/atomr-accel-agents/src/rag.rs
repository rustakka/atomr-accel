//! `RagPipeline` — query → embedding → vector search → context
//! assembly → LLM call.
//!
//! Wires:
//! - [`crate::embedding_cache::EmbeddingCache`] (LRU keyed on
//!   query bytes → embedding vector). On miss the configured
//!   `EmbeddingFn` runs and the result is inserted.
//! - [`crate::vector_index::CpuVectorIndex`] (top-k cosine
//!   similarity over the indexed corpus).
//! - A user-supplied `LlmFn` that takes `(query, retrieved_contexts)`
//!   and returns the final answer.
//!
//! All three steps run inside a single tokio task spawned per
//! query, so the actor's mailbox stays free.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;

use crate::embedding_cache::EmbeddingCacheMsg;
use crate::vector_index::VectorIndexMsg;

#[derive(Debug, Clone)]
pub struct RagQuery {
    pub text: String,
    pub top_k: usize,
}

#[derive(Debug, Clone)]
pub struct RagAnswer {
    pub answer: String,
    pub sources: Vec<u64>,
    pub embedding_was_cached: bool,
}

pub trait EmbeddingFn: Send + Sync + 'static {
    fn embed(&self, text: &str) -> Result<Vec<f32>, GpuError>;
}

impl<F> EmbeddingFn for F
where
    F: Fn(&str) -> Result<Vec<f32>, GpuError> + Send + Sync + 'static,
{
    fn embed(&self, text: &str) -> Result<Vec<f32>, GpuError> {
        self(text)
    }
}

pub trait LlmFn: Send + Sync + 'static {
    fn answer(&self, query: &str, contexts: &[u64]) -> Result<String, GpuError>;
}

impl<F> LlmFn for F
where
    F: Fn(&str, &[u64]) -> Result<String, GpuError> + Send + Sync + 'static,
{
    fn answer(&self, query: &str, contexts: &[u64]) -> Result<String, GpuError> {
        self(query, contexts)
    }
}

pub struct RagConfig {
    pub embedding: Arc<dyn EmbeddingFn>,
    pub embedding_cache: ActorRef<EmbeddingCacheMsg>,
    pub vector_index: ActorRef<VectorIndexMsg>,
    pub llm: Arc<dyn LlmFn>,
    pub timeout: Duration,
}

impl Clone for RagConfig {
    fn clone(&self) -> Self {
        Self {
            embedding: self.embedding.clone(),
            embedding_cache: self.embedding_cache.clone(),
            vector_index: self.vector_index.clone(),
            llm: self.llm.clone(),
            timeout: self.timeout,
        }
    }
}

pub enum RagMsg {
    Query {
        q: RagQuery,
        reply: oneshot::Sender<Result<RagAnswer, GpuError>>,
    },
}

pub struct RagPipeline {
    cfg: RagConfig,
}

impl RagPipeline {
    pub fn props(cfg: RagConfig) -> Props<Self> {
        Props::create(move || RagPipeline { cfg: cfg.clone() })
    }
}

#[async_trait]
impl Actor for RagPipeline {
    type Msg = RagMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: RagMsg) {
        match msg {
            RagMsg::Query { q, reply } => {
                let cfg = self.cfg.clone();
                tokio::spawn(async move {
                    let result = run_rag(cfg, q).await;
                    let _ = reply.send(result);
                });
            }
        }
    }
}

async fn run_rag(cfg: RagConfig, q: RagQuery) -> Result<RagAnswer, GpuError> {
    // 1. Embedding (with cache).
    let key = q.text.as_bytes().to_vec();
    let cached: Option<Vec<f32>> = cfg
        .embedding_cache
        .ask_with(
            move |tx| EmbeddingCacheMsg::Get { key, reply: tx },
            cfg.timeout,
        )
        .await
        .map_err(|e| GpuError::Unrecoverable(format!("rag: embed cache get: {e}")))?;
    let (embedding, was_cached) = match cached {
        Some(v) => (v, true),
        None => {
            let v = cfg.embedding.embed(&q.text)?;
            // Best-effort insert — ignore reply.
            let key = q.text.as_bytes().to_vec();
            let v_clone = v.clone();
            let _ = cfg
                .embedding_cache
                .ask_with(
                    move |tx| EmbeddingCacheMsg::Insert {
                        key,
                        value: v_clone,
                        reply: tx,
                    },
                    cfg.timeout,
                )
                .await;
            (v, false)
        }
    };

    // 2. Vector search.
    let top_k = q.top_k.max(1);
    let scored: Vec<(u64, f32)> = cfg
        .vector_index
        .ask_with(
            move |tx| VectorIndexMsg::Search {
                query: embedding,
                top_k,
                reply: tx,
            },
            cfg.timeout,
        )
        .await
        .map_err(|e| GpuError::Unrecoverable(format!("rag: vec search: {e}")))??;
    let sources: Vec<u64> = scored.into_iter().map(|(id, _)| id).collect();

    // 3. LLM step.
    let answer = cfg.llm.answer(&q.text, &sources)?;
    Ok(RagAnswer {
        answer,
        sources,
        embedding_was_cached: was_cached,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding_cache::{EmbeddingCache, EmbeddingCacheConfig};
    use crate::vector_index::{CpuVectorIndex, VectorEntry, VectorIndexMsg};
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn end_to_end_rag_query() {
        let sys = ActorSystem::create("rag-test", Config::empty())
            .await
            .unwrap();
        let cache = sys
            .actor_of(
                EmbeddingCache::props(EmbeddingCacheConfig {
                    capacity_entries: 8,
                    embedding_dim: 3,
                }),
                "cache",
            )
            .unwrap();
        let index = sys.actor_of(CpuVectorIndex::props(3), "idx").unwrap();

        // Seed the index with three docs.
        for (id, e) in [
            (1u64, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.7, 0.7, 0.0]),
        ] {
            let (tx, rx) = oneshot::channel();
            index.tell(VectorIndexMsg::Insert {
                entry: VectorEntry { id, embedding: e },
                reply: tx,
            });
            tokio::time::timeout(Duration::from_secs(2), rx)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        }

        // Trivial embedding fn: hash-by-length into a 3-vector.
        let embed_fn: Arc<dyn EmbeddingFn> = Arc::new(|text: &str| {
            let v = match text {
                "alpha" => vec![1.0, 0.0, 0.0],
                "beta" => vec![0.0, 1.0, 0.0],
                _ => vec![0.5, 0.5, 0.5],
            };
            Ok(v)
        });
        let llm: Arc<dyn LlmFn> =
            Arc::new(|q: &str, ctx: &[u64]| Ok(format!("answered '{q}' from {ctx:?}")));

        let rag = sys
            .actor_of(
                RagPipeline::props(RagConfig {
                    embedding: embed_fn,
                    embedding_cache: cache,
                    vector_index: index,
                    llm,
                    timeout: Duration::from_secs(2),
                }),
                "rag",
            )
            .unwrap();

        let (tx, rx) = oneshot::channel();
        rag.tell(RagMsg::Query {
            q: RagQuery {
                text: "alpha".into(),
                top_k: 2,
            },
            reply: tx,
        });
        let r = tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(r.answer.contains("alpha"));
        // First source should be id=1 (best cosine match for [1,0,0]).
        assert_eq!(r.sources[0], 1);
        assert!(!r.embedding_was_cached);

        // Second query for same text → cache hit.
        let (tx, rx) = oneshot::channel();
        rag.tell(RagMsg::Query {
            q: RagQuery {
                text: "alpha".into(),
                top_k: 2,
            },
            reply: tx,
        });
        let r2 = tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(r2.embedding_was_cached);

        sys.terminate().await;
    }
}
