//! Agentic / LLM GPU actor blueprints on rakka-accel-cuda.
//!
//! ```ignore
//! use rakka_accel_agents::prelude::*;
//! ```
//!
//! - [`shared_state::SharedGpuStateCoordinator`] — write-token
//!   coordination over a `ManagedRef<f32>` world buffer.
//! - [`embedding_cache::EmbeddingCache`] — LRU cache of
//!   `(input hash) -> Vec<f32>`.
//! - [`rag::RagPipeline`] — retrieval + generate harness.
//! - [`vector_index::CpuVectorIndex`] — top-k cosine similarity
//!   over a flat host index.
//! - [`langgraph_nodes::LangGraphGpuActor`] — DAG executor with
//!   cycle detection.

pub mod embedding_cache;
pub mod langgraph_nodes;
pub mod rag;
pub mod shared_state;
pub mod vector_index;

pub mod prelude {
    //! Canonical re-exports. `use rakka_accel_agents::prelude::*;`.
    pub use crate::embedding_cache::{
        CacheStats, EmbeddingCache, EmbeddingCacheConfig, EmbeddingCacheMsg,
    };
    pub use crate::langgraph_nodes::{LangGraphGpuActor, NodeEntry, NodeGraph, NodeGraphMsg, NodeId};
    pub use crate::rag::{RagAnswer, RagConfig, RagMsg, RagPipeline, RagQuery};
    pub use crate::shared_state::{
        SharedGpuStateCoordinator, SharedStateMsg, SharedStateStats, WriteToken,
    };
    pub use crate::vector_index::{CpuVectorIndex, VectorEntry, VectorIndexMsg};
}
