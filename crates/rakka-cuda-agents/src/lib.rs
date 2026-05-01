//! Agentic / LLM GPU actor blueprints (§7.4).
//!
//! F4 ships:
//! - [`shared_state::SharedGpuStateCoordinator`] — coordinates
//!   write tokens for N agents sharing a `ManagedRef<f32>`
//!   "world state" buffer.
//!
//! F5 adds:
//! - [`embedding_cache::EmbeddingCache`] — LRU cache of
//!   `(input hash) -> GpuRef<f32>`.
//! - [`rag::RagPipeline`] skeleton.
//! - [`vector_index::GpuVectorIndex`] (CPU stub now; GPU later).

pub mod embedding_cache;
pub mod langgraph_nodes;
pub mod rag;
pub mod shared_state;
pub mod vector_index;
