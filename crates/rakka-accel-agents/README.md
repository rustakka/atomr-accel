# rakka-accel-agents

Agentic / LLM GPU actor blueprints on
[rakka-accel-cuda](../rakka-accel-cuda): RAG, embedding cache, vector index,
shared-state coordination, and a LangGraph-style DAG executor.

## Add to your project

```toml
[dependencies]
rakka-accel        = "0.0"
rakka-accel-agents = "0.0"
```

```rust
use rakka_accel_agents::prelude::*;
```

Depends only on `rakka-accel-cuda` (for `ManagedRef`, errors). No
patterns / train / realtime coupling.

## What's inside

| Blueprint                       | Type                            |
|---------------------------------|---------------------------------|
| Retrieval-augmented generation  | `RagPipeline`                   |
| LRU embedding cache             | `EmbeddingCache`                |
| CPU vector index (top-k cosine) | `CpuVectorIndex`                |
| Shared-state write tokens       | `SharedGpuStateCoordinator`     |
| LangGraph DAG executor          | `LangGraphGpuActor<S>`          |

License: Apache-2.0.
