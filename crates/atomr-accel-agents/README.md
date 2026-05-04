# atomr-accel-agents

Agentic / LLM GPU actor blueprints on
[atomr-accel-cuda](../atomr-accel-cuda): RAG, embedding cache, vector index,
shared-state coordination, and a LangGraph-style DAG executor.

## Add to your project

```toml
[dependencies]
atomr-accel        = "0.0"
atomr-accel-agents = "0.0"
```

```rust
use atomr_accel_agents::prelude::*;
```

Depends only on `atomr-accel-cuda` (for `ManagedRef`, errors). No
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
