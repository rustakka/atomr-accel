"""``atomr_accel.agents`` — agentic / LLM GPU actor blueprints.

Phase 2 ships:

- ``SharedGpuStateCoordinator`` — write-token coordinator. Methods:
  ``spawn``, ``acquire_write``, ``release_write``.
- ``EmbeddingCache`` — LRU cache. Methods: ``spawn``, ``get``, ``insert``.
- ``CpuVectorIndex`` — top-k cosine search. Methods: ``spawn``, ``insert``,
  ``search``.
- ``RagPipeline``, ``LangGraphGpuActor`` — callback / generic-state, so
  shipped as structural anchors (Phase 2.5 widens the surface).
"""

from ._native import (
    SharedGpuStateCoordinator,
    EmbeddingCache,
    CpuVectorIndex,
    RagPipeline,
    LangGraphGpuActor,
)

__all__ = [
    "SharedGpuStateCoordinator",
    "EmbeddingCache",
    "CpuVectorIndex",
    "RagPipeline",
    "LangGraphGpuActor",
]
