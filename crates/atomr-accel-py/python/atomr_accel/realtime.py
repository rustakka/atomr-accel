"""``atomr_accel.realtime`` — interactive-rate GPU actor blueprints.

Phase 2 ships ``spawn`` constructors plus one representative method
per actor:

- ``ClothSimulationActor`` — ``spawn``, ``step``.
- ``FluidSimulationActor`` — ``spawn``, ``step``.
- ``ParticleSystemActor`` — ``spawn``, ``step``.
- ``SpatialIndexActor`` — ``spawn``, ``query_neighbors``.
- ``GpuHashMapActor`` — ``spawn``, ``insert``.
- ``ImageFilterPipeline`` — ``spawn``, ``process``.

Snapshot / Reset / Lookup / UpdateConfig / ProcessGpu follow in the
Phase 2.5 tracking issue.
"""

from ._native import (
    ClothSimulationActor,
    FluidSimulationActor,
    ParticleSystemActor,
    SpatialIndexActor,
    GpuHashMapActor,
    ImageFilterPipeline,
)

__all__ = [
    "ClothSimulationActor",
    "FluidSimulationActor",
    "ParticleSystemActor",
    "SpatialIndexActor",
    "GpuHashMapActor",
    "ImageFilterPipeline",
]
