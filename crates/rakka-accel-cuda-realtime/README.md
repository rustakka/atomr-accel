# rakka-accel-cuda-realtime

Interactive-rate GPU actor blueprints on
[rakka-accel-cuda](../rakka-accel-cuda): image filters, particle systems,
spatial index, cloth + fluid simulation, sparse SpMV, and a
typed multi-pass analysis pipeline. Bundles CUDA-C kernel sources
under `kernels/` for NVRTC dispatch.

## Add to your project

```toml
[dependencies]
rakka-accel          = "0.0"
rakka-accel-cuda-realtime = "0.0"

# Optional: enable JIT-compiled GPU paths
# rakka-accel-cuda-realtime = { version = "0.0", features = ["nvrtc"] }
```

```rust
use rakka_accel_cuda_realtime::prelude::*;
```

Depends only on `rakka-accel-cuda`. No patterns / train / agents
coupling.

## What's inside

| Actor                       | GPU path             |
|-----------------------------|----------------------|
| `ImageFilterPipeline`       | cuDNN (feature `cudnn`) |
| `ParticleSystemActor`       | NVRTC (`kernels/particle_step.cu`) |
| `ClothSimulationActor`      | NVRTC (`kernels/cloth_springs.cu`) |
| `FluidSimulationActor`      | CPU reference        |
| `SpatialIndexActor`         | CPU reference        |
| `GpuHashMapActor`           | NVRTC (`kernels/hashmap_probe.cu`) |
| `GpuSparseStructureActor`   | NVRTC + cuSPARSE (`kernels/coo_spmv.cu`) |
| `MultiPassAnalysisActor`    | configurable         |
| `ReductionAnalysisActor`    | CPU reference        |
| `VideoEffectsGraph`         | configurable         |

## Features

- `cudnn` — pass-through to `rakka-accel-cuda/cudnn`.
- `nvrtc` — pass-through to `rakka-accel-cuda/nvrtc`; enables
  `with_nvrtc(...)` constructors that JIT-compile the bundled
  CUDA-C sources at startup.

License: Apache-2.0.
