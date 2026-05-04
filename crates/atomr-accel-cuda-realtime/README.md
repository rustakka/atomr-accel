# atomr-accel-cuda-realtime

Interactive-rate GPU actor blueprints on
[atomr-accel-cuda](../atomr-accel-cuda): image filters, particle systems,
spatial index, cloth + fluid simulation, sparse SpMV, and a
typed multi-pass analysis pipeline. Bundles CUDA-C kernel sources
under `kernels/` for NVRTC dispatch.

## Add to your project

```toml
[dependencies]
atomr-accel          = "0.0"
atomr-accel-cuda-realtime = "0.0"

# Optional: enable JIT-compiled GPU paths
# atomr-accel-cuda-realtime = { version = "0.0", features = ["nvrtc"] }
```

```rust
use atomr_accel_cuda_realtime::prelude::*;
```

Depends only on `atomr-accel-cuda`. No patterns / train / agents
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

- `cudnn` — pass-through to `atomr-accel-cuda/cudnn`.
- `nvrtc` — pass-through to `atomr-accel-cuda/nvrtc`; enables
  `with_nvrtc(...)` constructors that JIT-compile the bundled
  CUDA-C sources at startup.

License: Apache-2.0.
