# atomr-accel-train

Distributed training blueprints on [atomr-accel-cuda](../atomr-accel-cuda):
data-parallel, pipeline-parallel, tensor-parallel, async parameter
server, plus typed `OptimizerKind` / `LossKind` enums.

## Add to your project

```toml
[dependencies]
atomr-accel       = "0.0"
atomr-accel-train = "0.0"
```

```rust
use atomr_accel_train::prelude::*;
```

This crate depends on `atomr-accel-cuda` (foundation) plus
`atomr-accel-patterns` (for replica routing). Nothing else.

## What's inside

| Blueprint                  | Type                          |
|----------------------------|-------------------------------|
| Data parallel              | `DataParallelTrainer<P>`      |
| Pipeline parallel          | `PipelineParallelTrainer<P>`  |
| Tensor parallel            | `TensorParallelTrainer<P>`    |
| Async parameter server     | `AsyncParameterServer`        |
| Optimizer enum             | `OptimizerKind` (SGD, AdamW)  |
| Loss enum                  | `LossKind` (MSE, CrossEntropy) |

License: Apache-2.0.
