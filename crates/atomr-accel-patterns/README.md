# atomr-accel-patterns

Universal GPU actor blueprints for [atomr-accel-cuda](../atomr-accel-cuda):
batching, cascade, replica pool, fair-share scheduler, hot-swap,
speculative decode, mixture-of-experts. Each pattern is a typed
atomr actor that you parameterize over your own request / response
types.

## Add to your project

```toml
[dependencies]
atomr-accel          = "0.0"
atomr-accel-patterns = "0.0"
```

```rust
use atomr_accel_patterns::prelude::*;
```

You only pull in what you `use`; the patterns crate itself depends
on `atomr-accel-cuda` (for `GpuRef`, errors) plus the standard atomr
foundation. No other sub-crate is involved.

## What's inside

| Pattern                      | Type                              |
|------------------------------|-----------------------------------|
| Dynamic batching             | `DynamicBatchingServer<Req, Resp>` |
| Inference cascade            | `InferenceCascade<Req, Resp>`      |
| Replica pool                 | `ModelReplicaPool<Msg>`            |
| Fair-share (WFQ) scheduler   | `FairShareScheduler<Req, Resp>`    |
| Hot-swap                     | `ModelHotSwapServer<P>`            |
| Speculative decode           | `SpeculativeDecoder`               |
| Mixture of experts           | `MoeRouter<P>`                     |
| CPU mock backend             | `GpuMockActor`                     |

Run the no-GPU demos:

```bash
cargo run -p atomr-accel-patterns --example batching_no_gpu
cargo run -p atomr-accel-patterns --example cascade_no_gpu
cargo run -p atomr-accel-patterns --example fair_share_no_gpu
cargo run -p atomr-accel-patterns --example moe_no_gpu
cargo run -p atomr-accel-patterns --example speculative_no_gpu
```

License: Apache-2.0.
