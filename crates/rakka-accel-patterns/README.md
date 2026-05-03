# rakka-accel-patterns

Universal GPU actor blueprints for [rakka-accel-cuda](../rakka-accel-cuda):
batching, cascade, replica pool, fair-share scheduler, hot-swap,
speculative decode, mixture-of-experts. Each pattern is a typed
rakka actor that you parameterize over your own request / response
types.

## Add to your project

```toml
[dependencies]
rakka-accel          = "0.0"
rakka-accel-patterns = "0.0"
```

```rust
use rakka_accel_patterns::prelude::*;
```

You only pull in what you `use`; the patterns crate itself depends
on `rakka-accel-cuda` (for `GpuRef`, errors) plus the standard rakka
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
cargo run -p rakka-accel-patterns --example batching_no_gpu
cargo run -p rakka-accel-patterns --example cascade_no_gpu
cargo run -p rakka-accel-patterns --example fair_share_no_gpu
cargo run -p rakka-accel-patterns --example moe_no_gpu
cargo run -p rakka-accel-patterns --example speculative_no_gpu
```

License: Apache-2.0.
