# vendored flash-attention csrc

This directory will, once populated, contain the minimum FA2 + FA3
csrc subset described in `NOTICE`. The Phase 7 deliverable ships the
license + notice + dispatch-table machinery; the actual `.cu` source
ships in a follow-up vendor commit gated behind `cuda-runtime-tests`.

See `../../include/atomr_flash_adapter.h` for the in-tree headers
that map the FA csrc onto atomr-accel's `GpuRef` / `AccelDtype`
abstractions.
