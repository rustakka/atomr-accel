//! `SingleStreamAllocator` (§5.7) — every `KernelActor` shares one
//! stream. Useful for resource-constrained edge devices or
//! deterministic-replay testing.

use std::sync::Arc;

use cudarc::driver::CudaStream;

use super::{ActorHints, StreamAllocator};

pub struct SingleStreamAllocator {
    stream: Arc<CudaStream>,
}

impl SingleStreamAllocator {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl StreamAllocator for SingleStreamAllocator {
    fn acquire(&self, _hints: ActorHints) -> Arc<CudaStream> {
        self.stream.clone()
    }
}
