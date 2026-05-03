//! `GpuHashMapActor` — open-addressing hashmap.
//!
//! F3.x ships a CPU-resident reference implementation so dependent
//! actors (`SpatialIndex`, `ParticleSystem`) can build against the
//! contract before the NVRTC kernel lands. F4+ swaps the table for
//! a GPU-resident slab driven by an NVRTC-compiled probe kernel;
//! the message surface stays identical.

use std::collections::HashMap;

use rakka_core::actor::{Context, Props};
use rakka_macros::Actor;
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

#[derive(Debug, Clone)]
pub struct GpuHashMapConfig {
    pub capacity: usize,
    pub key_size_bytes: usize,
    pub value_size_bytes: usize,
}

pub enum GpuHashMapMsg {
    Insert {
        keys: Vec<u8>,
        values: Vec<u8>,
        reply: oneshot::Sender<Result<u32, GpuError>>,
    },
    Lookup {
        keys: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, GpuError>>,
    },
    Clear {
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Stats {
        reply: oneshot::Sender<GpuHashMapStats>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuHashMapStats {
    pub occupancy: usize,
    pub capacity: usize,
}

#[derive(Actor)]
#[msg(GpuHashMapMsg)]
pub struct GpuHashMapActor {
    config: GpuHashMapConfig,
    table: HashMap<Vec<u8>, Vec<u8>>,
}

impl GpuHashMapActor {
    pub fn props(config: GpuHashMapConfig) -> Props<Self> {
        Props::create(move || GpuHashMapActor {
            config: config.clone(),
            table: HashMap::with_capacity(config.capacity),
        })
    }

    fn split_keys(&self, keys: &[u8]) -> Result<Vec<Vec<u8>>, GpuError> {
        if self.config.key_size_bytes == 0 {
            return Err(GpuError::Unrecoverable("GpuHashMap: key_size_bytes == 0".into()));
        }
        if keys.len() % self.config.key_size_bytes != 0 {
            return Err(GpuError::Unrecoverable(format!(
                "GpuHashMap: keys len {} not multiple of key_size {}",
                keys.len(),
                self.config.key_size_bytes
            )));
        }
        Ok(keys
            .chunks(self.config.key_size_bytes)
            .map(|c| c.to_vec())
            .collect())
    }
}

impl GpuHashMapActor {
    async fn handle_msg(&mut self, _ctx: &mut Context<Self>, msg: GpuHashMapMsg) {
        match msg {
            GpuHashMapMsg::Insert { keys, values, reply } => {
                let key_chunks = match self.split_keys(&keys) {
                    Ok(v) => v,
                    Err(e) => { let _ = reply.send(Err(e)); return; }
                };
                let n = key_chunks.len();
                if values.len() != n * self.config.value_size_bytes {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "GpuHashMap::Insert: values len {} != {n} × value_size {}",
                        values.len(),
                        self.config.value_size_bytes
                    ))));
                    return;
                }
                let mut inserted = 0u32;
                for (i, k) in key_chunks.into_iter().enumerate() {
                    if self.table.len() >= self.config.capacity && !self.table.contains_key(&k) {
                        break; // capacity reached
                    }
                    let lo = i * self.config.value_size_bytes;
                    let hi = lo + self.config.value_size_bytes;
                    self.table.insert(k, values[lo..hi].to_vec());
                    inserted += 1;
                }
                let _ = reply.send(Ok(inserted));
            }
            GpuHashMapMsg::Lookup { keys, reply } => {
                let key_chunks = match self.split_keys(&keys) {
                    Ok(v) => v,
                    Err(e) => { let _ = reply.send(Err(e)); return; }
                };
                let mut out = Vec::with_capacity(key_chunks.len() * self.config.value_size_bytes);
                for k in key_chunks {
                    match self.table.get(&k) {
                        Some(v) => out.extend_from_slice(v),
                        None => out.extend(std::iter::repeat(0u8).take(self.config.value_size_bytes)),
                    }
                }
                let _ = reply.send(Ok(out));
            }
            GpuHashMapMsg::Clear { reply } => {
                self.table.clear();
                let _ = reply.send(Ok(()));
            }
            GpuHashMapMsg::Stats { reply } => {
                let _ = reply.send(GpuHashMapStats {
                    occupancy: self.table.len(),
                    capacity: self.config.capacity,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn insert_lookup_roundtrip() {
        let cfg = GpuHashMapConfig {
            capacity: 16,
            key_size_bytes: 4,
            value_size_bytes: 4,
        };
        let sys = ActorSystem::create("hm-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(GpuHashMapActor::props(cfg), "hm").unwrap();

        // Two keys: [1,0,0,0] -> [42,0,0,0], [2,0,0,0] -> [99,0,0,0].
        let keys = vec![1u8, 0, 0, 0, 2, 0, 0, 0];
        let values = vec![42u8, 0, 0, 0, 99, 0, 0, 0];
        let (tx, rx) = oneshot::channel();
        actor.tell(GpuHashMapMsg::Insert { keys: keys.clone(), values, reply: tx });
        let inserted = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(inserted, 2);

        let (tx, rx) = oneshot::channel();
        actor.tell(GpuHashMapMsg::Lookup { keys, reply: tx });
        let out = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(out[0], 42);
        assert_eq!(out[4], 99);

        // Lookup missing key — returns zeros.
        let (tx, rx) = oneshot::channel();
        actor.tell(GpuHashMapMsg::Lookup { keys: vec![7, 0, 0, 0], reply: tx });
        let out = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(out, vec![0, 0, 0, 0]);

        sys.terminate().await;
    }
}
