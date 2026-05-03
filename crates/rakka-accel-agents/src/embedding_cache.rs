//! `EmbeddingCache` — LRU cache of `(input hash) -> Vec<f32>`.
//!
//! F4 ships a CPU-resident LRU keyed on a 64-bit hash of the input
//! bytes. F5 swaps the value type to `GpuRef<f32>` once the agents
//! crate has a stable model-actor surface to compute embeddings
//! against.

use std::collections::HashMap;
use std::collections::VecDeque;

use rakka_core::actor::{Context, Props};
use rakka_macros::Actor;
use tokio::sync::oneshot;

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub capacity: usize,
}

pub struct EmbeddingCacheConfig {
    pub capacity_entries: usize,
    pub embedding_dim: usize,
}

pub enum EmbeddingCacheMsg {
    /// Try the cache. On miss, returns `None` and the caller is
    /// responsible for computing + storing the embedding via
    /// `Insert`. F4 keeps cache and compute decoupled.
    Get {
        key: Vec<u8>,
        reply: oneshot::Sender<Option<Vec<f32>>>,
    },
    Insert {
        key: Vec<u8>,
        value: Vec<f32>,
        reply: oneshot::Sender<()>,
    },
    Invalidate {
        key: Vec<u8>,
        reply: oneshot::Sender<bool>,
    },
    Stats {
        reply: oneshot::Sender<CacheStats>,
    },
}

#[derive(Actor)]
#[msg(EmbeddingCacheMsg)]
pub struct EmbeddingCache {
    config: EmbeddingCacheConfig,
    cache: HashMap<u64, Vec<f32>>,
    /// LRU order: front is least-recent.
    order: VecDeque<u64>,
    stats: CacheStats,
}

fn hash_key(k: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    k.hash(&mut h);
    h.finish()
}

impl EmbeddingCache {
    pub fn props(config: EmbeddingCacheConfig) -> Props<Self> {
        Props::create(move || EmbeddingCache {
            config: EmbeddingCacheConfig {
                capacity_entries: config.capacity_entries,
                embedding_dim: config.embedding_dim,
            },
            cache: HashMap::with_capacity(config.capacity_entries),
            order: VecDeque::with_capacity(config.capacity_entries),
            stats: CacheStats { capacity: config.capacity_entries, ..Default::default() },
        })
    }

    fn touch(&mut self, k: u64) {
        if let Some(pos) = self.order.iter().position(|x| *x == k) {
            self.order.remove(pos);
        }
        self.order.push_back(k);
    }
}

impl EmbeddingCache {
    /// `#[derive(Actor)]` delegates to this method via the
    /// rakka-macros-generated `impl Actor`.
    async fn handle_msg(&mut self, _ctx: &mut Context<Self>, msg: EmbeddingCacheMsg) {
        match msg {
            EmbeddingCacheMsg::Get { key, reply } => {
                let h = hash_key(&key);
                if let Some(v) = self.cache.get(&h).cloned() {
                    self.stats.hits += 1;
                    self.touch(h);
                    let _ = reply.send(Some(v));
                } else {
                    self.stats.misses += 1;
                    let _ = reply.send(None);
                }
            }
            EmbeddingCacheMsg::Insert { key, value, reply } => {
                let h = hash_key(&key);
                if self.cache.len() >= self.config.capacity_entries && !self.cache.contains_key(&h) {
                    if let Some(victim) = self.order.pop_front() {
                        self.cache.remove(&victim);
                    }
                }
                self.cache.insert(h, value);
                self.touch(h);
                self.stats.size = self.cache.len();
                let _ = reply.send(());
            }
            EmbeddingCacheMsg::Invalidate { key, reply } => {
                let h = hash_key(&key);
                let removed = self.cache.remove(&h).is_some();
                if removed {
                    if let Some(pos) = self.order.iter().position(|x| *x == h) {
                        self.order.remove(pos);
                    }
                    self.stats.size = self.cache.len();
                }
                let _ = reply.send(removed);
            }
            EmbeddingCacheMsg::Stats { reply } => {
                let _ = reply.send(self.stats);
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
    async fn cache_hit_miss() {
        let sys = ActorSystem::create("embed-test", Config::empty()).await.unwrap();
        let actor = sys
            .actor_of(
                EmbeddingCache::props(EmbeddingCacheConfig {
                    capacity_entries: 4,
                    embedding_dim: 8,
                }),
                "cache",
            )
            .unwrap();

        let key = b"hello".to_vec();
        // Miss
        let (tx, rx) = oneshot::channel();
        actor.tell(EmbeddingCacheMsg::Get { key: key.clone(), reply: tx });
        let v = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert!(v.is_none());

        // Insert
        let (tx, rx) = oneshot::channel();
        actor.tell(EmbeddingCacheMsg::Insert {
            key: key.clone(),
            value: vec![1.0; 8],
            reply: tx,
        });
        tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();

        // Hit
        let (tx, rx) = oneshot::channel();
        actor.tell(EmbeddingCacheMsg::Get { key: key.clone(), reply: tx });
        let v = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert_eq!(v, Some(vec![1.0; 8]));

        sys.terminate().await;
    }
}
