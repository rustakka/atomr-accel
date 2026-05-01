//! `SpatialIndexActor` — uniform-grid spatial hash for fast 3D
//! neighbor queries on point clouds / particle systems.
//!
//! Each insert hashes the point's cell coordinate into a host-side
//! HashMap<CellKey, Vec<u64 ids>>. Queries return all ids in the
//! 3×3×3 cell region around the query point. F6 ships the CPU
//! reference; F7+ swaps the table for `GpuHashMapActor`-backed
//! storage with the same message surface.

use std::collections::HashMap;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Point3 {
    pub id: u64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone)]
pub struct SpatialIndexConfig {
    pub cell_size: f32,
}

pub enum SpatialMsg {
    /// Replace the entire point set.
    Rebuild {
        points: Vec<Point3>,
        reply: oneshot::Sender<Result<usize, GpuError>>,
    },
    /// Query for points in the 3×3×3 cell neighborhood of `(x,y,z)`.
    QueryNeighbors {
        x: f32,
        y: f32,
        z: f32,
        reply: oneshot::Sender<Vec<u64>>,
    },
    Stats {
        reply: oneshot::Sender<SpatialStats>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialStats {
    pub total_points: usize,
    pub occupied_cells: usize,
}

pub struct SpatialIndexActor {
    cfg: SpatialIndexConfig,
    cells: HashMap<CellKey, Vec<u64>>,
    total: usize,
}

impl SpatialIndexActor {
    pub fn props(cfg: SpatialIndexConfig) -> Props<Self> {
        Props::create(move || SpatialIndexActor {
            cfg: cfg.clone(),
            cells: HashMap::new(),
            total: 0,
        })
    }

    fn cell_for(&self, x: f32, y: f32, z: f32) -> CellKey {
        let cs = self.cfg.cell_size.max(1e-6);
        CellKey {
            x: (x / cs).floor() as i32,
            y: (y / cs).floor() as i32,
            z: (z / cs).floor() as i32,
        }
    }
}

#[async_trait]
impl Actor for SpatialIndexActor {
    type Msg = SpatialMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SpatialMsg) {
        match msg {
            SpatialMsg::Rebuild { points, reply } => {
                self.cells.clear();
                self.total = 0;
                for p in points {
                    let key = self.cell_for(p.x, p.y, p.z);
                    self.cells.entry(key).or_default().push(p.id);
                    self.total += 1;
                }
                let _ = reply.send(Ok(self.total));
            }
            SpatialMsg::QueryNeighbors { x, y, z, reply } => {
                let center = self.cell_for(x, y, z);
                let mut out = Vec::new();
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let k = CellKey {
                                x: center.x + dx,
                                y: center.y + dy,
                                z: center.z + dz,
                            };
                            if let Some(v) = self.cells.get(&k) {
                                out.extend_from_slice(v);
                            }
                        }
                    }
                }
                let _ = reply.send(out);
            }
            SpatialMsg::Stats { reply } => {
                let _ = reply.send(SpatialStats {
                    total_points: self.total,
                    occupied_cells: self.cells.len(),
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
    async fn neighbors_within_cell_neighborhood() {
        let sys = ActorSystem::create("spatial-test", Config::empty()).await.unwrap();
        let actor = sys
            .actor_of(
                SpatialIndexActor::props(SpatialIndexConfig { cell_size: 1.0 }),
                "idx",
            )
            .unwrap();

        let pts = vec![
            Point3 { id: 1, x: 0.5, y: 0.5, z: 0.5 },
            Point3 { id: 2, x: 1.2, y: 0.4, z: 0.4 }, // adjacent cell
            Point3 { id: 3, x: 5.0, y: 5.0, z: 5.0 }, // far away
        ];
        let (tx, rx) = oneshot::channel();
        actor.tell(SpatialMsg::Rebuild { points: pts, reply: tx });
        tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(SpatialMsg::QueryNeighbors { x: 0.5, y: 0.5, z: 0.5, reply: tx });
        let n = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert!(n.contains(&1));
        assert!(n.contains(&2));
        assert!(!n.contains(&3));

        sys.terminate().await;
    }
}
