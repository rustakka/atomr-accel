//! `ClothSimulationActor` — verlet mass-spring cloth grid.
//!
//! Each `Step { dt }` integrates positions via Verlet (`x_new =
//! 2x - x_prev + a * dt^2`), then enforces structural / shear
//! springs by clamping each spring's length toward its rest
//! length. F8 ships a CPU reference; F9+ replaces the per-spring
//! loop with a fused NVRTC kernel.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

use crate::particle::Vec3;

#[derive(Debug, Clone)]
pub struct ClothConfig {
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub gravity: Vec3,
    /// 0 = pinned (immovable). [w*h] flat row-major.
    pub pinned: Vec<bool>,
    pub stiffness: f32,
    pub iterations: u32,
}

impl Default for ClothConfig {
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            spacing: 0.1,
            gravity: Vec3 { x: 0.0, y: -9.8, z: 0.0 },
            pinned: vec![false; 256],
            stiffness: 0.5,
            iterations: 4,
        }
    }
}

pub enum ClothMsg {
    Reset {
        reply: oneshot::Sender<()>,
    },
    Step {
        dt: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Snapshot {
        reply: oneshot::Sender<Vec<Vec3>>,
    },
}

pub struct ClothSimulationActor {
    cfg: ClothConfig,
    pos: Vec<Vec3>,
    prev: Vec<Vec3>,
}

impl ClothSimulationActor {
    pub fn props(cfg: ClothConfig) -> Props<Self> {
        Props::create(move || {
            let n = cfg.width * cfg.height;
            let mut pos = Vec::with_capacity(n);
            for j in 0..cfg.height {
                for i in 0..cfg.width {
                    pos.push(Vec3 {
                        x: i as f32 * cfg.spacing,
                        y: j as f32 * cfg.spacing,
                        z: 0.0,
                    });
                }
            }
            let prev = pos.clone();
            ClothSimulationActor {
                cfg: cfg.clone(),
                pos,
                prev,
            }
        })
    }

    fn step(&mut self, dt: f32) {
        let g = self.cfg.gravity;
        let dt2 = dt * dt;
        // Verlet integration on positions.
        for i in 0..self.pos.len() {
            if self.cfg.pinned.get(i).copied().unwrap_or(false) {
                continue;
            }
            let x = self.pos[i];
            let xp = self.prev[i];
            let new_x = Vec3 {
                x: 2.0 * x.x - xp.x + g.x * dt2,
                y: 2.0 * x.y - xp.y + g.y * dt2,
                z: 2.0 * x.z - xp.z + g.z * dt2,
            };
            self.prev[i] = x;
            self.pos[i] = new_x;
        }
        // Enforce structural springs across each iteration.
        let w = self.cfg.width;
        let h = self.cfg.height;
        let rest = self.cfg.spacing;
        let stiff = self.cfg.stiffness.clamp(0.0, 1.0);
        for _ in 0..self.cfg.iterations {
            // Horizontal springs.
            for j in 0..h {
                for i in 0..w - 1 {
                    constrain(self, j * w + i, j * w + (i + 1), rest, stiff);
                }
            }
            // Vertical springs.
            for j in 0..h - 1 {
                for i in 0..w {
                    constrain(self, j * w + i, (j + 1) * w + i, rest, stiff);
                }
            }
        }
    }
}

fn constrain(s: &mut ClothSimulationActor, a: usize, b: usize, rest: f32, stiff: f32) {
    let pa = s.pos[a];
    let pb = s.pos[b];
    let dx = pb.x - pa.x;
    let dy = pb.y - pa.y;
    let dz = pb.z - pa.z;
    let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-8);
    let diff = (d - rest) / d * stiff * 0.5;
    let off_x = dx * diff;
    let off_y = dy * diff;
    let off_z = dz * diff;
    let pin_a = s.cfg.pinned.get(a).copied().unwrap_or(false);
    let pin_b = s.cfg.pinned.get(b).copied().unwrap_or(false);
    if !pin_a {
        s.pos[a].x += off_x;
        s.pos[a].y += off_y;
        s.pos[a].z += off_z;
    }
    if !pin_b {
        s.pos[b].x -= off_x;
        s.pos[b].y -= off_y;
        s.pos[b].z -= off_z;
    }
}

#[async_trait]
impl Actor for ClothSimulationActor {
    type Msg = ClothMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ClothMsg) {
        match msg {
            ClothMsg::Reset { reply } => {
                let cfg = self.cfg.clone();
                let n = cfg.width * cfg.height;
                let mut pos = Vec::with_capacity(n);
                for j in 0..cfg.height {
                    for i in 0..cfg.width {
                        pos.push(Vec3 {
                            x: i as f32 * cfg.spacing,
                            y: j as f32 * cfg.spacing,
                            z: 0.0,
                        });
                    }
                }
                self.prev = pos.clone();
                self.pos = pos;
                let _ = reply.send(());
            }
            ClothMsg::Step { dt, reply } => {
                self.step(dt);
                let _ = reply.send(Ok(()));
            }
            ClothMsg::Snapshot { reply } => {
                let _ = reply.send(self.pos.clone());
            }
        }
    }
}
