//! `FluidSimulationActor` — Eulerian 2D smoke-like advection on a
//! coarse grid. Densities advect along a velocity field, the
//! velocity field decays each step. No projection step (this is a
//! reference, not a physically-accurate fluid sim).

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;

#[derive(Debug, Clone)]
pub struct FluidConfig {
    pub width: usize,
    pub height: usize,
    pub viscosity: f32,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            viscosity: 0.0,
        }
    }
}

pub enum FluidMsg {
    AddDensity {
        x: usize,
        y: usize,
        amount: f32,
        reply: oneshot::Sender<()>,
    },
    AddVelocity {
        x: usize,
        y: usize,
        vx: f32,
        vy: f32,
        reply: oneshot::Sender<()>,
    },
    Step {
        dt: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    SnapshotDensity {
        reply: oneshot::Sender<Vec<f32>>,
    },
}

pub struct FluidSimulationActor {
    cfg: FluidConfig,
    density: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
}

impl FluidSimulationActor {
    pub fn props(cfg: FluidConfig) -> Props<Self> {
        Props::create(move || {
            let n = cfg.width * cfg.height;
            FluidSimulationActor {
                cfg: cfg.clone(),
                density: vec![0.0; n],
                vx: vec![0.0; n],
                vy: vec![0.0; n],
            }
        })
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.cfg.width + x
    }

    fn sample(&self, field: &[f32], x: f32, y: f32) -> f32 {
        let w = self.cfg.width as f32;
        let h = self.cfg.height as f32;
        let cx = x.clamp(0.0, w - 1.001);
        let cy = y.clamp(0.0, h - 1.001);
        let x0 = cx.floor() as usize;
        let y0 = cy.floor() as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let fx = cx - x0 as f32;
        let fy = cy - y0 as f32;
        let s00 = field[self.idx(x0, y0)];
        let s10 = field[self.idx(x1, y0)];
        let s01 = field[self.idx(x0, y1)];
        let s11 = field[self.idx(x1, y1)];
        let s0 = s00 * (1.0 - fx) + s10 * fx;
        let s1 = s01 * (1.0 - fx) + s11 * fx;
        s0 * (1.0 - fy) + s1 * fy
    }

    fn step(&mut self, dt: f32) {
        // Semi-Lagrangian advection of density along the velocity
        // field.
        let mut new_density = self.density.clone();
        for y in 0..self.cfg.height {
            for x in 0..self.cfg.width {
                let i = self.idx(x, y);
                let vx = self.vx[i];
                let vy = self.vy[i];
                let sx = x as f32 - vx * dt;
                let sy = y as f32 - vy * dt;
                new_density[i] = self.sample(&self.density, sx, sy);
            }
        }
        self.density = new_density;

        // Velocity decay.
        let decay = (1.0 - self.cfg.viscosity * dt).max(0.0);
        for v in &mut self.vx {
            *v *= decay;
        }
        for v in &mut self.vy {
            *v *= decay;
        }
    }
}

#[async_trait]
impl Actor for FluidSimulationActor {
    type Msg = FluidMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: FluidMsg) {
        match msg {
            FluidMsg::AddDensity {
                x,
                y,
                amount,
                reply,
            } => {
                if x < self.cfg.width && y < self.cfg.height {
                    let i = self.idx(x, y);
                    self.density[i] += amount;
                }
                let _ = reply.send(());
            }
            FluidMsg::AddVelocity {
                x,
                y,
                vx,
                vy,
                reply,
            } => {
                if x < self.cfg.width && y < self.cfg.height {
                    let i = self.idx(x, y);
                    self.vx[i] += vx;
                    self.vy[i] += vy;
                }
                let _ = reply.send(());
            }
            FluidMsg::Step { dt, reply } => {
                self.step(dt);
                let _ = reply.send(Ok(()));
            }
            FluidMsg::SnapshotDensity { reply } => {
                let _ = reply.send(self.density.clone());
            }
        }
    }
}
