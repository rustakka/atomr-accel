//! `ParticleSystemActor` — Newtonian per-particle update.
//!
//! Each `Step { dt }` advances every particle's position by
//! `velocity * dt + 0.5 * accel * dt^2` and velocity by
//! `accel * dt`. Drag is applied as a per-step multiplicative
//! velocity decay.
//!
//! F6 ships a CPU reference. The integration loop is pure data —
//! ideal for an NVRTC kernel in F7+. The public message surface
//! stays identical when the GPU backend lands.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
}

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
}

#[derive(Debug, Clone)]
pub struct ParticleSystemConfig {
    /// Per-step gravitational + external acceleration applied to
    /// every particle.
    pub gravity: Vec3,
    /// Per-step linear drag coefficient (velocity *= 1 - drag).
    pub drag: f32,
    /// Optional bounding box; particles that exit are clamped and
    /// their velocity along the violated axis is reflected with
    /// `bounce` restitution.
    pub bounds: Option<(Vec3, Vec3)>,
    pub bounce: f32,
}

impl Default for ParticleSystemConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3 { x: 0.0, y: -9.8, z: 0.0 },
            drag: 0.0,
            bounds: None,
            bounce: 0.5,
        }
    }
}

pub enum ParticleMsg {
    /// Replace the particle set. Returns the previous count.
    Reset {
        particles: Vec<Particle>,
        reply: oneshot::Sender<usize>,
    },
    /// Advance the simulation by `dt` seconds.
    Step {
        dt: f32,
        reply: oneshot::Sender<Result<usize, GpuError>>,
    },
    /// Pull the current particle state.
    Snapshot {
        reply: oneshot::Sender<Vec<Particle>>,
    },
    /// Update config in-place.
    UpdateConfig {
        cfg: ParticleSystemConfig,
        reply: oneshot::Sender<()>,
    },
}

pub struct ParticleSystemActor {
    cfg: ParticleSystemConfig,
    particles: Vec<Particle>,
}

impl ParticleSystemActor {
    pub fn props(cfg: ParticleSystemConfig) -> Props<Self> {
        Props::create(move || ParticleSystemActor {
            cfg: cfg.clone(),
            particles: Vec::new(),
        })
    }

    fn step(&mut self, dt: f32) {
        let g = self.cfg.gravity;
        let drag = self.cfg.drag;
        for p in &mut self.particles {
            // Velocity Verlet (simplified): v += a*dt; x += v*dt.
            p.velocity.x += g.x * dt;
            p.velocity.y += g.y * dt;
            p.velocity.z += g.z * dt;
            if drag > 0.0 {
                let f = 1.0 - drag;
                p.velocity.x *= f;
                p.velocity.y *= f;
                p.velocity.z *= f;
            }
            p.position.x += p.velocity.x * dt;
            p.position.y += p.velocity.y * dt;
            p.position.z += p.velocity.z * dt;
            if let Some((min, max)) = self.cfg.bounds {
                let bounce = self.cfg.bounce;
                clamp_axis(&mut p.position.x, &mut p.velocity.x, min.x, max.x, bounce);
                clamp_axis(&mut p.position.y, &mut p.velocity.y, min.y, max.y, bounce);
                clamp_axis(&mut p.position.z, &mut p.velocity.z, min.z, max.z, bounce);
            }
        }
    }
}

fn clamp_axis(p: &mut f32, v: &mut f32, lo: f32, hi: f32, bounce: f32) {
    if *p < lo {
        *p = lo;
        *v = -(*v) * bounce;
    } else if *p > hi {
        *p = hi;
        *v = -(*v) * bounce;
    }
}

#[async_trait]
impl Actor for ParticleSystemActor {
    type Msg = ParticleMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ParticleMsg) {
        match msg {
            ParticleMsg::Reset { particles, reply } => {
                let prev = self.particles.len();
                self.particles = particles;
                let _ = reply.send(prev);
            }
            ParticleMsg::Step { dt, reply } => {
                self.step(dt);
                let _ = reply.send(Ok(self.particles.len()));
            }
            ParticleMsg::Snapshot { reply } => {
                let _ = reply.send(self.particles.clone());
            }
            ParticleMsg::UpdateConfig { cfg, reply } => {
                self.cfg = cfg;
                let _ = reply.send(());
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
    async fn gravity_pulls_particle_down() {
        let cfg = ParticleSystemConfig {
            gravity: Vec3 { x: 0.0, y: -10.0, z: 0.0 },
            drag: 0.0,
            bounds: None,
            bounce: 0.0,
        };
        let sys = ActorSystem::create("particle-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(ParticleSystemActor::props(cfg), "particles").unwrap();

        let p = Particle {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
        };
        let (tx, rx) = oneshot::channel();
        actor.tell(ParticleMsg::Reset { particles: vec![p], reply: tx });
        tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();

        // Two steps of dt=0.1 → v_y = -2; y = -0.1 + -0.2 = -0.3.
        for _ in 0..2 {
            let (tx, rx) = oneshot::channel();
            actor.tell(ParticleMsg::Step { dt: 0.1, reply: tx });
            tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        }

        let (tx, rx) = oneshot::channel();
        actor.tell(ParticleMsg::Snapshot { reply: tx });
        let snap = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
        assert!((snap[0].velocity.y - (-2.0)).abs() < 1e-5);
        assert!(snap[0].position.y < 0.0);

        sys.terminate().await;
    }
}
