// Velocity-Verlet particle integration with optional drag and an
// axis-aligned bounding-box reflection. Mirrors the CPU reference in
// crates/rakka-cuda-realtime/src/particle.rs::step.
//
// One thread per particle. Positions and velocities are stored as
// SoA float3 to match cudarc's `f32` device buffers.
//
// Launch shape: grid = ceil_div(n, blockDim.x), block = 256.
extern "C" __global__ void particle_step(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    int n,
    float dt,
    float gx, float gy, float gz,
    float drag_factor,             // = 1.0 - drag (already pre-computed by host)
    int   has_bounds,              // 0 = no clamp, 1 = clamp + reflect
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float bounce)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float vx = vel_x[i] + gx * dt;
    float vy = vel_y[i] + gy * dt;
    float vz = vel_z[i] + gz * dt;

    if (drag_factor != 1.0f) {
        vx *= drag_factor;
        vy *= drag_factor;
        vz *= drag_factor;
    }

    float px = pos_x[i] + vx * dt;
    float py = pos_y[i] + vy * dt;
    float pz = pos_z[i] + vz * dt;

    if (has_bounds) {
        if (px < min_x) { px = min_x; vx = -vx * bounce; }
        else if (px > max_x) { px = max_x; vx = -vx * bounce; }
        if (py < min_y) { py = min_y; vy = -vy * bounce; }
        else if (py > max_y) { py = max_y; vy = -vy * bounce; }
        if (pz < min_z) { pz = min_z; vz = -vz * bounce; }
        else if (pz > max_z) { pz = max_z; vz = -vz * bounce; }
    }

    pos_x[i] = px; pos_y[i] = py; pos_z[i] = pz;
    vel_x[i] = vx; vel_y[i] = vy; vel_z[i] = vz;
}
