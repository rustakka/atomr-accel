// Verlet cloth integration + structural-spring constraint pass.
//
// Two kernels — invoke `cloth_verlet` once per Step, then
// `cloth_constrain_pass` `iterations` times. The constraint kernel
// processes one spring per thread; pinned vertices remain anchored.
//
// Mirrors the CPU reference in
// crates/rakka-cuda-realtime/src/cloth.rs::step.

extern "C" __global__ void cloth_verlet(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ prev_x,
    float* __restrict__ prev_y,
    float* __restrict__ prev_z,
    const unsigned char* __restrict__ pinned,   // 0 / 1 per vertex
    int n,
    float dt2,
    float gx, float gy, float gz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (pinned[i]) return;

    float xx = pos_x[i];
    float yy = pos_y[i];
    float zz = pos_z[i];

    float nx = 2.0f * xx - prev_x[i] + gx * dt2;
    float ny = 2.0f * yy - prev_y[i] + gy * dt2;
    float nz = 2.0f * zz - prev_z[i] + gz * dt2;

    prev_x[i] = xx; prev_y[i] = yy; prev_z[i] = zz;
    pos_x[i] = nx; pos_y[i] = ny; pos_z[i] = nz;
}

// One thread per (a, b) spring pair. Springs come in two flat lists
// (`springs_a`, `springs_b`), of length `n_springs`. The kernel uses
// atomic adds so simultaneous touches of the same vertex from
// different springs compose without races.
extern "C" __global__ void cloth_constrain_pass(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const int* __restrict__ springs_a,
    const int* __restrict__ springs_b,
    const unsigned char* __restrict__ pinned,
    int n_springs,
    float rest,
    float stiff)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_springs) return;
    int a = springs_a[s];
    int b = springs_b[s];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];
    float d2 = dx * dx + dy * dy + dz * dz;
    float d  = sqrtf(d2 < 1e-16f ? 1e-16f : d2);

    float diff = ((d - rest) / d) * stiff * 0.5f;
    float ox = dx * diff;
    float oy = dy * diff;
    float oz = dz * diff;

    if (!pinned[a]) {
        atomicAdd(&pos_x[a], ox);
        atomicAdd(&pos_y[a], oy);
        atomicAdd(&pos_z[a], oz);
    }
    if (!pinned[b]) {
        atomicAdd(&pos_x[b], -ox);
        atomicAdd(&pos_y[b], -oy);
        atomicAdd(&pos_z[b], -oz);
    }
}
