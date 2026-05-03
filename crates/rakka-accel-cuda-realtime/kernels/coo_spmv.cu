// COO-format sparse matrix-vector multiply: y = A * x.
//
// One thread per non-zero. atomicAdd accumulates partial products into
// the destination row. Suitable as the GpuSparseStructureActor::SpMv
// fast path when the cuSPARSE feature is off (cudarc < safe wrapper).
//
// Launch shape: grid = ceil_div(nnz, blockDim.x), block = 256.
//
// Buffers (device):
//   rows  : i32[nnz] — row index of each entry
//   cols  : i32[nnz] — column index of each entry  (kept for symmetry; unused)
//   vals  : f32[nnz] — value at (row, col)
//   x     : f32[cols_count] — input vector
//   y     : f32[rows_count] — output vector (must be zero-initialized by caller)
extern "C" __global__ void coo_spmv(
    const int* __restrict__ rows,
    const int* __restrict__ cols,
    const float* __restrict__ vals,
    const float* __restrict__ x,
    float* __restrict__ y,
    int nnz,
    int rows_count,
    int cols_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    int r = rows[idx];
    int c = cols[idx];
    if (r < 0 || r >= rows_count || c < 0 || c >= cols_count) return;
    atomicAdd(&y[r], vals[idx] * x[c]);
}
