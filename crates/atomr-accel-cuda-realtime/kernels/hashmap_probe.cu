// Open-addressing hashmap probe kernels for `GpuHashMapActor`.
//
// Layout (all device buffers):
//   slots_key    : u8[capacity * key_size]
//   slots_value  : u8[capacity * value_size]
//   slot_state   : u32[capacity]   // 0 = empty, 1 = occupied, 2 = tombstone
//
// One thread per query key for both kernels. Linear probing with the
// standard FNV-1a hash truncated to `capacity`. The kernels return
// per-query status codes via `out_status` (1 = found / inserted, 0 =
// missing / table-full).
//
// Mirrors the CPU reference in
// crates/rakka-cuda-realtime/src/hashmap.rs.

__device__ inline unsigned int fnv1a(const unsigned char* k, int len) {
    unsigned int h = 0x811c9dc5u;
    for (int i = 0; i < len; ++i) {
        h ^= (unsigned int)k[i];
        h *= 0x01000193u;
    }
    return h;
}

__device__ inline int key_eq(const unsigned char* a, const unsigned char* b, int len) {
    for (int i = 0; i < len; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

extern "C" __global__ void hashmap_lookup(
    const unsigned char* __restrict__ slots_key,
    const unsigned char* __restrict__ slots_value,
    const unsigned int* __restrict__ slot_state,
    const unsigned char* __restrict__ query_keys,
    unsigned char* __restrict__ out_values,
    unsigned int* __restrict__ out_status,
    int n_queries,
    int capacity,
    int key_size,
    int value_size,
    int max_probes)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries) return;

    const unsigned char* qk = query_keys + (size_t)q * key_size;
    unsigned int h = fnv1a(qk, key_size) % (unsigned int)capacity;

    int probes = max_probes < capacity ? max_probes : capacity;
    for (int i = 0; i < probes; ++i) {
        unsigned int slot = (h + (unsigned int)i) % (unsigned int)capacity;
        unsigned int st = slot_state[slot];
        if (st == 0u) {
            // Empty slot — key is absent.
            out_status[q] = 0u;
            for (int j = 0; j < value_size; ++j)
                out_values[(size_t)q * value_size + j] = 0;
            return;
        }
        if (st == 1u && key_eq(slots_key + (size_t)slot * key_size, qk, key_size)) {
            for (int j = 0; j < value_size; ++j)
                out_values[(size_t)q * value_size + j] = slots_value[(size_t)slot * value_size + j];
            out_status[q] = 1u;
            return;
        }
        // Tombstone or collision — keep probing.
    }
    out_status[q] = 0u;
    for (int j = 0; j < value_size; ++j)
        out_values[(size_t)q * value_size + j] = 0;
}

// Linear-probe insert. Uses atomicCAS on slot_state to claim a slot.
// Returns out_status[q] = 1 if the (key, value) was written, 0 if the
// table was full.
extern "C" __global__ void hashmap_insert(
    unsigned char* __restrict__ slots_key,
    unsigned char* __restrict__ slots_value,
    unsigned int* __restrict__ slot_state,
    const unsigned char* __restrict__ query_keys,
    const unsigned char* __restrict__ query_values,
    unsigned int* __restrict__ out_status,
    int n_queries,
    int capacity,
    int key_size,
    int value_size,
    int max_probes)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries) return;

    const unsigned char* qk = query_keys + (size_t)q * key_size;
    const unsigned char* qv = query_values + (size_t)q * value_size;
    unsigned int h = fnv1a(qk, key_size) % (unsigned int)capacity;

    int probes = max_probes < capacity ? max_probes : capacity;
    for (int i = 0; i < probes; ++i) {
        unsigned int slot = (h + (unsigned int)i) % (unsigned int)capacity;
        unsigned int prev = atomicCAS(&slot_state[slot], 0u, 1u);
        if (prev == 0u || prev == 2u) {
            // Won the slot (was empty or tombstoned).
            for (int j = 0; j < key_size; ++j)
                slots_key[(size_t)slot * key_size + j] = qk[j];
            for (int j = 0; j < value_size; ++j)
                slots_value[(size_t)slot * value_size + j] = qv[j];
            // If we took a tombstone we have to upgrade the state to
            // occupied; if we took an empty we already wrote 1u via CAS.
            if (prev == 2u) atomicExch(&slot_state[slot], 1u);
            out_status[q] = 1u;
            return;
        }
        // Slot occupied — check if it's our key (idempotent overwrite).
        if (key_eq(slots_key + (size_t)slot * key_size, qk, key_size)) {
            for (int j = 0; j < value_size; ++j)
                slots_value[(size_t)slot * value_size + j] = qv[j];
            out_status[q] = 1u;
            return;
        }
        // Otherwise keep probing.
    }
    out_status[q] = 0u;
}
