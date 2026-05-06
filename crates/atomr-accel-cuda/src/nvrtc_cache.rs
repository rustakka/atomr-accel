//! Persistent disk cache for NVRTC-compiled CUDA kernels (Phase 0.6).
//!
//! Modern CUDA kernels — Hopper/Blackwell hand-rolled CUDA-C, CUTLASS
//! template instantiations, FlashAttention 2/3 variants — take 10s to
//! 60s each through NVRTC. A persistent disk cache turns subsequent
//! runs into single-digit-millisecond hot starts.
//!
//! ## Design
//!
//! - **Key**: `(source_hash, arch, options_hash)` where `arch` is the
//!   SM compute capability (e.g. `80`, `90`, `100`) and `options_hash`
//!   is FNV-1a of the NVRTC compile options in their original order
//!   (callers should sort beforehand if they want order-insensitive
//!   keys — see [`hash_options`]).
//! - **Value**: serialised PTX (and optional CUBIN) bytes wrapped in
//!   [`CachedKernel`].
//! - **Storage**: filesystem under
//!   `$XDG_CACHE_HOME/atomr-accel/nvrtc/` (or
//!   `$HOME/.cache/atomr-accel/nvrtc/`, falling back to
//!   `$TMPDIR/atomr-accel/nvrtc/`). One file per cache entry, named
//!   `{source_hash:016x}-{arch}-{options_hash:016x}.bin`.
//! - **Format**: bincode of [`CachedKernel`]. Entries whose
//!   `atomr_accel_version` does not match
//!   [`env!("CARGO_PKG_VERSION")`] are rejected on load.
//! - **Concurrency**: in-process [`RwLock`]ed [`HashMap`] read-through
//!   cache. Cross-process safety via atomic file write
//!   (`<name>.tmp` then `rename`).
//!
//! ## Usage
//!
//! ```no_run
//! use atomr_accel_cuda::nvrtc_cache::{
//!     hash_options, hash_source, CachedKernel, NvrtcCache, NvrtcCacheKey,
//! };
//!
//! let cache = NvrtcCache::new().unwrap();
//! let src = "extern \"C\" __global__ void noop() {}";
//! let key = NvrtcCacheKey {
//!     source_hash: hash_source(src),
//!     arch: 80,
//!     options_hash: hash_options(["-std=c++17", "--use_fast_math"]),
//! };
//! if let Some(entry) = cache.get(key) {
//!     println!("hot: {} bytes of PTX", entry.ptx.len());
//! } else {
//!     // ... NVRTC compile ...
//!     let ptx: Vec<u8> = b"PTX...".to_vec();
//!     cache.insert(key, CachedKernel::new(ptx, None)).unwrap();
//! }
//! ```
//!
//! Phase 5 will wire `NvrtcActor` through this cache; this module ships
//! the storage layer alone.

use crate::error::GpuError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::hash::Hasher;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Composite cache key. `source_hash` and `options_hash` are produced
/// by [`hash_source`] / [`hash_options`]; `arch` is the SM compute
/// capability as an integer (e.g. `80`, `90`, `100`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvrtcCacheKey {
    pub source_hash: u64,
    pub arch: u32,
    pub options_hash: u64,
}

/// On-disk and in-memory cache value.
///
/// `atomr_accel_version` is checked on load: entries from older
/// crate versions are silently rejected so a cache built against a
/// stale `cudarc` / NVRTC ABI never gets loaded into a newer build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedKernel {
    pub ptx: Vec<u8>,
    pub cubin: Option<Vec<u8>>,
    pub atomr_accel_version: String,
}

impl CachedKernel {
    /// Build a [`CachedKernel`] stamped with the current crate version.
    pub fn new(ptx: Vec<u8>, cubin: Option<Vec<u8>>) -> Self {
        Self {
            ptx,
            cubin,
            atomr_accel_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Read-through disk cache for compiled NVRTC kernels.
///
/// Cross-process safe: writes go to a temp file and are atomically
/// renamed into place. Reads tolerate concurrent writers (a partial
/// `.tmp` is invisible to readers; a corrupt `.bin` returns `None`).
#[derive(Debug)]
pub struct NvrtcCache {
    dir: PathBuf,
    memory: RwLock<HashMap<NvrtcCacheKey, Arc<CachedKernel>>>,
}

impl NvrtcCache {
    /// Construct a cache rooted at the OS-default location.
    ///
    /// Probe order:
    /// 1. `$XDG_CACHE_HOME/atomr-accel/nvrtc/`
    /// 2. `$HOME/.cache/atomr-accel/nvrtc/` (via [`dirs::cache_dir`])
    /// 3. `<temp_dir>/atomr-accel/nvrtc/`
    pub fn new() -> Result<Self, GpuError> {
        Self::with_dir(default_cache_dir())
    }

    /// Construct a cache rooted at an explicit directory. Creates the
    /// directory (recursively) if it does not exist.
    pub fn with_dir(path: PathBuf) -> Result<Self, GpuError> {
        fs::create_dir_all(&path).map_err(|e| {
            GpuError::Unrecoverable(format!(
                "NvrtcCache: failed to create cache dir {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(Self {
            dir: path,
            memory: RwLock::new(HashMap::new()),
        })
    }

    /// Cache root directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Look up a cached kernel.
    ///
    /// Returns `None` if the entry is absent, unreadable, mis-encoded,
    /// or stamped with a non-matching `atomr_accel_version`. A read
    /// hit on disk but cold in memory promotes the entry into the
    /// in-memory map for subsequent lookups.
    pub fn get(&self, key: NvrtcCacheKey) -> Option<Arc<CachedKernel>> {
        if let Some(hit) = self
            .memory
            .read()
            .ok()
            .and_then(|guard| guard.get(&key).cloned())
        {
            return Some(hit);
        }

        let path = self.entry_path(&key);
        let bytes = fs::read(&path).ok()?;
        let entry: CachedKernel = bincode::deserialize(&bytes).ok()?;
        if entry.atomr_accel_version != env!("CARGO_PKG_VERSION") {
            return None;
        }
        let arc = Arc::new(entry);
        if let Ok(mut guard) = self.memory.write() {
            guard.insert(key, arc.clone());
        }
        Some(arc)
    }

    /// Store a kernel.
    ///
    /// The on-disk write is atomic (write to `<name>.tmp` then
    /// rename). Concurrent writers race the rename; the loser's bytes
    /// are silently overwritten — bincode payloads of the same key are
    /// expected to be identical so this is benign.
    pub fn insert(&self, key: NvrtcCacheKey, value: CachedKernel) -> Result<(), GpuError> {
        let bytes = bincode::serialize(&value).map_err(|e| {
            GpuError::Unrecoverable(format!("NvrtcCache: bincode serialize: {}", e))
        })?;
        let final_path = self.entry_path(&key);
        let tmp_path = final_path.with_extension("bin.tmp");

        {
            let mut f = fs::File::create(&tmp_path).map_err(|e| {
                GpuError::Unrecoverable(format!("NvrtcCache: create {}: {}", tmp_path.display(), e))
            })?;
            f.write_all(&bytes).map_err(|e| {
                GpuError::Unrecoverable(format!("NvrtcCache: write {}: {}", tmp_path.display(), e))
            })?;
            f.sync_all().map_err(|e| {
                GpuError::Unrecoverable(format!("NvrtcCache: fsync {}: {}", tmp_path.display(), e))
            })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| {
            // Best-effort clean up the temp file on rename failure.
            let _ = fs::remove_file(&tmp_path);
            GpuError::Unrecoverable(format!(
                "NvrtcCache: rename {} -> {}: {}",
                tmp_path.display(),
                final_path.display(),
                e
            ))
        })?;

        if let Ok(mut guard) = self.memory.write() {
            guard.insert(key, Arc::new(value));
        }
        Ok(())
    }

    /// Drop every in-memory entry. Disk contents are left untouched
    /// — subsequent `get` calls will re-populate from disk.
    pub fn clear_in_memory(&self) {
        if let Ok(mut guard) = self.memory.write() {
            guard.clear();
        }
    }

    fn entry_path(&self, key: &NvrtcCacheKey) -> PathBuf {
        self.dir.join(format!(
            "{:016x}-{}-{:016x}.bin",
            key.source_hash, key.arch, key.options_hash
        ))
    }
}

/// FNV-1a 64-bit hash of a kernel source string. Stable across
/// processes and across crate compilations.
pub fn hash_source(src: &str) -> u64 {
    let mut h = FnvHasher::new();
    h.write(src.as_bytes());
    h.finish()
}

/// FNV-1a 64-bit hash of an iterable of NVRTC compile options.
///
/// **Order-sensitive**: callers that want order-insensitive keys must
/// sort the iterable before passing it in. NVRTC's `--define-macro`
/// and `--include-path` flags are order-significant in general, so
/// the cache key preserves the caller's order.
pub fn hash_options<I, S>(opts: I) -> u64
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut h = FnvHasher::new();
    for opt in opts {
        let bytes = opt.as_ref().as_bytes();
        // Length prefix so ["ab", "c"] != ["a", "bc"].
        h.write_u64(bytes.len() as u64);
        h.write(bytes);
        // Separator byte to make the boundary unambiguous.
        h.write_u8(0xff);
    }
    h.finish()
}

fn default_cache_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        let p = PathBuf::from(xdg);
        if !p.as_os_str().is_empty() {
            return p.join("atomr-accel").join("nvrtc");
        }
    }
    if let Some(cache) = dirs::cache_dir() {
        return cache.join("atomr-accel").join("nvrtc");
    }
    std::env::temp_dir().join("atomr-accel").join("nvrtc")
}

// ---------------------------------------------------------------------------
// FNV-1a 64-bit. Tiny, dependency-free, deterministic — good enough for a
// kernel-source content hash. Not cryptographic.
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

struct FnvHasher(u64);

impl FnvHasher {
    fn new() -> Self {
        Self(FNV_OFFSET_BASIS_64)
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME_64);
        }
        self.0 = h;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_kernel(seed: u8) -> CachedKernel {
        CachedKernel::new(vec![seed; 64], Some(vec![seed.wrapping_add(1); 32]))
    }

    fn key(source_hash: u64, arch: u32, options_hash: u64) -> NvrtcCacheKey {
        NvrtcCacheKey {
            source_hash,
            arch,
            options_hash,
        }
    }

    #[test]
    fn round_trip_via_with_dir() {
        let tmp = tempdir().unwrap();
        let cache = NvrtcCache::with_dir(tmp.path().to_path_buf()).unwrap();
        let k = key(0xdead_beef, 80, 0x1234);
        let v = sample_kernel(7);

        assert!(cache.get(k).is_none(), "cold cache should miss");

        cache.insert(k, v.clone()).unwrap();

        let got = cache.get(k).expect("hot lookup must hit");
        assert_eq!(got.ptx, v.ptx);
        assert_eq!(got.cubin, v.cubin);
        assert_eq!(got.atomr_accel_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn cache_persists_across_fresh_handles() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let k = key(0x9999, 90, 0xabcd);
        let v = sample_kernel(42);

        {
            let cache = NvrtcCache::with_dir(dir.clone()).unwrap();
            cache.insert(k, v.clone()).unwrap();
        } // drop the cache, only the file remains

        let cache2 = NvrtcCache::with_dir(dir).unwrap();
        let got = cache2.get(k).expect("disk-backed entry must survive");
        assert_eq!(got.ptx, v.ptx);
        assert_eq!(got.cubin, v.cubin);
    }

    #[test]
    fn distinct_keys_distinct_paths() {
        let tmp = tempdir().unwrap();
        let cache = NvrtcCache::with_dir(tmp.path().to_path_buf()).unwrap();

        let k_src = key(1, 80, 0);
        let k_arch = key(0, 90, 0);
        let k_opts = key(0, 80, 1);
        let k_zero = key(0, 80, 0);

        let p_src = cache.entry_path(&k_src);
        let p_arch = cache.entry_path(&k_arch);
        let p_opts = cache.entry_path(&k_opts);
        let p_zero = cache.entry_path(&k_zero);

        assert_ne!(p_src, p_arch);
        assert_ne!(p_src, p_opts);
        assert_ne!(p_src, p_zero);
        assert_ne!(p_arch, p_opts);
        assert_ne!(p_arch, p_zero);
        assert_ne!(p_opts, p_zero);

        // Inserting under each key writes a separate file.
        cache.insert(k_src, sample_kernel(1)).unwrap();
        cache.insert(k_arch, sample_kernel(2)).unwrap();
        cache.insert(k_opts, sample_kernel(3)).unwrap();
        cache.insert(k_zero, sample_kernel(4)).unwrap();

        assert!(p_src.exists());
        assert!(p_arch.exists());
        assert!(p_opts.exists());
        assert!(p_zero.exists());

        // And reads come back distinct.
        assert_eq!(cache.get(k_src).unwrap().ptx, vec![1u8; 64]);
        assert_eq!(cache.get(k_arch).unwrap().ptx, vec![2u8; 64]);
        assert_eq!(cache.get(k_opts).unwrap().ptx, vec![3u8; 64]);
        assert_eq!(cache.get(k_zero).unwrap().ptx, vec![4u8; 64]);
    }

    #[test]
    fn version_mismatch_rejected_on_load() {
        let tmp = tempdir().unwrap();
        let cache = NvrtcCache::with_dir(tmp.path().to_path_buf()).unwrap();
        let k = key(11, 80, 22);

        // Hand-write an entry with a stale version stamp.
        let stale = CachedKernel {
            ptx: vec![0xaa; 16],
            cubin: None,
            atomr_accel_version: "0.0.0-impossible".to_string(),
        };
        let bytes = bincode::serialize(&stale).unwrap();
        let path = cache.entry_path(&k);
        fs::write(&path, &bytes).unwrap();

        // Skip the in-memory shortcut by clearing it explicitly.
        cache.clear_in_memory();
        assert!(
            cache.get(k).is_none(),
            "entry with mismatched atomr_accel_version must not be returned"
        );
    }

    #[test]
    fn hash_source_is_deterministic() {
        let a = hash_source("__global__ void k() {}");
        let b = hash_source("__global__ void k() {}");
        let c = hash_source("__global__ void other() {}");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn hash_options_is_deterministic_and_order_sensitive() {
        let a = hash_options(["-std=c++17", "--use_fast_math"]);
        let b = hash_options(["-std=c++17", "--use_fast_math"]);
        let c = hash_options(["--use_fast_math", "-std=c++17"]);
        let d = hash_options(["-std=c++17"]);
        let e = hash_options(Vec::<&str>::new());

        assert_eq!(a, b, "same input must produce same hash");
        assert_ne!(a, c, "option order must change the hash");
        assert_ne!(a, d, "option count must change the hash");
        assert_ne!(a, e);
        assert_ne!(d, e);

        // Length-prefix invariant: ["ab","c"] != ["a","bc"].
        let split1 = hash_options(["ab", "c"]);
        let split2 = hash_options(["a", "bc"]);
        assert_ne!(split1, split2);
    }

    #[test]
    fn clear_in_memory_keeps_disk() {
        let tmp = tempdir().unwrap();
        let cache = NvrtcCache::with_dir(tmp.path().to_path_buf()).unwrap();
        let k = key(7, 80, 7);
        cache.insert(k, sample_kernel(9)).unwrap();
        cache.clear_in_memory();
        let got = cache.get(k).expect("disk entry survives clear_in_memory");
        assert_eq!(got.ptx, vec![9u8; 64]);
    }
}
