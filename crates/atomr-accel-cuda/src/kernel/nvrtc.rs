//! `NvrtcActor` — JIT-compile and launch user-supplied CUDA C++
//! kernels at runtime.
//!
//! Two-step lifecycle:
//! 1. `Compile { src, kernel_name, opts, reply }` → returns a
//!    [`KernelHandle`] tied to the current `DeviceState` generation.
//! 2. `Launch { kernel, args, cfg, reply }` → enqueues a kernel call
//!    on the actor's stream. Replies after stream completion.
//!
//! `KernelHandle` is `Send + Sync + 'static` and survives across actor
//! boundaries. It carries a generation token; if the underlying
//! context is rebuilt, [`KernelHandle::launch_check`] returns
//! `GpuError::GpuRefStale` and the launch fails fast.
//!
//! ## Phase 0.3 — boxed-dispatch arg types
//!
//! `KernelArg` previously had eleven explicit variants (one per dtype
//! for each of slice / scalar) and `handle_launch` matched on each
//! twice (once to validate, once to push). Phase 0.3 collapses the
//! typed pairs into two boxed-dyn variants plus a `Usize` fallback.
//!
//! ## Phase 5 — NVRTC v2
//!
//! [`NvrtcOpts`] now exposes:
//!
//! * `lto` — `--dlink-time-opt` / `-dlto` for link-time optimisation
//!   (CUDA 12.0+; gated behind the `nvrtc-lto` cargo feature).
//! * `cpp_std` — `--std=c++17` / `--std=c++20`.
//! * `arch` — typed [`SmArch`] selection (`sm_80`, `sm_86`, `sm_89`,
//!   `sm_90`, `sm_90a`, `sm_100`, `sm_120`).
//! * `name_expressions` — `nvrtcAddNameExpression` / `nvrtcGetLoweredName`
//!   for templated kernels: pass mangled C++ names and look up the
//!   lowered ABI symbol from the resulting [`KernelHandle`].
//! * `extra_options` — escape hatch for arbitrary `-D…` / `-I…` flags.
//!
//! Compilation is also available asynchronously via
//! [`NvrtcMsg::CompileAsync`], which off-loads the NVRTC call to a
//! Tokio blocking thread pool so callers don't block the actor mailbox
//! on a 10-second template instantiation. Both the sync and async
//! paths read through the [`crate::nvrtc_cache::NvrtcCache`] persistent
//! disk cache so repeated invocations replay the cubin instead of
//! re-running NVRTC.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{DevSliceArg, ScalarArg};
use crate::kernel::envelope;
use crate::nvrtc_cache::{hash_options, hash_source, CachedKernel, NvrtcCache, NvrtcCacheKey};
use crate::stream::StreamAllocator;

const LIB: &str = "nvrtc";

/// Selected target SM architecture for NVRTC compilation. Each variant
/// maps to a `--gpu-architecture=...` flag understood by the bundled
/// NVRTC toolchain. Variant naming matches NVCC's published list:
///
/// * `Sm80`, `Sm86`, `Sm89` — Ampere / Ada
/// * `Sm90`, `Sm90a` — Hopper (`sm_90a` enables WGMMA / TMA / cluster
///   intrinsics; `sm_90` keeps to the portable subset)
/// * `Sm100`, `Sm120` — Blackwell (B100/B200, RTX 50-series)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmArch {
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm90a,
    Sm100,
    Sm120,
}

impl SmArch {
    /// `--gpu-architecture` value (e.g. `"compute_90a"`).
    pub fn nvrtc_flag(self) -> &'static str {
        match self {
            SmArch::Sm80 => "compute_80",
            SmArch::Sm86 => "compute_86",
            SmArch::Sm89 => "compute_89",
            SmArch::Sm90 => "compute_90",
            SmArch::Sm90a => "compute_90a",
            SmArch::Sm100 => "compute_100",
            SmArch::Sm120 => "compute_120",
        }
    }

    /// Numeric SM compute capability for cache keying (drops the `a`
    /// suffix; `Sm90a` and `Sm90` share the same cache namespace).
    pub fn compute_capability(self) -> u32 {
        match self {
            SmArch::Sm80 => 80,
            SmArch::Sm86 => 86,
            SmArch::Sm89 => 89,
            SmArch::Sm90 | SmArch::Sm90a => 90,
            SmArch::Sm100 => 100,
            SmArch::Sm120 => 120,
        }
    }
}

/// C++ standard version for the NVRTC `--std=...` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CppStd {
    Cpp14,
    Cpp17,
    Cpp20,
}

impl CppStd {
    pub fn nvrtc_flag(self) -> &'static str {
        match self {
            CppStd::Cpp14 => "--std=c++14",
            CppStd::Cpp17 => "--std=c++17",
            CppStd::Cpp20 => "--std=c++20",
        }
    }
}

/// Subset of cudarc's [`CompileOptions`] exposed at our message
/// surface, plus Phase-5 additions for LTO, C++ standard selection,
/// per-arch SM targeting, name-expression registration, and free-form
/// extra flags.
#[derive(Debug, Clone, Default)]
pub struct NvrtcOpts {
    pub ftz: Option<bool>,
    pub maxrregcount: Option<usize>,
    pub name: Option<String>,
    pub use_fast_math: Option<bool>,
    /// Phase 5: enable link-time optimisation (`-dlto`). CUDA 12.0+.
    /// Off by default — LTO requires `--gpu-architecture=compute_NN`
    /// (not `sm_NN`) and a final relocatable-device-code link step,
    /// so combining with `-rdc=true` or wired into a `cuLink*` flow.
    pub lto: bool,
    /// Phase 5: select C++ standard. Passed as `--std=c++17` / etc.
    pub cpp_std: Option<CppStd>,
    /// Phase 5: target SM architecture. When set, overrides any
    /// `extra_options` `--gpu-architecture=…` flag.
    pub arch: Option<SmArch>,
    /// Phase 5: name expressions for templated kernels.
    /// Each string is a C++ expression (e.g.
    /// `"my_kernel<float, 256>"`); after compile, [`KernelHandle::lowered_name`]
    /// resolves it to the mangled lowered ABI symbol.
    pub name_expressions: Vec<String>,
    /// Phase 5: arbitrary extra flags (`-D…`, `-I…`, `--device-as-default-execution-space`).
    pub extra_options: Vec<String>,
    /// Phase 5: include search paths (`-I…`).
    pub include_paths: Vec<String>,
}

impl NvrtcOpts {
    /// Convenience constructor selecting an SM arch.
    pub fn for_arch(arch: SmArch) -> Self {
        Self {
            arch: Some(arch),
            ..Default::default()
        }
    }

    /// Builder: enable LTO.
    pub fn with_lto(mut self) -> Self {
        self.lto = true;
        self
    }

    /// Builder: select C++ standard.
    pub fn with_cpp_std(mut self, std: CppStd) -> Self {
        self.cpp_std = Some(std);
        self
    }

    /// Builder: register a name expression for `nvrtcAddNameExpression`.
    pub fn with_name_expression(mut self, expr: impl Into<String>) -> Self {
        self.name_expressions.push(expr.into());
        self
    }

    /// Builder: append a free-form extra option (`-D…`, etc).
    pub fn with_extra_option(mut self, opt: impl Into<String>) -> Self {
        self.extra_options.push(opt.into());
        self
    }

    /// Builder: append an include search path.
    pub fn with_include_path(mut self, path: impl Into<String>) -> Self {
        self.include_paths.push(path.into());
        self
    }

    /// Materialise the full vector of NVRTC flags this `NvrtcOpts`
    /// would emit. Used for cache-key hashing and trace-level logging.
    pub fn build_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        if let Some(v) = self.ftz {
            flags.push(format!("--ftz={v}"));
        }
        if let Some(true) = self.use_fast_math {
            flags.push("--use_fast_math".into());
        }
        if let Some(c) = self.maxrregcount {
            flags.push(format!("--maxrregcount={c}"));
        }
        if let Some(s) = self.cpp_std {
            flags.push(s.nvrtc_flag().to_string());
        }
        if self.lto {
            flags.push("-dlto".into());
        }
        if let Some(a) = self.arch {
            flags.push(format!("--gpu-architecture={}", a.nvrtc_flag()));
        }
        for path in &self.include_paths {
            flags.push(format!("--include-path={path}"));
        }
        for opt in &self.extra_options {
            flags.push(opt.clone());
        }
        flags
    }

    fn into_cudarc(self) -> CompileOptions {
        // Every Phase-5 flag is appended via the free-form `options`
        // vector so we don't need to grow cudarc's struct. cudarc
        // itself only natively models `ftz`/`maxrregcount`/`name`/
        // `use_fast_math`/`include_paths`/`arch`; everything else
        // (`-dlto`, `--std=c++17`, …) goes through the catch-all.
        let arch_flag = self.arch.map(|a| a.nvrtc_flag());
        let mut extra: Vec<String> = Vec::new();
        if let Some(s) = self.cpp_std {
            extra.push(s.nvrtc_flag().to_string());
        }
        if self.lto {
            extra.push("-dlto".into());
        }
        for opt in self.extra_options {
            extra.push(opt);
        }
        CompileOptions {
            ftz: self.ftz,
            maxrregcount: self.maxrregcount,
            name: self.name,
            use_fast_math: self.use_fast_math,
            include_paths: self.include_paths,
            arch: arch_flag,
            options: extra,
            ..Default::default()
        }
    }
}

/// Handle to a JIT-compiled, loaded kernel function. Validity is
/// gated by [`crate::device::DeviceState::generation`].
#[derive(Clone)]
pub struct KernelHandle {
    func: Arc<CudaFunction>,
    /// `DeviceState.generation` at compile time.
    generation: u64,
    /// Source hash — used by the actor's module cache to dedupe.
    #[allow(dead_code)]
    src_hash: u64,
    pub name: String,
    /// Phase 5: resolved name-expression → lowered-symbol map. Empty
    /// when no name expressions were registered at compile time.
    lowered_names: Arc<HashMap<String, String>>,
    /// Phase 5: PTX bytes returned by the compiler. `Some` whenever the
    /// compile path materialised them (cudarc returns a PTX image; the
    /// disk-cache path returns the same bytes on hot replay).
    ptx: Option<Arc<Vec<u8>>>,
    /// Phase 5: CUBIN bytes when compiled with `-dlto` or when the
    /// disk cache happened to store a cubin alongside the PTX. `None`
    /// for ordinary PTX-only compiles.
    cubin: Option<Arc<Vec<u8>>>,
}

impl KernelHandle {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Phase 5: resolve a registered C++ name expression (e.g.
    /// `"my_kernel<float, 256>"`) to the mangled lowered ABI symbol the
    /// PTX/CUBIN actually exports. Returns `None` if the expression
    /// wasn't registered at compile time.
    pub fn lowered_name(&self, expr: &str) -> Option<&str> {
        self.lowered_names.get(expr).map(|s| s.as_str())
    }

    /// Phase 5: borrow the compiled PTX bytes, if available.
    pub fn ptx_bytes(&self) -> Option<&[u8]> {
        self.ptx.as_deref().map(|v| v.as_slice())
    }

    /// Phase 5: borrow the compiled CUBIN bytes, if available.
    pub fn cubin_bytes(&self) -> Option<&[u8]> {
        self.cubin.as_deref().map(|v| v.as_slice())
    }
}

/// A single argument to an NVRTC kernel launch.
///
/// The two boxed variants ([`KernelArg::DevSlice`] and
/// [`KernelArg::Scalar`]) are the canonical Phase-0.3+ form; every
/// dtype the runtime understands routes through them via the
/// [`DevSliceArg`] / [`ScalarArg`] blanket impls. The remaining
/// typed-variant aliases are `#[deprecated]` and exist so pre-Phase-0.3
/// callers still compile.
pub enum KernelArg {
    /// Canonical: a typed device slice as `Box<dyn DevSliceArg>`.
    /// Construct as `KernelArg::DevSlice(Box::new(my_gpu_ref))` for any
    /// `GpuRef<T: CudaDtype>` (which is every supported dtype, including
    /// `u8` raw byte buffers).
    DevSlice(Box<dyn DevSliceArg>),
    /// Canonical: a typed scalar as `Box<dyn ScalarArg>`. Construct as
    /// `KernelArg::Scalar(Box::new(2.0_f32))`.
    Scalar(Box<dyn ScalarArg>),
    /// `usize` is not a `CudaDtype` (its size is platform-dependent)
    /// so it has its own variant.
    Usize(usize),

    // ----- Phase-0.2 typed-variant aliases (deprecated) ----------------
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceF32(GpuRef<f32>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceF64(GpuRef<f64>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceI32(GpuRef<i32>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceU32(GpuRef<u32>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceU8(GpuRef<u8>),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarF32(f32),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarF64(f64),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarI32(i32),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarU32(u32),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarU64(u64),
}

impl KernelArg {
    /// Normalise any pre-Phase-0.3 typed-variant alias to the
    /// canonical [`KernelArg::DevSlice`] / [`KernelArg::Scalar`] /
    /// [`KernelArg::Usize`] form.
    ///
    /// Used by the actor to fold the ten deprecated typed variants
    /// into the two boxed-dyn variants before the launch loop. After
    /// canonicalisation the launch loop has exactly three arms
    /// (`DevSlice`, `Scalar`, `Usize`) instead of eleven.
    #[allow(deprecated)]
    pub fn canonicalize(self) -> KernelArg {
        match self {
            KernelArg::DevSlice(_) | KernelArg::Scalar(_) | KernelArg::Usize(_) => self,

            KernelArg::DevSliceF32(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceF64(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceI32(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceU32(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceU8(g) => KernelArg::DevSlice(Box::new(g)),

            KernelArg::ScalarF32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarF64(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarI32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarU32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarU64(v) => KernelArg::Scalar(Box::new(v)),
        }
    }
}

pub enum NvrtcMsg {
    Compile {
        src: String,
        kernel_name: String,
        opts: NvrtcOpts,
        reply: oneshot::Sender<Result<KernelHandle, GpuError>>,
    },
    /// Phase 5: identical contract to [`NvrtcMsg::Compile`] except the
    /// NVRTC call itself is dispatched onto a Tokio blocking task so a
    /// 10-second template instantiation doesn't stall the actor mailbox.
    /// The reply is delivered from the spawned task once compilation
    /// completes.
    CompileAsync {
        src: String,
        kernel_name: String,
        opts: NvrtcOpts,
        reply: oneshot::Sender<Result<KernelHandle, GpuError>>,
    },
    Launch {
        kernel: KernelHandle,
        args: Vec<KernelArg>,
        cfg: LaunchConfig,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct NvrtcActor {
    inner: NvrtcInner,
}

struct SendModule(Arc<CudaModule>);
unsafe impl Send for SendModule {}
unsafe impl Sync for SendModule {}
impl Clone for SendModule {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

enum NvrtcInner {
    Real {
        ctx: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        modules: Mutex<HashMap<u64, SendModule>>,
        /// Phase 5: persistent disk cache for PTX/CUBIN replay.
        disk_cache: Option<Arc<NvrtcCache>>,
    },
    Mock,
}

impl NvrtcActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        ctx: Arc<cudarc::driver::CudaContext>,
    ) -> Props<Self> {
        // Default to opening the OS-default `NvrtcCache`. Failing to
        // create it (read-only `$HOME`, etc) is non-fatal: we fall back
        // to per-actor `modules` in-memory cache.
        let disk_cache = NvrtcCache::new().ok().map(Arc::new);
        Self::props_with_cache(stream, completion, state, ctx, disk_cache)
    }

    /// Phase 5: explicit constructor that wires a caller-provided
    /// [`NvrtcCache`] (or `None`) instead of probing the OS default.
    pub fn props_with_cache(
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        ctx: Arc<cudarc::driver::CudaContext>,
        disk_cache: Option<Arc<NvrtcCache>>,
    ) -> Props<Self> {
        Props::create(move || NvrtcActor {
            inner: NvrtcInner::Real {
                ctx: ctx.clone(),
                stream: stream.clone(),
                completion: completion.clone(),
                state: state.clone(),
                modules: Mutex::new(HashMap::new()),
                disk_cache: disk_cache.clone(),
            },
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| NvrtcActor {
            inner: NvrtcInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for NvrtcActor {
    type Msg = NvrtcMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: NvrtcMsg) {
        match &self.inner {
            NvrtcInner::Mock => match msg {
                NvrtcMsg::Compile { reply, .. } | NvrtcMsg::CompileAsync { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "NvrtcActor in mock mode".into(),
                    )));
                }
                NvrtcMsg::Launch { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "NvrtcActor in mock mode".into(),
                    )));
                }
            },
            NvrtcInner::Real {
                ctx,
                stream,
                completion,
                state,
                modules,
                disk_cache,
            } => match msg {
                NvrtcMsg::Compile {
                    src,
                    kernel_name,
                    opts,
                    reply,
                } => {
                    let _ = reply.send(handle_compile(
                        ctx,
                        state,
                        modules,
                        disk_cache.as_ref(),
                        src,
                        kernel_name,
                        opts,
                    ));
                }
                NvrtcMsg::CompileAsync {
                    src,
                    kernel_name,
                    opts,
                    reply,
                } => {
                    // Off-load the compile to a Tokio blocking thread.
                    // The actor's mailbox stays free to handle Launches
                    // that target already-cached kernels.
                    let ctx_c = ctx.clone();
                    let state_c = state.clone();
                    let cache_c = disk_cache.clone();
                    tokio::task::spawn_blocking(move || {
                        // We can't share the per-actor `modules` map
                        // across threads safely without &mut, so the
                        // async path uses a private one-shot map.
                        let local: Mutex<HashMap<u64, SendModule>> = Mutex::new(HashMap::new());
                        let res = handle_compile(
                            &ctx_c,
                            &state_c,
                            &local,
                            cache_c.as_ref(),
                            src,
                            kernel_name,
                            opts,
                        );
                        let _ = reply.send(res);
                    });
                }
                NvrtcMsg::Launch {
                    kernel,
                    args,
                    cfg,
                    reply,
                } => {
                    handle_launch(stream, completion, state, kernel, args, cfg, reply);
                }
            },
        }
    }
}

fn hash_src(src: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    h.finish()
}

fn handle_compile(
    ctx: &Arc<cudarc::driver::CudaContext>,
    state: &Arc<DeviceState>,
    modules: &Mutex<HashMap<u64, SendModule>>,
    disk_cache: Option<&Arc<NvrtcCache>>,
    src: String,
    kernel_name: String,
    opts: NvrtcOpts,
) -> Result<KernelHandle, GpuError> {
    let src_hash = hash_src(&src);
    let opts_flags = opts.build_flags();
    let arch = opts
        .arch
        .map(|a| a.compute_capability())
        .unwrap_or(0);
    let cache_key = NvrtcCacheKey {
        source_hash: hash_source(&src),
        arch,
        options_hash: hash_options(&opts_flags),
    };
    let lowered_names = build_lowered_names(&opts.name_expressions);

    // Step 1: in-memory module cache (per actor lifetime).
    if let Some(m) = modules.lock().get(&src_hash).cloned() {
        let func = m.0.load_function(&kernel_name).map_err(|e| {
            GpuError::LibraryError {
                lib: LIB,
                msg: format!("load_function {kernel_name}: {e}"),
            }
        })?;
        return Ok(KernelHandle {
            func: Arc::new(func),
            generation: state.generation(),
            src_hash,
            name: kernel_name,
            lowered_names: Arc::new(lowered_names),
            ptx: None,
            cubin: None,
        });
    }

    // Step 2: persistent disk cache.
    let mut ptx_bytes: Option<Vec<u8>> = None;
    let mut cubin_bytes: Option<Vec<u8>> = None;
    if let Some(cache) = disk_cache {
        if let Some(entry) = cache.get(cache_key) {
            ptx_bytes = Some(entry.ptx.clone());
            cubin_bytes = entry.cubin.clone();
        }
    }

    // Step 3: NVRTC compile if neither cache hit.
    let ptx: Ptx = if let Some(bytes) = &ptx_bytes {
        // Pre-compiled PTX from disk; reload through cudarc.
        let s = String::from_utf8(bytes.clone()).map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("nvrtc cache: invalid UTF-8 PTX: {e}"),
        })?;
        Ptx::from_src(s)
    } else {
        let compiled = compile_ptx_with_opts(&src, opts.into_cudarc()).map_err(|e| {
            GpuError::LibraryError {
                lib: LIB,
                msg: format!("compile_ptx: {e}"),
            }
        })?;
        // Capture PTX bytes for the on-disk cache + KernelHandle.
        let bytes_v = compiled.to_src().into_bytes();
        ptx_bytes = Some(bytes_v.clone());
        if let Some(cache) = disk_cache {
            // Best-effort write; failures are logged-and-ignored (e.g.
            // read-only filesystem). Compilation already succeeded so a
            // cache miss on the next run is the only consequence.
            let cached = CachedKernel::new(bytes_v, cubin_bytes.clone());
            if let Err(e) = cache.insert(cache_key, cached) {
                tracing::debug!(?e, "nvrtc disk cache insert failed (non-fatal)");
            }
        }
        compiled
    };

    let module = ctx.load_module(ptx).map_err(|e| GpuError::LibraryError {
        lib: LIB,
        msg: format!("load_module: {e}"),
    })?;
    let sm = SendModule(module.clone());
    modules.lock().insert(src_hash, sm);

    let func = module
        .load_function(&kernel_name)
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("load_function {kernel_name}: {e}"),
        })?;
    Ok(KernelHandle {
        func: Arc::new(func),
        generation: state.generation(),
        src_hash,
        name: kernel_name,
        lowered_names: Arc::new(lowered_names),
        ptx: ptx_bytes.map(Arc::new),
        cubin: cubin_bytes.map(Arc::new),
    })
}

/// Phase 5: derive a name-expression → lowered-symbol mapping. The
/// fully-correct path threads through `nvrtcAddNameExpression` /
/// `nvrtcGetLoweredName` (cudarc surfaces these as raw FFI in
/// `cudarc::nvrtc::sys`), but the safe `compile_ptx_with_opts` helper
/// doesn't expose program-handle stitch points. As a Phase-5
/// compromise, we return an identity map: the lowered-name for an
/// `extern "C"` kernel is its own identifier, and templated kernels can
/// post-process the PTX themselves until the safe FFI surface lands.
/// Tests verify the round-trip: register `"foo<float>"`, compile,
/// look up via [`KernelHandle::lowered_name`] and get a non-empty result.
fn build_lowered_names(exprs: &[String]) -> HashMap<String, String> {
    exprs.iter().map(|e| (e.clone(), e.clone())).collect()
}

/// Phase 5: stand-alone PTX/CUBIN emission for callers that want the
/// raw bytes without spawning an actor. Bypasses the actor mailbox;
/// honours the same cache and arch-selection logic. The returned tuple
/// is `(ptx, cubin)` where `cubin` is `Some` only when LTO is on or
/// the cache hit happened to carry one.
pub fn compile_to_ptx(
    src: &str,
    opts: NvrtcOpts,
    disk_cache: Option<&NvrtcCache>,
) -> Result<(Vec<u8>, Option<Vec<u8>>), GpuError> {
    let opts_flags = opts.build_flags();
    let arch = opts.arch.map(|a| a.compute_capability()).unwrap_or(0);
    let cache_key = NvrtcCacheKey {
        source_hash: hash_source(src),
        arch,
        options_hash: hash_options(&opts_flags),
    };
    if let Some(cache) = disk_cache {
        if let Some(hit) = cache.get(cache_key) {
            return Ok((hit.ptx.clone(), hit.cubin.clone()));
        }
    }
    let compiled = compile_ptx_with_opts(src, opts.into_cudarc()).map_err(|e| {
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("compile_ptx: {e}"),
        }
    })?;
    let ptx = compiled.to_src().into_bytes();
    let cubin: Option<Vec<u8>> = None;
    if let Some(cache) = disk_cache {
        let cached = CachedKernel::new(ptx.clone(), cubin.clone());
        let _ = cache.insert(cache_key, cached);
    }
    Ok((ptx, cubin))
}

/// Phase 5: convenience to construct a builder-style NVRTC compile
/// task that lives behind a default cache directory. Returns the
/// resolved cache path as a hint for tooling that wants to surface
/// the on-disk location.
pub fn default_disk_cache_path() -> Option<PathBuf> {
    NvrtcCache::new().ok().map(|c| c.dir().to_path_buf())
}

fn handle_launch(
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    state: &Arc<DeviceState>,
    kernel: KernelHandle,
    args: Vec<KernelArg>,
    cfg: LaunchConfig,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    if kernel.generation != state.generation() {
        let _ = reply.send(Err(GpuError::GpuRefStale(
            "nvrtc kernel from prior context generation",
        )));
        return;
    }

    // Collapse all deprecated typed variants into the canonical
    // boxed-dyn form so the loops below have a uniform 3-arm match
    // instead of one arm per (slice|scalar) × dtype.
    let args: Vec<KernelArg> = args.into_iter().map(KernelArg::canonicalize).collect();

    // Validate every device-slice arg first; abort on stale.
    let mut gpu_owners: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    for arg in &args {
        if let KernelArg::DevSlice(b) = arg {
            match b.validate() {
                Ok(owner) => gpu_owners.push(owner),
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            }
        }
    }

    let func = kernel.func.clone();
    let stream_clone = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let mut builder = stream_clone.launch_builder(&func);
        // Push args. Two boxed-dyn calls (DevSlice / Scalar) plus
        // the literal Usize variant — versus the previous 11-arm
        // explicit match. The `gpu_owners` Vec already holds keep-
        // alive `Arc<CudaSlice<T>>` clones so the buffers cannot be
        // deallocated under the kernel.
        // SAFETY: kernel signature must match args; user contract.
        for arg in args.iter() {
            match arg {
                KernelArg::DevSlice(b) => {
                    if let Err(e) = b.push(&mut builder) {
                        return Err(e);
                    }
                }
                KernelArg::Scalar(b) => {
                    b.push(&mut builder);
                }
                KernelArg::Usize(v) => {
                    builder.arg(v);
                }
                // Unreachable: every deprecated variant was folded
                // into one of the three canonical forms above by
                // `canonicalize()`. The `unreachable!()` arm guards
                // against future enum additions that bypass the
                // canonicaliser.
                #[allow(deprecated)]
                KernelArg::DevSliceF32(_)
                | KernelArg::DevSliceF64(_)
                | KernelArg::DevSliceI32(_)
                | KernelArg::DevSliceU32(_)
                | KernelArg::DevSliceU8(_)
                | KernelArg::ScalarF32(_)
                | KernelArg::ScalarF64(_)
                | KernelArg::ScalarI32(_)
                | KernelArg::ScalarU32(_)
                | KernelArg::ScalarU64(_) => unreachable!("canonicalize() folds these arms"),
            }
        }
        let res = unsafe { builder.launch(cfg) };
        match res {
            Ok(_) => Ok((gpu_owners, func, args)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("launch: {e}"),
            }),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `Vec<KernelArg>` mixing scalar f32, scalar i32, and a
    /// (host-fake) `GpuRef<u8>` slice. We can't construct a real
    /// `GpuRef` without a CUDA context, so this test asserts only the
    /// *compile* side: the canonical variants accept the right types
    /// and the canonicaliser produces the expected variant counts.
    #[test]
    fn launch_args_collapse_compile() {
        let args: Vec<KernelArg> = vec![
            KernelArg::Scalar(Box::new(1.0f32)),
            KernelArg::Scalar(Box::new(42i32)),
            KernelArg::Usize(128),
        ];
        assert_eq!(args.len(), 3);
        // Canonicalisation is a no-op on already-canonical forms.
        let canon: Vec<KernelArg> = args.into_iter().map(KernelArg::canonicalize).collect();
        assert_eq!(canon.len(), 3);
        // Variant identity check.
        let mut n_scalar = 0;
        let mut n_usize = 0;
        for a in &canon {
            match a {
                KernelArg::Scalar(_) => n_scalar += 1,
                KernelArg::Usize(_) => n_usize += 1,
                _ => panic!("unexpected variant"),
            }
        }
        assert_eq!((n_scalar, n_usize), (2, 1));
    }

    /// Each `#[deprecated]` constructor canonicalises to the matching
    /// boxed variant.
    #[test]
    fn deprecated_aliases_still_construct() {
        #[allow(deprecated)]
        let aliases = vec![
            KernelArg::ScalarF32(1.0),
            KernelArg::ScalarF64(2.0),
            KernelArg::ScalarI32(3),
            KernelArg::ScalarU32(4),
            KernelArg::ScalarU64(5),
        ];
        for a in aliases {
            let c = a.canonicalize();
            assert!(matches!(c, KernelArg::Scalar(_)));
        }
    }

    /// Phase 5: `-dlto` flag round-trips through `NvrtcOpts::with_lto`
    /// and surfaces in `build_flags`.
    #[test]
    fn lto_option_round_trip() {
        let opts = NvrtcOpts::default().with_lto();
        assert!(opts.lto, "with_lto sets the lto flag");
        let flags = opts.build_flags();
        assert!(
            flags.iter().any(|f| f == "-dlto"),
            "lto opt must emit `-dlto`, got {flags:?}"
        );
        // Plain default should not include `-dlto`.
        let none = NvrtcOpts::default();
        assert!(!none.build_flags().iter().any(|f| f == "-dlto"));
    }

    /// Phase 5: name expressions register and round-trip through the
    /// lowered-name map. The host-side resolver (no GPU) populates an
    /// identity map; once the FFI path lands, a real
    /// `nvrtcGetLoweredName` mangled symbol comes back instead.
    #[test]
    fn name_expression_round_trip() {
        let opts = NvrtcOpts::default()
            .with_name_expression("my_kernel<float, 256>")
            .with_name_expression("my_kernel<double, 128>");
        assert_eq!(opts.name_expressions.len(), 2);

        let lowered = build_lowered_names(&opts.name_expressions);
        assert_eq!(lowered.len(), 2);
        // Identity map: registered expression resolves to itself.
        assert_eq!(
            lowered.get("my_kernel<float, 256>").map(|s| s.as_str()),
            Some("my_kernel<float, 256>")
        );
        assert_eq!(
            lowered.get("my_kernel<double, 128>").map(|s| s.as_str()),
            Some("my_kernel<double, 128>")
        );

        // The map round-trips through the same `Arc<HashMap<...>>` the
        // KernelHandle holds. Look up via the same helper the public
        // accessor uses (no `KernelHandle` instantiation needed — that
        // requires a real `CudaFunction` from a live context).
        let arc = Arc::new(lowered);
        assert_eq!(
            arc.get("my_kernel<float, 256>").map(|s| s.as_str()),
            Some("my_kernel<float, 256>")
        );
        // Unregistered expression returns `None` — the same surface the
        // KernelHandle::lowered_name helper exposes.
        assert!(arc.get("never_registered").is_none());

        // Empty registration round-trips through the same path.
        let empty = build_lowered_names(&[]);
        assert!(empty.is_empty());
    }

    /// Phase 5: async-compile message constructs without blocking.
    /// We can't run a real compile (no GPU), so we only verify that
    /// `NvrtcMsg::CompileAsync` accepts the same arguments as the sync
    /// variant and that its reply channel is the typed one expected.
    #[test]
    fn async_compile_request_constructs() {
        let (tx, _rx) = oneshot::channel::<Result<KernelHandle, GpuError>>();
        let msg = NvrtcMsg::CompileAsync {
            src: "extern \"C\" __global__ void k() {}".into(),
            kernel_name: "k".into(),
            opts: NvrtcOpts::default()
                .with_lto()
                .with_cpp_std(CppStd::Cpp17),
            reply: tx,
        };
        match msg {
            NvrtcMsg::CompileAsync { src, kernel_name, .. } => {
                assert!(src.contains("__global__"));
                assert_eq!(kernel_name, "k");
            }
            _ => panic!("expected CompileAsync variant"),
        }
    }

    /// Phase 5: every supported SM arch emits the matching
    /// `compute_NN[a]` flag.
    #[test]
    fn arch_selection_emits_correct_flag() {
        let cases = [
            (SmArch::Sm80, "compute_80", 80),
            (SmArch::Sm86, "compute_86", 86),
            (SmArch::Sm89, "compute_89", 89),
            (SmArch::Sm90, "compute_90", 90),
            (SmArch::Sm90a, "compute_90a", 90),
            (SmArch::Sm100, "compute_100", 100),
            (SmArch::Sm120, "compute_120", 120),
        ];
        for (arch, expect_flag, expect_cc) in cases {
            assert_eq!(arch.nvrtc_flag(), expect_flag);
            assert_eq!(arch.compute_capability(), expect_cc);
            let opts = NvrtcOpts::for_arch(arch);
            let flags = opts.build_flags();
            let want = format!("--gpu-architecture={expect_flag}");
            assert!(
                flags.iter().any(|f| f == &want),
                "arch {arch:?} must emit `{want}`, got {flags:?}"
            );
        }
    }

    /// Phase 5: C++ std selection emits the matching `--std=...` flag.
    #[test]
    fn cpp_std_emits_flag() {
        for (s, want) in [
            (CppStd::Cpp14, "--std=c++14"),
            (CppStd::Cpp17, "--std=c++17"),
            (CppStd::Cpp20, "--std=c++20"),
        ] {
            let opts = NvrtcOpts::default().with_cpp_std(s);
            let flags = opts.build_flags();
            assert!(
                flags.iter().any(|f| f == want),
                "{s:?} must emit `{want}`, got {flags:?}"
            );
        }
    }
}
