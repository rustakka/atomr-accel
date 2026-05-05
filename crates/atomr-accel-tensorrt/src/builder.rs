//! Safe wrappers for `nvinfer1::IBuilder` and
//! `nvinfer1::IBuilderConfig`.
//!
//! Construction is GPU-free and panics-free even when libnvinfer is
//! not installed: the `IBuilderConfig` struct is a pure-Rust value
//! that records the requested knobs and is later replayed against the
//! C++ builder via the FFI shim under `tensorrt-link`.

use bitflags::bitflags;

bitflags! {
    /// Mirror of `nvinfer1::BuilderFlag` as a bitfield. Each
    /// flag toggles a single TensorRT optimisation knob; combine with
    /// `|`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct BuilderFlags: u32 {
        const FP16                       = 1 <<  0;
        const INT8                       = 1 <<  1;
        const DEBUG_KERNELS              = 1 <<  2;
        const GPU_FALLBACK               = 1 <<  3;
        const REFIT                      = 1 <<  4;
        const DISABLE_TIMING_CACHE       = 1 <<  5;
        const TF32                       = 1 <<  6;
        const SPARSE_WEIGHTS             = 1 <<  7;
        const SAFETY_SCOPE               = 1 <<  8;
        const OBEY_PRECISION_CONSTRAINTS = 1 <<  9;
        const PREFER_PRECISION_CONSTRAINTS = 1 << 10;
        const DIRECT_IO                  = 1 << 11;
        const REJECT_EMPTY_ALGORITHMS    = 1 << 12;
        const BF16                       = 1 << 13;
        const FP8                        = 1 << 14;
        const STRIP_PLAN                 = 1 << 15;
        const VERSION_COMPATIBLE         = 1 << 16;
        const EXCLUDE_LEAN_RUNTIME       = 1 << 17;
    }
}

bitflags! {
    /// Tactic sources to enable (mirrors `nvinfer1::TacticSource`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct TacticSources: u32 {
        const CUBLAS                 = 1 << 0;
        const CUBLAS_LT              = 1 << 1;
        const CUDNN                  = 1 << 2;
        const EDGE_MASK_CONVOLUTIONS = 1 << 3;
        const JIT_CONVOLUTIONS       = 1 << 4;
    }
}

/// High-level inference precision policy. Maps to a combination of
/// `BuilderFlags` (e.g. `BEST` ⇒ FP16 | INT8 | TF32 | BF16 | FP8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    Fp32,
    Fp16,
    Bf16,
    Int8,
    Fp8,
    /// Enable everything; let the builder pick the fastest tactic.
    Best,
}

impl Precision {
    pub fn flags(self) -> BuilderFlags {
        match self {
            Precision::Fp32 => BuilderFlags::TF32,
            Precision::Fp16 => BuilderFlags::FP16 | BuilderFlags::TF32,
            Precision::Bf16 => BuilderFlags::BF16 | BuilderFlags::TF32,
            Precision::Int8 => BuilderFlags::INT8 | BuilderFlags::TF32,
            Precision::Fp8 => BuilderFlags::FP8 | BuilderFlags::FP16 | BuilderFlags::TF32,
            Precision::Best => {
                BuilderFlags::FP16
                    | BuilderFlags::BF16
                    | BuilderFlags::INT8
                    | BuilderFlags::FP8
                    | BuilderFlags::TF32
            }
        }
    }
}

/// Default GPU/DLA target. DLA is the Jetson AI accelerator;
/// `Dla(core)` selects a specific DLA core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceType {
    #[default]
    Gpu,
    Dla(i32),
}

/// Engine refit policy. `OnDemand` opts into `BuilderFlags::REFIT`,
/// `WeightsStreaming` further enables `STRIP_PLAN` so weights live
/// outside the engine plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefitPolicy {
    #[default]
    Disabled,
    OnDemand,
    WeightsStreaming,
}

/// Pure-Rust mirror of `nvinfer1::IBuilderConfig`. Holds the requested
/// knobs in a side table; the FFI shim under `tensorrt-link` replays
/// them against the C++ object inside `BuilderActor::build`.
#[derive(Debug, Clone)]
pub struct IBuilderConfig {
    pub precision: Precision,
    pub device_type: DeviceType,
    /// Enable structured 2:4 sparsity (Ampere+).
    pub structured_sparsity: bool,
    /// Tactic-source allow-list (default: all on).
    pub tactic_sources: TacticSources,
    /// Persist the per-build timing cache. `None` = no cache.
    pub timing_cache: Option<Vec<u8>>,
    /// Engine refit policy.
    pub refit: RefitPolicy,
    /// Workspace memory pool budget (bytes).
    pub workspace_bytes: usize,
    /// DLA SRAM pool budget (bytes), only honoured when `device_type ==
    /// Dla(_)`.
    pub dla_sram_bytes: usize,
    /// Extra builder flags merged in on top of `precision.flags()` —
    /// allows callers to toggle e.g. `DEBUG_KERNELS` without losing
    /// the high-level precision policy.
    pub extra_flags: BuilderFlags,
}

impl Default for IBuilderConfig {
    fn default() -> Self {
        Self {
            precision: Precision::default(),
            device_type: DeviceType::default(),
            structured_sparsity: false,
            tactic_sources: TacticSources::CUBLAS
                | TacticSources::CUBLAS_LT
                | TacticSources::CUDNN
                | TacticSources::EDGE_MASK_CONVOLUTIONS
                | TacticSources::JIT_CONVOLUTIONS,
            timing_cache: None,
            refit: RefitPolicy::default(),
            workspace_bytes: 1 << 30, // 1 GiB
            dla_sram_bytes: 0,
            extra_flags: BuilderFlags::empty(),
        }
    }
}

impl IBuilderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }

    pub fn with_device(mut self, dt: DeviceType) -> Self {
        self.device_type = dt;
        self
    }

    pub fn with_sparsity(mut self, on: bool) -> Self {
        self.structured_sparsity = on;
        self
    }

    pub fn with_tactic_sources(mut self, ts: TacticSources) -> Self {
        self.tactic_sources = ts;
        self
    }

    pub fn with_timing_cache(mut self, cache: Vec<u8>) -> Self {
        self.timing_cache = Some(cache);
        self
    }

    pub fn with_refit(mut self, refit: RefitPolicy) -> Self {
        self.refit = refit;
        self
    }

    pub fn with_workspace_bytes(mut self, bytes: usize) -> Self {
        self.workspace_bytes = bytes;
        self
    }

    pub fn with_extra_flags(mut self, flags: BuilderFlags) -> Self {
        self.extra_flags = flags;
        self
    }

    /// Compute the final `BuilderFlags` bitmask the FFI shim would
    /// pass to `IBuilderConfig::setFlag()`. Combines the precision
    /// policy with refit + sparsity + caller-supplied extras.
    pub fn effective_flags(&self) -> BuilderFlags {
        let mut f = self.precision.flags() | self.extra_flags;
        if self.structured_sparsity {
            f |= BuilderFlags::SPARSE_WEIGHTS;
        }
        match self.refit {
            RefitPolicy::Disabled => {}
            RefitPolicy::OnDemand => f |= BuilderFlags::REFIT,
            RefitPolicy::WeightsStreaming => {
                f |= BuilderFlags::REFIT | BuilderFlags::STRIP_PLAN;
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_config_round_trip() {
        let cfg = IBuilderConfig::new()
            .with_precision(Precision::Best)
            .with_device(DeviceType::Dla(1))
            .with_sparsity(true)
            .with_refit(RefitPolicy::WeightsStreaming)
            .with_workspace_bytes(2 << 30)
            .with_extra_flags(BuilderFlags::DEBUG_KERNELS)
            .with_tactic_sources(TacticSources::CUBLAS | TacticSources::CUDNN)
            .with_timing_cache(vec![1, 2, 3, 4]);

        assert_eq!(cfg.precision, Precision::Best);
        assert!(cfg.structured_sparsity);
        assert!(matches!(cfg.refit, RefitPolicy::WeightsStreaming));
        assert!(matches!(cfg.device_type, DeviceType::Dla(1)));
        assert_eq!(cfg.workspace_bytes, 2 << 30);
        assert_eq!(cfg.timing_cache.as_deref(), Some(&[1u8, 2, 3, 4][..]));
        assert!(cfg.tactic_sources.contains(TacticSources::CUBLAS));
        assert!(!cfg.tactic_sources.contains(TacticSources::CUBLAS_LT));

        let flags = cfg.effective_flags();
        // Best ⇒ FP16/BF16/INT8/FP8/TF32, plus REFIT|STRIP_PLAN from
        // WeightsStreaming, plus SPARSE_WEIGHTS, plus DEBUG_KERNELS.
        assert!(flags.contains(BuilderFlags::FP16));
        assert!(flags.contains(BuilderFlags::BF16));
        assert!(flags.contains(BuilderFlags::INT8));
        assert!(flags.contains(BuilderFlags::FP8));
        assert!(flags.contains(BuilderFlags::TF32));
        assert!(flags.contains(BuilderFlags::REFIT));
        assert!(flags.contains(BuilderFlags::STRIP_PLAN));
        assert!(flags.contains(BuilderFlags::SPARSE_WEIGHTS));
        assert!(flags.contains(BuilderFlags::DEBUG_KERNELS));
    }

    #[test]
    fn precision_flag_mapping_is_stable() {
        assert!(Precision::Fp16.flags().contains(BuilderFlags::FP16));
        assert!(Precision::Bf16.flags().contains(BuilderFlags::BF16));
        assert!(Precision::Int8.flags().contains(BuilderFlags::INT8));
        assert!(Precision::Fp8.flags().contains(BuilderFlags::FP8));
        let best = Precision::Best.flags();
        for f in [
            BuilderFlags::FP16,
            BuilderFlags::BF16,
            BuilderFlags::INT8,
            BuilderFlags::FP8,
            BuilderFlags::TF32,
        ] {
            assert!(best.contains(f), "Best is missing {:?}", f);
        }
    }

    #[test]
    fn refit_disabled_does_not_set_refit_flag() {
        let cfg = IBuilderConfig::new();
        let f = cfg.effective_flags();
        assert!(!f.contains(BuilderFlags::REFIT));
        assert!(!f.contains(BuilderFlags::STRIP_PLAN));
    }
}
