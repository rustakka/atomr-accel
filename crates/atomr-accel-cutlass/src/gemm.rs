//! GEMM template request and dispatch surface.
//!
//! A [`GemmRequest<T>`] is a typed, host-side description of a CUTLASS
//! `gemm_universal<...>` instantiation. The actor lifts it to a
//! `Box<dyn CutlassGemmDispatch>` so the per-dtype monomorphisations
//! can share a single mailbox.

use core::marker::PhantomData;

use crate::dtype::{CutlassDtype, GemmSupported, SmArch};
use crate::kernels;
use crate::plan_cache::PlanKey;

/// Row- vs column-major layout tags used at the API surface.
///
/// CUTLASS uses `cutlass::layout::RowMajor` / `cutlass::layout::ColumnMajor`
/// internally; the emitter maps these directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmLayout {
    RowMajor,
    ColMajor,
}

impl GemmLayout {
    pub fn cutlass_layout(self) -> &'static str {
        match self {
            GemmLayout::RowMajor => "cutlass::layout::RowMajor",
            GemmLayout::ColMajor => "cutlass::layout::ColumnMajor",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            GemmLayout::RowMajor => "rm",
            GemmLayout::ColMajor => "cm",
        }
    }
}

/// `(M, N, K)` problem shape for a single GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl GemmShape {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

/// Epilogue selector. The `Linear { alpha, beta }` arm is the default
/// `D = alpha * A @ B + beta * C` epilogue. `LinearReLU` and
/// `LinearGelu` are the most common fused activations. The richer
/// epilogue surface (multi-output, quantize, reduce) lives in
/// [`crate::evt`] behind the `evt` cargo feature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GemmEpilogue {
    Linear { alpha: f32, beta: f32 },
    LinearReLU { alpha: f32, beta: f32 },
    LinearGelu { alpha: f32, beta: f32 },
}

impl Default for GemmEpilogue {
    fn default() -> Self {
        GemmEpilogue::Linear { alpha: 1.0, beta: 0.0 }
    }
}

impl Eq for GemmEpilogue {}

impl core::hash::Hash for GemmEpilogue {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // Bit-cast the f32 fields so the hash matches `Eq` on
        // bit-pattern equality. CUTLASS-side template id only depends
        // on the discriminant, but the plan cache also wants to
        // dedupe at runtime parameter values.
        match *self {
            GemmEpilogue::Linear { alpha, beta } => {
                0u8.hash(state);
                alpha.to_bits().hash(state);
                beta.to_bits().hash(state);
            }
            GemmEpilogue::LinearReLU { alpha, beta } => {
                1u8.hash(state);
                alpha.to_bits().hash(state);
                beta.to_bits().hash(state);
            }
            GemmEpilogue::LinearGelu { alpha, beta } => {
                2u8.hash(state);
                alpha.to_bits().hash(state);
                beta.to_bits().hash(state);
            }
        }
    }
}

impl GemmEpilogue {
    /// Stable short name used in plan-cache keys and template ids.
    pub fn short_name(self) -> &'static str {
        match self {
            GemmEpilogue::Linear { .. } => "linear",
            GemmEpilogue::LinearReLU { .. } => "linear_relu",
            GemmEpilogue::LinearGelu { .. } => "linear_gelu",
        }
    }
}

/// Typed GEMM request. `T` is the element type for `A` and `B`; the
/// accumulator and output dtypes are derived from `T` by CUTLASS
/// (configurable via `accum_dtype` / `output_dtype`).
#[derive(Debug, Clone)]
pub struct GemmRequest<T: GemmSupported> {
    pub shape: GemmShape,
    pub layout_a: GemmLayout,
    pub layout_b: GemmLayout,
    pub layout_c: GemmLayout,
    pub epilogue: GemmEpilogue,
    /// Override the accumulator dtype. Defaults to fp32.
    pub accum_dtype: CutlassDtype,
    /// Override the output dtype. Defaults to `T::DTYPE`.
    pub output_dtype: CutlassDtype,
    /// Target compute architecture.
    pub arch: SmArch,
    /// Use a CUTLASS persistent (Hopper / Blackwell) kernel.
    pub persistent: bool,
    _t: PhantomData<fn() -> T>,
}

impl<T: GemmSupported> GemmRequest<T> {
    /// Canonical constructor.
    pub fn new(shape: GemmShape, arch: SmArch) -> Self {
        Self {
            shape,
            layout_a: GemmLayout::RowMajor,
            layout_b: GemmLayout::RowMajor,
            layout_c: GemmLayout::RowMajor,
            epilogue: GemmEpilogue::default(),
            accum_dtype: CutlassDtype::F32,
            output_dtype: T::DTYPE,
            arch,
            persistent: arch.supports_persistent_kernels(),
            _t: PhantomData,
        }
    }

    /// Deprecated 5-argument constructor. Pre-Phase-6 callers passed
    /// `(m, n, k, layout, alpha)`; we keep this path so out-of-tree
    /// downstreams compile against the 0.3.0 API surface.
    #[deprecated(
        note = "use `GemmRequest::new(shape, arch)` plus the builder methods instead"
    )]
    pub fn legacy(m: u32, n: u32, k: u32, layout: GemmLayout, alpha: f32) -> Self {
        let mut req = Self::new(GemmShape::new(m, n, k), SmArch::Sm80);
        req.layout_a = layout;
        req.layout_b = layout;
        req.layout_c = layout;
        req.epilogue = GemmEpilogue::Linear { alpha, beta: 0.0 };
        req
    }

    pub fn with_layouts(mut self, a: GemmLayout, b: GemmLayout, c: GemmLayout) -> Self {
        self.layout_a = a;
        self.layout_b = b;
        self.layout_c = c;
        self
    }

    pub fn with_epilogue(mut self, ep: GemmEpilogue) -> Self {
        self.epilogue = ep;
        self
    }

    pub fn with_accum_dtype(mut self, dt: CutlassDtype) -> Self {
        self.accum_dtype = dt;
        self
    }

    pub fn with_output_dtype(mut self, dt: CutlassDtype) -> Self {
        self.output_dtype = dt;
        self
    }

    pub fn with_persistent(mut self, persistent: bool) -> Self {
        self.persistent = persistent;
        self
    }

    /// Stable plan-cache key for this request. Used by the actor to
    /// dedupe NVRTC compilations.
    pub fn plan_key(&self) -> PlanKey {
        PlanKey::gemm::<T>(
            self.shape,
            self.layout_a,
            self.layout_b,
            self.layout_c,
            self.epilogue,
            self.accum_dtype,
            self.output_dtype,
            self.arch,
            self.persistent,
        )
    }

    /// Render the `.cu` source for this template. Returns the source
    /// plus the lowered kernel name to look up after NVRTC compile.
    pub fn render_cu(&self) -> (String, String) {
        kernels::render_gemm::<T>(self)
    }
}

/// Erased dispatch surface so the actor mailbox is `Sized`. Each
/// `GemmRequest<T>` boxes itself as `Box<dyn CutlassGemmDispatch>`.
pub trait CutlassGemmDispatch: Send + 'static {
    fn plan_key(&self) -> PlanKey;
    fn render_cu(&self) -> (String, String);
    fn dtype(&self) -> CutlassDtype;
    fn arch(&self) -> SmArch;
    fn shape(&self) -> GemmShape;
}

impl<T: GemmSupported> CutlassGemmDispatch for GemmRequest<T> {
    fn plan_key(&self) -> PlanKey {
        GemmRequest::plan_key(self)
    }

    fn render_cu(&self) -> (String, String) {
        GemmRequest::render_cu(self)
    }

    fn dtype(&self) -> CutlassDtype {
        T::DTYPE
    }

    fn arch(&self) -> SmArch {
        self.arch
    }

    fn shape(&self) -> GemmShape {
        self.shape
    }
}

/// Reply payload for [`crate::actor::CutlassMsg::Refit`]. Replaces a
/// previously compiled plan's weight buffer in place without
/// recompiling the kernel.
#[derive(Debug)]
pub struct RefitMsg {
    pub plan_key: PlanKey,
    /// Opaque weight bytes; the actor forwards them to the kernel's
    /// allocated workspace via the existing memory actors.
    pub weights: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{Bf16, F16, F4E2m1, F8E4m3, F8E5m2};

    #[test]
    fn gemm_request_round_trip_for_every_dtype() {
        // f32 — every arch
        let req = GemmRequest::<f32>::new(GemmShape::new(128, 256, 64), SmArch::Sm80);
        assert_eq!(req.dtype(), CutlassDtype::F32);
        assert_eq!(req.shape().m, 128);
        let (src, name) = req.render_cu();
        assert!(src.contains("cutlass::gemm::device::GemmUniversal"));
        assert!(name.starts_with("atomr_cutlass_gemm_"));

        // f64
        let req = GemmRequest::<f64>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);
        assert_eq!(req.dtype(), CutlassDtype::F64);

        // f16 / bf16
        let req = GemmRequest::<F16>::new(GemmShape::new(64, 64, 64), SmArch::Sm80)
            .with_layouts(GemmLayout::ColMajor, GemmLayout::RowMajor, GemmLayout::RowMajor);
        let key1 = req.plan_key();
        let req2 = GemmRequest::<F16>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);
        assert_ne!(key1, req2.plan_key());

        let _ = GemmRequest::<Bf16>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);

        // fp8 e4m3 / e5m2 — Hopper
        let req = GemmRequest::<F8E4m3>::new(GemmShape::new(128, 128, 128), SmArch::Sm90a)
            .with_epilogue(GemmEpilogue::LinearReLU { alpha: 1.0, beta: 0.0 });
        assert_eq!(req.dtype(), CutlassDtype::F8E4m3);
        assert!(req.persistent);
        let _ = GemmRequest::<F8E5m2>::new(GemmShape::new(64, 64, 64), SmArch::Sm90a);

        // fp4 — Blackwell
        let req = GemmRequest::<F4E2m1>::new(GemmShape::new(64, 64, 64), SmArch::Sm100);
        assert_eq!(req.dtype(), CutlassDtype::F4E2m1);

        // i8 / i32 / u8
        let _ = GemmRequest::<i8>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);
        let _ = GemmRequest::<i32>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);
        let _ = GemmRequest::<u8>::new(GemmShape::new(64, 64, 64), SmArch::Sm80);
    }

    #[test]
    fn deprecated_constructor_paths_compile() {
        // The legacy 5-arg form must still build so 0.2.x callers are
        // not broken. We allow the deprecation warning here on
        // purpose.
        #[allow(deprecated)]
        let req = GemmRequest::<f32>::legacy(64, 64, 64, GemmLayout::RowMajor, 1.0);
        assert_eq!(req.shape, GemmShape::new(64, 64, 64));
        match req.epilogue {
            GemmEpilogue::Linear { alpha, beta } => {
                assert_eq!(alpha, 1.0);
                assert_eq!(beta, 0.0);
            }
            _ => panic!("legacy constructor should produce Linear epilogue"),
        }
    }

    #[test]
    fn persistent_default_tracks_arch() {
        assert!(!GemmRequest::<f32>::new(GemmShape::new(1, 1, 1), SmArch::Sm80).persistent);
        assert!(GemmRequest::<f32>::new(GemmShape::new(1, 1, 1), SmArch::Sm90a).persistent);
        assert!(GemmRequest::<f32>::new(GemmShape::new(1, 1, 1), SmArch::Sm100).persistent);
    }
}
