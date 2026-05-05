//! Grouped / batched GEMM template request.
//!
//! Gated behind the `grouped` cargo feature. Targets Hopper
//! (sm_90a, persistent grouped GEMM) and Blackwell (sm_100, fp8 / fp4
//! group GEMM). On Ampere we fall back to a strided-batched variant.

use core::marker::PhantomData;

use crate::dtype::{CutlassDtype, GemmSupported, SmArch};
use crate::gemm::{GemmEpilogue, GemmLayout, GemmShape};
use crate::kernels;
use crate::plan_cache::PlanKey;

/// Layout discriminator for a grouped GEMM. Each variant has its own
/// CUTLASS template specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupedLayout {
    /// All groups share the same `(M, N, K)` and stride.
    Uniform,
    /// `M` varies per group, `N` and `K` are uniform (variable-K is
    /// implemented via a separate template).
    VariableM,
    /// `M`, `N`, `K` all vary per group.
    Variable,
}

impl GroupedLayout {
    pub fn short_name(self) -> &'static str {
        match self {
            GroupedLayout::Uniform => "uniform",
            GroupedLayout::VariableM => "var_m",
            GroupedLayout::Variable => "var",
        }
    }
}

/// Per-group problem shape. For `Uniform`, only the first entry is
/// used; for `Variable*`, the actor uploads the full vector.
#[derive(Debug, Clone)]
pub struct GroupedGemmShape {
    pub shapes: Vec<GemmShape>,
}

impl GroupedGemmShape {
    pub fn new(shapes: Vec<GemmShape>) -> Self {
        Self { shapes }
    }

    pub fn group_count(&self) -> usize {
        self.shapes.len()
    }

    /// Hash-friendly summary that can live in a `PlanKey`.
    pub fn summary(&self) -> (u32, u32, u32, usize) {
        let m = self.shapes.iter().map(|s| s.m).max().unwrap_or(0);
        let n = self.shapes.iter().map(|s| s.n).max().unwrap_or(0);
        let k = self.shapes.iter().map(|s| s.k).max().unwrap_or(0);
        (m, n, k, self.shapes.len())
    }
}

/// Typed grouped-GEMM request.
#[derive(Debug, Clone)]
pub struct GroupedGemmRequest<T: GemmSupported> {
    pub shape: GroupedGemmShape,
    pub layout_a: GemmLayout,
    pub layout_b: GemmLayout,
    pub layout_c: GemmLayout,
    pub grouped_layout: GroupedLayout,
    pub epilogue: GemmEpilogue,
    pub accum_dtype: CutlassDtype,
    pub output_dtype: CutlassDtype,
    pub arch: SmArch,
    pub persistent: bool,
    _t: PhantomData<fn() -> T>,
}

impl<T: GemmSupported> GroupedGemmRequest<T> {
    pub fn new(shape: GroupedGemmShape, arch: SmArch) -> Self {
        Self {
            shape,
            layout_a: GemmLayout::RowMajor,
            layout_b: GemmLayout::RowMajor,
            layout_c: GemmLayout::RowMajor,
            grouped_layout: GroupedLayout::Uniform,
            epilogue: GemmEpilogue::default(),
            accum_dtype: CutlassDtype::F32,
            output_dtype: T::DTYPE,
            arch,
            persistent: arch.supports_persistent_kernels(),
            _t: PhantomData,
        }
    }

    pub fn with_grouped_layout(mut self, gl: GroupedLayout) -> Self {
        self.grouped_layout = gl;
        self
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

    pub fn plan_key(&self) -> PlanKey {
        PlanKey::grouped_gemm::<T>(
            self.shape.summary(),
            self.layout_a,
            self.layout_b,
            self.layout_c,
            self.grouped_layout,
            self.epilogue,
            self.accum_dtype,
            self.output_dtype,
            self.arch,
            self.persistent,
        )
    }

    pub fn render_cu(&self) -> (String, String) {
        kernels::render_grouped_gemm::<T>(self)
    }
}

/// Erased dispatch surface for grouped GEMM messages.
pub trait CutlassGroupedGemmDispatch: Send + 'static {
    fn plan_key(&self) -> PlanKey;
    fn render_cu(&self) -> (String, String);
    fn group_count(&self) -> usize;
    fn dtype(&self) -> CutlassDtype;
    fn arch(&self) -> SmArch;
}

impl<T: GemmSupported> CutlassGroupedGemmDispatch for GroupedGemmRequest<T> {
    fn plan_key(&self) -> PlanKey {
        GroupedGemmRequest::plan_key(self)
    }

    fn render_cu(&self) -> (String, String) {
        GroupedGemmRequest::render_cu(self)
    }

    fn group_count(&self) -> usize {
        self.shape.group_count()
    }

    fn dtype(&self) -> CutlassDtype {
        T::DTYPE
    }

    fn arch(&self) -> SmArch {
        self.arch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{F16, F8E4m3};

    #[test]
    fn grouped_gemm_request_round_trip() {
        let shapes = vec![
            GemmShape::new(64, 64, 64),
            GemmShape::new(128, 128, 64),
            GemmShape::new(64, 256, 32),
        ];
        let req = GroupedGemmRequest::<F16>::new(GroupedGemmShape::new(shapes), SmArch::Sm90a)
            .with_grouped_layout(GroupedLayout::Variable);
        assert_eq!(req.group_count(), 3);
        assert_eq!(req.dtype(), CutlassDtype::F16);
        assert_eq!(req.arch(), SmArch::Sm90a);

        let key1 = req.plan_key();
        let key2 = GroupedGemmRequest::<F16>::new(
            GroupedGemmShape::new(vec![GemmShape::new(64, 64, 64)]),
            SmArch::Sm90a,
        )
        .plan_key();
        assert_ne!(key1, key2);

        let (src, name) = req.render_cu();
        assert!(src.contains("GroupedGemm") || src.contains("grouped"));
        assert!(name.starts_with("atomr_cutlass_grouped_gemm_"));

        // fp8 grouped on Hopper
        let _ = GroupedGemmRequest::<F8E4m3>::new(
            GroupedGemmShape::new(vec![GemmShape::new(128, 128, 128)]),
            SmArch::Sm90a,
        );
    }

    #[test]
    fn grouped_layouts_have_distinct_keys() {
        let shapes = vec![GemmShape::new(64, 64, 64)];
        let s = GroupedGemmShape::new(shapes);
        let a = GroupedGemmRequest::<F16>::new(s.clone(), SmArch::Sm90a)
            .with_grouped_layout(GroupedLayout::Uniform)
            .plan_key();
        let b = GroupedGemmRequest::<F16>::new(s.clone(), SmArch::Sm90a)
            .with_grouped_layout(GroupedLayout::VariableM)
            .plan_key();
        let c = GroupedGemmRequest::<F16>::new(s, SmArch::Sm90a)
            .with_grouped_layout(GroupedLayout::Variable)
            .plan_key();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }
}
