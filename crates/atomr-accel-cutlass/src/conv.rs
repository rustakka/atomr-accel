//! Implicit-GEMM convolution requests.
//!
//! CUTLASS exposes three implicit-GEMM kernels per layout:
//! `Conv2dFprop` (forward), `Conv2dDgrad` (gradient w.r.t. input),
//! `Conv2dWgrad` (gradient w.r.t. filter). We mirror that surface
//! here. Layout selection (NHWC / NCHW) and dtype propagate through
//! the same template render pipeline as GEMM.

use core::marker::PhantomData;

use crate::dtype::{CutlassDtype, GemmSupported, SmArch};
use crate::kernels;
use crate::plan_cache::PlanKey;

/// Tensor layout for the convolution. CUTLASS's implicit-GEMM kernels
/// are NHWC-first; NCHW is a translated fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvLayout {
    Nhwc,
    Nchw,
}

impl ConvLayout {
    pub fn cutlass_layout(self) -> &'static str {
        match self {
            ConvLayout::Nhwc => "cutlass::layout::TensorNHWC",
            ConvLayout::Nchw => "cutlass::layout::TensorNCHW",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            ConvLayout::Nhwc => "nhwc",
            ConvLayout::Nchw => "nchw",
        }
    }
}

/// `(N, H, W, C)` × `(R, S)` × stride / pad / dilation. `(K, P, Q)` is
/// derived inside the template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvShape {
    pub n: u32,
    pub h: u32,
    pub w: u32,
    pub c: u32,
    pub k: u32,
    pub r: u32,
    pub s: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dil_h: u32,
    pub dil_w: u32,
}

impl ConvShape {
    /// Convenience builder: stride / pad / dilation default to 1 / 0 / 1.
    pub fn nhwc(n: u32, h: u32, w: u32, c: u32, k: u32, r: u32, s: u32) -> Self {
        Self {
            n,
            h,
            w,
            c,
            k,
            r,
            s,
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            dil_h: 1,
            dil_w: 1,
        }
    }
}

/// Discriminator for which convolution gradient we're emitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ConvKind {
    Fprop,
    Dgrad,
    Wgrad,
}

impl ConvKind {
    pub(crate) fn short_name(self) -> &'static str {
        match self {
            ConvKind::Fprop => "fprop",
            ConvKind::Dgrad => "dgrad",
            ConvKind::Wgrad => "wgrad",
        }
    }

    pub(crate) fn cutlass_kernel(self) -> &'static str {
        match self {
            ConvKind::Fprop => "cutlass::conv::device::ImplicitGemmConvolution",
            ConvKind::Dgrad => "cutlass::conv::device::ImplicitGemmConvolutionDgrad",
            ConvKind::Wgrad => "cutlass::conv::device::ImplicitGemmConvolutionWgrad",
        }
    }
}

macro_rules! conv_request {
    ($name:ident, $kind:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name<T: GemmSupported> {
            pub shape: ConvShape,
            pub layout: ConvLayout,
            pub accum_dtype: CutlassDtype,
            pub output_dtype: CutlassDtype,
            pub arch: SmArch,
            _t: PhantomData<fn() -> T>,
        }

        impl<T: GemmSupported> $name<T> {
            pub fn new(shape: ConvShape, arch: SmArch) -> Self {
                Self {
                    shape,
                    layout: ConvLayout::Nhwc,
                    accum_dtype: CutlassDtype::F32,
                    output_dtype: T::DTYPE,
                    arch,
                    _t: PhantomData,
                }
            }

            pub fn with_layout(mut self, layout: ConvLayout) -> Self {
                self.layout = layout;
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

            pub fn plan_key(&self) -> PlanKey {
                PlanKey::conv::<T>(
                    $kind,
                    self.shape,
                    self.layout,
                    self.accum_dtype,
                    self.output_dtype,
                    self.arch,
                )
            }

            pub fn render_cu(&self) -> (String, String) {
                kernels::render_conv::<T>(
                    $kind,
                    self.shape,
                    self.layout,
                    self.accum_dtype,
                    self.output_dtype,
                    self.arch,
                )
            }
        }
    };
}

conv_request!(ConvFwdRequest, ConvKind::Fprop);
conv_request!(ConvDgradRequest, ConvKind::Dgrad);
conv_request!(ConvWgradRequest, ConvKind::Wgrad);

/// Erased dispatch surface for any convolution gradient.
pub trait CutlassConvDispatch: Send + 'static {
    fn plan_key(&self) -> PlanKey;
    fn render_cu(&self) -> (String, String);
    fn dtype(&self) -> CutlassDtype;
    fn arch(&self) -> SmArch;
    fn shape(&self) -> ConvShape;
    fn kind_name(&self) -> &'static str;
}

macro_rules! impl_dispatch {
    ($name:ident, $kind:expr) => {
        impl<T: GemmSupported> CutlassConvDispatch for $name<T> {
            fn plan_key(&self) -> PlanKey {
                $name::plan_key(self)
            }

            fn render_cu(&self) -> (String, String) {
                $name::render_cu(self)
            }

            fn dtype(&self) -> CutlassDtype {
                T::DTYPE
            }

            fn arch(&self) -> SmArch {
                self.arch
            }

            fn shape(&self) -> ConvShape {
                self.shape
            }

            fn kind_name(&self) -> &'static str {
                $kind.short_name()
            }
        }
    };
}

impl_dispatch!(ConvFwdRequest, ConvKind::Fprop);
impl_dispatch!(ConvDgradRequest, ConvKind::Dgrad);
impl_dispatch!(ConvWgradRequest, ConvKind::Wgrad);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::F16;

    #[test]
    fn conv_fwd_dgrad_wgrad_round_trip() {
        let shape = ConvShape::nhwc(8, 56, 56, 64, 128, 3, 3);

        let fwd = ConvFwdRequest::<F16>::new(shape, SmArch::Sm80);
        let dgrad = ConvDgradRequest::<F16>::new(shape, SmArch::Sm80);
        let wgrad = ConvWgradRequest::<F16>::new(shape, SmArch::Sm80);

        // All three keys distinct
        let kf = fwd.plan_key();
        let kd = dgrad.plan_key();
        let kw = wgrad.plan_key();
        assert_ne!(kf, kd);
        assert_ne!(kd, kw);
        assert_ne!(kf, kw);

        let (src_f, name_f) = fwd.render_cu();
        assert!(name_f.contains("fprop"));
        assert!(src_f.contains("ImplicitGemmConvolution"));

        let (_, name_d) = dgrad.render_cu();
        assert!(name_d.contains("dgrad"));

        let (_, name_w) = wgrad.render_cu();
        assert!(name_w.contains("wgrad"));

        // dispatch trait
        assert_eq!(fwd.kind_name(), "fprop");
        assert_eq!(dgrad.kind_name(), "dgrad");
        assert_eq!(wgrad.kind_name(), "wgrad");

        // Layout swap changes the key.
        let fwd_nchw = ConvFwdRequest::<F16>::new(shape, SmArch::Sm80).with_layout(ConvLayout::Nchw);
        assert_ne!(fwd.plan_key(), fwd_nchw.plan_key());
    }
}
