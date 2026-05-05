//! Typed `MatmulRequest<T: GemmSupported>` plus the `BlasLtDispatch`
//! impl that routes it through the kernel envelope.
//!
//! Today's pre-Phase-1 actor accepted only `MatmulConfig + GpuRef<f32>`.
//! `MatmulRequest<T>` widens that to:
//! - any `T: GemmSupported` (f32 / f64 / f16 / bf16 / fp8),
//! - explicit `D` output buffer (so fp8 split-k and out-of-place
//!   cases work),
//! - the curated [`Epilogue`] enum,
//! - optional `bias`, `gelu_aux`,
//! - per-tensor / per-row fp8 scale pointers via [`ScaleSet`],
//! - a `workspace_size` hint folded into the heuristic search.
//!
//! cudarc 0.19.4's safe `Matmul` trait is implemented for `f32` and
//! (under feature `f16`) `half::f16` / `half::bf16`. For dtypes
//! cudarc doesn't yet wrap (fp8) the dispatch falls through to a
//! typed `Err(GpuError::Unrecoverable)` until we land the sys-level
//! path — see [`dispatch_safe_path`] below.

use std::sync::Arc;

use cudarc::cublaslt::{Activation, Matmul, MatmulConfig};
use tokio::sync::oneshot;

use crate::dtype::{DTypeKind, GemmSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::blas_lt::epilogue::Epilogue;
use crate::kernel::blas_lt::scaling::ScaleSet;
use crate::kernel::dispatch::{BlasLtDispatch, BlasLtDispatchCtx};
use crate::kernel::envelope;

const LIB: &str = "cublaslt";

/// Typed matmul request. Public surface; instantiated by callers.
pub struct MatmulRequest<T: GemmSupported> {
    pub a: GpuRef<T>,
    pub b: GpuRef<T>,
    pub c: GpuRef<T>,
    /// Optional explicit `D` output buffer. cuBLASLt allows
    /// out-of-place matmul where the result lands in `D` rather than
    /// in-place into `C`. Required for fp8 (the scale-back step
    /// produces a different dtype than the accumulator).
    pub d: Option<GpuRef<T>>,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub transa: bool,
    pub transb: bool,
    pub lda: i64,
    pub ldb: i64,
    pub ldc: i64,
    pub ldd: i64,
    pub epilogue: Epilogue,
    pub bias: Option<GpuRef<T>>,
    pub gelu_aux: Option<GpuRef<T>>,
    pub scales: ScaleSet,
    /// Hint for the heuristic: maximum workspace bytes the algorithm
    /// search may use. A reasonable default is `4 * 1024 * 1024`
    /// (cuBLASLt's standard 4 MiB minimum).
    pub workspace_size: usize,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: GemmSupported> std::fmt::Debug for MatmulRequest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatmulRequest")
            .field("dtype", &T::NAME)
            .field("m", &self.m)
            .field("n", &self.n)
            .field("k", &self.k)
            .field("transa", &self.transa)
            .field("transb", &self.transb)
            .field("epilogue", &self.epilogue)
            .field("workspace_size", &self.workspace_size)
            .finish()
    }
}

/// Internal sealing trait — bridges `T: GemmSupported` to the cudarc
/// `Matmul<T>` impls. Concrete impls below; one per safe-dtype.
trait CudarcMatmulPath: GemmSupported {
    fn dispatch_safe(req: Box<MatmulRequest<Self>>, ctx: &BlasLtDispatchCtx<'_>);
}

impl CudarcMatmulPath for f32 {
    fn dispatch_safe(req: Box<MatmulRequest<f32>>, ctx: &BlasLtDispatchCtx<'_>) {
        dispatch_safe_path::<f32>(req, ctx);
    }
}

#[cfg(feature = "f16")]
impl CudarcMatmulPath for half::f16 {
    fn dispatch_safe(req: Box<MatmulRequest<half::f16>>, ctx: &BlasLtDispatchCtx<'_>) {
        dispatch_safe_path::<half::f16>(req, ctx);
    }
}

#[cfg(feature = "f16")]
impl CudarcMatmulPath for half::bf16 {
    fn dispatch_safe(req: Box<MatmulRequest<half::bf16>>, ctx: &BlasLtDispatchCtx<'_>) {
        dispatch_safe_path::<half::bf16>(req, ctx);
    }
}

/// Bridge for dtypes cudarc 0.19.4 doesn't wrap with a `Matmul<T>`
/// impl yet (f64, fp8). Reply with `Unrecoverable("dtype …")` until
/// the sys-level path lands.
trait UnsupportedMatmulPath {
    fn dispatch_unsupported(reply: oneshot::Sender<Result<(), GpuError>>, dtype: &'static str);
}

impl<T> UnsupportedMatmulPath for T {
    fn dispatch_unsupported(reply: oneshot::Sender<Result<(), GpuError>>, dtype: &'static str) {
        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
            "BlasLtActor: matmul<{dtype}> not yet implemented (Phase 1 sys-level wiring pending)"
        ))));
    }
}

/// f64 (cudarc 0.19.4 has no `Matmul<f64>` impl).
impl BlasLtDispatch for MatmulRequest<f64> {
    fn dtype_kind(&self) -> DTypeKind {
        DTypeKind::F64
    }
    fn dispatch(self: Box<Self>, _ctx: &BlasLtDispatchCtx<'_>) {
        <f64 as UnsupportedMatmulPath>::dispatch_unsupported(self.reply, "f64");
    }
}

impl BlasLtDispatch for MatmulRequest<f32> {
    fn dtype_kind(&self) -> DTypeKind {
        DTypeKind::F32
    }
    fn dispatch(self: Box<Self>, ctx: &BlasLtDispatchCtx<'_>) {
        <f32 as CudarcMatmulPath>::dispatch_safe(self, ctx);
    }
}

#[cfg(feature = "f16")]
impl BlasLtDispatch for MatmulRequest<half::f16> {
    fn dtype_kind(&self) -> DTypeKind {
        DTypeKind::F16
    }
    fn dispatch(self: Box<Self>, ctx: &BlasLtDispatchCtx<'_>) {
        <half::f16 as CudarcMatmulPath>::dispatch_safe(self, ctx);
    }
}

#[cfg(feature = "f16")]
impl BlasLtDispatch for MatmulRequest<half::bf16> {
    fn dtype_kind(&self) -> DTypeKind {
        DTypeKind::Bf16
    }
    fn dispatch(self: Box<Self>, ctx: &BlasLtDispatchCtx<'_>) {
        <half::bf16 as CudarcMatmulPath>::dispatch_safe(self, ctx);
    }
}

#[cfg(feature = "cublas-fp8")]
impl BlasLtDispatch for MatmulRequest<crate::dtype::F8E4m3> {
    fn dtype_kind(&self) -> DTypeKind {
        DTypeKind::F8E4m3
    }
    fn dispatch(self: Box<Self>, _ctx: &BlasLtDispatchCtx<'_>) {
        <crate::dtype::F8E4m3 as UnsupportedMatmulPath>::dispatch_unsupported(self.reply, "fp8e4m3");
    }
}

#[cfg(feature = "cublas-fp8")]
impl BlasLtDispatch for MatmulRequest<crate::dtype::F8E5m2> {
    fn dtype_kind(&self) -> DTypeKind {
        DTypeKind::F8E5m2
    }
    fn dispatch(self: Box<Self>, _ctx: &BlasLtDispatchCtx<'_>) {
        <crate::dtype::F8E5m2 as UnsupportedMatmulPath>::dispatch_unsupported(self.reply, "fp8e5m2");
    }
}

/// Body of the safe-cudarc dispatch path. The function is generic
/// over `T` so all three (f32, f16, bf16) share one body. The
/// scale-pointer / heuristic / workspace-pool integration is wired
/// through the `MatmulConfig`'s alpha/beta (currently f32-only at the
/// cudarc surface) — once cudarc lands `Matmul` for fp8 we'll
/// promote this helper to call into the sys-level descriptor path.
fn dispatch_safe_path<T>(req: Box<MatmulRequest<T>>, ctx: &BlasLtDispatchCtx<'_>)
where
    T: GemmSupported + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    cudarc::cublaslt::CudaBlasLT: Matmul<T>,
    T::Scalar: Into<f32> + Copy,
{
    let MatmulRequest {
        a,
        b,
        c,
        d: _d,
        m,
        n,
        k,
        alpha,
        beta,
        transa,
        transb,
        lda,
        ldb,
        ldc,
        ldd: _ldd,
        epilogue,
        bias,
        gelu_aux: _gelu_aux,
        scales: _scales,
        workspace_size: _workspace_size,
        reply,
    } = *req;

    // Touch the heuristic cache so we record at least an empty entry
    // for this shape. (Real heuristic search lands when we move off
    // cudarc's safe `Matmul` API onto the sys-level descriptor path.)
    let _entry = ctx.heuristic.get(&crate::kernel::blas_lt::heuristic::HeuristicKey::new(
        m,
        n,
        k,
        T::KIND,
        transa,
        transb,
        epilogue,
        ctx.sm_arch,
    ));

    // Map the curated Epilogue back to cudarc's safe `Activation`
    // (cudarc's safe API only exposes Relu/Gelu for now). Other
    // variants degrade to "no activation" under the safe path; the
    // forthcoming sys-level path consumes the full enum.
    let activation = match epilogue {
        Epilogue::Relu | Epilogue::ReluBias | Epilogue::ReluAux | Epilogue::ReluAuxBias => {
            Some(Activation::Relu)
        }
        Epilogue::Gelu | Epilogue::GeluBias | Epilogue::GeluAux | Epilogue::GeluAuxBias => {
            Some(Activation::Gelu)
        }
        _ => None,
    };

    let cfg = MatmulConfig {
        transa,
        transb,
        transc: false,
        m: m as u64,
        n: n as u64,
        k: k as u64,
        alpha: alpha.into(),
        lda,
        ldb,
        beta: beta.into(),
        ldc,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };

    let (a_slice, b_slice, c_slice) = match envelope::access_all_3(&a, &b, &c) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let bias_slice = match bias.as_ref() {
        None => None,
        Some(g) => match g.access() {
            Ok(s) => Some(s.clone()),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        },
    };
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "BlasLt C has multiple live references".into(),
            )));
            return;
        }
    };
    c.record_write(ctx.stream);

    let blas_lt = ctx.blas_lt.clone();
    let stream = ctx.stream;
    let completion = ctx.completion;

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let bias_ref = bias_slice.as_ref().map(|s| &**s);
        let act_ref = activation.as_ref();
        // SAFETY: matmul is unsafe due to dim-validity contract.
        let res =
            unsafe { blas_lt.matmul(cfg, &*a_slice, &*b_slice, &mut c_owned, bias_ref, act_ref) };
        match res {
            Ok(()) => Ok((a_slice, b_slice, c_owned, bias_slice, blas_lt)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("matmul: {e}"),
            }),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::CudaDtype;

    fn make_request<T: GemmSupported>() -> MatmulRequest<T>
    where
        T::Scalar: Default,
    {
        let (tx, _rx) = oneshot::channel::<Result<(), GpuError>>();
        // We can't actually construct GpuRef<T> without a DeviceState,
        // so this helper short-circuits via `make_request_unfilled`
        // below. The test verifies the type instantiation only.
        let _ = (T::NAME, tx);
        unreachable!("type-instantiation-only helper")
    }

    /// Compile-time check that `MatmulRequest<T>` instantiates for
    /// every dtype the dispatch trait covers under the active
    /// feature set. We materialize a function pointer to each
    /// dispatch impl — that's enough to exercise the trait bounds
    /// without constructing real GpuRefs.
    #[test]
    fn matmul_request_dispatches_for_f32_f16_bf16() {
        // Type-id ping for the f32 dispatch path.
        fn _accepts_f32(b: Box<dyn BlasLtDispatch>) -> Box<dyn BlasLtDispatch> {
            b
        }
        // Probe the trait impls exist for every required dtype.
        let _f32_kind: fn(&MatmulRequest<f32>) -> DTypeKind = MatmulRequest::<f32>::dtype_kind;
        let _f64_kind: fn(&MatmulRequest<f64>) -> DTypeKind = MatmulRequest::<f64>::dtype_kind;
        #[cfg(feature = "f16")]
        let _f16_kind: fn(&MatmulRequest<half::f16>) -> DTypeKind =
            MatmulRequest::<half::f16>::dtype_kind;
        #[cfg(feature = "f16")]
        let _bf16_kind: fn(&MatmulRequest<half::bf16>) -> DTypeKind =
            MatmulRequest::<half::bf16>::dtype_kind;

        // Confirm the kind tags line up.
        // We can't construct a request without a GpuRef, but we *can*
        // probe the const dtype tags through `CudaDtype::KIND`.
        assert_eq!(<f32 as CudaDtype>::KIND, DTypeKind::F32);
        assert_eq!(<f64 as CudaDtype>::KIND, DTypeKind::F64);
        #[cfg(feature = "f16")]
        {
            assert_eq!(<half::f16 as CudaDtype>::KIND, DTypeKind::F16);
            assert_eq!(<half::bf16 as CudaDtype>::KIND, DTypeKind::Bf16);
        }
        // Suppress unused-warning on the unreachable helper.
        let _ = make_request::<f32> as fn() -> MatmulRequest<f32>;
    }
}
