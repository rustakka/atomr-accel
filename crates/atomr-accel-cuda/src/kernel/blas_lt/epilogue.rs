//! `Epilogue` enum — atomr-accel's curated mapping over cuBLASLt's
//! `cublasLtEpilogue_t`.
//!
//! cuBLASLt fuses post-matmul ops (bias add, activation, gradient
//! aux/preact) into the kernel itself. The full set is large and
//! version-dependent; we expose the variants that matter for
//! transformer training/inference: bias, ReLU/GeLU forward + aux,
//! ReLU/GeLU backward (`drelu`/`dgelu`) with optional bias gradient,
//! and the `BGRADA`/`BGRADB` reduction-only variants used by mixed
//! optimizer/data-parallel pipelines.
//!
//! Cache key compatibility: `Epilogue` derives `Hash + Eq` so
//! `HeuristicKey` can fold it into the `(m,n,k,dtype,layout,
//! epilogue,arch)` cache without a custom `impl`.

use cudarc::cublaslt::sys::cublasLtEpilogue_t;

/// Curated epilogue matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Epilogue {
    /// No fused activation, no bias. Identity epilogue.
    None,
    /// Fused ReLU.
    Relu,
    /// Bias-add only.
    Bias,
    /// Bias-add + ReLU.
    ReluBias,
    /// ReLU forward storing the masked-input "aux" tensor for the
    /// matching backward pass.
    ReluAux,
    /// ReLU forward + bias + aux storage.
    ReluAuxBias,
    /// Fused GeLU.
    Gelu,
    /// GeLU forward storing the unactivated preact for backward.
    GeluAux,
    /// Bias-add + GeLU.
    GeluBias,
    /// Bias-add + GeLU forward + aux storage.
    GeluAuxBias,
    /// ReLU backward (`drelu`) — gradient w.r.t. the preact.
    DRelu,
    /// ReLU backward + reduce-sum producing bias gradient.
    DReluBgrad,
    /// GeLU backward (`dgelu`).
    DGelu,
    /// GeLU backward + reduce-sum producing bias gradient.
    DGeluBgrad,
    /// Reduction-only along the `A` (M) dimension — produces the
    /// bias-grad without any activation.
    BgradA,
    /// Reduction-only along the `B` (N) dimension.
    BgradB,
}

impl Epilogue {
    /// Map to the cuBLASLt sys-level enum value.
    pub fn to_cublas(self) -> cublasLtEpilogue_t {
        match self {
            Self::None => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            Self::Relu => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
            Self::Bias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            Self::ReluBias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
            Self::ReluAux => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX,
            Self::ReluAuxBias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
            Self::Gelu => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            Self::GeluAux => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX,
            Self::GeluBias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
            Self::GeluAuxBias => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
            // `CUBLASLT_EPILOGUE_DRELU`/`_DGELU` (pure backward, no
            // bias-grad reduction) are only present on cudarc's
            // CUDA ≥ 11.6 cfg branch, which we can't easily detect
            // from this crate. Map both pure-backward variants to the
            // BGRAD form — callers wanting the pure-backward output
            // simply ignore the bias-grad side output.
            Self::DRelu | Self::DReluBgrad => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DRELU_BGRAD,
            Self::DGelu | Self::DGeluBgrad => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DGELU_BGRAD,
            Self::BgradA => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADA,
            Self::BgradB => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADB,
        }
    }

    /// Does this epilogue read or write a bias vector?
    pub fn uses_bias(self) -> bool {
        matches!(
            self,
            Self::Bias
                | Self::ReluBias
                | Self::ReluAuxBias
                | Self::GeluBias
                | Self::GeluAuxBias
        )
    }

    /// Does this epilogue store or consume an `epilogue_aux` tensor
    /// (the activation preact / mask used by the matching backward)?
    pub fn uses_aux(self) -> bool {
        matches!(
            self,
            Self::ReluAux
                | Self::ReluAuxBias
                | Self::GeluAux
                | Self::GeluAuxBias
                | Self::DRelu
                | Self::DReluBgrad
                | Self::DGelu
                | Self::DGeluBgrad
        )
    }

    /// Does this epilogue produce a bias gradient as a side output
    /// (`BGRADA`, `BGRADB`, or `D{Relu,Gelu}_BGRAD`)?
    pub fn produces_bias_grad(self) -> bool {
        matches!(
            self,
            Self::BgradA | Self::BgradB | Self::DReluBgrad | Self::DGeluBgrad
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip every variant through `to_cublas()` and check it
    /// maps to a non-zero, non-default sys-level value (or is `None`,
    /// which is the only legitimate `DEFAULT`).
    #[test]
    fn epilogue_round_trip() {
        let cases = [
            (Epilogue::None, cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT),
            (Epilogue::Relu, cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU),
            (Epilogue::Bias, cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS),
            (
                Epilogue::ReluBias,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
            ),
            (
                Epilogue::ReluAux,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX,
            ),
            (
                Epilogue::ReluAuxBias,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
            ),
            (Epilogue::Gelu, cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU),
            (
                Epilogue::GeluAux,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX,
            ),
            (
                Epilogue::GeluBias,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
            ),
            (
                Epilogue::GeluAuxBias,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
            ),
            (
                Epilogue::DReluBgrad,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DRELU_BGRAD,
            ),
            (
                Epilogue::DGeluBgrad,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DGELU_BGRAD,
            ),
            (
                Epilogue::BgradA,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADA,
            ),
            (
                Epilogue::BgradB,
                cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADB,
            ),
        ];
        for (lhs, rhs) in cases {
            assert_eq!(lhs.to_cublas(), rhs, "{lhs:?}");
        }
    }

    #[test]
    fn epilogue_capability_flags() {
        assert!(Epilogue::Bias.uses_bias());
        assert!(!Epilogue::None.uses_bias());
        assert!(Epilogue::ReluBias.uses_bias());
        assert!(Epilogue::GeluAux.uses_aux());
        assert!(Epilogue::DReluBgrad.produces_bias_grad());
        assert!(Epilogue::BgradA.produces_bias_grad());
        assert!(!Epilogue::Relu.produces_bias_grad());
    }

    #[test]
    fn epilogue_default_is_none_variant() {
        // Sanity: ensure default discriminant equality holds across
        // cudarc cuda-version cfgs (both 11.x and 12.x bindings emit
        // CUBLASLT_EPILOGUE_DEFAULT = 1).
        assert_eq!(
            Epilogue::None.to_cublas() as u32,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT as u32
        );
    }
}
