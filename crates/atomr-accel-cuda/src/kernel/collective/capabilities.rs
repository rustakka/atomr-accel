//! Runtime probe for NCCL capabilities.
//!
//! Surfaces version + opt-in feature gates: fp8 reduction (NCCL >=
//! 2.20), NVLS (NCCL >= 2.18 on supported topologies), SHARP. The
//! probe is best-effort: if NCCL isn't loadable on this host (e.g.
//! a CPU-only CI runner), the probe returns
//! [`NcclCapabilities::zeroed`] rather than panicking.

/// Static description of the loaded NCCL library's capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NcclCapabilities {
    /// `(major, minor, patch)`. `(0, 0, 0)` if NCCL isn't loadable.
    pub version: (i32, i32, i32),
    /// True iff `nccl-fp8` feature is enabled and NCCL >= 2.20.
    pub has_fp8: bool,
    /// True iff `nccl-nvls` feature is enabled. Whether NVLS is
    /// actually usable depends on topology — this flag indicates
    /// only that the build path is compiled in.
    pub has_nvls: bool,
    /// SHARP support is reported via NCCL_NET_PLUGIN; this probe
    /// reports `false` until we wire the plugin query.
    pub has_sharp: bool,
}

impl NcclCapabilities {
    /// All-zero capabilities — the value returned when NCCL isn't
    /// initialised on this host.
    pub fn zeroed() -> Self {
        Self::default()
    }
}

/// Best-effort capability probe. Calls `ncclGetVersion` via cudarc's
/// safe wrapper; on any error returns [`NcclCapabilities::zeroed`].
pub fn probe_capabilities() -> NcclCapabilities {
    let version_int =
        std::panic::catch_unwind(cudarc::nccl::result::get_nccl_version).unwrap_or(Ok(0));
    let v = match version_int {
        Ok(v) => v,
        Err(_) => return NcclCapabilities::zeroed(),
    };
    if v == 0 {
        return NcclCapabilities::zeroed();
    }
    // NCCL packs version as MAJOR*10000 + MINOR*100 + PATCH (NCCL >= 2.9)
    // or MAJOR*1000 + MINOR*100 + PATCH (older). Detect by magnitude.
    let (major, minor, patch) = if v >= 20000 {
        (v / 10000, (v / 100) % 100, v % 100)
    } else {
        (v / 1000, (v / 100) % 10, v % 100)
    };

    let supports_fp8 = (major, minor) >= (2, 20);

    NcclCapabilities {
        version: (major, minor, patch),
        has_fp8: cfg!(feature = "nccl-fp8") && supports_fp8,
        has_nvls: cfg!(feature = "nccl-nvls"),
        has_sharp: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a host without a working NCCL install, `probe_capabilities`
    /// must not panic — it must return `zeroed()`.
    #[test]
    fn probe_returns_zeroed_when_nccl_uninit() {
        // Whatever the host has, the probe must succeed without
        // panicking and either return zeros (no NCCL) or a real
        // version. Both shapes are acceptable; we only assert the
        // call returns.
        let caps = probe_capabilities();
        if caps.version == (0, 0, 0) {
            assert_eq!(caps, NcclCapabilities::zeroed());
        } else {
            // Real NCCL: version major must be sane (>=2 in practice
            // but we accept >=1 to avoid version-pinning the test).
            assert!(caps.version.0 >= 1);
        }
    }
}
