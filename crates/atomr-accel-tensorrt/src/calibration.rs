//! INT8 PTQ calibration helpers.
//!
//! Two algorithms are exposed: entropy (Karpathy / kullback-leibler)
//! and minmax. Both implement the `Calibrator` trait below, which the
//! `TrtActor::Build` path wires into `IBuilderConfig::setInt8Calibrator`
//! under the `tensorrt-link` feature.
//!
//! When the `tensorrt-fp8` feature is also on, the same trait is used
//! for FP8 PTQ — the only difference is the dtype announced to the
//! builder.

#![cfg(feature = "tensorrt-int8")]

use std::sync::Arc;

use crate::error::TrtError;

/// Algorithm dispatch tag — mirrors `nvinfer1::CalibrationAlgoType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationAlgo {
    /// `IInt8EntropyCalibrator2` — recommended for most networks.
    EntropyV2,
    /// `IInt8EntropyCalibrator` — legacy KL-divergence.
    Entropy,
    /// `IInt8MinMaxCalibrator` — used for transformer/attention nets.
    MinMax,
    /// `IInt8LegacyCalibrator` — pre-TRT5; rarely needed.
    Legacy,
}

/// Trait every calibrator implements. The actor reads `next_batch`
/// repeatedly until it returns `None`, then optionally
/// reads/writes the calibration cache.
pub trait Calibrator: Send + Sync {
    /// Algorithm tag — selects the concrete TensorRT class.
    fn algorithm(&self) -> CalibrationAlgo;

    /// Pull the next calibration batch. Returns `None` when the
    /// dataset is exhausted. Each `(name, device_ptr, bytes)` tuple
    /// names an input tensor, a CUDA device address, and the
    /// in-bytes batch size.
    fn next_batch(&mut self) -> Option<Vec<CalibrationBinding>>;

    /// Read the calibration cache (if any). Returning `None` forces a
    /// fresh pass.
    fn read_cache(&self) -> Option<Vec<u8>> {
        None
    }

    /// Persist a calibration cache produced by the previous pass.
    fn write_cache(&mut self, _blob: &[u8]) {}
}

/// One input tensor for a single calibration batch.
#[derive(Debug, Clone)]
pub struct CalibrationBinding {
    pub name: String,
    pub device_ptr: u64,
    pub bytes: usize,
}

/// Min-max calibrator. Stores per-batch device pointers + an
/// optional persistent cache.
pub struct MinMaxCalibrator {
    batches: Vec<Vec<CalibrationBinding>>,
    cursor: usize,
    cache: Option<Vec<u8>>,
}

impl MinMaxCalibrator {
    pub fn new(batches: Vec<Vec<CalibrationBinding>>) -> Self {
        Self {
            batches,
            cursor: 0,
            cache: None,
        }
    }

    pub fn with_cache(mut self, blob: Vec<u8>) -> Self {
        self.cache = Some(blob);
        self
    }

    pub fn into_arc(self) -> Arc<parking_lot::Mutex<dyn Calibrator>> {
        Arc::new(parking_lot::Mutex::new(self))
    }
}

impl Calibrator for MinMaxCalibrator {
    fn algorithm(&self) -> CalibrationAlgo {
        CalibrationAlgo::MinMax
    }

    fn next_batch(&mut self) -> Option<Vec<CalibrationBinding>> {
        if self.cursor >= self.batches.len() {
            None
        } else {
            let b = self.batches[self.cursor].clone();
            self.cursor += 1;
            Some(b)
        }
    }

    fn read_cache(&self) -> Option<Vec<u8>> {
        self.cache.clone()
    }

    fn write_cache(&mut self, blob: &[u8]) {
        self.cache = Some(blob.to_vec());
    }
}

/// Entropy calibrator (KL-divergence). Same shape as `MinMaxCalibrator`
/// but reports `EntropyV2` so the C++ side instantiates the matching
/// class.
pub struct EntropyCalibrator {
    batches: Vec<Vec<CalibrationBinding>>,
    cursor: usize,
    cache: Option<Vec<u8>>,
    legacy: bool,
}

impl EntropyCalibrator {
    pub fn new(batches: Vec<Vec<CalibrationBinding>>) -> Self {
        Self {
            batches,
            cursor: 0,
            cache: None,
            legacy: false,
        }
    }

    /// Use the legacy `IInt8EntropyCalibrator` (V1) instead of V2.
    pub fn legacy(mut self) -> Self {
        self.legacy = true;
        self
    }
}

impl Calibrator for EntropyCalibrator {
    fn algorithm(&self) -> CalibrationAlgo {
        if self.legacy {
            CalibrationAlgo::Entropy
        } else {
            CalibrationAlgo::EntropyV2
        }
    }

    fn next_batch(&mut self) -> Option<Vec<CalibrationBinding>> {
        if self.cursor >= self.batches.len() {
            None
        } else {
            let b = self.batches[self.cursor].clone();
            self.cursor += 1;
            Some(b)
        }
    }

    fn read_cache(&self) -> Option<Vec<u8>> {
        self.cache.clone()
    }

    fn write_cache(&mut self, blob: &[u8]) {
        self.cache = Some(blob.to_vec());
    }
}

/// FP8 PTQ calibrator. Behaves identically to `MinMaxCalibrator` but
/// is wrapped in a separate type so `TrtActor::Build` can route it to
/// `BuilderFlag::FP8` instead of `INT8`.
#[cfg(feature = "tensorrt-fp8")]
pub struct Fp8Calibrator {
    batches: Vec<Vec<CalibrationBinding>>,
    cursor: usize,
}

#[cfg(feature = "tensorrt-fp8")]
impl Fp8Calibrator {
    pub fn new(batches: Vec<Vec<CalibrationBinding>>) -> Self {
        Self {
            batches,
            cursor: 0,
        }
    }
}

#[cfg(feature = "tensorrt-fp8")]
impl Calibrator for Fp8Calibrator {
    fn algorithm(&self) -> CalibrationAlgo {
        // FP8 PTQ rides on the entropy-V2 algorithm in TRT 9+.
        CalibrationAlgo::EntropyV2
    }

    fn next_batch(&mut self) -> Option<Vec<CalibrationBinding>> {
        if self.cursor >= self.batches.len() {
            None
        } else {
            let b = self.batches[self.cursor].clone();
            self.cursor += 1;
            Some(b)
        }
    }
}

/// Helper: poll a calibrator to completion, accumulating cache bytes.
/// Useful for testing the trait surface without any FFI.
pub fn drain<C: Calibrator>(c: &mut C) -> Result<usize, TrtError> {
    let mut count = 0usize;
    while let Some(batch) = c.next_batch() {
        count += batch.len();
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n: usize) -> Vec<Vec<CalibrationBinding>> {
        (0..n)
            .map(|i| {
                vec![CalibrationBinding {
                    name: "input".into(),
                    device_ptr: 0xCAFE_0000 + i as u64,
                    bytes: 1024 * (i + 1),
                }]
            })
            .collect()
    }

    #[test]
    fn int8_minmax_calibrator_constructs() {
        let mut c = MinMaxCalibrator::new(fixture(3));
        assert_eq!(c.algorithm(), CalibrationAlgo::MinMax);
        assert_eq!(drain(&mut c).unwrap(), 3);
        // After exhaustion, repeated polls return None.
        assert!(c.next_batch().is_none());

        c.write_cache(&[0xAB, 0xCD]);
        assert_eq!(c.read_cache().as_deref(), Some(&[0xAB, 0xCD][..]));
    }

    #[test]
    fn entropy_v2_default_and_legacy() {
        let c = EntropyCalibrator::new(fixture(1));
        assert_eq!(c.algorithm(), CalibrationAlgo::EntropyV2);
        let c = EntropyCalibrator::new(fixture(1)).legacy();
        assert_eq!(c.algorithm(), CalibrationAlgo::Entropy);
    }

    #[cfg(feature = "tensorrt-fp8")]
    #[test]
    fn fp8_calibrator_uses_entropy_v2() {
        let c = Fp8Calibrator::new(fixture(2));
        assert_eq!(c.algorithm(), CalibrationAlgo::EntropyV2);
    }
}
