//! `AccelDtype` — backend-agnostic numeric data-type trait.
//!
//! Every numeric kernel actor in atomr-accel works over `T: AccelDtype`
//! so allocation, copy, and op messages can be dtype-generic without
//! exploding into one variant per (op, dtype) pair.
//!
//! The trait captures only what every backend agrees on (size, identity
//! values, NaN, a discriminant for runtime dispatch). Backend-specific
//! traits like `CudaDtype` layer on top with the binding-specific
//! mappings (cudarc enums for CUDA, Metal `MTLDataType` for Metal,
//! `hipblasDatatype_t` for ROCm, …).

use std::fmt::Debug;

/// Marker for any numeric type that can be a typed device buffer
/// element across atomr-accel backends.
///
/// `AccelDtype` is intentionally *narrower* than what individual
/// backends can support. cuBLAS f64 GEMM is gated by a `GemmSupported`
/// marker on the CUDA side; this trait says only that the type is a
/// recognised dtype.
pub trait AccelDtype: Copy + Send + Sync + 'static + Debug {
    /// Companion scalar type for host-side parameters (alpha/beta,
    /// mean/std, scaling factors). `Self` for full-precision dtypes;
    /// `f32` for fp8 / fp4 wrappers because the upstream APIs accept
    /// f32 scales.
    type Scalar: Copy + Send + Sync + 'static + Debug;

    /// Runtime discriminant.
    const KIND: DType;

    /// Bytes per element including representation padding.
    const SIZE: usize;

    /// Human-readable name used in tracing and error messages.
    const NAME: &'static str;

    fn zero() -> Self;
    fn one() -> Self;

    /// `Some(NaN)` for floats, `None` for integers.
    fn nan() -> Option<Self>;
}

/// Compact discriminant for [`AccelDtype::KIND`].
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[non_exhaustive]
pub enum DType {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    /// 8-bit float, E4M3 (sign 1, exp 4, mant 3). Hopper+ fp8 GEMM, FlashAttention v3.
    F8E4m3,
    /// 8-bit float, E5M2 (sign 1, exp 5, mant 2).
    F8E5m2,
    /// 4-bit float, E2M1. Blackwell fp4 inference.
    F4E2m1,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::F8E4m3 | DType::F8E5m2 | DType::F4E2m1 => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::F8E4m3 => "f8_e4m3",
            DType::F8E5m2 => "f8_e5m2",
            DType::F4E2m1 => "f4_e2m1",
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F32
                | DType::F64
                | DType::F16
                | DType::Bf16
                | DType::F8E4m3
                | DType::F8E5m2
                | DType::F4E2m1
        )
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        )
    }

    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::F32
                | DType::F64
                | DType::F16
                | DType::Bf16
                | DType::F8E4m3
                | DType::F8E5m2
                | DType::F4E2m1
        )
    }
}

macro_rules! impl_accel_dtype_int {
    ($t:ty, $kind:expr, $name:literal) => {
        impl AccelDtype for $t {
            type Scalar = Self;
            const KIND: DType = $kind;
            const SIZE: usize = std::mem::size_of::<Self>();
            const NAME: &'static str = $name;

            #[inline]
            fn zero() -> Self {
                0
            }
            #[inline]
            fn one() -> Self {
                1
            }
            #[inline]
            fn nan() -> Option<Self> {
                None
            }
        }
    };
}

macro_rules! impl_accel_dtype_float {
    ($t:ty, $kind:expr, $name:literal) => {
        impl AccelDtype for $t {
            type Scalar = Self;
            const KIND: DType = $kind;
            const SIZE: usize = std::mem::size_of::<Self>();
            const NAME: &'static str = $name;

            #[inline]
            fn zero() -> Self {
                0.0
            }
            #[inline]
            fn one() -> Self {
                1.0
            }
            #[inline]
            fn nan() -> Option<Self> {
                Some(<$t>::NAN)
            }
        }
    };
}

impl_accel_dtype_float!(f32, DType::F32, "f32");
impl_accel_dtype_float!(f64, DType::F64, "f64");
impl_accel_dtype_int!(i8, DType::I8, "i8");
impl_accel_dtype_int!(i16, DType::I16, "i16");
impl_accel_dtype_int!(i32, DType::I32, "i32");
impl_accel_dtype_int!(i64, DType::I64, "i64");
impl_accel_dtype_int!(u8, DType::U8, "u8");
impl_accel_dtype_int!(u16, DType::U16, "u16");
impl_accel_dtype_int!(u32, DType::U32, "u32");
impl_accel_dtype_int!(u64, DType::U64, "u64");

#[cfg(feature = "f16")]
impl AccelDtype for half::f16 {
    type Scalar = Self;
    const KIND: DType = DType::F16;
    const SIZE: usize = std::mem::size_of::<Self>();
    const NAME: &'static str = "f16";
    #[inline]
    fn zero() -> Self {
        half::f16::ZERO
    }
    #[inline]
    fn one() -> Self {
        half::f16::ONE
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(half::f16::NAN)
    }
}

#[cfg(feature = "f16")]
impl AccelDtype for half::bf16 {
    type Scalar = Self;
    const KIND: DType = DType::Bf16;
    const SIZE: usize = std::mem::size_of::<Self>();
    const NAME: &'static str = "bf16";
    #[inline]
    fn zero() -> Self {
        half::bf16::ZERO
    }
    #[inline]
    fn one() -> Self {
        half::bf16::ONE
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(half::bf16::NAN)
    }
}

/// 8-bit float, E4M3 layout. Storage is one byte; conversions to/from
/// f32 are saturating.
#[cfg(feature = "f8")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F8E4m3(pub u8);

#[cfg(feature = "f8")]
impl F8E4m3 {
    pub const ZERO: Self = F8E4m3(0x00);
    pub const ONE: Self = F8E4m3(0x38);
    pub const NAN: Self = F8E4m3(0x7f);

    /// Saturating round-to-nearest-even conversion from f32.
    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self::NAN;
        }
        let max = 448.0_f32;
        let clamped = x.clamp(-max, max);
        let bits = clamped.to_bits();
        let sign = ((bits >> 31) as u8) << 7;
        let abs = clamped.abs();
        if abs == 0.0 {
            return F8E4m3(sign);
        }
        let f32_exp = ((bits >> 23) & 0xff) as i32 - 127;
        let f32_mant = bits & 0x007f_ffff;
        let e4_exp = f32_exp + 7;
        if e4_exp <= 0 {
            let shift = 21 + (1 - e4_exp) as u32;
            let m = ((f32_mant | 0x0080_0000) >> shift) as u8;
            return F8E4m3(sign | (m & 0x07));
        }
        let mant = (f32_mant >> 20) as u8;
        let round_bit = ((f32_mant >> 19) & 1) as u8;
        let sticky = ((f32_mant & 0x0007_ffff) != 0) as u8;
        let mut e = e4_exp as u8;
        let mut m = mant & 0x07;
        if round_bit == 1 && (sticky == 1 || (m & 1) == 1) {
            m = m.wrapping_add(1);
            if m == 0x08 {
                m = 0;
                e = e.wrapping_add(1);
            }
        }
        if e >= 0x0f {
            return F8E4m3(sign | 0x7e);
        }
        F8E4m3(sign | (e << 3) | m)
    }

    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = (self.0 >> 3) & 0x0f;
        let mant = self.0 & 0x07;
        if exp == 0 && mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        if exp == 0x0f && mant == 0x07 {
            return f32::NAN;
        }
        let (e, m) = if exp == 0 {
            let lz = (mant.leading_zeros() as i32) - 5;
            (1 - 7 - lz, ((mant as u32) << (lz + 1)) & 0x07)
        } else {
            (exp as i32 - 7, mant as u32)
        };
        let bits = ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 20);
        f32::from_bits(bits)
    }
}

#[cfg(feature = "f8")]
impl AccelDtype for F8E4m3 {
    type Scalar = f32;
    const KIND: DType = DType::F8E4m3;
    const SIZE: usize = 1;
    const NAME: &'static str = "f8_e4m3";
    #[inline]
    fn zero() -> Self {
        F8E4m3::ZERO
    }
    #[inline]
    fn one() -> Self {
        F8E4m3::ONE
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(F8E4m3::NAN)
    }
}

/// 8-bit float, E5M2 layout. Storage is one byte; conversions to/from
/// f32 are saturating.
#[cfg(feature = "f8")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F8E5m2(pub u8);

#[cfg(feature = "f8")]
impl F8E5m2 {
    pub const ZERO: Self = F8E5m2(0x00);
    pub const ONE: Self = F8E5m2(0x3c);
    pub const NAN: Self = F8E5m2(0x7e);
    pub const INFINITY: Self = F8E5m2(0x7c);

    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self::NAN;
        }
        let bits = x.to_bits();
        let sign = ((bits >> 31) as u8) << 7;
        let f32_exp = ((bits >> 23) & 0xff) as i32 - 127;
        let f32_mant = bits & 0x007f_ffff;
        if x == 0.0 {
            return F8E5m2(sign);
        }
        let e5_exp = f32_exp + 15;
        if e5_exp >= 0x1f {
            return F8E5m2(sign | 0x7c);
        }
        if e5_exp <= 0 {
            let shift = 22 + (1 - e5_exp) as u32;
            let m = ((f32_mant | 0x0080_0000) >> shift) as u8;
            return F8E5m2(sign | (m & 0x03));
        }
        let mant = (f32_mant >> 21) as u8;
        let round_bit = ((f32_mant >> 20) & 1) as u8;
        let sticky = ((f32_mant & 0x000f_ffff) != 0) as u8;
        let mut e = e5_exp as u8;
        let mut m = mant & 0x03;
        if round_bit == 1 && (sticky == 1 || (m & 1) == 1) {
            m = m.wrapping_add(1);
            if m == 0x04 {
                m = 0;
                e = e.wrapping_add(1);
            }
        }
        if e >= 0x1f {
            return F8E5m2(sign | 0x7c);
        }
        F8E5m2(sign | (e << 2) | m)
    }

    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = (self.0 >> 2) & 0x1f;
        let mant = self.0 & 0x03;
        if exp == 0 && mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        if exp == 0x1f {
            return if mant == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }
        let (e, m) = if exp == 0 {
            let lz = (mant.leading_zeros() as i32) - 6;
            (1 - 15 - lz, ((mant as u32) << (lz + 1)) & 0x03)
        } else {
            (exp as i32 - 15, mant as u32)
        };
        let bits = ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 21);
        f32::from_bits(bits)
    }
}

#[cfg(feature = "f8")]
impl AccelDtype for F8E5m2 {
    type Scalar = f32;
    const KIND: DType = DType::F8E5m2;
    const SIZE: usize = 1;
    const NAME: &'static str = "f8_e5m2";
    #[inline]
    fn zero() -> Self {
        F8E5m2::ZERO
    }
    #[inline]
    fn one() -> Self {
        F8E5m2::ONE
    }
    #[inline]
    fn nan() -> Option<Self> {
        Some(F8E5m2::NAN)
    }
}

/// 4-bit float, E2M1 layout. The byte stores one element in the low
/// nibble; the upper nibble is zero. Two F4E2m1 values are commonly
/// packed into one byte by the kernel layer — that packing is not the
/// concern of this newtype.
#[cfg(feature = "f4")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F4E2m1(pub u8);

#[cfg(feature = "f4")]
impl F4E2m1 {
    pub const ZERO: Self = F4E2m1(0x0);
    pub const ONE: Self = F4E2m1(0x4);

    pub fn to_f32(self) -> f32 {
        let n = self.0 & 0x0f;
        let sign = if (n >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        let exp = (n >> 1) & 0x03;
        let mant = n & 0x01;
        let value = match (exp, mant) {
            (0, 0) => 0.0,
            (0, 1) => 0.5,
            (e, m) => {
                let mantissa = 1.0 + (m as f32) * 0.5;
                mantissa * 2.0_f32.powi(e as i32 - 1)
            }
        };
        sign * value
    }
}

#[cfg(feature = "f4")]
impl AccelDtype for F4E2m1 {
    type Scalar = f32;
    const KIND: DType = DType::F4E2m1;
    const SIZE: usize = 1;
    const NAME: &'static str = "f4_e2m1";
    #[inline]
    fn zero() -> Self {
        F4E2m1::ZERO
    }
    #[inline]
    fn one() -> Self {
        F4E2m1::ONE
    }
    #[inline]
    fn nan() -> Option<Self> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_size_matches_trait() {
        assert_eq!(<f32 as AccelDtype>::SIZE, DType::F32.size_bytes());
        assert_eq!(<f64 as AccelDtype>::SIZE, DType::F64.size_bytes());
        assert_eq!(<i8 as AccelDtype>::SIZE, DType::I8.size_bytes());
        assert_eq!(<i32 as AccelDtype>::SIZE, DType::I32.size_bytes());
        assert_eq!(<u32 as AccelDtype>::SIZE, DType::U32.size_bytes());
        assert_eq!(<u64 as AccelDtype>::SIZE, DType::U64.size_bytes());
    }

    #[test]
    fn dtype_classifiers() {
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());
        assert!(DType::I32.is_integer());
        assert!(DType::I32.is_signed());
        assert!(!DType::U32.is_signed());
        assert!(DType::F32.is_signed());
    }

    #[test]
    fn dtype_names_match() {
        assert_eq!(DType::F32.name(), <f32 as AccelDtype>::NAME);
        assert_eq!(DType::F64.name(), <f64 as AccelDtype>::NAME);
        assert_eq!(DType::U8.name(), <u8 as AccelDtype>::NAME);
    }

    #[test]
    fn float_nan_is_some() {
        assert!(<f32 as AccelDtype>::nan().is_some());
        assert!(<f64 as AccelDtype>::nan().is_some());
    }

    #[test]
    fn integer_nan_is_none() {
        assert!(<i32 as AccelDtype>::nan().is_none());
        assert!(<u64 as AccelDtype>::nan().is_none());
    }

    #[test]
    fn zero_one_round_trip() {
        assert_eq!(<f32 as AccelDtype>::zero(), 0.0);
        assert_eq!(<f32 as AccelDtype>::one(), 1.0);
        assert_eq!(<i32 as AccelDtype>::zero(), 0);
        assert_eq!(<i32 as AccelDtype>::one(), 1);
    }

    #[cfg(feature = "f8")]
    #[test]
    fn f8e4m3_round_trip_simple() {
        assert_eq!(F8E4m3::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(F8E4m3::from_f32(1.0).to_f32(), 1.0);
        assert_eq!(F8E4m3::from_f32(2.0).to_f32(), 2.0);
        assert_eq!(F8E4m3::from_f32(-1.0).to_f32(), -1.0);
    }

    #[cfg(feature = "f8")]
    #[test]
    fn f8e5m2_round_trip_simple() {
        assert_eq!(F8E5m2::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(F8E5m2::from_f32(1.0).to_f32(), 1.0);
        assert_eq!(F8E5m2::from_f32(2.0).to_f32(), 2.0);
        assert_eq!(F8E5m2::from_f32(-1.0).to_f32(), -1.0);
    }
}
