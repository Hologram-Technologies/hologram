//! RingWord: the carrier type trait for ring elements.
//!
//! Implemented for u8, u16, u32, u64, u128. Each method compiles to
//! a single ALU instruction. All methods are #[inline].

/// The carrier type for a ring element at a given quantum level.
///
/// Every method delegates to Rust's built-in wrapping arithmetic intrinsics.
/// Zero-cost: monomorphized at compile time, inlined to single ALU instructions.
pub trait RingWord:
    Copy
    + Eq
    + Ord
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::BitXor<Output = Self>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitOr<Output = Self>
    + core::ops::Not<Output = Self>
{
    /// The zero element (additive identity).
    const ZERO: Self;
    /// The one element (multiplicative identity).
    const ONE: Self;
    /// The maximum element: 2^BITS - 1.
    const MAX: Self;
    /// Number of bits in this word.
    const BITS: u32;

    /// Wrapping negation: `(-self) mod 2^BITS`.
    fn wrapping_neg(self) -> Self;
    /// Wrapping addition: `(self + other) mod 2^BITS`.
    fn wrapping_add(self, other: Self) -> Self;
    /// Wrapping subtraction: `(self - other) mod 2^BITS`.
    fn wrapping_sub(self, other: Self) -> Self;
    /// Wrapping multiplication: `(self * other) mod 2^BITS`.
    fn wrapping_mul(self, other: Self) -> Self;
    /// Population count (Hamming weight).
    fn count_ones(self) -> u32;
    /// Number of leading zeros.
    fn leading_zeros(self) -> u32;
    /// Number of trailing zeros.
    fn trailing_zeros(self) -> u32;
    /// Convert from u64 (truncating).
    fn from_u64(v: u64) -> Self;
    /// Convert to u64 (zero-extending).
    fn to_u64(self) -> u64;
    /// Convert to u128 (zero-extending). Used by `Address::canonical_bytes`
    /// per ADR-052 / Amendment 43 §2 — works uniformly for u8..u128.
    fn to_u128_le(self) -> u128;
}

macro_rules! impl_ring_word {
    ($ty:ty) => {
        impl RingWord for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MAX: Self = <$ty>::MAX;
            const BITS: u32 = <$ty>::BITS;

            #[inline]
            fn wrapping_neg(self) -> Self {
                <$ty>::wrapping_neg(self)
            }
            #[inline]
            fn wrapping_add(self, other: Self) -> Self {
                <$ty>::wrapping_add(self, other)
            }
            #[inline]
            fn wrapping_sub(self, other: Self) -> Self {
                <$ty>::wrapping_sub(self, other)
            }
            #[inline]
            fn wrapping_mul(self, other: Self) -> Self {
                <$ty>::wrapping_mul(self, other)
            }
            #[inline]
            fn count_ones(self) -> u32 {
                <$ty>::count_ones(self)
            }
            #[inline]
            fn leading_zeros(self) -> u32 {
                <$ty>::leading_zeros(self)
            }
            #[inline]
            fn trailing_zeros(self) -> u32 {
                <$ty>::trailing_zeros(self)
            }
            #[inline]
            fn from_u64(v: u64) -> Self {
                v as Self
            }
            #[inline]
            fn to_u64(self) -> u64 {
                self as u64
            }
            #[inline]
            fn to_u128_le(self) -> u128 {
                self as u128
            }
        }
    };
}

impl_ring_word!(u8);
impl_ring_word!(u16);
impl_ring_word!(u32);
impl_ring_word!(u64);
impl_ring_word!(u128);
