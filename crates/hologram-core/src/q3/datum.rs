//! Q3 (32-bit) datum — element of Z/2^32 Z.

use crate::quantum::{q3_curvature, q3_stratum};
use crate::HoloPrimitives;

/// Element of Z/2^32 Z at quantum level 3.
///
/// Stores value (full 32 bits), spectrum (32-char binary string), and
/// a Braille address (6 Braille glyphs, encoding 32 bits with 4 bits padding).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuadDatum {
    value: u32,
    spectrum_buf: [u8; 32],
    address: QuadAddress,
}

impl QuadDatum {
    /// Additive identity.
    pub const ZERO: Self = Self::new(0);
    /// Multiplicative identity / ring generator.
    pub const PI1: Self = Self::new(1);

    /// Create a datum from a raw 32-bit value.
    #[inline]
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self {
            value,
            spectrum_buf: Self::make_spectrum(value),
            address: QuadAddress::from_quad(value),
        }
    }

    const fn make_spectrum(value: u32) -> [u8; 32] {
        let mut buf = [b'0'; 32];
        let mut i = 0;
        while i < 32 {
            if value & (1 << (31 - i)) != 0 {
                buf[i] = b'1';
            }
            i += 1;
        }
        buf
    }

    /// Raw 32-bit value.
    #[inline(always)]
    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    /// Binary spectrum as a 32-character string slice.
    #[inline]
    #[must_use]
    pub fn spectrum(&self) -> &str {
        // SAFETY: spectrum_buf contains only b'0' and b'1'.
        unsafe { core::str::from_utf8_unchecked(&self.spectrum_buf) }
    }

    /// The Braille address for this datum.
    #[inline]
    #[must_use]
    pub const fn address(&self) -> &QuadAddress {
        &self.address
    }

    /// Ring reflection (additive inverse).
    #[inline(always)]
    #[must_use]
    pub fn ring_neg(self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    /// Hypercube reflection: bnot(x) = !x.
    #[inline(always)]
    #[must_use]
    pub fn bnot(self) -> Self {
        Self::new(!self.value)
    }

    /// Successor: (x + 1) mod 2^32.
    #[inline(always)]
    #[must_use]
    pub fn succ(self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: (x - 1) mod 2^32.
    #[inline(always)]
    #[must_use]
    pub fn pred(self) -> Self {
        Self::new(self.value.wrapping_sub(1))
    }

    /// Hamming weight (popcount) of the 32-bit value.
    #[inline]
    #[must_use]
    pub fn stratum(self) -> u8 {
        q3_stratum(self.value)
    }

    /// Curvature: hamming(x, x+1).
    #[inline]
    #[must_use]
    pub fn curvature(self) -> u8 {
        q3_curvature(self.value)
    }

    /// Add two Q3 datums.
    #[inline]
    #[must_use]
    pub fn ring_add(self, rhs: Self) -> Self {
        Self::new(self.value.wrapping_add(rhs.value))
    }
}

impl core::fmt::Debug for QuadDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "QuadDatum({:#x})", self.value)
    }
}

impl core::fmt::Display for QuadDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Default for QuadDatum {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<u32> for QuadDatum {
    #[inline]
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<QuadDatum> for u32 {
    #[inline]
    fn from(datum: QuadDatum) -> Self {
        datum.value
    }
}

// --- QuadAddress ---

/// Braille address for a 32-bit datum: 6 Braille characters.
///
/// 32 bits ÷ 6 bits/glyph = 5.33 → 6 glyphs (last has 2 data bits + 4 padding).
/// Braille U+2800..U+283F → UTF-8: E2 A0 (80 + 6-bit group).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuadAddress {
    value: u32,
    glyph_buf: [u8; 18], // 6 × 3-byte UTF-8 Braille chars
}

impl QuadAddress {
    /// Create a Braille address from a 32-bit value.
    #[must_use]
    pub const fn from_quad(value: u32) -> Self {
        let g0 = (value & 0x3F) as u8;
        let g1 = ((value >> 6) & 0x3F) as u8;
        let g2 = ((value >> 12) & 0x3F) as u8;
        let g3 = ((value >> 18) & 0x3F) as u8;
        let g4 = ((value >> 24) & 0x3F) as u8;
        let g5 = ((value >> 30) & 0x03) as u8; // only 2 bits
        Self {
            value,
            glyph_buf: [
                0xE2,
                0xA0,
                0x80 + g0,
                0xE2,
                0xA0,
                0x80 + g1,
                0xE2,
                0xA0,
                0x80 + g2,
                0xE2,
                0xA0,
                0x80 + g3,
                0xE2,
                0xA0,
                0x80 + g4,
                0xE2,
                0xA0,
                0x80 + g5,
            ],
        }
    }

    /// The Braille string representation (6 glyphs).
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: glyph_buf contains valid UTF-8 Braille chars only.
        unsafe { core::str::from_utf8_unchecked(&self.glyph_buf) }
    }
}

impl core::fmt::Debug for QuadAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "QuadAddress({:#x})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl uor_foundation::kernel::schema::Datum<HoloPrimitives> for QuadDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn quantum(&self) -> u64 {
        32
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> &str {
        self.spectrum()
    }

    type Address = QuadAddress;

    fn glyph(&self) -> &Self::Address {
        &self.address
    }
}

impl uor_foundation::kernel::address::Address<HoloPrimitives> for QuadAddress {
    fn glyph(&self) -> &str {
        self.as_str()
    }

    fn length(&self) -> u64 {
        6 // 6 Braille glyphs
    }

    fn addresses(&self) -> &str {
        ""
    }

    fn digest(&self) -> &str {
        self.as_str()
    }

    fn quantum(&self) -> u64 {
        32
    }

    #[inline]
    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    #[inline]
    fn canonical_bytes(&self) -> &str {
        self.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_pi1() {
        assert_eq!(QuadDatum::ZERO.value(), 0);
        assert_eq!(QuadDatum::PI1.value(), 1);
    }

    #[test]
    fn neg_involution() {
        let x = QuadDatum::new(0xDEAD_BEEF);
        assert_eq!(x.ring_neg().ring_neg(), x);
    }

    #[test]
    fn bnot_involution() {
        let x = QuadDatum::new(0xDEAD_BEEF);
        assert_eq!(x.bnot().bnot(), x);
    }

    #[test]
    fn critical_identity() {
        for x in [0u32, 1, 0xFF, 0xFFFF, u32::MAX] {
            let d = QuadDatum::new(x);
            assert_eq!(d.bnot().ring_neg(), d.succ());
        }
    }

    #[test]
    fn spectrum_is_32_chars() {
        let d = QuadDatum::new(0);
        assert_eq!(d.spectrum().len(), 32);
        assert_eq!(d.spectrum(), "00000000000000000000000000000000");
        let d = QuadDatum::new(u32::MAX);
        assert_eq!(d.spectrum(), "11111111111111111111111111111111");
    }

    #[test]
    fn address_length() {
        use uor_foundation::kernel::address::Address;
        let a = QuadAddress::from_quad(0);
        assert_eq!(Address::<HoloPrimitives>::length(&a), 6);
        assert_eq!(a.as_str().chars().count(), 6);
    }

    #[test]
    fn round_trip_value() {
        for x in [0u32, 1, 0xFF, 0xFFFF, 0xDEAD_BEEF, u32::MAX] {
            let d = QuadDatum::new(x);
            assert_eq!(d.value(), x);
        }
    }

    #[test]
    fn datum_trait_impl() {
        use uor_foundation::kernel::schema::Datum;
        let d = QuadDatum::new(0xDEAD_BEEF);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 0xDEAD_BEEF);
        assert_eq!(Datum::<HoloPrimitives>::quantum(&d), 32);
    }

    #[test]
    fn from_u32_round_trip() {
        let d: QuadDatum = 0xCAFE_BABEu32.into();
        assert_eq!(d.value(), 0xCAFE_BABE);
        let v: u32 = d.into();
        assert_eq!(v, 0xCAFE_BABE);
    }
}
