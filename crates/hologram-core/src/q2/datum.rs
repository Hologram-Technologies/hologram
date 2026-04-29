//! Q2 (24-bit) datum — element of Z/2^24 Z.

use crate::q2::arith::{add_q2, bnot_q2, neg_q2, pred_q2, succ_q2};
use crate::quantum::{q2_curvature, q2_stratum};
use crate::HoloPrimitives;

/// Element of Z/2^24 Z at quantum level 2.
///
/// Stores value (low 24 bits), spectrum (24-char binary string), and
/// a Braille address (4 Braille glyphs, 6 bits each).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleDatum {
    value: u32,
    spectrum_buf: [u8; 24],
    address: TripleAddress,
}

impl TripleDatum {
    /// Additive identity.
    pub fn zero() -> Self {
        Self::new(0)
    }
    /// Multiplicative identity / ring generator.
    pub fn pi1() -> Self {
        Self::new(1)
    }

    /// Create a datum from a raw 24-bit value. High byte is masked to 0.
    #[inline]
    #[must_use]
    pub fn new(value: u32) -> Self {
        let v = value & 0x00FF_FFFF;
        Self {
            value: v,
            spectrum_buf: Self::make_spectrum(v),
            address: TripleAddress::from_triple(v),
        }
    }

    const fn make_spectrum(value: u32) -> [u8; 24] {
        let mut buf = [b'0'; 24];
        let mut i = 0;
        while i < 24 {
            if value & (1 << (23 - i)) != 0 {
                buf[i] = b'1';
            }
            i += 1;
        }
        buf
    }

    /// Raw 24-bit value (high byte always 0).
    #[inline(always)]
    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    /// Binary spectrum as a 24-character string slice.
    #[inline]
    #[must_use]
    pub fn spectrum(&self) -> &str {
        // SAFETY: spectrum_buf contains only b'0' and b'1'.
        unsafe { core::str::from_utf8_unchecked(&self.spectrum_buf) }
    }

    /// The Braille address for this datum.
    #[inline]
    #[must_use]
    pub const fn address(&self) -> &TripleAddress {
        &self.address
    }

    /// Ring reflection (additive inverse): ring_neg(x) = (2^24 - x) mod 2^24.
    #[inline(always)]
    #[must_use]
    pub fn ring_neg(self) -> Self {
        Self::new(neg_q2(self.value))
    }

    /// Hypercube reflection: bnot(x) = (2^24 - 1) ^ x.
    #[inline(always)]
    #[must_use]
    pub fn bnot(self) -> Self {
        Self::new(bnot_q2(self.value))
    }

    /// Successor: succ(x) = (x + 1) mod 2^24.
    #[inline(always)]
    #[must_use]
    pub fn succ(self) -> Self {
        Self::new(succ_q2(self.value))
    }

    /// Predecessor: pred(x) = (x - 1) mod 2^24.
    #[inline(always)]
    #[must_use]
    pub fn pred(self) -> Self {
        Self::new(pred_q2(self.value))
    }

    /// Hamming weight (popcount) of the 24-bit value.
    #[inline]
    #[must_use]
    pub fn stratum(self) -> u8 {
        q2_stratum(self.value)
    }

    /// Curvature: hamming(x, x+1) masked to 24 bits.
    #[inline]
    #[must_use]
    pub fn curvature(self) -> u8 {
        q2_curvature(self.value)
    }

    /// Add two Q2 datums.
    #[inline]
    #[must_use]
    pub fn ring_add(self, rhs: Self) -> Self {
        Self::new(add_q2(self.value, rhs.value))
    }
}

impl core::fmt::Debug for TripleDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TripleDatum({:#x})", self.value)
    }
}

impl core::fmt::Display for TripleDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Default for TripleDatum {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl From<u32> for TripleDatum {
    #[inline]
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<TripleDatum> for u32 {
    #[inline]
    fn from(datum: TripleDatum) -> Self {
        datum.value
    }
}

// --- TripleAddress ---

/// Braille address for a 24-bit datum: 4 Braille characters (6 bits each).
///
/// Braille U+2800..U+283F → UTF-8: E2 A0 (80 + 6-bit group).
/// 24 bits split into four 6-bit groups: bits [5:0], [11:6], [17:12], [23:18].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleAddress {
    value: u32,
    glyph_buf: [u8; 12], // 4 × 3-byte UTF-8 Braille chars
    /// Per ADR-052 / Amendment 43 §2: `header(2) || le_bytes(x, 3)`.
    canonical_buf: [u8; 4],
    /// Pre-computed `"blake3:<hex>"` digest of `canonical_buf`.
    digest_buf: [u8; crate::element::DIGEST_STR_LEN],
}

impl TripleAddress {
    /// Create a Braille address from a 24-bit value.
    #[must_use]
    pub fn from_triple(value: u32) -> Self {
        let v = value & 0x00FF_FFFF;
        let g0 = (v & 0x3F) as u8;
        let g1 = ((v >> 6) & 0x3F) as u8;
        let g2 = ((v >> 12) & 0x3F) as u8;
        let g3 = ((v >> 18) & 0x3F) as u8;
        let canonical_buf = [0x02, v as u8, (v >> 8) as u8, (v >> 16) as u8];
        let digest_buf = crate::element::blake3_digest_str(&canonical_buf);
        Self {
            value: v,
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
            ],
            canonical_buf,
            digest_buf,
        }
    }

    /// The Braille string representation (4 glyphs).
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: glyph_buf contains valid UTF-8 Braille chars only.
        unsafe { core::str::from_utf8_unchecked(&self.glyph_buf) }
    }
}

impl core::fmt::Debug for TripleAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TripleAddress({:#x})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl uor_foundation::kernel::schema::Datum<HoloPrimitives> for TripleDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        24
    }

    /// Per ADR-052: stratum is the ring-level index k. Q2 → 2.
    fn stratum(&self) -> u64 {
        2
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = TripleAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl uor_foundation::kernel::address::Element<HoloPrimitives> for TripleAddress {
    fn length(&self) -> u64 {
        4 // 4 Braille glyphs
    }

    fn addresses(&self) -> &str {
        self.as_str()
    }

    fn digest(&self) -> &str {
        // SAFETY: digest_buf is ASCII (`blake3:` + lowercase hex).
        unsafe { core::str::from_utf8_unchecked(&self.digest_buf) }
    }

    fn witt_length(&self) -> u64 {
        24
    }

    #[inline]
    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    #[inline]
    fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_pi1() {
        assert_eq!(TripleDatum::zero().value(), 0);
        assert_eq!(TripleDatum::pi1().value(), 1);
    }

    #[test]
    fn neg_involution() {
        let x = TripleDatum::new(12345);
        assert_eq!(x.ring_neg().ring_neg(), x);
    }

    #[test]
    fn bnot_involution() {
        let x = TripleDatum::new(0xABCDEF);
        assert_eq!(x.bnot().bnot(), TripleDatum::new(0xABCDEF));
    }

    #[test]
    fn critical_identity() {
        // ring_neg(bnot(x)) = succ(x)
        let vals = [
            TripleDatum::new(0),
            TripleDatum::new(1),
            TripleDatum::new(0xFF),
            TripleDatum::new(0x00FF_FFFF),
        ];
        for x in vals {
            assert_eq!(x.bnot().ring_neg(), x.succ());
        }
    }

    #[test]
    fn stratum_and_curvature() {
        assert_eq!(TripleDatum::new(0).stratum(), 0);
        assert_eq!(TripleDatum::new(0x00FF_FFFF).stratum(), 24);
        assert_eq!(TripleDatum::new(0).curvature(), 1); // 0 → 1: 1 bit changes
    }

    #[test]
    fn round_trip_value() {
        for x in [0u32, 1, 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF] {
            let d = TripleDatum::new(x);
            assert_eq!(d.value(), x & 0x00FF_FFFF);
        }
    }

    #[test]
    fn spectrum_is_binary_string() {
        let d = TripleDatum::new(0);
        assert_eq!(d.spectrum(), "000000000000000000000000");
        let d = TripleDatum::new(0x00FF_FFFF);
        assert_eq!(d.spectrum(), "111111111111111111111111");
    }

    #[test]
    fn datum_trait_impl() {
        use uor_foundation::kernel::schema::Datum;
        let d = TripleDatum::new(0xABCDEF);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 0xABCDEF);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 24);
        // Per ADR-052: stratum is the ring-level index k. Q2 → 2.
        assert_eq!(Datum::<HoloPrimitives>::stratum(&d), 2);
    }

    #[test]
    fn address_length() {
        use uor_foundation::kernel::address::Element;
        let a = TripleAddress::from_triple(0);
        assert_eq!(Element::<HoloPrimitives>::length(&a), 4);
        assert_eq!(Element::<HoloPrimitives>::witt_length(&a), 24);
        assert_eq!(a.as_str().chars().count(), 4); // 4 Braille glyphs
    }

    #[test]
    fn spectrum_matches_value_bits() {
        let v = 0b10_1010_1010_1010_1010_1010_u32;
        let d = TripleDatum::new(v);
        let s = d.spectrum();
        assert_eq!(s.len(), 24);
        for (i, ch) in s.chars().enumerate() {
            let bit = (d.value() >> (23 - i as u32)) & 1;
            assert_eq!(ch, if bit == 1 { '1' } else { '0' });
        }
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(TripleDatum::default(), TripleDatum::zero());
    }

    #[test]
    fn from_u32_round_trip() {
        let d: TripleDatum = 0xABCDEFu32.into();
        assert_eq!(d.value(), 0xABCDEF);
        let v: u32 = d.into();
        assert_eq!(v, 0xABCDEF);
    }
}
