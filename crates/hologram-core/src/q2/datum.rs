//! Q2 (24-bit) datum — element of Z/2^24 Z.

use crate::q2::arith::{add_q2, bnot_q2, neg_q2, pred_q2, succ_q2};
use crate::quantum::{q2_curvature, q2_stratum};
use crate::HoloPrimitives;

/// Element of Z/2^24 Z at quantum level 2.
///
/// Stores value (low 24 bits), spectrum (24-char binary string), and
/// a Braille address (4 Braille glyphs, 6 bits each).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TripleDatum {
    value: u32,
    spectrum_buf: [u8; 24],
    address: TripleAddress,
}

impl TripleDatum {
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
    pub const fn value(&self) -> u32 {
        self.value
    }

    /// Binary spectrum as a 24-character string slice.
    #[inline]
    #[must_use]
    pub fn spectrum(&self) -> &str {
        // SAFETY: spectrum_buf contains only b'0' and b'1'.
        unsafe { core::str::from_utf8_unchecked(&self.spectrum_buf) }
    }

    /// The content-addressed identifier for this datum.
    #[inline]
    #[must_use]
    pub fn address(&self) -> &TripleAddress {
        &self.address
    }

    /// Ring reflection (additive inverse): ring_neg(x) = (2^24 - x) mod 2^24.
    #[inline(always)]
    #[must_use]
    pub fn ring_neg(&self) -> Self {
        Self::new(neg_q2(self.value))
    }

    /// Hypercube reflection: bnot(x) = (2^24 - 1) ^ x.
    #[inline(always)]
    #[must_use]
    pub fn bnot(&self) -> Self {
        Self::new(bnot_q2(self.value))
    }

    /// Successor: succ(x) = (x + 1) mod 2^24.
    #[inline(always)]
    #[must_use]
    pub fn succ(&self) -> Self {
        Self::new(succ_q2(self.value))
    }

    /// Predecessor: pred(x) = (x - 1) mod 2^24.
    #[inline(always)]
    #[must_use]
    pub fn pred(&self) -> Self {
        Self::new(pred_q2(self.value))
    }

    /// Hamming weight (popcount) of the 24-bit value.
    #[inline]
    #[must_use]
    pub fn stratum(&self) -> u8 {
        q2_stratum(self.value)
    }

    /// Curvature: hamming(x, x+1) masked to 24 bits.
    #[inline]
    #[must_use]
    pub fn curvature(&self) -> u8 {
        q2_curvature(self.value)
    }

    /// Add two Q2 datums.
    #[inline]
    #[must_use]
    pub fn ring_add(&self, rhs: &Self) -> Self {
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
        Self::new(0)
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

/// Content-addressed identifier for a 24-bit datum at W24.
///
/// Per uor-foundation v0.2.0 Amendment 43 section 2:
/// - `canonical_bytes` = hex(`header(2) || le_bytes(value, 3)`) = 8 hex chars
/// - `digest` = hex(BLAKE3(`canonical_raw`)) = 64 hex chars
/// - `digest_algorithm` = `"blake3"`
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TripleAddress {
    value: u32,
    /// hex(header(2) || le_bytes(value, 3)) — 8 hex chars for 4 raw bytes.
    canonical_hex: [u8; 8],
    /// hex(BLAKE3(canonical_raw)) — 64 lowercase hex chars.
    digest_hex: [u8; 64],
}

impl TripleAddress {
    /// The zero address (triple value 0).
    pub fn zero() -> Self {
        Self::from_triple(0)
    }

    /// Create a content-addressed identifier from a 24-bit value.
    ///
    /// Computes Amendment 43 canonical bytes and BLAKE3 digest at construction.
    #[must_use]
    pub fn from_triple(value: u32) -> Self {
        use crate::datum::hex_encode;

        let v = value & 0x00FF_FFFF;
        // Amendment 43: canonical = header(k) || le_bytes(x, k+1)
        // For W24 (k=2): header = 0x02, le_bytes(value, 3) = first 3 bytes of u32 LE
        let le = v.to_le_bytes();
        let canonical_raw = [0x02u8, le[0], le[1], le[2]];

        let mut canonical_hex = [0u8; 8];
        hex_encode(&canonical_raw, &mut canonical_hex);

        let hash = blake3::hash(&canonical_raw);
        let mut digest_hex = [0u8; 64];
        hex_encode(hash.as_bytes(), &mut digest_hex);

        Self {
            value: v,
            canonical_hex,
            digest_hex,
        }
    }

    /// The raw 24-bit value this address represents.
    pub fn value(&self) -> u32 {
        self.value
    }
}

impl core::fmt::Debug for TripleAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TripleAddress({:#x})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl hologram_foundation::schema::Datum<HoloPrimitives> for TripleDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        24
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = TripleAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl hologram_foundation::address::Element<HoloPrimitives> for TripleAddress {
    /// Byte length of the canonical encoding (4 raw bytes for W24).
    fn length(&self) -> u64 {
        4
    }

    fn addresses(&self) -> &str {
        // SAFETY: digest_hex is valid ASCII hex.
        unsafe { core::str::from_utf8_unchecked(&self.digest_hex) }
    }

    /// BLAKE3 content hash of the canonical bytes, as 64 lowercase hex chars.
    fn digest(&self) -> &str {
        // SAFETY: digest_hex is valid ASCII hex.
        unsafe { core::str::from_utf8_unchecked(&self.digest_hex) }
    }

    fn witt_length(&self) -> u64 {
        24
    }

    #[inline]
    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Amendment 43 canonical encoding: hex(header(2) || le_bytes(value, 3)).
    #[inline]
    fn canonical_bytes(&self) -> &str {
        // SAFETY: canonical_hex is valid ASCII hex.
        unsafe { core::str::from_utf8_unchecked(&self.canonical_hex) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_pi1() {
        assert_eq!(TripleDatum::new(0).value(), 0);
        assert_eq!(TripleDatum::new(1).value(), 1);
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
        use hologram_foundation::schema::Datum;
        let d = TripleDatum::new(0xABCDEF);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 0xABCDEF);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 24);
        assert_eq!(
            Datum::<HoloPrimitives>::stratum(&d),
            (0xABCDEFu32).count_ones() as u64
        );
    }

    #[test]
    fn triple_address_digest_is_blake3() {
        use crate::datum::hex_encode;
        use hologram_foundation::address::Element;
        let addr = TripleAddress::from_triple(0xABCDEF);
        // digest() returns 64 hex chars (256-bit BLAKE3)
        assert_eq!(addr.digest().len(), 64);
        // canonical_bytes() returns 8 hex chars (4 raw bytes: header(2) || le_bytes)
        assert_eq!(addr.canonical_bytes().len(), 8);
        // digest_algorithm is "blake3"
        assert_eq!(addr.digest_algorithm(), "blake3");
        // Verify: digest = hex(blake3(canonical_raw))
        let v = 0xABCDEFu32;
        let le = v.to_le_bytes();
        let canonical_raw = [0x02u8, le[0], le[1], le[2]];
        let expected_hash = blake3::hash(&canonical_raw);
        let mut expected_hex = [0u8; 64];
        hex_encode(expected_hash.as_bytes(), &mut expected_hex);
        let expected = core::str::from_utf8(&expected_hex).unwrap();
        assert_eq!(addr.digest(), expected);
        // length() returns raw byte count
        assert_eq!(Element::<HoloPrimitives>::length(&addr), 4);
        assert_eq!(Element::<HoloPrimitives>::witt_length(&addr), 24);
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
        assert_eq!(TripleDatum::default(), TripleDatum::new(0));
    }

    #[test]
    fn from_u32_round_trip() {
        let d: TripleDatum = 0xABCDEFu32.into();
        assert_eq!(d.value(), 0xABCDEF);
        let v: u32 = d.into();
        assert_eq!(v, 0xABCDEF);
    }
}
