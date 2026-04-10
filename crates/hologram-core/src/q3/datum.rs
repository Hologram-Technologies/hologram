//! Q3 (32-bit) datum — element of Z/2^32 Z.

use crate::quantum::{q3_curvature, q3_stratum};
use crate::HoloPrimitives;

/// Element of Z/2^32 Z at quantum level 3.
///
/// Stores value (full 32 bits), spectrum (32-char binary string), and
/// a Braille address (6 Braille glyphs, encoding 32 bits with 4 bits padding).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct QuadDatum {
    value: u32,
    spectrum_buf: [u8; 32],
    address: QuadAddress,
}

impl QuadDatum {
    /// Create a datum from a raw 32-bit value.
    #[inline]
    #[must_use]
    pub fn new(value: u32) -> Self {
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
    pub const fn value(&self) -> u32 {
        self.value
    }

    /// Binary spectrum as a 32-character string slice.
    #[inline]
    #[must_use]
    pub fn spectrum(&self) -> &str {
        // SAFETY: spectrum_buf contains only b'0' and b'1'.
        unsafe { core::str::from_utf8_unchecked(&self.spectrum_buf) }
    }

    /// The content-addressed identifier for this datum.
    #[inline]
    #[must_use]
    pub fn address(&self) -> &QuadAddress {
        &self.address
    }

    /// Ring reflection (additive inverse).
    #[inline(always)]
    #[must_use]
    pub fn ring_neg(&self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    /// Hypercube reflection: bnot(x) = !x.
    #[inline(always)]
    #[must_use]
    pub fn bnot(&self) -> Self {
        Self::new(!self.value)
    }

    /// Successor: (x + 1) mod 2^32.
    #[inline(always)]
    #[must_use]
    pub fn succ(&self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: (x - 1) mod 2^32.
    #[inline(always)]
    #[must_use]
    pub fn pred(&self) -> Self {
        Self::new(self.value.wrapping_sub(1))
    }

    /// Hamming weight (popcount) of the 32-bit value.
    #[inline]
    #[must_use]
    pub fn stratum(&self) -> u8 {
        q3_stratum(self.value)
    }

    /// Curvature: hamming(x, x+1).
    #[inline]
    #[must_use]
    pub fn curvature(&self) -> u8 {
        q3_curvature(self.value)
    }

    /// Add two Q3 datums.
    #[inline]
    #[must_use]
    pub fn ring_add(&self, rhs: &Self) -> Self {
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
        Self::new(0)
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

/// Content-addressed identifier for a 32-bit datum at W32.
///
/// Per uor-foundation v0.2.0 Amendment 43 section 2:
/// - `canonical_bytes` = hex(`header(3) || le_bytes(value, 4)`) = 10 hex chars
/// - `digest` = hex(BLAKE3(`canonical_raw`)) = 64 hex chars
/// - `digest_algorithm` = `"blake3"`
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct QuadAddress {
    value: u32,
    /// hex(header(3) || le_bytes(value, 4)) — 10 hex chars for 5 raw bytes.
    canonical_hex: [u8; 10],
    /// hex(BLAKE3(canonical_raw)) — 64 lowercase hex chars.
    digest_hex: [u8; 64],
}

impl QuadAddress {
    /// The zero address (quad value 0).
    pub fn zero() -> Self {
        Self::from_quad(0)
    }

    /// Create a content-addressed identifier from a 32-bit value.
    ///
    /// Computes Amendment 43 canonical bytes and BLAKE3 digest at construction.
    #[must_use]
    pub fn from_quad(value: u32) -> Self {
        use crate::datum::hex_encode;

        // Amendment 43: canonical = header(k) || le_bytes(x, k+1)
        // For W32 (k=3): header = 0x03, le_bytes(value, 4) = value.to_le_bytes()
        let le = value.to_le_bytes();
        let canonical_raw = [0x03u8, le[0], le[1], le[2], le[3]];

        let mut canonical_hex = [0u8; 10];
        hex_encode(&canonical_raw, &mut canonical_hex);

        let hash = blake3::hash(&canonical_raw);
        let mut digest_hex = [0u8; 64];
        hex_encode(hash.as_bytes(), &mut digest_hex);

        Self {
            value,
            canonical_hex,
            digest_hex,
        }
    }

    /// The raw 32-bit value this address represents.
    pub fn value(&self) -> u32 {
        self.value
    }
}

impl core::fmt::Debug for QuadAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "QuadAddress({:#x})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl hologram_foundation::schema::Datum<HoloPrimitives> for QuadDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        32
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = QuadAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl hologram_foundation::address::Element<HoloPrimitives> for QuadAddress {
    /// Byte length of the canonical encoding (5 raw bytes for W32).
    fn length(&self) -> u64 {
        5
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
        32
    }

    #[inline]
    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Amendment 43 canonical encoding: hex(header(3) || le_bytes(value, 4)).
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
        assert_eq!(QuadDatum::new(0).value(), 0);
        assert_eq!(QuadDatum::new(1).value(), 1);
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
    fn quad_address_digest_is_blake3() {
        use crate::datum::hex_encode;
        use hologram_foundation::address::Element;
        let addr = QuadAddress::from_quad(0xDEAD_BEEF);
        // digest() returns 64 hex chars (256-bit BLAKE3)
        assert_eq!(addr.digest().len(), 64);
        // canonical_bytes() returns 10 hex chars (5 raw bytes: header(3) || le_bytes)
        assert_eq!(addr.canonical_bytes().len(), 10);
        // digest_algorithm is "blake3"
        assert_eq!(addr.digest_algorithm(), "blake3");
        // Verify: digest = hex(blake3(canonical_raw))
        let le = 0xDEAD_BEEFu32.to_le_bytes();
        let canonical_raw = [0x03u8, le[0], le[1], le[2], le[3]];
        let expected_hash = blake3::hash(&canonical_raw);
        let mut expected_hex = [0u8; 64];
        hex_encode(expected_hash.as_bytes(), &mut expected_hex);
        let expected = core::str::from_utf8(&expected_hex).unwrap();
        assert_eq!(addr.digest(), expected);
        // length() returns raw byte count
        assert_eq!(Element::<HoloPrimitives>::length(&addr), 5);
        assert_eq!(Element::<HoloPrimitives>::witt_length(&addr), 32);
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
        use hologram_foundation::schema::Datum;
        let d = QuadDatum::new(0xDEAD_BEEF);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 0xDEAD_BEEF);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 32);
    }

    #[test]
    fn from_u32_round_trip() {
        let d: QuadDatum = 0xCAFE_BABEu32.into();
        assert_eq!(d.value(), 0xCAFE_BABE);
        let v: u32 = d.into();
        assert_eq!(v, 0xCAFE_BABE);
    }
}
