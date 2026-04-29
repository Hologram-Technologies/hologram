//! WordDatum: an element of Z/65536Z implementing uor-foundation Datum.

use crate::q1::observables;
use crate::HoloPrimitives;

/// An element of Z/65536Z at quantum level 1 (16-bit).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct WordDatum {
    value: u16,
    spectrum_buf: [u8; 16],
    address: WordAddress,
}

impl WordDatum {
    /// Create a datum from a raw word value.
    #[inline]
    #[must_use]
    pub fn new(value: u16) -> Self {
        let spectrum = Self::make_spectrum(value);
        let address = WordAddress::from_word(value);
        Self {
            value,
            spectrum_buf: spectrum,
            address,
        }
    }

    const fn make_spectrum(value: u16) -> [u8; 16] {
        let mut buf = [b'0'; 16];
        let mut i = 0;
        while i < 16 {
            if value & (1 << (15 - i)) != 0 {
                buf[i] = b'1';
            }
            i += 1;
        }
        buf
    }

    /// The raw word value.
    #[inline]
    #[must_use]
    pub const fn val(&self) -> u16 {
        self.value
    }

    /// Stratum (popcount) of this datum.
    #[inline]
    #[must_use]
    pub const fn stratum(&self) -> u8 {
        observables::stratum_q1(self.value)
    }

    /// Binary spectrum as a string slice.
    #[inline]
    #[must_use]
    pub fn spectrum(&self) -> &str {
        // SAFETY: spectrum_buf only contains b'0' and b'1'.
        unsafe { core::str::from_utf8_unchecked(&self.spectrum_buf) }
    }

    /// The Braille address for this datum.
    #[inline]
    #[must_use]
    pub const fn address(&self) -> &WordAddress {
        &self.address
    }

    /// Negation: `(-x) mod 65536`.
    #[inline]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    /// Bitwise complement.
    #[inline]
    #[must_use]
    pub fn bnot(self) -> Self {
        Self::new(!self.value)
    }

    /// Successor: `(x + 1) mod 65536`.
    #[inline]
    #[must_use]
    pub fn succ(self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: `(x - 1) mod 65536`.
    #[inline]
    #[must_use]
    pub fn pred(self) -> Self {
        Self::new(self.value.wrapping_sub(1))
    }

    /// The zero datum (additive identity).
    pub fn zero() -> Self {
        Self::new(0)
    }

    /// The generator pi_1 (value = 1).
    pub fn pi1() -> Self {
        Self::new(1)
    }
}

impl core::fmt::Debug for WordDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "WordDatum({})", self.value)
    }
}

impl core::fmt::Display for WordDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Default for WordDatum {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl From<u16> for WordDatum {
    #[inline]
    fn from(value: u16) -> Self {
        Self::new(value)
    }
}

impl From<WordDatum> for u16 {
    #[inline]
    fn from(datum: WordDatum) -> Self {
        datum.value
    }
}

// --- WordAddress ---

/// Braille address for a word datum at Q1.
///
/// Each Braille character encodes 6 bits. For 16-bit values,
/// we use 3 characters (covering lo-6, mid-6, and hi-4 bits).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct WordAddress {
    value: u16,
    glyph_buf: [u8; 9],
    /// Per ADR-052 / Amendment 43 §2: `header(k) || le_bytes(x, k+1)`.
    /// For Q1, k = 1, so header byte = 0x01 followed by 2 LE value bytes.
    canonical_buf: [u8; 3],
    /// Pre-computed `"blake3:<64 hex>"` digest of `canonical_buf`.
    digest_buf: [u8; crate::element::DIGEST_STR_LEN],
}

impl WordAddress {
    /// Create a Braille address from a word value.
    #[must_use]
    pub fn from_word(value: u16) -> Self {
        let lo6 = (value & 0x3F) as u8;
        let mid6 = ((value >> 6) & 0x3F) as u8;
        let hi4 = ((value >> 12) & 0x0F) as u8;
        // Canonical bytes per Amendment 43 §2 (k=1, 2 value bytes LE).
        let canonical_buf = [0x01, value as u8, (value >> 8) as u8];
        let digest_buf = crate::element::blake3_digest_str(&canonical_buf);
        Self {
            value,
            glyph_buf: [
                0xE2,
                0xA0,
                0x80 + lo6,
                0xE2,
                0xA0,
                0x80 + mid6,
                0xE2,
                0xA0,
                0x80 + hi4,
            ],
            canonical_buf,
            digest_buf,
        }
    }

    /// The Braille string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: glyph_buf contains valid UTF-8 Braille characters.
        unsafe { core::str::from_utf8_unchecked(&self.glyph_buf) }
    }
}

impl core::fmt::Debug for WordAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "WordAddress({})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl uor_foundation::kernel::schema::Datum<HoloPrimitives> for WordDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        16
    }

    /// Per ADR-052: stratum is the ring-level index k. Q1 → 1.
    fn stratum(&self) -> u64 {
        1
    }

    /// Per ADR-052: spectrum is the underlying ring value cast to u64.
    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = WordAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl uor_foundation::kernel::address::Element<HoloPrimitives> for WordAddress {
    fn length(&self) -> u64 {
        3
    }

    fn addresses(&self) -> &str {
        self.as_str()
    }

    fn digest(&self) -> &str {
        // SAFETY: digest_buf is ASCII (`blake3:` + lowercase hex).
        unsafe { core::str::from_utf8_unchecked(&self.digest_buf) }
    }

    fn witt_length(&self) -> u64 {
        16
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
    fn datum_new() {
        let d = WordDatum::new(1000);
        assert_eq!(d.val(), 1000);
    }

    #[test]
    fn datum_spectrum() {
        assert_eq!(WordDatum::new(0).spectrum(), "0000000000000000");
        assert_eq!(WordDatum::new(0xFFFF).spectrum(), "1111111111111111");
        assert_eq!(WordDatum::new(0x00FF).spectrum(), "0000000011111111");
        assert_eq!(WordDatum::new(0xFF00).spectrum(), "1111111100000000");
        assert_eq!(WordDatum::new(0xAAAA).spectrum(), "1010101010101010");
    }

    #[test]
    fn datum_stratum() {
        assert_eq!(WordDatum::new(0).stratum(), 0);
        assert_eq!(WordDatum::new(1).stratum(), 1);
        assert_eq!(WordDatum::new(0xFFFF).stratum(), 16);
        assert_eq!(WordDatum::new(0xAAAA).stratum(), 8);
    }

    #[test]
    fn datum_neg() {
        assert_eq!(WordDatum::new(0).neg().val(), 0);
        assert_eq!(WordDatum::new(1).neg().val(), 65535);
        assert_eq!(WordDatum::new(32768).neg().val(), 32768);
    }

    #[test]
    fn datum_bnot() {
        assert_eq!(WordDatum::new(0).bnot().val(), 0xFFFF);
        assert_eq!(WordDatum::new(0xFFFF).bnot().val(), 0);
        assert_eq!(WordDatum::new(0xAAAA).bnot().val(), 0x5555);
    }

    #[test]
    fn critical_identity() {
        // neg(bnot(x)) == succ(x) for all x
        for i in (0u32..=65535).step_by(256) {
            let d = WordDatum::new(i as u16);
            assert_eq!(d.bnot().neg().val(), d.succ().val());
        }
        // Edge cases
        let d = WordDatum::new(65535);
        assert_eq!(d.bnot().neg().val(), d.succ().val());
    }

    #[test]
    fn involution_property() {
        for i in (0u32..=65535).step_by(256) {
            let d = WordDatum::new(i as u16);
            assert_eq!(d.neg().neg().val(), d.val());
            assert_eq!(d.bnot().bnot().val(), d.val());
        }
    }

    #[test]
    fn succ_pred_inverse() {
        for i in (0u32..=65535).step_by(500) {
            let d = WordDatum::new(i as u16);
            assert_eq!(d.succ().pred().val(), d.val());
            assert_eq!(d.pred().succ().val(), d.val());
        }
    }

    #[test]
    fn word_address_glyph_count() {
        let addr = WordAddress::from_word(0);
        assert_eq!(addr.as_str().chars().count(), 3);
        let addr = WordAddress::from_word(0xFFFF);
        assert_eq!(addr.as_str().chars().count(), 3);
    }

    #[test]
    fn word_address_distinct() {
        // Different values should produce different addresses
        let a1 = WordAddress::from_word(0);
        let a2 = WordAddress::from_word(1);
        let a3 = WordAddress::from_word(0xFFFF);
        assert_ne!(a1, a2);
        assert_ne!(a1, a3);
        assert_ne!(a2, a3);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(WordDatum::default(), WordDatum::zero());
    }

    #[test]
    fn from_u16() {
        let d: WordDatum = 1000u16.into();
        assert_eq!(d.val(), 1000);
        let v: u16 = d.into();
        assert_eq!(v, 1000);
    }

    #[test]
    fn datum_trait_impl() {
        use uor_foundation::kernel::schema::Datum;
        let d = WordDatum::new(1000);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 1000);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 16);
        // Per ADR-052: stratum is the ring-level index k. Q1 → 1.
        assert_eq!(Datum::<HoloPrimitives>::stratum(&d), 1);
    }

    #[test]
    fn address_trait_impl() {
        use uor_foundation::kernel::address::Element;
        let a = WordAddress::from_word(1000);
        assert_eq!(Element::<HoloPrimitives>::length(&a), 3);
        assert_eq!(Element::<HoloPrimitives>::witt_length(&a), 16);
    }

    #[test]
    fn rkyv_round_trip() {
        let d = WordDatum::new(42000);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&d).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<WordDatum>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.value, 42000);

        let addr = WordAddress::from_word(42000);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&addr).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<WordAddress>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.value, 42000);
    }
}
