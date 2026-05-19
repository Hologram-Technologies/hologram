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
    pub const fn new(value: u16) -> Self {
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
    pub const fn neg(self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    /// Bitwise complement.
    #[inline]
    #[must_use]
    pub const fn bnot(self) -> Self {
        Self::new(!self.value)
    }

    /// Successor: `(x + 1) mod 65536`.
    #[inline]
    #[must_use]
    pub const fn succ(self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: `(x - 1) mod 65536`.
    #[inline]
    #[must_use]
    pub const fn pred(self) -> Self {
        Self::new(self.value.wrapping_sub(1))
    }

    /// The zero datum (additive identity).
    pub const ZERO: Self = Self::new(0);

    /// The generator pi_1 (value = 1).
    pub const PI1: Self = Self::new(1);
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
        Self::ZERO
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
}

impl WordAddress {
    /// Create a Braille address from a word value.
    #[must_use]
    pub const fn from_word(value: u16) -> Self {
        let lo6 = (value & 0x3F) as u8;
        let mid6 = ((value >> 6) & 0x3F) as u8;
        let hi4 = ((value >> 12) & 0x0F) as u8;
        // Braille U+2800..U+283F → UTF-8: E2 A0 {80+offset}
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

    fn quantum(&self) -> u64 {
        16
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Address = WordAddress;

    fn glyph(&self) -> &Self::Address {
        &self.address
    }
}

impl uor_foundation::kernel::address::Address<HoloPrimitives> for WordAddress {
    fn glyph(&self) -> &str {
        self.as_str()
    }

    fn length(&self) -> u64 {
        3
    }

    fn addresses(&self) -> &str {
        ""
    }

    fn digest(&self) -> &str {
        self.as_str()
    }

    fn quantum(&self) -> u64 {
        16
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
        assert_eq!(WordDatum::default(), WordDatum::ZERO);
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
        assert_eq!(Datum::<HoloPrimitives>::quantum(&d), 16);
        assert_eq!(
            Datum::<HoloPrimitives>::stratum(&d),
            1000u16.count_ones() as u64
        );
    }

    #[test]
    fn address_trait_impl() {
        use uor_foundation::kernel::address::Address;
        let a = WordAddress::from_word(1000);
        assert_eq!(Address::<HoloPrimitives>::length(&a), 3);
        assert_eq!(Address::<HoloPrimitives>::quantum(&a), 16);
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
