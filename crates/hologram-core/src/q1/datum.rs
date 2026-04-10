//! WordDatum: an element of Z/65536Z implementing uor-foundation Datum.

use crate::q1::observables;
use crate::HoloPrimitives;

/// An element of Z/65536Z at quantum level 1 (16-bit).
#[derive(Clone, PartialEq, Eq, Hash)]
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

    /// The content-addressed identifier for this datum.
    #[inline]
    #[must_use]
    pub fn address(&self) -> &WordAddress {
        &self.address
    }

    /// Negation: `(-x) mod 65536`.
    #[inline]
    #[must_use]
    pub fn neg(&self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    /// Bitwise complement.
    #[inline]
    #[must_use]
    pub fn bnot(&self) -> Self {
        Self::new(!self.value)
    }

    /// Successor: `(x + 1) mod 65536`.
    #[inline]
    #[must_use]
    pub fn succ(&self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: `(x - 1) mod 65536`.
    #[inline]
    #[must_use]
    pub fn pred(&self) -> Self {
        Self::new(self.value.wrapping_sub(1))
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
        Self::new(0)
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

/// Content-addressed identifier for a word datum at W16.
///
/// Per uor-foundation v0.2.0 Amendment 43 section 2:
/// - `canonical_bytes` = hex(`header(1) || le_bytes(value, 2)`) = 6 hex chars
/// - `digest` = hex(BLAKE3(`canonical_raw`)) = 64 hex chars
/// - `digest_algorithm` = `"blake3"`
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct WordAddress {
    value: u16,
    /// hex(header(1) || le_bytes(value, 2)) — 6 hex chars for 3 raw bytes.
    canonical_hex: [u8; 6],
    /// hex(BLAKE3(canonical_raw)) — 64 lowercase hex chars.
    digest_hex: [u8; 64],
}

impl WordAddress {
    /// The zero address (word value 0).
    pub fn zero() -> Self {
        Self::from_word(0)
    }

    /// Create a content-addressed identifier from a word value.
    ///
    /// Computes Amendment 43 canonical bytes and BLAKE3 digest at construction.
    #[must_use]
    pub fn from_word(value: u16) -> Self {
        use crate::datum::hex_encode;

        // Amendment 43: canonical = header(k) || le_bytes(x, k+1)
        // For W16 (k=1): header = 0x01, le_bytes(value, 2) = value.to_le_bytes()
        let le = value.to_le_bytes();
        let canonical_raw = [0x01u8, le[0], le[1]];

        let mut canonical_hex = [0u8; 6];
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

    /// The raw word value this address represents.
    pub fn value(&self) -> u16 {
        self.value
    }
}

impl core::fmt::Debug for WordAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "WordAddress({})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl hologram_foundation::schema::Datum<HoloPrimitives> for WordDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        16
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = WordAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl hologram_foundation::address::Element<HoloPrimitives> for WordAddress {
    /// Byte length of the canonical encoding (3 raw bytes for W16).
    fn length(&self) -> u64 {
        3
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
        16
    }

    #[inline]
    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Amendment 43 canonical encoding: hex(header(1) || le_bytes(value, 2)).
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
    fn word_address_digest_is_blake3() {
        use crate::datum::hex_encode;
        use hologram_foundation::address::Element;
        let addr = WordAddress::from_word(1000);
        // digest() returns 64 hex chars (256-bit BLAKE3)
        assert_eq!(addr.digest().len(), 64);
        // canonical_bytes() returns 6 hex chars (3 raw bytes: header(1) || le_bytes(1000))
        assert_eq!(addr.canonical_bytes().len(), 6);
        // canonical_bytes is "01e803" (header=0x01, 1000=0x03E8 le=[0xE8, 0x03])
        assert_eq!(addr.canonical_bytes(), "01e803");
        // digest_algorithm is "blake3"
        assert_eq!(addr.digest_algorithm(), "blake3");
        // Verify: digest = hex(blake3(canonical_raw))
        let le = 1000u16.to_le_bytes();
        let canonical_raw = [0x01u8, le[0], le[1]];
        let expected_hash = blake3::hash(&canonical_raw);
        let mut expected_hex = [0u8; 64];
        hex_encode(expected_hash.as_bytes(), &mut expected_hex);
        let expected = core::str::from_utf8(&expected_hex).unwrap();
        assert_eq!(addr.digest(), expected);
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
        assert_eq!(WordDatum::default(), WordDatum::new(0));
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
        use hologram_foundation::schema::Datum;
        let d = WordDatum::new(1000);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 1000);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 16);
        assert_eq!(
            Datum::<HoloPrimitives>::stratum(&d),
            1000u16.count_ones() as u64
        );
    }

    #[test]
    fn address_trait_impl() {
        use hologram_foundation::address::Element;
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
        let bytes_a = rkyv::to_bytes::<rkyv::rancor::Error>(&addr).unwrap();
        let archived_a =
            rkyv::access::<rkyv::Archived<WordAddress>, rkyv::rancor::Error>(&bytes_a).unwrap();
        assert_eq!(archived_a.value, 42000);
    }
}
