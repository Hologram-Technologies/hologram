//! ByteDatum: an element of Z/256Z implementing uor-foundation Datum.

use crate::lut::q0;
use crate::HoloPrimitives;

/// An element of Z/256Z at quantum level 0 (8-bit).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct ByteDatum {
    value: u8,
    spectrum_buf: [u8; 8],
    address: ByteAddress,
}

impl ByteDatum {
    /// Create a datum from a raw byte value.
    #[inline]
    #[must_use]
    pub fn new(value: u8) -> Self {
        let spectrum = Self::make_spectrum(value);
        let address = ByteAddress::from_byte(value);
        Self {
            value,
            spectrum_buf: spectrum,
            address,
        }
    }

    const fn make_spectrum(value: u8) -> [u8; 8] {
        let mut buf = [b'0'; 8];
        let mut i = 0;
        while i < 8 {
            if value & (1 << (7 - i)) != 0 {
                buf[i] = b'1';
            }
            i += 1;
        }
        buf
    }

    /// The raw byte value.
    #[inline]
    #[must_use]
    pub const fn val(&self) -> u8 {
        self.value
    }

    /// Stratum (popcount) of this datum.
    #[inline]
    #[must_use]
    pub const fn stratum(&self) -> u8 {
        q0::stratum_q0(self.value)
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
    pub const fn address(&self) -> &ByteAddress {
        &self.address
    }

    /// Negation: `(-x) mod 256`.
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

    /// Successor: `(x + 1) mod 256`.
    #[inline]
    #[must_use]
    pub fn succ(self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: `(x - 1) mod 256`.
    #[inline]
    #[must_use]
    pub fn pred(self) -> Self {
        Self::new(self.value.wrapping_sub(1))
    }

    /// The zero datum (additive identity).
    #[must_use]
    pub fn zero() -> Self {
        Self::new(0)
    }

    /// The generator pi_1 (value = 1).
    #[must_use]
    pub fn pi1() -> Self {
        Self::new(1)
    }
}

impl core::fmt::Debug for ByteDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ByteDatum({})", self.value)
    }
}

impl core::fmt::Display for ByteDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Default for ByteDatum {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl From<u8> for ByteDatum {
    #[inline]
    fn from(value: u8) -> Self {
        Self::new(value)
    }
}

impl From<ByteDatum> for u8 {
    #[inline]
    fn from(datum: ByteDatum) -> Self {
        datum.value
    }
}

// --- ByteAddress ---

/// Braille address for a byte datum at Q0.
///
/// Each Braille character encodes 6 bits. For 8-bit values,
/// we use 2 characters (covering lo-6 and hi-2 bits).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct ByteAddress {
    value: u8,
    glyph_buf: [u8; 6],
    /// Per ADR-052 / Amendment 43 §2: `header(0) || le_bytes(x, 1)`.
    canonical_buf: [u8; 2],
    /// Pre-computed `"blake3:<hex>"` digest string.
    digest_buf: [u8; crate::element::DIGEST_STR_LEN],
}

impl ByteAddress {
    /// The zero address (byte value 0).
    #[must_use]
    pub fn zero() -> Self {
        Self::from_byte(0)
    }

    /// Create a Braille address from a byte value.
    #[must_use]
    pub fn from_byte(value: u8) -> Self {
        let lo6 = value & 0x3F;
        let hi2 = (value >> 6) & 0x03;
        let canonical_buf = [0x00, value];
        let digest_buf = crate::element::blake3_digest_str(&canonical_buf);
        Self {
            value,
            glyph_buf: [0xE2, 0xA0, 0x80 + lo6, 0xE2, 0xA0, 0x80 + hi2],
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

impl core::fmt::Debug for ByteAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ByteAddress({})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl uor_foundation::kernel::schema::Datum<HoloPrimitives> for ByteDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        8
    }

    /// Per ADR-052: stratum is the ring-level index k. Q0 → 0.
    fn stratum(&self) -> u64 {
        0
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = ByteAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl uor_foundation::kernel::address::Element<HoloPrimitives> for ByteAddress {
    fn length(&self) -> u64 {
        2
    }

    fn addresses(&self) -> &str {
        self.as_str()
    }

    fn digest(&self) -> &str {
        // SAFETY: digest_buf is ASCII (`blake3:` + lowercase hex).
        unsafe { core::str::from_utf8_unchecked(&self.digest_buf) }
    }

    fn witt_length(&self) -> u64 {
        8
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

// ── RingDatum: unified datum for any quantum level ──────────────────────────

use crate::op::QuantumLevelExt;

/// A ring element at any quantum level (Q0–Q7+).
///
/// Stores the value as `u64`, which covers all native levels up to Q7 (64-bit).
/// The `level` field determines the ring modulus: Z/(2^(8*(k+1)))Z.
///
/// This unified type replaces the per-level `ByteDatum`/`WordDatum`/`TripleDatum`/`QuadDatum`
/// for code that needs to work with arbitrary precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RingDatum {
    /// The ring element value, masked to the level's byte width.
    pub value: u64,
    /// The quantum level of this datum.
    pub level: uor_foundation::WittLevel,
}

impl RingDatum {
    /// Create a datum at the given quantum level.
    /// The value is masked to the ring modulus.
    #[inline]
    pub fn new(value: u64, level: uor_foundation::WittLevel) -> Self {
        let bw = level.byte_width();
        let bits = (bw as u32) * 8;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        Self {
            value: value & mask,
            level,
        }
    }

    /// Byte width of this datum's ring element.
    #[inline]
    pub fn byte_width(&self) -> u8 {
        self.level.byte_width()
    }

    /// Stratum (Hamming weight / popcount).
    #[inline]
    pub fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    /// Spectrum (numeric value).
    #[inline]
    pub fn spectrum(&self) -> u64 {
        self.value
    }

    /// Construct from a ByteDatum (Q0).
    #[inline]
    pub fn from_byte_datum(d: &ByteDatum) -> Self {
        Self::new(d.val() as u64, uor_foundation::WittLevel::W8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_datum_masks_value() {
        let d = RingDatum::new(300, uor_foundation::WittLevel::W8);
        assert_eq!(d.value, 44); // 300 & 0xFF = 44
    }

    #[test]
    fn ring_datum_q7_no_mask() {
        // Q7 = native u64 = witt_length 64 in 0.3.0 naming.
        let d = RingDatum::new(u64::MAX, uor_foundation::WittLevel::new(64));
        assert_eq!(d.value, u64::MAX);
    }

    #[test]
    fn ring_datum_from_byte_datum() {
        let bd = ByteDatum::new(42);
        let rd = RingDatum::from_byte_datum(&bd);
        assert_eq!(rd.value, 42);
        assert_eq!(rd.level, uor_foundation::WittLevel::W8);
    }

    #[test]
    fn datum_new() {
        let d = ByteDatum::new(42);
        assert_eq!(d.val(), 42);
    }

    #[test]
    fn datum_spectrum() {
        assert_eq!(ByteDatum::new(0).spectrum(), "00000000");
        assert_eq!(ByteDatum::new(255).spectrum(), "11111111");
        assert_eq!(ByteDatum::new(0b10101010).spectrum(), "10101010");
    }

    #[test]
    fn datum_stratum() {
        assert_eq!(ByteDatum::new(0).stratum(), 0);
        assert_eq!(ByteDatum::new(1).stratum(), 1);
        assert_eq!(ByteDatum::new(255).stratum(), 8);
        assert_eq!(ByteDatum::new(0b10101010).stratum(), 4);
    }

    #[test]
    fn datum_neg() {
        assert_eq!(ByteDatum::new(0).neg().val(), 0);
        assert_eq!(ByteDatum::new(1).neg().val(), 255);
        assert_eq!(ByteDatum::new(128).neg().val(), 128);
    }

    #[test]
    fn datum_bnot() {
        assert_eq!(ByteDatum::new(0).bnot().val(), 255);
        assert_eq!(ByteDatum::new(255).bnot().val(), 0);
        assert_eq!(ByteDatum::new(0xAA).bnot().val(), 0x55);
    }

    #[test]
    fn critical_identity() {
        for i in 0..=255u8 {
            let d = ByteDatum::new(i);
            assert_eq!(d.bnot().neg().val(), d.succ().val());
        }
    }

    #[test]
    fn involution_property() {
        for i in 0..=255u8 {
            let d = ByteDatum::new(i);
            assert_eq!(d.neg().neg().val(), i);
            assert_eq!(d.bnot().bnot().val(), i);
        }
    }

    #[test]
    fn succ_pred_inverse() {
        for i in 0..=255u8 {
            let d = ByteDatum::new(i);
            assert_eq!(d.succ().pred().val(), i);
            assert_eq!(d.pred().succ().val(), i);
        }
    }

    #[test]
    fn byte_address_glyph_count() {
        let addr = ByteAddress::from_byte(0);
        assert_eq!(addr.as_str().chars().count(), 2);
        let addr = ByteAddress::from_byte(255);
        assert_eq!(addr.as_str().chars().count(), 2);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(ByteDatum::default(), ByteDatum::zero());
    }

    #[test]
    fn from_u8() {
        let d: ByteDatum = 42u8.into();
        assert_eq!(d.val(), 42);
        let v: u8 = d.into();
        assert_eq!(v, 42);
    }

    #[test]
    fn datum_trait_impl() {
        use uor_foundation::kernel::schema::Datum;
        let d = ByteDatum::new(42);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 42);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 8);
        // Per ADR-052: stratum is the ring-level index k. Q0 → 0.
        // (The old popcount interpretation lives on the inherent
        // `Datum::stratum` u8 method.)
        assert_eq!(Datum::<HoloPrimitives>::stratum(&d), 0);
    }

    #[test]
    fn address_trait_impl() {
        use uor_foundation::kernel::address::Element;
        let a = ByteAddress::from_byte(42);
        assert_eq!(Element::<HoloPrimitives>::length(&a), 2);
        assert_eq!(Element::<HoloPrimitives>::witt_length(&a), 8);
    }
}
