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
    pub const fn new(value: u8) -> Self {
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
    pub const fn neg(self) -> Self {
        Self::new(self.value.wrapping_neg())
    }

    /// Bitwise complement.
    #[inline]
    #[must_use]
    pub const fn bnot(self) -> Self {
        Self::new(!self.value)
    }

    /// Successor: `(x + 1) mod 256`.
    #[inline]
    #[must_use]
    pub const fn succ(self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: `(x - 1) mod 256`.
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
        Self::ZERO
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
}

impl ByteAddress {
    /// The zero address (byte value 0).
    pub const ZERO: Self = Self::from_byte(0);

    /// Create a Braille address from a byte value.
    #[must_use]
    pub const fn from_byte(value: u8) -> Self {
        let lo6 = value & 0x3F;
        let hi2 = (value >> 6) & 0x03;
        // Braille U+2800..U+283F → UTF-8: E2 A0 {80+offset}
        Self {
            value,
            glyph_buf: [0xE2, 0xA0, 0x80 + lo6, 0xE2, 0xA0, 0x80 + hi2],
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

    fn quantum(&self) -> u64 {
        8
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Address = ByteAddress;

    fn glyph(&self) -> &Self::Address {
        &self.address
    }
}

impl uor_foundation::kernel::address::Address<HoloPrimitives> for ByteAddress {
    fn glyph(&self) -> &str {
        self.as_str()
    }

    fn length(&self) -> u64 {
        2
    }

    fn addresses(&self) -> &str {
        ""
    }

    fn digest(&self) -> &str {
        self.as_str()
    }

    fn quantum(&self) -> u64 {
        8
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
    pub level: uor_foundation::QuantumLevel,
}

impl RingDatum {
    /// Create a datum at the given quantum level.
    /// The value is masked to the ring modulus.
    #[inline]
    pub fn new(value: u64, level: uor_foundation::QuantumLevel) -> Self {
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
        Self::new(d.val() as u64, uor_foundation::QuantumLevel::Q0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_datum_masks_value() {
        let d = RingDatum::new(300, uor_foundation::QuantumLevel::Q0);
        assert_eq!(d.value, 44); // 300 & 0xFF = 44
    }

    #[test]
    fn ring_datum_q7_no_mask() {
        let d = RingDatum::new(u64::MAX, uor_foundation::QuantumLevel::new(7));
        assert_eq!(d.value, u64::MAX);
    }

    #[test]
    fn ring_datum_from_byte_datum() {
        let bd = ByteDatum::new(42);
        let rd = RingDatum::from_byte_datum(&bd);
        assert_eq!(rd.value, 42);
        assert_eq!(rd.level, uor_foundation::QuantumLevel::Q0);
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
        assert_eq!(ByteDatum::default(), ByteDatum::ZERO);
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
        assert_eq!(Datum::<HoloPrimitives>::quantum(&d), 8);
        assert_eq!(Datum::<HoloPrimitives>::stratum(&d), 3);
    }

    #[test]
    fn address_trait_impl() {
        use uor_foundation::kernel::address::Address;
        let a = ByteAddress::from_byte(42);
        assert_eq!(Address::<HoloPrimitives>::length(&a), 2);
        assert_eq!(Address::<HoloPrimitives>::quantum(&a), 8);
    }
}
