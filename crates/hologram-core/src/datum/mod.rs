//! ByteDatum: an element of Z/256Z implementing uor-foundation Datum.

use crate::lut::q0;
use crate::HoloPrimitives;

/// An element of Z/256Z at Witt level W8 (8-bit).
#[derive(Clone, PartialEq, Eq, Hash)]
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

    /// The content-addressed identifier for this datum.
    #[inline]
    #[must_use]
    pub fn address(&self) -> &ByteAddress {
        &self.address
    }

    /// Negation: `(-x) mod 256`.
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

    /// Successor: `(x + 1) mod 256`.
    #[inline]
    #[must_use]
    pub fn succ(&self) -> Self {
        Self::new(self.value.wrapping_add(1))
    }

    /// Predecessor: `(x - 1) mod 256`.
    #[inline]
    #[must_use]
    pub fn pred(&self) -> Self {
        Self::new(self.value.wrapping_sub(1))
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
        Self::new(0)
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

/// Content-addressed identifier for a byte datum at W8.
///
/// Per uor-foundation v0.2.0 Amendment 43 section 2:
/// - `canonical_bytes` = hex(`header(0) || le_bytes(value, 1)`) = 4 hex chars
/// - `digest` = hex(BLAKE3(`canonical_raw`)) = 64 hex chars
/// - `digest_algorithm` = `"blake3"`
///
/// The hex buffers are pre-computed at construction for O(1) access via
/// the `Element` trait. The struct is larger than v0.1.4's glyph-based
/// `ByteAddress` (72 bytes vs 8 bytes) but lives only in the compiler
/// layer, never in the execution tape or serialised graph.
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct ByteAddress {
    value: u8,
    /// hex(header(0) || le_bytes(value, 1)) — 4 hex chars for 2 raw bytes.
    canonical_hex: [u8; 4],
    /// hex(BLAKE3(canonical_raw)) — 64 lowercase hex chars.
    digest_hex: [u8; 64],
}

/// Static hex table for fast nibble→char conversion.
pub(crate) const HEX: [u8; 16] = *b"0123456789abcdef";

/// Encode raw bytes as lowercase hex into a fixed-size buffer.
/// Returns the number of hex chars written (always `raw.len() * 2`).
pub(crate) fn hex_encode(raw: &[u8], out: &mut [u8]) -> usize {
    for (i, &b) in raw.iter().enumerate() {
        out[i * 2] = HEX[(b >> 4) as usize];
        out[i * 2 + 1] = HEX[(b & 0x0F) as usize];
    }
    raw.len() * 2
}

impl ByteAddress {
    /// The zero address (byte value 0).
    pub fn zero() -> Self {
        Self::from_byte(0)
    }

    /// Create a content-addressed identifier from a byte value.
    ///
    /// Computes Amendment 43 canonical bytes and BLAKE3 digest at construction.
    #[must_use]
    pub fn from_byte(value: u8) -> Self {
        // Amendment 43: canonical = header(k) || le_bytes(x, k+1)
        // For W8 (k=0): header = 0x00, le_bytes(value, 1) = [value]
        let canonical_raw = [0x00u8, value];

        let mut canonical_hex = [0u8; 4];
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

    /// The raw byte value this address represents.
    pub fn value(&self) -> u8 {
        self.value
    }
}

impl core::fmt::Debug for ByteAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ByteAddress({})", self.value)
    }
}

// --- uor-foundation trait implementations ---

impl hologram_foundation::schema::Datum<HoloPrimitives> for ByteDatum {
    fn value(&self) -> u64 {
        self.value as u64
    }

    fn witt_length(&self) -> u64 {
        8
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        self.value as u64
    }

    type Element = ByteAddress;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}

impl hologram_foundation::address::Element<HoloPrimitives> for ByteAddress {
    /// Byte length of the canonical encoding (2 raw bytes for W8).
    fn length(&self) -> u64 {
        2
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
        8
    }

    #[inline]
    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Amendment 43 canonical encoding: hex(header(0) || le_bytes(value, 1)).
    #[inline]
    fn canonical_bytes(&self) -> &str {
        // SAFETY: canonical_hex is valid ASCII hex.
        unsafe { core::str::from_utf8_unchecked(&self.canonical_hex) }
    }
}

// ── RingDatum: unified datum for any quantum level ──────────────────────────

use crate::op::WittLevelExt;

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
    pub level: hologram_foundation::WittLevel,
}

impl RingDatum {
    /// Create a datum at the given quantum level.
    /// The value is masked to the ring modulus.
    #[inline]
    pub fn new(value: u64, level: hologram_foundation::WittLevel) -> Self {
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
        Self::new(d.val() as u64, hologram_foundation::WittLevel::W8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_datum_masks_value() {
        let d = RingDatum::new(300, hologram_foundation::WittLevel::W8);
        assert_eq!(d.value, 44); // 300 & 0xFF = 44
    }

    #[test]
    fn ring_datum_w64_no_mask() {
        // v0.2.0: `WittLevel::new(64)` constructs the 64-bit witt level
        // (formerly called Q7). At 64 bits, the mask saturates to u64::MAX
        // and no truncation happens.
        let d = RingDatum::new(u64::MAX, hologram_foundation::WittLevel::new(64));
        assert_eq!(d.value, u64::MAX);
    }

    #[test]
    fn ring_datum_from_byte_datum() {
        let bd = ByteDatum::new(42);
        let rd = RingDatum::from_byte_datum(&bd);
        assert_eq!(rd.value, 42);
        assert_eq!(rd.level, hologram_foundation::WittLevel::W8);
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
    fn byte_address_digest_is_blake3() {
        use hologram_foundation::address::Element;
        let addr = ByteAddress::from_byte(42);
        // digest() returns 64 hex chars (256-bit BLAKE3)
        assert_eq!(addr.digest().len(), 64);
        // canonical_bytes() returns 4 hex chars (2 raw bytes: header(0) || [42])
        assert_eq!(addr.canonical_bytes().len(), 4);
        // canonical_bytes is "002a" (header=0x00, value=0x2a=42)
        assert_eq!(addr.canonical_bytes(), "002a");
        // digest_algorithm is "blake3"
        assert_eq!(addr.digest_algorithm(), "blake3");
        // Verify: digest = hex(blake3(unhex(canonical_bytes)))
        let canonical_raw = [0x00u8, 42u8];
        let expected_hash = blake3::hash(&canonical_raw);
        let mut expected_hex = [0u8; 64];
        hex_encode(expected_hash.as_bytes(), &mut expected_hex);
        let expected = core::str::from_utf8(&expected_hex).unwrap();
        assert_eq!(addr.digest(), expected);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(ByteDatum::default(), ByteDatum::new(0));
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
        use hologram_foundation::schema::Datum;
        let d = ByteDatum::new(42);
        assert_eq!(Datum::<HoloPrimitives>::value(&d), 42);
        assert_eq!(Datum::<HoloPrimitives>::witt_length(&d), 8);
        assert_eq!(Datum::<HoloPrimitives>::stratum(&d), 3);
    }

    #[test]
    fn address_trait_impl() {
        use hologram_foundation::address::Element;
        let a = ByteAddress::from_byte(42);
        assert_eq!(Element::<HoloPrimitives>::length(&a), 2);
        assert_eq!(Element::<HoloPrimitives>::witt_length(&a), 8);
    }
}
