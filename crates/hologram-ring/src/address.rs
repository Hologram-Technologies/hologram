//! Address: content-addressed identifier for a ring element.
//!
//! Per uor-foundation v0.2.0 Amendment 43:
//! - `canonical_bytes` = hex(`header(k) || le_bytes(value, k+1)`)
//! - `digest` = hex(BLAKE3(`canonical_raw`)) = 64 hex chars
//! - `digest_algorithm` = `"blake3"`
//!
//! In v0.2.0 the foundation renamed the `kernel::address::Address` trait to
//! `kernel::address::Element`. The hologram type `Address<W>` keeps its
//! local name, but its trait impl points at v0.2.0's `Element`.

use crate::level::WittLevelMarker;
use crate::word::RingWord;
use crate::PrismPrimitives;

/// Static hex table for fast nibble→char conversion.
const HEX: [u8; 16] = *b"0123456789abcdef";

/// Encode raw bytes as lowercase hex into a fixed-size buffer.
fn hex_encode(raw: &[u8], out: &mut [u8]) -> usize {
    for (i, &b) in raw.iter().enumerate() {
        out[i * 2] = HEX[(b >> 4) as usize];
        out[i * 2 + 1] = HEX[(b & 0x0F) as usize];
    }
    raw.len() * 2
}

/// Content-addressed identifier for a ring element at Witt level W.
///
/// Stores the Amendment 43 canonical encoding as hex and its BLAKE3 digest.
/// Max canonical_raw = 17 bytes (1 header + 16 for W128) → 34 hex chars.
pub struct Address<W: WittLevelMarker> {
    /// hex(header(k) || le_bytes(value, k+1)) — up to 34 hex chars.
    canonical_hex: [u8; 34],
    /// Number of valid hex chars in canonical_hex.
    canonical_len: u8,
    /// hex(BLAKE3(canonical_raw)) — 64 lowercase hex chars.
    digest_hex: [u8; 64],
    bits: u32,
    _phantom: core::marker::PhantomData<W>,
}

impl<W: WittLevelMarker> Address<W> {
    /// Create a content-addressed identifier from a ring word value.
    ///
    /// Computes Amendment 43 canonical bytes and BLAKE3 digest at construction.
    pub fn from_word(value: W::Word) -> Self {
        let bits = W::BITS;
        let byte_width = (bits / 8) as usize;
        let header = (byte_width - 1) as u8;

        // Build canonical_raw = header || le_bytes(value, byte_width)
        // Max 17 bytes (1 header + 16 for W128)
        let mut canonical_raw = [0u8; 17];
        canonical_raw[0] = header;
        let val = value.to_u64();
        let le = val.to_le_bytes();
        let copy_len = byte_width.min(8);
        canonical_raw[1..1 + copy_len].copy_from_slice(&le[..copy_len]);
        let raw_len = 1 + byte_width;

        let mut canonical_hex = [0u8; 34];
        let canonical_len = hex_encode(&canonical_raw[..raw_len], &mut canonical_hex);

        let hash = blake3::hash(&canonical_raw[..raw_len]);
        let mut digest_hex = [0u8; 64];
        hex_encode(hash.as_bytes(), &mut digest_hex);

        Self {
            canonical_hex,
            canonical_len: canonical_len as u8,
            digest_hex,
            bits,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<W: WittLevelMarker> hologram_foundation::address::Element<PrismPrimitives> for Address<W> {
    /// Byte length of the canonical encoding (1 + byte_width raw bytes).
    fn length(&self) -> u64 {
        (1 + self.bits / 8) as u64
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

    /// v0.2.0 renamed `quantum()` to `witt_length()`.
    fn witt_length(&self) -> u64 {
        self.bits as u64
    }

    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    /// Amendment 43 canonical encoding as hex.
    fn canonical_bytes(&self) -> &str {
        // SAFETY: canonical_hex is valid ASCII hex.
        unsafe {
            core::str::from_utf8_unchecked(&self.canonical_hex[..self.canonical_len as usize])
        }
    }
}
