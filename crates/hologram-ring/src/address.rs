//! Element: content-addressable identifier for a ring element.
//!
//! Per ADR-052, `Element::digest_algorithm` is BLAKE3 and
//! `canonical_bytes` follow Amendment 43 §2 — `header(k) || le_bytes(x, k+1)`.
//! The header byte is the ring-level index k (0..3 for Q0..Q3, 7 for
//! Q7, 15 for Q15); the value bytes are the underlying ring word as
//! little-endian.
//!
//! The Braille glyph string from the legacy address is still computed
//! and exposed via [`Address::glyph_str`] for display — the trait
//! itself no longer requires it.

use crate::level::QuantumLevel;
use crate::word::RingWord;
use crate::PrismPrimitives;

/// Maximum canonical-bytes length: 1 header byte + 16 value bytes (Q15).
const MAX_CANONICAL_BYTES: usize = 1 + 16;
/// Digest string format: `"blake3:" + 64 hex chars` = 71 bytes.
const DIGEST_STR_LEN: usize = 7 + 64;

/// Content-addressable element for a ring datum at level Q.
///
/// Implements [`uor_foundation::kernel::address::Element`].
pub struct Address<Q: QuantumLevel> {
    /// Braille glyph buffer: max 24 chars × 3 bytes UTF-8.
    glyph_buf: [u8; 72],
    glyph_len: u8,
    /// Canonical bytes per Amendment 43 §2.
    canonical_buf: [u8; MAX_CANONICAL_BYTES],
    canonical_len: u8,
    /// Pre-computed `"blake3:<hex>"` digest string.
    digest_buf: [u8; DIGEST_STR_LEN],
    bits: u32,
    _phantom: core::marker::PhantomData<Q>,
}

impl<Q: QuantumLevel> Address<Q> {
    /// Create an address from a ring word value.
    pub fn from_word(value: Q::Word) -> Self {
        let bits = Q::BITS;
        let index = Q::INDEX;
        // Braille rendering (legacy display surface).
        let num_glyphs = bits.div_ceil(6) as usize;
        let mut glyph_buf = [0u8; 72];
        let mut glyph_len = 0u8;
        let val = value.to_u64();
        for g in 0..num_glyphs {
            let shift = if bits <= 64 {
                let total_bits = num_glyphs * 6;
                total_bits.saturating_sub((g + 1) * 6)
            } else {
                0
            };
            let six_bits = ((val >> shift) & 0x3F) as u8;
            let code = 0x2800 + six_bits as u32;
            let b1 = 0xE0 | ((code >> 12) & 0x0F) as u8;
            let b2 = 0x80 | ((code >> 6) & 0x3F) as u8;
            let b3 = 0x80 | (code & 0x3F) as u8;
            let offset = g * 3;
            if offset + 3 <= 72 {
                glyph_buf[offset] = b1;
                glyph_buf[offset + 1] = b2;
                glyph_buf[offset + 2] = b3;
                glyph_len = (offset + 3) as u8;
            }
        }

        // Canonical bytes per Amendment 43 §2: `header(k) || le_bytes(x, k+1)`.
        let mut canonical_buf = [0u8; MAX_CANONICAL_BYTES];
        canonical_buf[0] = index as u8;
        let value_bytes = (index as usize) + 1;
        // RingWord is one of u8/u16/u32/u64/u128; copy the LE representation.
        // We do a generic dance via u128 since every variant fits.
        let v = value.to_u128_le();
        for i in 0..value_bytes {
            canonical_buf[1 + i] = (v >> (i * 8)) as u8;
        }
        let canonical_len = 1 + value_bytes as u8;

        // Pre-compute the BLAKE3 digest of the canonical bytes.
        let hash = blake3::hash(&canonical_buf[..canonical_len as usize]);
        let mut digest_buf = [0u8; DIGEST_STR_LEN];
        digest_buf[..7].copy_from_slice(b"blake3:");
        for (i, byte) in hash.as_bytes().iter().enumerate() {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            digest_buf[7 + i * 2] = HEX[(byte >> 4) as usize];
            digest_buf[7 + i * 2 + 1] = HEX[(byte & 0x0F) as usize];
        }

        Self {
            glyph_buf,
            glyph_len,
            canonical_buf,
            canonical_len,
            digest_buf,
            bits,
            _phantom: core::marker::PhantomData,
        }
    }

    /// The Braille glyph string for display (legacy surface).
    pub fn glyph_str(&self) -> &str {
        // SAFETY: only valid UTF-8 Braille sequences are written above.
        unsafe { core::str::from_utf8_unchecked(&self.glyph_buf[..self.glyph_len as usize]) }
    }
}

impl<Q: QuantumLevel> uor_foundation::kernel::address::Element<PrismPrimitives> for Address<Q> {
    fn length(&self) -> u64 {
        Q::BITS.div_ceil(6) as u64
    }

    fn addresses(&self) -> &str {
        self.glyph_str()
    }

    fn digest(&self) -> &str {
        // SAFETY: digest_buf only contains ASCII bytes (`blake3:` + lowercase hex).
        unsafe { core::str::from_utf8_unchecked(&self.digest_buf) }
    }

    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_buf[..self.canonical_len as usize]
    }

    fn witt_length(&self) -> u64 {
        self.bits as u64
    }
}
