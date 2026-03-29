//! Address: Braille-encoded address for a ring element.
//!
//! Each Braille character U+2800..U+283F encodes 6 bits.
//! UTF-8 encoding: [0xE2, 0xA0|hi, 0x80|lo] per glyph.

use crate::level::QuantumLevel;
use crate::word::RingWord;
use crate::PrismPrimitives;

/// Braille-encoded address for a ring element at quantum level Q.
pub struct Address<Q: QuantumLevel> {
    glyph_buf: [u8; 72], // max 24 Braille chars × 3 bytes = 72
    glyph_len: u8,
    bits: u32,
    _phantom: core::marker::PhantomData<Q>,
}

impl<Q: QuantumLevel> Address<Q> {
    /// Create an address from a ring word value.
    pub fn from_word(value: Q::Word) -> Self {
        let bits = Q::BITS;
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
            // Braille U+2800 + six_bits → UTF-8: E2 A0 (80+six_bits)
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

        Self {
            glyph_buf,
            glyph_len,
            bits,
            _phantom: core::marker::PhantomData,
        }
    }

    /// The Braille glyph string.
    pub fn as_str(&self) -> &str {
        // SAFETY: we only write valid UTF-8 Braille sequences
        unsafe { core::str::from_utf8_unchecked(&self.glyph_buf[..self.glyph_len as usize]) }
    }
}

impl<Q: QuantumLevel> uor_foundation::kernel::address::Address<PrismPrimitives> for Address<Q> {
    fn glyph(&self) -> &str {
        self.as_str()
    }

    fn length(&self) -> u64 {
        self.bits.div_ceil(6) as u64
    }

    fn addresses(&self) -> &str {
        ""
    }

    fn digest(&self) -> &str {
        self.as_str()
    }

    fn quantum(&self) -> u64 {
        self.bits as u64
    }

    fn digest_algorithm(&self) -> &str {
        "blake3"
    }

    fn canonical_bytes(&self) -> &str {
        self.as_str()
    }
}
