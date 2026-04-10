//! Datum: an element of the ring R_n at Witt level W.

use crate::address::Address;
use crate::level::WittLevelMarker;
use crate::word::RingWord;
use crate::PrismPrimitives;

/// An element of the ring R_n at Witt level W.
/// const-constructible, zero allocation.
pub struct Datum<W: WittLevelMarker> {
    value: W::Word,
    spectrum_buf: [u8; 128], // binary string, max 128 bits
    spectrum_len: u8,
    address: Address<W>,
}

impl<W: WittLevelMarker> Datum<W> {
    /// Create a datum from a raw ring value.
    pub fn new(value: W::Word) -> Self {
        let mut spectrum_buf = [b'0'; 128];
        let bits = W::BITS.min(128) as usize;
        let val_u64 = value.to_u64();
        for (i, byte) in spectrum_buf[..bits.min(64)].iter_mut().enumerate() {
            if val_u64 & (1u64 << (bits.min(64) - 1 - i)) != 0 {
                *byte = b'1';
            }
        }
        Self {
            value,
            spectrum_buf,
            spectrum_len: bits as u8,
            address: Address::from_word(value),
        }
    }

    /// The raw ring value.
    #[inline]
    pub fn val(&self) -> W::Word {
        self.value
    }

    /// Stratum (popcount).
    #[inline]
    pub fn stratum(&self) -> u32 {
        self.value.count_ones()
    }

    /// Binary spectrum string.
    pub fn spectrum(&self) -> &str {
        // SAFETY: spectrum_buf only contains b'0' and b'1'
        unsafe { core::str::from_utf8_unchecked(&self.spectrum_buf[..self.spectrum_len as usize]) }
    }
}

impl<W: WittLevelMarker> hologram_foundation::schema::Datum<PrismPrimitives> for Datum<W> {
    fn value(&self) -> u64 {
        self.value.to_u64()
    }

    /// v0.2.0 renamed `quantum()` to `witt_length()`. Returns the bit width
    /// of this datum's ring (8/16/32/64/128).
    fn witt_length(&self) -> u64 {
        W::BITS as u64
    }

    fn stratum(&self) -> u64 {
        self.value.count_ones() as u64
    }

    fn spectrum(&self) -> u64 {
        // v0.2.0 `spectrum` returns P::NonNegativeInteger (u64 for
        // PrismPrimitives). Return the underlying numeric value; the
        // binary-string representation is still available via the inherent
        // `Datum::spectrum` method.
        self.value.to_u64()
    }

    /// v0.2.0 renamed the associated type `Address` to `Element` and
    /// changed its bound from `kernel::address::Address<P>` to
    /// `kernel::address::Element<P>`.
    type Element = Address<W>;

    /// v0.2.0 renamed the accessor `glyph()` to `element()`.
    fn element(&self) -> &Self::Element {
        &self.address
    }
}
