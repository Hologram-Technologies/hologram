//! Datum: an element of the ring R_n at quantum level Q.

use crate::address::Address;
use crate::level::QuantumLevel;
use crate::word::RingWord;
use crate::PrismPrimitives;

/// An element of the ring R_n at quantum level Q.
/// const-constructible, zero allocation.
pub struct Datum<Q: QuantumLevel> {
    value: Q::Word,
    spectrum_buf: [u8; 128], // binary string, max 128 bits
    spectrum_len: u8,
    address: Address<Q>,
}

impl<Q: QuantumLevel> Datum<Q> {
    /// Create a datum from a raw ring value.
    pub fn new(value: Q::Word) -> Self {
        let mut spectrum_buf = [b'0'; 128];
        let bits = Q::BITS.min(128) as usize;
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
    pub fn val(&self) -> Q::Word {
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

impl<Q: QuantumLevel> uor_foundation::kernel::schema::Datum<PrismPrimitives> for Datum<Q> {
    fn value(&self) -> u64 {
        self.value.to_u64()
    }

    /// Witt level n of this datum, where the ring is Z/(2^n)Z.
    /// Was named `quantum` in uor-foundation 0.1.x.
    fn witt_length(&self) -> u64 {
        Q::BITS as u64
    }

    /// Per ADR-052: `stratum` is the ring-level index k (where k = INDEX
    /// in `QuantumLevel`). The previous u8-popcount interpretation is
    /// kept on the inherent method `Datum::stratum` for callers that
    /// want it.
    fn stratum(&self) -> u64 {
        Q::INDEX as u64
    }

    /// Per ADR-052: `spectrum` is the underlying ring value cast to
    /// u64. The binary-string representation remains available via
    /// the inherent method `Datum::spectrum`.
    fn spectrum(&self) -> u64 {
        self.value.to_u64()
    }

    type Element = Address<Q>;

    fn element(&self) -> &Self::Element {
        &self.address
    }
}
