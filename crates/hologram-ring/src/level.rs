//! QuantumLevel: marker trait for quantum levels.
//!
//! Each level is a zero-sized type (ZST) — monomorphized away at compile time.
//! The quantum level determines the bit width, ring, and carrier type.

use crate::word::RingWord;

/// Marker trait for quantum levels. Zero-sized — monomorphized away.
///
/// The bit width follows the formula: `BITS = 8 * (INDEX + 1)`.
/// The ring at level k is R_n = Z/(2^n)Z where n = BITS.
pub trait QuantumLevel: Copy + core::fmt::Debug + 'static {
    /// Bit width: 8 * (INDEX + 1).
    const BITS: u32;
    /// Level index k.
    const INDEX: u32;
    /// The carrier word type for this level.
    type Word: RingWord;
}

/// Q0: 8-bit, Z/256Z, carrier u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q0;

impl QuantumLevel for Q0 {
    const BITS: u32 = 8;
    const INDEX: u32 = 0;
    type Word = u8;
}

/// Q1: 16-bit, Z/65536Z, carrier u16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q1;

impl QuantumLevel for Q1 {
    const BITS: u32 = 16;
    const INDEX: u32 = 1;
    type Word = u16;
}

/// Q3: 32-bit, Z/2^32Z, carrier u32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q3;

impl QuantumLevel for Q3 {
    const BITS: u32 = 32;
    const INDEX: u32 = 3;
    type Word = u32;
}

/// Q7: 64-bit, Z/2^64Z, carrier u64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q7;

impl QuantumLevel for Q7 {
    const BITS: u32 = 64;
    const INDEX: u32 = 7;
    type Word = u64;
}

/// Q15: 128-bit, Z/2^128Z, carrier u128.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q15;

impl QuantumLevel for Q15 {
    const BITS: u32 = 128;
    const INDEX: u32 = 15;
    type Word = u128;
}
