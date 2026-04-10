//! Witt level markers: type-level dispatchers for the parametric ring.
//!
//! Each level is a zero-sized type (ZST) — monomorphized away at compile
//! time. The level determines the bit width, ring R_n = Z/(2^n)Z, and the
//! carrier word type.
//!
//! # Naming
//!
//! In v0.1.4 these markers were named `Q0`/`Q1`/`Q3`/`Q7`/`Q15` after their
//! quantum-level *index*. v0.2.0 names them by *bit width* — `W8`/`W16`/
//! `W32`/`W64`/`W128` — to align with `hologram_foundation::WittLevel`'s
//! naming convention. The trait `QuantumLevel` becomes `WittLevelMarker` to
//! disambiguate from the foundation's runtime `WittLevel` struct.
//!
//! # Performance
//!
//! Zero runtime cost — every type is a ZST, every constant is `const`, and
//! all generics are monomorphized at compile time. **Perf: NEUTRAL** vs
//! v0.1.4 (pure rename, no behavioral change).

use crate::word::RingWord;

/// Type-level Witt level marker. Zero-sized; monomorphized away.
///
/// The bit width follows the convention `BITS = 8 * (INDEX + 1)` where
/// `INDEX` is the historical Q-level index from v0.1.4. New code should
/// rely on `BITS` directly; `INDEX` is preserved for compatibility with
/// `hologram_core::RingLevel` which still uses 0/1/2/3 indices.
pub trait WittLevelMarker: Copy + core::fmt::Debug + 'static {
    /// Bit width of the ring at this level. Equal to the foundation's
    /// `WittLevel::witt_length()`.
    const BITS: u32;
    /// Historical index `k` such that `BITS = 8 * (k + 1)`.
    const INDEX: u32;
    /// Carrier word type for this level (u8/u16/u32/u64/u128).
    type Word: RingWord;
}

/// Witt level 8: Z/256Z, carrier u8. Maps to `hologram_foundation::WittLevel::W8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct W8;

impl WittLevelMarker for W8 {
    const BITS: u32 = 8;
    const INDEX: u32 = 0;
    type Word = u8;
}

/// Witt level 16: Z/65536Z, carrier u16. Maps to `hologram_foundation::WittLevel::W16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct W16;

impl WittLevelMarker for W16 {
    const BITS: u32 = 16;
    const INDEX: u32 = 1;
    type Word = u16;
}

/// Witt level 32: Z/2^32 Z, carrier u32. Maps to `hologram_foundation::WittLevel::W32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct W32;

impl WittLevelMarker for W32 {
    const BITS: u32 = 32;
    const INDEX: u32 = 3;
    type Word = u32;
}

/// Witt level 64: Z/2^64 Z, carrier u64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct W64;

impl WittLevelMarker for W64 {
    const BITS: u32 = 64;
    const INDEX: u32 = 7;
    type Word = u64;
}

/// Witt level 128: Z/2^128 Z, carrier u128.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct W128;

impl WittLevelMarker for W128 {
    const BITS: u32 = 128;
    const INDEX: u32 = 15;
    type Word = u128;
}
