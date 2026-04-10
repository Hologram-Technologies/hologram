//! Ring implementations for hologram's quantum levels.
//!
//! The `UnifiedRing` type provides ring arithmetic at any quantum level,
//! replacing the per-level `ByteRing`/`WordRing`/`TripleRing`/`OctonionRing`.

pub mod byte_io;
mod byte_ring;
pub mod const_eval;
pub mod grounding;

pub use byte_ring::ByteInvolution;
pub use byte_ring::ByteRing;
pub use byte_ring::HoloDivisionAlgebra;
pub(crate) use byte_ring::{Q1_ALGEBRA, Q2_ALGEBRA, Q3_ALGEBRA};

use crate::op::{PrimOp, WittLevelExt};
use hologram_foundation::WittLevel;

/// Ring implementation for any quantum level.
///
/// All arithmetic delegates to `PrimOp::apply_*_u64` with the level's byte width.
/// This replaces the 4 separate ring types (`ByteRing`, `WordRing`, `TripleRing`,
/// `OctonionRing`) with a single parameterized implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnifiedRing {
    pub level: WittLevel,
}

impl UnifiedRing {
    /// Create a ring at the given quantum level.
    #[inline]
    pub const fn new(level: WittLevel) -> Self {
        Self { level }
    }

    /// Byte width of ring elements (1 for Q0, 2 for Q1, ... 8 for Q7).
    #[inline]
    pub fn byte_width(&self) -> u8 {
        self.level.byte_width()
    }

    /// Ring modulus: 2^bits for the underlying Witt length.
    /// Returns 0 for W64+ where the modulus overflows u128 at W128.
    #[inline]
    pub fn modulus(&self) -> u128 {
        let bits = self.level.witt_length() as u128;
        if bits >= 128 {
            0
        } else {
            1u128 << bits
        }
    }

    /// Ring bit width.
    #[inline]
    pub fn bits_width(&self) -> u32 {
        self.level.bits_width()
    }

    /// Cayley-Dickson algebra dimension: doubles per level up to 8 (octonions).
    /// R=1 (W8), C=2 (W16), H=4 (W24), O=8 (W32+).
    #[inline]
    pub fn algebra_dimension(&self) -> u64 {
        let k = (self.level.witt_length() / 8).saturating_sub(1) as u64;
        1u64 << k.min(3)
    }

    /// Whether the algebra is commutative (R and C only — W8 and W16).
    #[inline]
    pub fn is_commutative(&self) -> bool {
        self.level.witt_length() <= 16
    }

    /// Whether the algebra is associative (R, C, H only — W8, W16, W24).
    #[inline]
    pub fn is_associative(&self) -> bool {
        self.level.witt_length() <= 24
    }

    /// Apply a unary ring operation.
    #[inline(always)]
    pub fn apply_unary(&self, op: PrimOp, x: u64) -> u64 {
        op.apply_unary_u64(x, self.byte_width())
    }

    /// Apply a binary ring operation.
    #[inline(always)]
    pub fn apply_binary(&self, op: PrimOp, a: u64, b: u64) -> u64 {
        op.apply_binary_u64(a, b, self.byte_width())
    }
}

/// Named constants for the standard quantum levels.
pub const BYTE_RING_U: UnifiedRing = UnifiedRing::new(WittLevel::W8);
pub const WORD_RING_U: UnifiedRing = UnifiedRing::new(WittLevel::W16);
pub const TRIPLE_RING_U: UnifiedRing = UnifiedRing::new(WittLevel::W24);
pub const OCTONION_RING_U: UnifiedRing = UnifiedRing::new(WittLevel::W32);

/// The involution enum for the unified ring — Neg and Bnot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnifiedInvolution {
    /// Additive inverse: neg(x) = -x mod 2^n.
    Neg,
    /// Bitwise complement: bnot(x) = ~x.
    Bnot,
}

impl UnifiedInvolution {
    /// Apply the involution at the given byte width.
    #[inline(always)]
    pub fn apply(self, x: u64, byte_width: u8) -> u64 {
        match self {
            Self::Neg => PrimOp::Neg.apply_unary_u64(x, byte_width),
            Self::Bnot => PrimOp::Bnot.apply_unary_u64(x, byte_width),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_ring_q0_matches_byte_ring() {
        let ring = UnifiedRing::new(WittLevel::W8);
        assert_eq!(ring.byte_width(), 1);
        assert_eq!(ring.bits_width(), 8);
        assert_eq!(ring.modulus(), 256);
        assert_eq!(ring.algebra_dimension(), 1);
        assert!(ring.is_commutative());
        assert!(ring.is_associative());
    }

    #[test]
    fn unified_ring_q3_matches_octonion_ring() {
        let ring = UnifiedRing::new(WittLevel::W32);
        assert_eq!(ring.byte_width(), 4);
        assert_eq!(ring.bits_width(), 32);
        assert_eq!(ring.modulus(), 4_294_967_296);
        assert_eq!(ring.algebra_dimension(), 8);
        assert!(!ring.is_commutative());
        assert!(!ring.is_associative());
    }

    #[test]
    fn unified_ring_w64_native_u64() {
        // v0.2.0: `WittLevel::new(64)` is the 64-bit Witt level (formerly Q7).
        let ring = UnifiedRing::new(WittLevel::new(64));
        assert_eq!(ring.byte_width(), 8);
        assert_eq!(ring.bits_width(), 64);
        assert_eq!(ring.algebra_dimension(), 8); // capped at octonion
    }

    #[test]
    fn unified_ring_arithmetic() {
        let ring = UnifiedRing::new(WittLevel::W8);
        assert_eq!(ring.apply_binary(PrimOp::Add, 200, 100), 44); // 300 mod 256
        assert_eq!(ring.apply_unary(PrimOp::Neg, 1), 255);

        let ring = UnifiedRing::new(WittLevel::W32);
        assert_eq!(ring.apply_binary(PrimOp::Add, u32::MAX as u64, 1), 0);
    }

    #[test]
    fn unified_ring_critical_identity_all_levels() {
        // v0.2.0: iterate Witt levels W8 .. W64 (formerly Q0 .. Q7).
        for k in 0..=7u32 {
            let bits = 8 * (k + 1);
            let ring = UnifiedRing::new(WittLevel::new(bits));
            let max_test = if k == 0 { 256 } else { 64 };
            for x in 0..max_test as u64 {
                let lhs = ring.apply_unary(PrimOp::Neg, ring.apply_unary(PrimOp::Bnot, x));
                let rhs = ring.apply_unary(PrimOp::Succ, x);
                assert_eq!(lhs, rhs, "critical identity failed at W{bits}, x={x}");
            }
        }
    }

    #[test]
    fn unified_involution_apply() {
        assert_eq!(UnifiedInvolution::Neg.apply(1, 1), 255);
        assert_eq!(UnifiedInvolution::Bnot.apply(0, 1), 255);
        assert_eq!(UnifiedInvolution::Neg.apply(1, 4), u32::MAX as u64);
    }

    #[test]
    fn cayley_dickson_dimension_chain() {
        assert_eq!(UnifiedRing::new(WittLevel::W8).algebra_dimension(), 1); // R
        assert_eq!(UnifiedRing::new(WittLevel::W16).algebra_dimension(), 2); // C
        assert_eq!(UnifiedRing::new(WittLevel::W24).algebra_dimension(), 4); // H
        assert_eq!(UnifiedRing::new(WittLevel::W32).algebra_dimension(), 8); // O
                                                                             // Beyond W32 (40-bit and up) the algebra is capped at octonion (8).
        assert_eq!(UnifiedRing::new(WittLevel::new(40)).algebra_dimension(), 8);
    }
}
