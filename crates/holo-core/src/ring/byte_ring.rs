//! ByteRing: the ring Z/256Z at quantum level Q0.

use crate::datum::ByteDatum;
use crate::lut::arith;
use crate::HoloPrimitives;
use uor_foundation::enums::{GeometricCharacter, QuantumLevel};

/// The ring Z/256Z at quantum level 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteRing;

impl ByteRing {
    /// Ring bit width.
    pub const QUANTUM: u64 = 8;

    /// Ring modulus (2^8 = 256).
    pub const MODULUS: u64 = 256;

    /// Add two byte values via LUT.
    #[inline]
    #[must_use]
    pub fn add(a: u8, b: u8) -> u8 {
        arith::add_q0(a, b)
    }

    /// Subtract two byte values via LUT.
    #[inline]
    #[must_use]
    pub fn sub(a: u8, b: u8) -> u8 {
        arith::sub_q0(a, b)
    }

    /// Multiply two byte values via LUT.
    #[inline]
    #[must_use]
    pub fn mul(a: u8, b: u8) -> u8 {
        arith::mul_q0(a, b)
    }
}

static GENERATOR: ByteDatum = ByteDatum::PI1;
static NEG_INV: ByteInvolution = ByteInvolution::Neg;
static BNOT_INV: ByteInvolution = ByteInvolution::Bnot;

impl uor_foundation::kernel::schema::Ring<HoloPrimitives> for ByteRing {
    fn ring_quantum(&self) -> u64 {
        Self::QUANTUM
    }

    fn modulus(&self) -> u64 {
        Self::MODULUS
    }

    type Datum = ByteDatum;

    fn generator(&self) -> &Self::Datum {
        &GENERATOR
    }

    type Involution = ByteInvolution;

    fn negation(&self) -> &Self::Involution {
        &NEG_INV
    }

    fn complement(&self) -> &Self::Involution {
        &BNOT_INV
    }

    fn at_quantum_level(&self) -> QuantumLevel {
        QuantumLevel::Q0
    }
}

/// The two involutions of the ring Z/256Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteInvolution {
    /// Ring reflection: neg(x) = (-x) mod 256.
    Neg,
    /// Hypercube reflection: bnot(x) = 255 ^ x.
    Bnot,
}

impl ByteInvolution {
    /// Apply this involution to a byte value.
    #[inline]
    #[must_use]
    pub const fn apply(self, x: u8) -> u8 {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
        }
    }
}

impl uor_foundation::kernel::op::Operation<HoloPrimitives> for ByteInvolution {
    fn arity(&self) -> u64 {
        1
    }

    fn has_geometric_character(&self) -> GeometricCharacter {
        match self {
            Self::Neg => GeometricCharacter::RingReflection,
            Self::Bnot => GeometricCharacter::HypercubeReflection,
        }
    }

    type OperationTarget = ByteInvolution;

    fn inverse(&self) -> &Self::OperationTarget {
        // Involutions are self-inverse
        match self {
            Self::Neg => &NEG_INV,
            Self::Bnot => &BNOT_INV,
        }
    }

    fn composed_of(&self) -> &str {
        match self {
            Self::Neg => "neg",
            Self::Bnot => "bnot",
        }
    }
}

impl uor_foundation::kernel::op::UnaryOp<HoloPrimitives> for ByteInvolution {}
impl uor_foundation::kernel::op::Involution<HoloPrimitives> for ByteInvolution {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_add() {
        assert_eq!(ByteRing::add(100, 200), 44); // wrapping
        assert_eq!(ByteRing::add(0, 0), 0);
        assert_eq!(ByteRing::add(1, 255), 0);
    }

    #[test]
    fn ring_sub() {
        assert_eq!(ByteRing::sub(0, 1), 255);
        assert_eq!(ByteRing::sub(100, 50), 50);
    }

    #[test]
    fn ring_mul() {
        assert_eq!(ByteRing::mul(3, 5), 15);
        assert_eq!(ByteRing::mul(16, 16), 0); // 256 mod 256
    }

    #[test]
    fn involution_neg() {
        let neg = ByteInvolution::Neg;
        for i in 0..=255u8 {
            assert_eq!(neg.apply(neg.apply(i)), i);
        }
    }

    #[test]
    fn involution_bnot() {
        let bnot = ByteInvolution::Bnot;
        for i in 0..=255u8 {
            assert_eq!(bnot.apply(bnot.apply(i)), i);
        }
    }

    #[test]
    fn ring_trait() {
        use uor_foundation::kernel::schema::Ring;
        let r = ByteRing;
        assert_eq!(r.ring_quantum(), 8);
        assert_eq!(r.modulus(), 256);
        assert_eq!(r.at_quantum_level(), QuantumLevel::Q0);
        assert_eq!(r.generator().val(), 1);
    }

    #[test]
    fn critical_identity_via_involutions() {
        let neg = ByteInvolution::Neg;
        let bnot = ByteInvolution::Bnot;
        for i in 0..=255u8 {
            assert_eq!(neg.apply(bnot.apply(i)), i.wrapping_add(1));
        }
    }
}
