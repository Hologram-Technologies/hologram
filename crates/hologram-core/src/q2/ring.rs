//! TripleRing: the ring Z/2^24Z at quantum level Q2.

use crate::q2::{arith, datum::TripleDatum};
use crate::HoloPrimitives;
use uor_foundation::enums::GeometricCharacter;
use uor_foundation::WittLevel as QuantumLevel;

/// The ring Z/2^24Z at quantum level 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleRing;

impl TripleRing {
    /// Ring bit width.
    pub const QUANTUM: u64 = 24;

    /// Ring modulus (2^24 = 16_777_216).
    pub const MODULUS: u64 = 0x01_000000;

    /// Add two triple values (wrapping).
    #[inline(always)]
    #[must_use]
    pub fn add(a: u32, b: u32) -> u32 {
        arith::add_q2(a, b)
    }

    /// Subtract two triple values (wrapping).
    #[inline(always)]
    #[must_use]
    pub fn sub(a: u32, b: u32) -> u32 {
        arith::sub_q2(a, b)
    }

    /// Multiply two triple values (wrapping).
    #[inline(always)]
    #[must_use]
    pub fn mul(a: u32, b: u32) -> u32 {
        arith::mul_q2(a, b)
    }
}

use std::sync::OnceLock;
fn generator_triple() -> &'static TripleDatum {
    static G: OnceLock<TripleDatum> = OnceLock::new();
    G.get_or_init(TripleDatum::pi1)
}
static NEG_INV: TripleInvolution = TripleInvolution::Neg;
static BNOT_INV: TripleInvolution = TripleInvolution::Bnot;

impl uor_foundation::kernel::schema::Ring<HoloPrimitives> for TripleRing {
    fn ring_witt_length(&self) -> u64 {
        Self::QUANTUM
    }

    fn modulus(&self) -> u64 {
        Self::MODULUS
    }

    type Datum = TripleDatum;

    fn generator(&self) -> &Self::Datum {
        generator_triple()
    }

    type Involution = TripleInvolution;

    fn negation(&self) -> &Self::Involution {
        &NEG_INV
    }

    fn complement(&self) -> &Self::Involution {
        &BNOT_INV
    }

    fn at_witt_level(&self) -> QuantumLevel {
        QuantumLevel::W24
    }
}

/// The two involutions of the ring Z/2^24Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TripleInvolution {
    /// Ring reflection: neg(x) = (-x) mod 2^24.
    Neg,
    /// Hypercube reflection: bnot(x) = (2^24 - 1) ^ x.
    Bnot,
}

impl TripleInvolution {
    /// Apply this involution to a 24-bit value.
    #[inline(always)]
    #[must_use]
    pub const fn apply(self, x: u32) -> u32 {
        match self {
            Self::Neg => arith::neg_q2(x),
            Self::Bnot => arith::bnot_q2(x),
        }
    }
}

impl uor_foundation::kernel::op::Operation<HoloPrimitives> for TripleInvolution {
    fn arity(&self) -> u64 {
        1
    }

    fn has_geometric_character(&self) -> GeometricCharacter {
        match self {
            Self::Neg => GeometricCharacter::RingReflection,
            Self::Bnot => GeometricCharacter::HypercubeReflection,
        }
    }

    type OperationTarget = TripleInvolution;

    fn inverse(&self) -> &Self::OperationTarget {
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

    fn is_ring_op(&self) -> bool {
        true
    }
}

impl uor_foundation::kernel::op::UnaryOp<HoloPrimitives> for TripleInvolution {}
impl uor_foundation::kernel::op::Involution<HoloPrimitives> for TripleInvolution {}

static GROUP_GENERATORS: [TripleInvolution; 2] = [TripleInvolution::Neg, TripleInvolution::Bnot];

impl uor_foundation::kernel::op::Group<HoloPrimitives> for TripleRing {
    type Operation = TripleInvolution;

    #[inline]
    fn generated_by(&self) -> &[Self::Operation] {
        &GROUP_GENERATORS
    }

    #[inline]
    fn order(&self) -> u64 {
        0x01_000000
    }
}

impl uor_foundation::kernel::op::DihedralGroup<HoloPrimitives> for TripleRing {}

/// Marker for the Z/2^24Z multiplication table.
pub struct TripleMultTable;

impl uor_foundation::kernel::division::MultiplicationTable<HoloPrimitives> for TripleMultTable {}

static TRIPLE_MULT_TABLE: TripleMultTable = TripleMultTable;

impl uor_foundation::kernel::division::NormedDivisionAlgebra<HoloPrimitives> for TripleRing {
    #[inline]
    fn algebra_dimension(&self) -> u64 {
        4
    }

    #[inline]
    fn is_commutative(&self) -> bool {
        true
    }

    #[inline]
    fn is_associative(&self) -> bool {
        true
    }

    #[inline]
    fn basis_elements(&self) -> &str {
        "{1, i, j, k}"
    }

    type MultiplicationTable = TripleMultTable;

    #[inline]
    fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
        &TRIPLE_MULT_TABLE
    }
}

impl uor_foundation::kernel::division::AlgebraCommutator<HoloPrimitives> for TripleRing {}

impl uor_foundation::kernel::division::AlgebraAssociator<HoloPrimitives> for TripleRing {}

impl uor_foundation::kernel::division::CayleyDicksonConstruction<HoloPrimitives> for TripleRing {
    type NormedDivisionAlgebra = crate::ring::HoloDivisionAlgebra;

    #[inline]
    fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
        &crate::ring::Q2_ALGEBRA
    }

    #[inline]
    fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
        &crate::ring::Q3_ALGEBRA
    }

    #[inline]
    fn adjoined_element(&self) -> &str {
        "l"
    }

    #[inline]
    fn conjugation_rule(&self) -> &str {
        "l^2 = -1 mod 2^32"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_quantum_and_modulus() {
        use uor_foundation::kernel::schema::Ring;
        let r = TripleRing;
        assert_eq!(r.ring_witt_length(), 24);
        assert_eq!(r.modulus(), 0x01_000000);
        assert_eq!(r.at_witt_level(), QuantumLevel::W24);
    }

    #[test]
    fn ring_generator() {
        use uor_foundation::kernel::schema::Ring;
        let r = TripleRing;
        assert_eq!(r.generator().value(), 1);
    }

    #[test]
    fn ring_add() {
        assert_eq!(TripleRing::add(100, 200), 300);
        assert_eq!(TripleRing::add(0x00FF_FFFF, 1), 0);
        assert_eq!(TripleRing::add(0, 0), 0);
    }

    #[test]
    fn ring_sub() {
        assert_eq!(TripleRing::sub(0, 1), 0x00FF_FFFF);
        assert_eq!(TripleRing::sub(300, 200), 100);
    }

    #[test]
    fn ring_mul() {
        assert_eq!(TripleRing::mul(3, 4), 12);
        assert_eq!(TripleRing::mul(0x01_0000, 0x01_0000), 0); // 2^32 mod 2^24 = 0
    }

    #[test]
    fn involution_neg_involution() {
        for x in [0u32, 1, 127, 0xFFFF, 0xFFFFFF] {
            assert_eq!(
                TripleInvolution::Neg.apply(TripleInvolution::Neg.apply(x)),
                x & 0x00FF_FFFF
            );
        }
    }

    #[test]
    fn involution_bnot_involution() {
        for x in [0u32, 1, 0xFF, 0xFFFF, 0xFFFFFF] {
            assert_eq!(
                TripleInvolution::Bnot.apply(TripleInvolution::Bnot.apply(x)),
                x & 0x00FF_FFFF
            );
        }
    }

    #[test]
    fn critical_identity() {
        // ring_neg(bnot(x)) = succ(x) must hold for all 24-bit values (spot check).
        for x in [0u32, 1, 0xFF, 0xFFFF, 0xFFFFFE, 0xFFFFFF] {
            let neg_bnot = arith::neg_q2(arith::bnot_q2(x));
            let succ = arith::succ_q2(x);
            assert_eq!(neg_bnot, succ, "critical identity failed at {x:#x}");
        }
    }

    #[test]
    fn ring_arithmetic_exhaustive_q2_add() {
        // Add inverse: a + (-a) == 0 for spot values.
        for a in [0u32, 1, 255, 65535, 0xFFFFFF] {
            assert_eq!(arith::add_q2(a, arith::neg_q2(a)), 0);
        }
    }

    #[test]
    fn group_order_and_generators() {
        use uor_foundation::kernel::op::Group;
        let ring = TripleRing;
        assert_eq!(ring.order(), 0x01_000000);
        assert_eq!(ring.generated_by().len(), 2);
    }

    #[test]
    fn normed_division_algebra_dimension() {
        use uor_foundation::kernel::division::NormedDivisionAlgebra;
        let r = TripleRing;
        assert_eq!(r.algebra_dimension(), 4);
        assert_eq!(r.basis_elements(), "{1, i, j, k}");
        assert!(r.is_commutative());
        assert!(r.is_associative());
    }

    #[test]
    fn operation_trait() {
        use uor_foundation::kernel::op::Operation;
        let neg = TripleInvolution::Neg;
        assert_eq!(neg.arity(), 1);
        assert_eq!(
            neg.has_geometric_character(),
            GeometricCharacter::RingReflection
        );
        assert_eq!(neg.composed_of(), "neg");

        let bnot = TripleInvolution::Bnot;
        assert_eq!(
            bnot.has_geometric_character(),
            GeometricCharacter::HypercubeReflection
        );
        assert_eq!(bnot.composed_of(), "bnot");
    }
}
