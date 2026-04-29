//! OctonionRing: the ring Z/2^32Z at quantum level Q3.
//!
//! The first non-associative, non-commutative algebra in the Cayley-Dickson chain.
//! R(Q0) → C(Q1) → H(Q2) → O(Q3).

use crate::q3::datum::QuadDatum;
use crate::quantum;
use crate::HoloPrimitives;
use uor_foundation::enums::GeometricCharacter;
use uor_foundation::WittLevel as QuantumLevel;

/// The ring Z/2^32Z at quantum level 3.
///
/// Algebraic classification: octonion (dimension 8).
/// Non-commutative, non-associative — the first algebra in the
/// Cayley-Dickson chain to lose associativity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OctonionRing;

impl OctonionRing {
    /// Ring bit width.
    pub const QUANTUM: u64 = 32;

    /// Ring modulus (2^32 = 4_294_967_296).
    pub const MODULUS: u64 = 4_294_967_296;

    /// Add two values (wrapping).
    #[inline(always)]
    #[must_use]
    pub fn add(a: u32, b: u32) -> u32 {
        quantum::q3_add(a, b)
    }

    /// Subtract two values (wrapping).
    #[inline(always)]
    #[must_use]
    pub fn sub(a: u32, b: u32) -> u32 {
        quantum::q3_sub(a, b)
    }

    /// Multiply two values (wrapping).
    #[inline(always)]
    #[must_use]
    pub fn mul(a: u32, b: u32) -> u32 {
        quantum::q3_mul(a, b)
    }
}

use std::sync::OnceLock;
fn generator_quad() -> &'static QuadDatum {
    static G: OnceLock<QuadDatum> = OnceLock::new();
    G.get_or_init(QuadDatum::pi1)
}
static NEG_INV: OctonionInvolution = OctonionInvolution::Neg;
static BNOT_INV: OctonionInvolution = OctonionInvolution::Bnot;

impl uor_foundation::kernel::schema::Ring<HoloPrimitives> for OctonionRing {
    fn ring_witt_length(&self) -> u64 {
        Self::QUANTUM
    }

    fn modulus(&self) -> u64 {
        Self::MODULUS
    }

    type Datum = QuadDatum;

    fn generator(&self) -> &Self::Datum {
        generator_quad()
    }

    type Involution = OctonionInvolution;

    fn negation(&self) -> &Self::Involution {
        &NEG_INV
    }

    fn complement(&self) -> &Self::Involution {
        &BNOT_INV
    }

    fn at_witt_level(&self) -> QuantumLevel {
        QuantumLevel::W32
    }
}

/// The two involutions of the ring Z/2^32Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OctonionInvolution {
    /// Ring reflection: neg(x) = (-x) mod 2^32.
    Neg,
    /// Hypercube reflection: bnot(x) = !x.
    Bnot,
}

impl OctonionInvolution {
    /// Apply this involution to a 32-bit value.
    #[inline(always)]
    #[must_use]
    pub const fn apply(self, x: u32) -> u32 {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
        }
    }
}

impl uor_foundation::kernel::op::Operation<HoloPrimitives> for OctonionInvolution {
    fn arity(&self) -> u64 {
        1
    }

    fn has_geometric_character(&self) -> GeometricCharacter {
        match self {
            Self::Neg => GeometricCharacter::RingReflection,
            Self::Bnot => GeometricCharacter::HypercubeReflection,
        }
    }

    type OperationTarget = OctonionInvolution;

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

impl uor_foundation::kernel::op::UnaryOp<HoloPrimitives> for OctonionInvolution {}
impl uor_foundation::kernel::op::Involution<HoloPrimitives> for OctonionInvolution {}

static GROUP_GENERATORS: [OctonionInvolution; 2] =
    [OctonionInvolution::Neg, OctonionInvolution::Bnot];

impl uor_foundation::kernel::op::Group<HoloPrimitives> for OctonionRing {
    type Operation = OctonionInvolution;

    #[inline]
    fn generated_by(&self) -> &[Self::Operation] {
        &GROUP_GENERATORS
    }

    #[inline]
    fn order(&self) -> u64 {
        Self::MODULUS
    }
}

impl uor_foundation::kernel::op::DihedralGroup<HoloPrimitives> for OctonionRing {}

/// Marker for the Z/2^32Z multiplication table.
pub struct OctonionMultTable;

impl uor_foundation::kernel::division::MultiplicationTable<HoloPrimitives> for OctonionMultTable {}

static OCTONION_MULT_TABLE: OctonionMultTable = OctonionMultTable;

impl uor_foundation::kernel::division::NormedDivisionAlgebra<HoloPrimitives> for OctonionRing {
    #[inline]
    fn algebra_dimension(&self) -> u64 {
        8
    }

    #[inline]
    fn is_commutative(&self) -> bool {
        false
    }

    #[inline]
    fn is_associative(&self) -> bool {
        false
    }

    #[inline]
    fn basis_elements(&self) -> &str {
        "{1, e1, e2, e3, e4, e5, e6, e7}"
    }

    type MultiplicationTable = OctonionMultTable;

    #[inline]
    fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
        &OCTONION_MULT_TABLE
    }
}

impl uor_foundation::kernel::division::AlgebraCommutator<HoloPrimitives> for OctonionRing {}

impl uor_foundation::kernel::division::AlgebraAssociator<HoloPrimitives> for OctonionRing {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_quantum_and_modulus() {
        use uor_foundation::kernel::schema::Ring;
        let r = OctonionRing;
        assert_eq!(r.ring_witt_length(), 32);
        assert_eq!(r.modulus(), 4_294_967_296);
        assert_eq!(r.at_witt_level(), QuantumLevel::W32);
    }

    #[test]
    fn ring_generator() {
        use uor_foundation::kernel::schema::Ring;
        let r = OctonionRing;
        assert_eq!(r.generator().value(), 1);
    }

    #[test]
    fn ring_arithmetic() {
        assert_eq!(OctonionRing::add(u32::MAX, 1), 0);
        assert_eq!(OctonionRing::mul(2, 3), 6);
        assert_eq!(OctonionRing::sub(0, 1), u32::MAX);
    }

    #[test]
    fn involution_neg() {
        for x in [0u32, 1, 127, u32::MAX] {
            assert_eq!(
                OctonionInvolution::Neg.apply(OctonionInvolution::Neg.apply(x)),
                x
            );
        }
    }

    #[test]
    fn involution_bnot() {
        for x in [0u32, 1, 0xFF, u32::MAX] {
            assert_eq!(
                OctonionInvolution::Bnot.apply(OctonionInvolution::Bnot.apply(x)),
                x
            );
        }
    }

    #[test]
    fn normed_division_algebra() {
        use uor_foundation::kernel::division::NormedDivisionAlgebra;
        let r = OctonionRing;
        assert_eq!(r.algebra_dimension(), 8);
        assert!(!r.is_commutative());
        assert!(!r.is_associative());
        assert_eq!(r.basis_elements(), "{1, e1, e2, e3, e4, e5, e6, e7}");
    }

    #[test]
    fn group_order_and_generators() {
        use uor_foundation::kernel::op::Group;
        let r = OctonionRing;
        assert_eq!(r.order(), 4_294_967_296);
        assert_eq!(r.generated_by().len(), 2);
    }

    #[test]
    fn operation_trait() {
        use uor_foundation::kernel::op::Operation;
        let neg = OctonionInvolution::Neg;
        assert_eq!(neg.arity(), 1);
        assert_eq!(
            neg.has_geometric_character(),
            GeometricCharacter::RingReflection
        );
    }
}
