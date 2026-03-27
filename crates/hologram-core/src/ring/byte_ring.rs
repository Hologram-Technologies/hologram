//! ByteRing: the ring Z/256Z at quantum level Q0.

use crate::datum::ByteDatum;
use crate::lut::arith;
use crate::q1::ring::WordRing;
use crate::q2::ring::TripleRing;
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

static GROUP_GENERATORS: [ByteInvolution; 2] = [ByteInvolution::Neg, ByteInvolution::Bnot];

impl uor_foundation::kernel::op::Group<HoloPrimitives> for ByteRing {
    type Operation = ByteInvolution;

    #[inline]
    fn generated_by(&self) -> &[Self::Operation] {
        &GROUP_GENERATORS
    }

    #[inline]
    fn order(&self) -> u64 {
        256
    }
}

impl uor_foundation::kernel::op::DihedralGroup<HoloPrimitives> for ByteRing {}

/// Marker for the Z/256Z multiplication table (the ring is trivially specified).
pub struct ByteMultTable;

impl uor_foundation::kernel::division::MultiplicationTable<HoloPrimitives> for ByteMultTable {}

static BYTE_MULT_TABLE: ByteMultTable = ByteMultTable;

impl uor_foundation::kernel::division::NormedDivisionAlgebra<HoloPrimitives> for ByteRing {
    #[inline]
    fn algebra_dimension(&self) -> u64 {
        1
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
        "{1}"
    }

    type MultiplicationTable = ByteMultTable;

    #[inline]
    fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
        &BYTE_MULT_TABLE
    }
}

impl uor_foundation::kernel::division::AlgebraCommutator<HoloPrimitives> for ByteRing {
    #[inline]
    fn commutator_formula(&self) -> &str {
        "[a,b] = 0"
    }
}

impl uor_foundation::kernel::division::AlgebraAssociator<HoloPrimitives> for ByteRing {
    #[inline]
    fn associator_formula(&self) -> &str {
        "[a,b,c] = 0"
    }
}

/// Unified division algebra enum bridging Q0–Q3 across the Cayley-Dickson chain.
///
/// Used as the `NormedDivisionAlgebra` associated type for `CayleyDicksonConstruction`
/// on `ByteRing`, `WordRing`, and `TripleRing`.
#[derive(Debug, Clone, Copy)]
pub enum HoloDivisionAlgebra {
    /// Q0 level: the real algebra R ≅ Z/256Z.
    Q0(ByteRing),
    /// Q1 level: the complex algebra C ≅ Z/65536Z.
    Q1(WordRing),
    /// Q2 level: the quaternion algebra H ≅ Z/2^24Z.
    Q2(TripleRing),
    /// Q3 level: the octonion algebra O ≅ Z/2^32Z.
    Q3(crate::q3::OctonionRing),
}

impl uor_foundation::kernel::division::NormedDivisionAlgebra<HoloPrimitives>
    for HoloDivisionAlgebra
{
    type MultiplicationTable = ByteMultTable;

    #[inline]
    fn algebra_dimension(&self) -> u64 {
        match self {
            Self::Q0(_) => 1,
            Self::Q1(_) => 2,
            Self::Q2(_) => 4,
            Self::Q3(_) => 8,
        }
    }

    #[inline]
    fn is_commutative(&self) -> bool {
        !matches!(self, Self::Q3(_))
    }

    #[inline]
    fn is_associative(&self) -> bool {
        !matches!(self, Self::Q3(_))
    }

    #[inline]
    fn basis_elements(&self) -> &str {
        match self {
            Self::Q0(_) => "{1}",
            Self::Q1(_) => "{1, i}",
            Self::Q2(_) => "{1, i, j, k}",
            Self::Q3(_) => "{1, e1, e2, e3, e4, e5, e6, e7}",
        }
    }

    #[inline]
    fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
        &BYTE_MULT_TABLE
    }
}

static Q0_ALGEBRA: HoloDivisionAlgebra = HoloDivisionAlgebra::Q0(ByteRing);
pub(crate) static Q1_ALGEBRA: HoloDivisionAlgebra = HoloDivisionAlgebra::Q1(WordRing);
pub(crate) static Q2_ALGEBRA: HoloDivisionAlgebra = HoloDivisionAlgebra::Q2(TripleRing);
pub(crate) static Q3_ALGEBRA: HoloDivisionAlgebra =
    HoloDivisionAlgebra::Q3(crate::q3::OctonionRing);

impl uor_foundation::kernel::division::CayleyDicksonConstruction<HoloPrimitives> for ByteRing {
    type NormedDivisionAlgebra = HoloDivisionAlgebra;

    #[inline]
    fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
        &Q0_ALGEBRA
    }

    #[inline]
    fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
        &Q1_ALGEBRA
    }

    #[inline]
    fn adjoined_element(&self) -> &str {
        "i"
    }

    #[inline]
    fn conjugation_rule(&self) -> &str {
        "i\u{00b2} = \u{2212}1 mod 256"
    }
}

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

    #[test]
    fn group_order_and_generators() {
        use uor_foundation::kernel::op::Group;
        let ring = ByteRing;
        assert_eq!(ring.order(), 256);
        assert_eq!(ring.generated_by().len(), 2);
    }

    #[test]
    fn cayley_dickson_r_to_c_grounding() {
        use uor_foundation::kernel::division::{CayleyDicksonConstruction, NormedDivisionAlgebra};
        let ring = ByteRing;
        let src = ring.cayley_dickson_source();
        let tgt = ring.cayley_dickson_target();
        assert_eq!(src.algebra_dimension(), 1); // R = Q0
        assert_eq!(tgt.algebra_dimension(), 2); // C = Q1
        assert_eq!(ring.adjoined_element(), "i");
        assert!(ring.conjugation_rule().contains("mod 256"));
    }

    #[test]
    fn holo_division_algebra_q2() {
        use uor_foundation::kernel::division::NormedDivisionAlgebra;
        let q2 = HoloDivisionAlgebra::Q2(TripleRing);
        assert_eq!(q2.algebra_dimension(), 4);
        assert_eq!(q2.basis_elements(), "{1, i, j, k}");
        assert!(q2.is_commutative());
        assert!(q2.is_associative());
    }
}
