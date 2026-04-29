//! WordRing: the ring Z/65536Z at quantum level Q1.

use crate::q1::arith;
use crate::q1::datum::WordDatum;
use crate::HoloPrimitives;
use uor_foundation::enums::GeometricCharacter;
use uor_foundation::WittLevel as QuantumLevel;

/// The ring Z/65536Z at quantum level 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WordRing;

impl WordRing {
    /// Ring bit width.
    pub const QUANTUM: u64 = 16;

    /// Ring modulus (2^16 = 65536).
    pub const MODULUS: u64 = 65536;

    /// Add two word values (wrapping).
    #[inline]
    #[must_use]
    pub fn add(a: u16, b: u16) -> u16 {
        arith::add_q1(a, b)
    }

    /// Subtract two word values (wrapping).
    #[inline]
    #[must_use]
    pub fn sub(a: u16, b: u16) -> u16 {
        arith::sub_q1(a, b)
    }

    /// Multiply two word values (wrapping).
    #[inline]
    #[must_use]
    pub fn mul(a: u16, b: u16) -> u16 {
        arith::mul_q1(a, b)
    }
}

use std::sync::OnceLock;
fn generator_word() -> &'static WordDatum {
    static G: OnceLock<WordDatum> = OnceLock::new();
    G.get_or_init(WordDatum::pi1)
}
static NEG_INV: WordInvolution = WordInvolution::Neg;
static BNOT_INV: WordInvolution = WordInvolution::Bnot;

impl uor_foundation::kernel::schema::Ring<HoloPrimitives> for WordRing {
    fn ring_witt_length(&self) -> u64 {
        Self::QUANTUM
    }

    fn modulus(&self) -> u64 {
        Self::MODULUS
    }

    type Datum = WordDatum;

    fn generator(&self) -> &Self::Datum {
        generator_word()
    }

    type Involution = WordInvolution;

    fn negation(&self) -> &Self::Involution {
        &NEG_INV
    }

    fn complement(&self) -> &Self::Involution {
        &BNOT_INV
    }

    fn at_witt_level(&self) -> QuantumLevel {
        QuantumLevel::W16
    }
}

impl uor_foundation::kernel::schema::W16Ring<HoloPrimitives> for WordRing {
    fn w16bit_width(&self) -> u64 {
        16
    }

    fn w16capacity(&self) -> u64 {
        65536
    }
}

/// The two involutions of the ring Z/65536Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WordInvolution {
    /// Ring reflection: neg(x) = (-x) mod 65536.
    Neg,
    /// Hypercube reflection: bnot(x) = 65535 ^ x.
    Bnot,
}

impl WordInvolution {
    /// Apply this involution to a word value.
    #[inline]
    #[must_use]
    pub const fn apply(self, x: u16) -> u16 {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
        }
    }
}

impl uor_foundation::kernel::op::Operation<HoloPrimitives> for WordInvolution {
    fn arity(&self) -> u64 {
        1
    }

    fn has_geometric_character(&self) -> GeometricCharacter {
        match self {
            Self::Neg => GeometricCharacter::RingReflection,
            Self::Bnot => GeometricCharacter::HypercubeReflection,
        }
    }

    type OperationTarget = WordInvolution;

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

impl uor_foundation::kernel::op::UnaryOp<HoloPrimitives> for WordInvolution {}
impl uor_foundation::kernel::op::Involution<HoloPrimitives> for WordInvolution {}

static GROUP_GENERATORS: [WordInvolution; 2] = [WordInvolution::Neg, WordInvolution::Bnot];

impl uor_foundation::kernel::op::Group<HoloPrimitives> for WordRing {
    type Operation = WordInvolution;

    #[inline]
    fn generated_by(&self) -> &[Self::Operation] {
        &GROUP_GENERATORS
    }

    #[inline]
    fn order(&self) -> u64 {
        65536
    }
}

impl uor_foundation::kernel::op::DihedralGroup<HoloPrimitives> for WordRing {}

/// Marker for the Z/65536Z multiplication table.
pub struct WordMultTable;

impl uor_foundation::kernel::division::MultiplicationTable<HoloPrimitives> for WordMultTable {}

static WORD_MULT_TABLE: WordMultTable = WordMultTable;

impl uor_foundation::kernel::division::NormedDivisionAlgebra<HoloPrimitives> for WordRing {
    #[inline]
    fn algebra_dimension(&self) -> u64 {
        2
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
        "{1, i}"
    }

    type MultiplicationTable = WordMultTable;

    #[inline]
    fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
        &WORD_MULT_TABLE
    }
}

impl uor_foundation::kernel::division::AlgebraCommutator<HoloPrimitives> for WordRing {}

impl uor_foundation::kernel::division::AlgebraAssociator<HoloPrimitives> for WordRing {}

impl uor_foundation::kernel::division::CayleyDicksonConstruction<HoloPrimitives> for WordRing {
    type NormedDivisionAlgebra = crate::ring::HoloDivisionAlgebra;

    #[inline]
    fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
        &crate::ring::Q1_ALGEBRA
    }

    #[inline]
    fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
        &crate::ring::Q2_ALGEBRA
    }

    #[inline]
    fn adjoined_element(&self) -> &str {
        "j"
    }

    #[inline]
    fn conjugation_rule(&self) -> &str {
        "j\u{00b2} = \u{2212}1 mod 65536"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_add() {
        assert_eq!(WordRing::add(100, 200), 300);
        assert_eq!(WordRing::add(65535, 1), 0);
        assert_eq!(WordRing::add(0, 0), 0);
        assert_eq!(WordRing::add(32768, 32768), 0);
    }

    #[test]
    fn ring_sub() {
        assert_eq!(WordRing::sub(0, 1), 65535);
        assert_eq!(WordRing::sub(1000, 500), 500);
    }

    #[test]
    fn ring_mul() {
        assert_eq!(WordRing::mul(3, 5), 15);
        assert_eq!(WordRing::mul(256, 256), 0); // 65536 mod 65536
    }

    #[test]
    fn involution_neg() {
        let neg = WordInvolution::Neg;
        for i in (0u32..=65535).step_by(256) {
            let v = i as u16;
            assert_eq!(neg.apply(neg.apply(v)), v);
        }
        // Check boundaries
        assert_eq!(neg.apply(neg.apply(0)), 0);
        assert_eq!(neg.apply(neg.apply(1)), 1);
        assert_eq!(neg.apply(neg.apply(65535)), 65535);
    }

    #[test]
    fn involution_bnot() {
        let bnot = WordInvolution::Bnot;
        for i in (0u32..=65535).step_by(256) {
            let v = i as u16;
            assert_eq!(bnot.apply(bnot.apply(v)), v);
        }
    }

    #[test]
    fn ring_trait() {
        use uor_foundation::kernel::schema::Ring;
        let r = WordRing;
        assert_eq!(r.ring_witt_length(), 16);
        assert_eq!(r.modulus(), 65536);
        assert_eq!(r.at_witt_level(), QuantumLevel::W16);
        assert_eq!(r.generator().val(), 1);
    }

    #[test]
    fn q1ring_trait() {
        use uor_foundation::kernel::schema::W16Ring;
        let r = WordRing;
        assert_eq!(r.w16bit_width(), 16);
        assert_eq!(r.w16capacity(), 65536);
    }

    #[test]
    fn critical_identity_via_involutions() {
        let neg = WordInvolution::Neg;
        let bnot = WordInvolution::Bnot;
        // neg(bnot(x)) == succ(x) for all x
        for i in (0u32..=65535).step_by(256) {
            let v = i as u16;
            assert_eq!(neg.apply(bnot.apply(v)), v.wrapping_add(1));
        }
        // Check edge cases
        assert_eq!(neg.apply(bnot.apply(0)), 1);
        assert_eq!(neg.apply(bnot.apply(65534)), 65535);
        assert_eq!(neg.apply(bnot.apply(65535)), 0);
    }

    #[test]
    fn cayley_dickson_chain_q1_to_q2() {
        use uor_foundation::kernel::division::{CayleyDicksonConstruction, NormedDivisionAlgebra};
        let r1 = WordRing;
        assert_eq!(r1.cayley_dickson_source().algebra_dimension(), 2); // C = Q1
        assert_eq!(r1.cayley_dickson_target().algebra_dimension(), 4); // H = Q2
        assert_eq!(r1.cayley_dickson_target().basis_elements(), "{1, i, j, k}");
        assert_eq!(r1.adjoined_element(), "j");
    }

    #[test]
    fn neg_values() {
        let neg = WordInvolution::Neg;
        assert_eq!(neg.apply(0), 0);
        assert_eq!(neg.apply(1), 65535);
        assert_eq!(neg.apply(32768), 32768); // self-inverse point
        assert_eq!(neg.apply(65535), 1);
    }

    #[test]
    fn bnot_values() {
        let bnot = WordInvolution::Bnot;
        assert_eq!(bnot.apply(0), 65535);
        assert_eq!(bnot.apply(65535), 0);
        assert_eq!(bnot.apply(0xAAAA), 0x5555);
        assert_eq!(bnot.apply(0x00FF), 0xFF00);
    }

    #[test]
    fn operation_trait() {
        use uor_foundation::kernel::op::Operation;
        let neg = WordInvolution::Neg;
        assert_eq!(neg.arity(), 1);
        assert_eq!(
            neg.has_geometric_character(),
            GeometricCharacter::RingReflection
        );
        assert_eq!(neg.composed_of(), "neg");

        let bnot = WordInvolution::Bnot;
        assert_eq!(
            bnot.has_geometric_character(),
            GeometricCharacter::HypercubeReflection
        );
        assert_eq!(bnot.composed_of(), "bnot");
    }
}
