//! UOR trait conformance tests.
//!
//! Verifies that prism types implement the UOR ontological hierarchy correctly.

use hologram_ring::datum::Datum;
use hologram_ring::ring::PrismRing;
use hologram_ring::{Involution, PrimOp, Q0, Q1, Q3, Q7};
use uor_foundation::enums::GeometricCharacter;
use uor_foundation::kernel::address::Address as UorAddress;
use uor_foundation::kernel::division::{
    AlgebraAssociator, AlgebraCommutator, CayleyDicksonConstruction, NormedDivisionAlgebra,
};
use uor_foundation::kernel::op::{
    BinaryOp, DihedralGroup, Group, Involution as UorInvolution, Operation, UnaryOp,
};
use uor_foundation::kernel::schema::{Datum as UorDatum, Ring};

// ── Datum ────────────────────────────────────────────────────────────────

#[test]
fn datum_value_q0() {
    let d = Datum::<Q0>::new(42u8);
    assert_eq!(UorDatum::value(&d), 42);
    assert_eq!(UorDatum::quantum(&d), 8);
    assert_eq!(UorDatum::stratum(&d), 42u8.count_ones() as u64);
}

#[test]
fn datum_value_q3() {
    let d = Datum::<Q3>::new(0xDEAD_BEEFu32);
    assert_eq!(UorDatum::value(&d), 0xDEAD_BEEF);
    assert_eq!(UorDatum::quantum(&d), 32);
    assert_eq!(UorDatum::stratum(&d), 0xDEAD_BEEFu32.count_ones() as u64);
}

#[test]
fn datum_spectrum_q0() {
    let d = Datum::<Q0>::new(0b1010_0101u8);
    // uor-foundation 0.1.4: `spectrum` on the trait returns the underlying
    // numeric value (P::NonNegativeInteger = u64). The binary-string form
    // is still available via the inherent `Datum::spectrum` method.
    assert_eq!(UorDatum::spectrum(&d), 0b1010_0101u64);
    assert_eq!(d.spectrum(), "10100101");
}

#[test]
fn datum_address_q0() {
    let d = Datum::<Q0>::new(42u8);
    let addr = UorDatum::glyph(&d);
    let glyph = UorAddress::glyph(addr);
    assert!(!glyph.is_empty(), "address glyph should be non-empty");
    assert_eq!(UorAddress::length(addr), 2); // ceil(8/6) = 2 Braille chars
    assert_eq!(UorAddress::quantum(addr), 8);
    assert_eq!(UorAddress::digest_algorithm(addr), "blake3");
}

#[test]
fn datum_address_q3() {
    let d = Datum::<Q3>::new(1000u32);
    let addr = UorDatum::glyph(&d);
    assert_eq!(UorAddress::length(addr), 6); // ceil(32/6) = 6
    assert_eq!(UorAddress::quantum(addr), 32);
}

// ── Ring ──────────────────────────────────────────────────────────────────

#[test]
fn ring_q0() {
    let r = PrismRing::<Q0>::new();
    assert_eq!(Ring::ring_quantum(&r), 8);
    assert_eq!(Ring::modulus(&r), 256);
    assert_eq!(UorDatum::value(Ring::generator(&r)), 1);
    assert_eq!(
        Ring::at_quantum_level(&r),
        uor_foundation::enums::QuantumLevel::Q0
    );
}

#[test]
fn ring_q3() {
    let r = PrismRing::<Q3>::new();
    assert_eq!(Ring::ring_quantum(&r), 32);
    assert_eq!(Ring::modulus(&r), 4_294_967_296);
    assert_eq!(UorDatum::value(Ring::generator(&r)), 1);
}

#[test]
fn ring_q7() {
    let r = PrismRing::<Q7>::new();
    assert_eq!(Ring::ring_quantum(&r), 64);
    // modulus overflows u64 for Q7 → returns 0 to signal this
    assert_eq!(Ring::modulus(&r), 0);
}

// ── Group + DihedralGroup ────────────────────────────────────────────────

#[test]
fn group_q0() {
    let r = PrismRing::<Q0>::new();
    assert_eq!(Group::generated_by(&r).len(), 2);
    assert_eq!(Group::order(&r), 256);
}

#[test]
fn dihedral_group_exists() {
    fn assert_dihedral<T: DihedralGroup<hologram_ring::PrismPrimitives>>(_: &T) {}
    assert_dihedral(&PrismRing::<Q0>::new());
    assert_dihedral(&PrismRing::<Q3>::new());
}

// ── NormedDivisionAlgebra ────────────────────────────────────────────────

#[test]
fn nda_dimensions() {
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<Q0>::new()),
        1
    );
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<Q1>::new()),
        2
    );
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<Q3>::new()),
        4
    );
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<Q7>::new()),
        8
    );
}

#[test]
fn nda_commutativity() {
    assert!(NormedDivisionAlgebra::is_commutative(
        &PrismRing::<Q0>::new()
    ));
    assert!(NormedDivisionAlgebra::is_commutative(
        &PrismRing::<Q1>::new()
    ));
    assert!(!NormedDivisionAlgebra::is_commutative(
        &PrismRing::<Q3>::new()
    ));
    assert!(!NormedDivisionAlgebra::is_commutative(
        &PrismRing::<Q7>::new()
    ));
}

#[test]
fn nda_associativity() {
    assert!(NormedDivisionAlgebra::is_associative(
        &PrismRing::<Q0>::new()
    ));
    assert!(NormedDivisionAlgebra::is_associative(
        &PrismRing::<Q3>::new()
    ));
    assert!(!NormedDivisionAlgebra::is_associative(
        &PrismRing::<Q7>::new()
    ));
}

// ── CayleyDicksonConstruction ────────────────────────────────────────────

#[test]
fn cayley_dickson_q0_to_q1() {
    let r = PrismRing::<Q0>::new();
    let src = CayleyDicksonConstruction::cayley_dickson_source(&r);
    let tgt = CayleyDicksonConstruction::cayley_dickson_target(&r);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(src), 1);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(tgt), 2);
    assert_eq!(CayleyDicksonConstruction::adjoined_element(&r), "i");
}

#[test]
fn cayley_dickson_q1_to_q3() {
    let r = PrismRing::<Q1>::new();
    let src = CayleyDicksonConstruction::cayley_dickson_source(&r);
    let tgt = CayleyDicksonConstruction::cayley_dickson_target(&r);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(src), 2);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(tgt), 4);
    assert_eq!(CayleyDicksonConstruction::adjoined_element(&r), "j");
}

#[test]
fn cayley_dickson_q3_to_q7() {
    let r = PrismRing::<Q3>::new();
    let src = CayleyDicksonConstruction::cayley_dickson_source(&r);
    let tgt = CayleyDicksonConstruction::cayley_dickson_target(&r);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(src), 4);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(tgt), 8);
    assert_eq!(CayleyDicksonConstruction::adjoined_element(&r), "l");
}

// ── Involution UOR traits ────────────────────────────────────────────────

#[test]
fn involution_operation_traits() {
    let neg: Involution<Q0> = Involution::Neg;
    assert_eq!(Operation::arity(&neg), 1);
    assert_eq!(
        Operation::has_geometric_character(&neg),
        GeometricCharacter::RingReflection
    );
    assert_eq!(Operation::composed_of(&neg), "neg");

    let bnot: Involution<Q0> = Involution::Bnot;
    assert_eq!(
        Operation::has_geometric_character(&bnot),
        GeometricCharacter::HypercubeReflection
    );

    // Marker traits compile
    fn assert_involution<T: UorInvolution<hologram_ring::PrismPrimitives>>(_: &T) {}
    fn assert_unary<T: UnaryOp<hologram_ring::PrismPrimitives>>(_: &T) {}
    assert_involution(&neg);
    assert_unary(&neg);
}

// ── PrimOp UOR traits ────────────────────────────────────────────────────

#[test]
fn primop_operation_traits() {
    assert_eq!(Operation::arity(&PrimOp::Add), 2);
    assert_eq!(Operation::arity(&PrimOp::Neg), 1);
    assert_eq!(Operation::composed_of(&PrimOp::Add), "add");

    assert!(BinaryOp::commutative(&PrimOp::Add));
    assert!(BinaryOp::associative(&PrimOp::Add));
    assert_eq!(BinaryOp::identity(&PrimOp::Add), 0);
    assert_eq!(BinaryOp::identity(&PrimOp::Mul), 1);
    assert!(!BinaryOp::commutative(&PrimOp::Sub));
}

// ── Algebra marker traits ────────────────────────────────────────────────
//
// uor-foundation 0.1.4 collapsed `AlgebraCommutator` and `AlgebraAssociator`
// into empty marker traits (the formula-string methods were removed). These
// tests now just assert that `PrismRing<Q>` still implements the markers.

#[test]
fn algebra_commutator_markers_implemented() {
    fn assert_impl<P, T>()
    where
        P: uor_foundation::Primitives,
        T: AlgebraCommutator<P>,
    {
    }
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<Q0>>();
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<Q3>>();
}

#[test]
fn algebra_associator_markers_implemented() {
    fn assert_impl<P, T>()
    where
        P: uor_foundation::Primitives,
        T: AlgebraAssociator<P>,
    {
    }
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<Q0>>();
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<Q7>>();
}
