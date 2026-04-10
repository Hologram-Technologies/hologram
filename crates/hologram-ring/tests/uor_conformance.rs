//! UOR trait conformance tests.
//!
//! Verifies that prism types implement the v0.2.0 ontological hierarchy
//! correctly, accessed through the `hologram-foundation` re-export shim.

use hologram_foundation::address::Element as UorElement;
use hologram_foundation::division::{
    AlgebraAssociator, AlgebraCommutator, CayleyDicksonConstruction, NormedDivisionAlgebra,
};
use hologram_foundation::enums::GeometricCharacter;
use hologram_foundation::op::{
    BinaryOp, DihedralGroup, Group, Involution as UorInvolution, Operation, UnaryOp,
};
use hologram_foundation::schema::{Datum as UorDatum, Ring};
use hologram_foundation::WittLevel;
use hologram_ring::datum::Datum;
use hologram_ring::ring::PrismRing;
use hologram_ring::{Involution, PrimOp, W16, W32, W64, W8};

// ── Datum ────────────────────────────────────────────────────────────────

#[test]
fn datum_value_w8() {
    let d = Datum::<W8>::new(42u8);
    assert_eq!(UorDatum::value(&d), 42);
    assert_eq!(UorDatum::witt_length(&d), 8);
    assert_eq!(UorDatum::stratum(&d), 42u8.count_ones() as u64);
}

#[test]
fn datum_value_w32() {
    let d = Datum::<W32>::new(0xDEAD_BEEFu32);
    assert_eq!(UorDatum::value(&d), 0xDEAD_BEEF);
    assert_eq!(UorDatum::witt_length(&d), 32);
    assert_eq!(UorDatum::stratum(&d), 0xDEAD_BEEFu32.count_ones() as u64);
}

#[test]
fn datum_spectrum_w8() {
    let d = Datum::<W8>::new(0b1010_0101u8);
    // v0.2.0: `spectrum` on the trait returns the underlying numeric value
    // (P::NonNegativeInteger = u64). The binary-string form is still
    // available via the inherent `Datum::spectrum` method.
    assert_eq!(UorDatum::spectrum(&d), 0b1010_0101u64);
    assert_eq!(d.spectrum(), "10100101");
}

#[test]
fn datum_element_w8() {
    let d = Datum::<W8>::new(42u8);
    let element = UorDatum::element(&d);
    // v0.2.0 Element trait removed `glyph()`. Length and digest stay.
    assert_eq!(UorElement::length(element), 2); // 1 header + 1 value byte
    assert_eq!(UorElement::witt_length(element), 8);
    assert_eq!(UorElement::digest_algorithm(element), "blake3");
}

#[test]
fn datum_element_w32() {
    let d = Datum::<W32>::new(1000u32);
    let element = UorDatum::element(&d);
    assert_eq!(UorElement::length(element), 5); // 1 header + 4 value bytes
    assert_eq!(UorElement::witt_length(element), 32);
}

// ── Ring ──────────────────────────────────────────────────────────────────

#[test]
fn ring_w8() {
    let r = PrismRing::<W8>::new();
    assert_eq!(Ring::ring_witt_length(&r), 8);
    assert_eq!(Ring::modulus(&r), 256);
    assert_eq!(UorDatum::value(Ring::generator(&r)), 1);
    assert_eq!(Ring::at_witt_level(&r), WittLevel::W8);
}

#[test]
fn ring_w32() {
    let r = PrismRing::<W32>::new();
    assert_eq!(Ring::ring_witt_length(&r), 32);
    assert_eq!(Ring::modulus(&r), 4_294_967_296);
    assert_eq!(UorDatum::value(Ring::generator(&r)), 1);
}

#[test]
fn ring_w64() {
    let r = PrismRing::<W64>::new();
    assert_eq!(Ring::ring_witt_length(&r), 64);
    // modulus overflows u64 for W64 → returns 0 to signal this
    assert_eq!(Ring::modulus(&r), 0);
}

// ── Group + DihedralGroup ────────────────────────────────────────────────

#[test]
fn group_w8() {
    let r = PrismRing::<W8>::new();
    assert_eq!(Group::generated_by(&r).len(), 2);
    assert_eq!(Group::order(&r), 256);
}

#[test]
fn dihedral_group_exists() {
    fn assert_dihedral<T: DihedralGroup<hologram_ring::PrismPrimitives>>(_: &T) {}
    assert_dihedral(&PrismRing::<W8>::new());
    assert_dihedral(&PrismRing::<W32>::new());
}

// ── NormedDivisionAlgebra ────────────────────────────────────────────────

#[test]
fn nda_dimensions() {
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<W8>::new()),
        1
    );
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<W16>::new()),
        2
    );
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<W32>::new()),
        4
    );
    assert_eq!(
        NormedDivisionAlgebra::algebra_dimension(&PrismRing::<W64>::new()),
        8
    );
}

#[test]
fn nda_commutativity() {
    assert!(NormedDivisionAlgebra::is_commutative(
        &PrismRing::<W8>::new()
    ));
    assert!(NormedDivisionAlgebra::is_commutative(
        &PrismRing::<W16>::new()
    ));
    assert!(!NormedDivisionAlgebra::is_commutative(
        &PrismRing::<W32>::new()
    ));
    assert!(!NormedDivisionAlgebra::is_commutative(
        &PrismRing::<W64>::new()
    ));
}

#[test]
fn nda_associativity() {
    assert!(NormedDivisionAlgebra::is_associative(
        &PrismRing::<W8>::new()
    ));
    assert!(NormedDivisionAlgebra::is_associative(
        &PrismRing::<W32>::new()
    ));
    assert!(!NormedDivisionAlgebra::is_associative(
        &PrismRing::<W64>::new()
    ));
}

// ── CayleyDicksonConstruction ────────────────────────────────────────────

#[test]
fn cayley_dickson_w8_to_w16() {
    let r = PrismRing::<W8>::new();
    let src = CayleyDicksonConstruction::cayley_dickson_source(&r);
    let tgt = CayleyDicksonConstruction::cayley_dickson_target(&r);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(src), 1);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(tgt), 2);
    assert_eq!(CayleyDicksonConstruction::adjoined_element(&r), "i");
}

#[test]
fn cayley_dickson_w16_to_w32() {
    let r = PrismRing::<W16>::new();
    let src = CayleyDicksonConstruction::cayley_dickson_source(&r);
    let tgt = CayleyDicksonConstruction::cayley_dickson_target(&r);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(src), 2);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(tgt), 4);
    assert_eq!(CayleyDicksonConstruction::adjoined_element(&r), "j");
}

#[test]
fn cayley_dickson_w32_to_w64() {
    let r = PrismRing::<W32>::new();
    let src = CayleyDicksonConstruction::cayley_dickson_source(&r);
    let tgt = CayleyDicksonConstruction::cayley_dickson_target(&r);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(src), 4);
    assert_eq!(NormedDivisionAlgebra::algebra_dimension(tgt), 8);
    assert_eq!(CayleyDicksonConstruction::adjoined_element(&r), "l");
}

// ── Involution UOR traits ────────────────────────────────────────────────

#[test]
fn involution_operation_traits() {
    let neg: Involution<W8> = Involution::Neg;
    assert_eq!(Operation::arity(&neg), 1);
    assert_eq!(
        Operation::has_geometric_character(&neg),
        GeometricCharacter::RingReflection
    );
    assert_eq!(Operation::composed_of(&neg), "neg");

    let bnot: Involution<W8> = Involution::Bnot;
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
// v0.2.0 keeps `AlgebraCommutator` and `AlgebraAssociator` as empty marker
// traits (the formula-string methods are gone). These tests assert that
// `PrismRing<W>` still implements the markers.

#[test]
fn algebra_commutator_markers_implemented() {
    fn assert_impl<P, T>()
    where
        P: hologram_foundation::Primitives,
        T: AlgebraCommutator<P>,
    {
    }
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<W8>>();
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<W32>>();
}

#[test]
fn algebra_associator_markers_implemented() {
    fn assert_impl<P, T>()
    where
        P: hologram_foundation::Primitives,
        T: AlgebraAssociator<P>,
    {
    }
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<W8>>();
    assert_impl::<hologram_ring::PrismPrimitives, PrismRing<W64>>();
}
