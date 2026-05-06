//! Spec XII.3: every dtype/shape/tensor declaration produces a parseable IRI.

use uor_foundation::pipeline::ConstrainedTypeShape;
use hologram_types::*;

#[test]
fn dtype_iris_under_namespace() {
    assert!(DTypeF32::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeF16::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeBf16::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeF64::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeI64::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeI32::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeI8::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeU64::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeU8::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
    assert!(DTypeBool::IRI.starts_with("https://hologram.uor.foundation/type/dtype/"));
}

#[test]
fn dtype_bit_widths_match() {
    assert_eq!(DTypeF32::BIT_WIDTH, 32);
    assert_eq!(DTypeF16::BIT_WIDTH, 16);
    assert_eq!(DTypeBf16::BIT_WIDTH, 16);
    assert_eq!(DTypeF64::BIT_WIDTH, 64);
    assert_eq!(DTypeI64::BIT_WIDTH, 64);
    assert_eq!(DTypeI32::BIT_WIDTH, 32);
    assert_eq!(DTypeI8::BIT_WIDTH, 8);
    assert_eq!(DTypeBool::BIT_WIDTH, 1);
}

#[test]
fn dim_constraint_is_affine() {
    assert_eq!(<Dim<128> as ConstrainedTypeShape>::SITE_COUNT, 1);
    let cs = <Dim<128> as ConstrainedTypeShape>::CONSTRAINTS;
    assert_eq!(cs.len(), 1);
}

#[test]
fn fingerprint_is_32_sites() {
    assert_eq!(<Fingerprint as ConstrainedTypeShape>::SITE_COUNT, 32);
}
