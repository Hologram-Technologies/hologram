//! Spec XII.3: every dtype / shape marker hologram contributes produces
//! a parseable IRI under the hologram namespace, and the canonical
//! prism shape carriers reach hologram callers through this crate's
//! re-exports.

use hologram_types::*;
use uor_foundation::pipeline::ConstrainedTypeShape;

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
fn prism_matrix_shape_reachable_through_hologram_types() {
    // Smoke: the prism-tensor MatrixShape is re-exported from this
    // crate's surface and resolves as a ConstrainedTypeShape with
    // `SITE_COUNT = R*C*E`.
    type M = MatrixShape<4, 4, 1>;
    assert_eq!(<M as ConstrainedTypeShape>::SITE_COUNT, 16);
}

#[test]
fn prism_digest_reachable_through_hologram_types() {
    type D32 = Digest<32>;
    assert_eq!(<D32 as ConstrainedTypeShape>::SITE_COUNT, 32);
}
