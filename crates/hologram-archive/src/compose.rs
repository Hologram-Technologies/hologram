//! **Non-commutative ordered composition on the BLAKE3 σ-axis** —
//! hologram's realization of the ADR-061 ordered product that uor-addr
//! does not ship (it ships only the commutative / equivalence operators
//! `g2`/`f4`/`e6`/`e7`/`e8`). Built on the public prism SDK exactly as
//! `prism-btc`'s `compose_ordered_product` is, but bound to
//! [`Blake3Hasher`] + [`AddressResolverTuple`].
//!
//! Used by [`crate::address::derive_label_witnessed`] to address a computed
//! value by *ordered* composition of its operands' κ-labels, yielding a
//! replayable TC-05 witness — so `f(A,B)` and `f(B,A)` get distinct,
//! verifiable addresses *by the operator's algebra*, not by a fold trick.

extern crate alloc;
use alloc::vec::Vec;

use prism::crypto::Blake3Hasher;
use prism::operation::TermValue;
use prism::pipeline::{
    output_shape, prism_model, ConstrainedTypeShape, ConstraintRef, EmptyCommitment,
    IntoBindingValue, PartitionProductFields, PrismModel,
};
use prism::vocabulary::DefaultHostTypes;
use uor_addr::composition::CompositionFailure;
use uor_addr::{AddrBounds, AddressOutcome, AddressResolverTuple, KappaLabel};

/// `blake3:<64hex>` κ-label width.
const LABEL: usize = 71;

static COMP_SITES: [ConstraintRef; LABEL] = uor_addr::label::site_constraints::<LABEL>();

/// Ordered-composition input carrier — a borrow of the canonical-form
/// bytes (`left_digest ‖ right_digest`) flowing through the ψ-tower as an
/// ADR-060 `Borrowed` carrier.
#[derive(Clone, Copy, Debug)]
pub struct OrderedCarrier<'a>(&'a [u8]);

impl<'a> OrderedCarrier<'a> {
    #[must_use]
    pub fn new(canonical_bytes: &'a [u8]) -> Self {
        Self(canonical_bytes)
    }
}

impl ConstrainedTypeShape for OrderedCarrier<'_> {
    const IRI: &'static str = "https://hologram.uor.foundation/addr/composition/OrderedCarrier";
    const SITE_COUNT: usize = 1;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
    const CYCLE_SIZE: u64 = u64::MAX;
}

impl prism::uor_foundation::pipeline::__sdk_seal::Sealed for OrderedCarrier<'_> {}

impl<'a> IntoBindingValue<'a> for OrderedCarrier<'a> {
    fn as_binding_value<const INLINE_BYTES: usize>(&self) -> TermValue<'a, INLINE_BYTES> {
        TermValue::borrowed(self.0)
    }
}

impl PartitionProductFields for OrderedCarrier<'_> {
    const FIELDS: &'static [(u32, u32)] = &[];
    const FIELD_NAMES: &'static [&'static str] = &[];
}

output_shape! {
    pub struct OrderedLabel;
    impl ConstrainedTypeShape for OrderedLabel {
        const IRI: &'static str =
            "https://hologram.uor.foundation/addr/composition/ordered-product/blake3";
        const SITE_COUNT: usize = LABEL;
        const CONSTRAINTS: &'static [ConstraintRef] = &COMP_SITES;
    }
}

prism::pipeline::verb! {
    pub fn compose_ordered_inference(input: OrderedCarrier<'_>) -> OrderedLabel {
        k_invariants(homotopy_groups(postnikov_tower(nerve(input))))
    }
}

prism_model! {
    pub struct OrderedModel;
    pub struct OrderedRoute;
    impl PrismModel<
        DefaultHostTypes,
        AddrBounds,
        Blake3Hasher,
        AddressResolverTuple<Blake3Hasher>,
        EmptyCommitment
    > for OrderedModel {
        type Input = OrderedCarrier<'a>;
        type Output = OrderedLabel;
        type Route = OrderedRoute;
        fn route(input: Self::Input) -> Self::Output {
            compose_ordered_inference(input)
        }
    }
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        _ => None,
    }
}

/// Extract the 32-byte digest from a `blake3:<64hex>` κ-label.
fn decode_blake3(operand: &KappaLabel<LABEL>) -> Result<[u8; 32], CompositionFailure> {
    let axis = operand
        .sigma_axis()
        .ok_or(CompositionFailure::MalformedOperand)?;
    if axis != "blake3" {
        return Err(CompositionFailure::MalformedOperand);
    }
    let hex = operand
        .sigma_axis_digest_hex()
        .ok_or(CompositionFailure::MalformedOperand)?;
    if hex.len() != 64 {
        return Err(CompositionFailure::MalformedOperand);
    }
    let mut raw = [0u8; 32];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_nibble(pair[0]).ok_or(CompositionFailure::MalformedOperand)?;
        let lo = hex_nibble(pair[1]).ok_or(CompositionFailure::MalformedOperand)?;
        raw[i] = (hi << 4) | lo;
    }
    Ok(raw)
}

/// Non-commutative ordered product on the BLAKE3 σ-axis: grounds the
/// ordered concatenation `left_digest ‖ right_digest` through the ψ-tower,
/// minting a witnessed κ-label. `compose_ordered_blake3(a, b)` differs from
/// `compose_ordered_blake3(b, a)` by the operator's byte discipline.
///
/// # Errors
///
/// [`CompositionFailure`] if an operand is not a well-formed `blake3`
/// κ-label, or (defensively) on a pipeline shape violation.
/// Address-only replay of [`compose_ordered_blake3`]: the same ordered
/// product, minting only the κ-label. The ψ-tower grounds the canonical
/// bytes (`left_digest ‖ right_digest`) on the BLAKE3 σ-axis, so the
/// grounded address **is** the σ-axis fold of those bytes — this computes
/// it directly, skipping the TC-05 witness scaffolding the executor's hot
/// path immediately drops. The witnessed form remains the authority: it is
/// re-derivable on demand for any address that must be independently
/// replayed, and `ordered_address_matches_witnessed` pins pointwise
/// equality, so any future algebra change fails closed instead of
/// silently diverging.
///
/// # Errors
///
/// [`CompositionFailure`] if an operand is not a well-formed `blake3`
/// κ-label.
pub fn compose_ordered_blake3_address(
    left: &KappaLabel<LABEL>,
    right: &KappaLabel<LABEL>,
) -> Result<KappaLabel<LABEL>, CompositionFailure> {
    let l = decode_blake3(left)?;
    let r = decode_blake3(right)?;
    let mut canon = [0u8; 64];
    canon[..32].copy_from_slice(&l);
    canon[32..].copy_from_slice(&r);
    Ok(crate::address::address_bytes(&canon))
}

pub fn compose_ordered_blake3(
    left: &KappaLabel<LABEL>,
    right: &KappaLabel<LABEL>,
) -> Result<AddressOutcome<LABEL>, CompositionFailure> {
    let l = decode_blake3(left)?;
    let r = decode_blake3(right)?;
    let mut canon = Vec::with_capacity(64);
    canon.extend_from_slice(&l);
    canon.extend_from_slice(&r);
    let grounded = OrderedModel::forward(OrderedCarrier::new(&canon))
        .map_err(|_| CompositionFailure::PipelineFailure)?;
    AddressOutcome::<LABEL>::from_grounded(&grounded)
        .map_err(|_| CompositionFailure::PipelineFailure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::address_bytes;

    #[test]
    fn ordered_address_matches_witnessed() {
        // The address-only ordered product must equal the witnessed
        // grounding's address pointwise — the fail-closed pin that keeps
        // the fast path honest if the composition algebra ever changes.
        let labels: alloc::vec::Vec<_> = (0..6u8)
            .map(|i| address_bytes(&[i, 0x5A, i.wrapping_mul(101)]))
            .collect();
        for l in &labels {
            for r in &labels {
                let witnessed = compose_ordered_blake3(l, r).expect("witnessed").address;
                let fast = compose_ordered_blake3_address(l, r).expect("fast");
                assert_eq!(witnessed.as_bytes(), fast.as_bytes());
            }
        }
    }
}
