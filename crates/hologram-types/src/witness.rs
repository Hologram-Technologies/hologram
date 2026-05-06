//! Per-op witness records (spec IV.7).

use core::marker::PhantomData;
use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

/// Witness record for an operation marker `Op`. `Op` is a hologram-ops marker
/// type. The IRI is the op's IRI suffixed with `/witness`. Concrete IRIs are
/// provided by per-op impls in hologram-ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessRecord<Op>(PhantomData<Op>);

impl<Op> Default for WitnessRecord<Op> {
    fn default() -> Self { Self(PhantomData) }
}

impl<Op> ConstrainedTypeShape for WitnessRecord<Op> {
    // Generic witness record. Per-op specializations
    // (`WitnessRecord<MatMulOp>` etc.) inherit this IRI; ops that need a
    // tighter IRI define a newtype wrapper in `hologram-ops`. Spec C-3
    // sanctions hologram introducing types in its own IRI namespace.
    const IRI: &'static str = "https://hologram.uor.foundation/type/witness";
    const SITE_COUNT: usize = 0;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
