//! Tensor type (spec IV.6).

use core::marker::PhantomData;
use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};
use uor_foundation::HostBounds;

/// Tensor with shape `S`, dtype `D`, host bounds `B`, and aggregate site
/// count `SITES`.
///
/// `SITES` is the product of dimension sizes; the compiler supplies it at
/// monomorphization. `B::WITT_LEVEL_MAX_BITS` governs which `WittLevel` the
/// compile path targets — it is not encoded as a constraint here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor<S, D, B, const SITES: usize>(PhantomData<(S, D, B)>)
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<S, D, B, const SITES: usize> Default for Tensor<S, D, B, SITES>
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S, D, B, const SITES: usize> ConstrainedTypeShape for Tensor<S, D, B, SITES>
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    const IRI: &'static str = "https://hologram.uor.foundation/type/tensor";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
