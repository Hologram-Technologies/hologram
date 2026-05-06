//! Weight + Constant types (spec IV.7).

use core::marker::PhantomData;
use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};
use uor_foundation::HostBounds;

/// Model weight: a content-addressed tensor body. `SITES = D::SITE_COUNT + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Weight<D, B, const SITES: usize>(PhantomData<(D, B)>)
where
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<D, B, const SITES: usize> Default for Weight<D, B, SITES>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self { Self(PhantomData) }
}

impl<D, B, const SITES: usize> ConstrainedTypeShape for Weight<D, B, SITES>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    const IRI: &'static str = "https://hologram.uor.foundation/type/weight";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}

/// Inline graph constant. Identical shape to `Weight` but distinct IRI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Constant<D, B, const SITES: usize>(PhantomData<(D, B)>)
where
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<D, B, const SITES: usize> Default for Constant<D, B, SITES>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self { Self(PhantomData) }
}

impl<D, B, const SITES: usize> ConstrainedTypeShape for Constant<D, B, SITES>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    const IRI: &'static str = "https://hologram.uor.foundation/type/constant";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
