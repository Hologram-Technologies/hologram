//! Layout type (spec IV.7).

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

/// Storage layout: strides + alignment + byte order.
/// `SITES = RANK + 2` (RANK strides + alignment + byte_order).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout<const RANK: usize, const SITES: usize>;

impl<const RANK: usize, const SITES: usize> ConstrainedTypeShape for Layout<RANK, SITES> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/layout";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
