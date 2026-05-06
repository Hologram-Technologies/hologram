//! Region type (spec IV.7).

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region;

impl ConstrainedTypeShape for Region {
    const IRI: &'static str = "https://hologram.uor.foundation/type/region";
    const SITE_COUNT: usize = 2; // (offset, length)
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
