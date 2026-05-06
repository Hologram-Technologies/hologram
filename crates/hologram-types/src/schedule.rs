//! Schedule type (spec IV.7).

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

/// `LEVELS` parallel-execution levels, each carrying a NodeId set.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Schedule<const LEVELS: usize>;

impl<const LEVELS: usize> ConstrainedTypeShape for Schedule<LEVELS> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/schedule";
    const SITE_COUNT: usize = LEVELS;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
