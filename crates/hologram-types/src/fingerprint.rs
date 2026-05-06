//! Fingerprint type (spec IV.7). 32 bytes = 32 W8 sites.

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint;

impl ConstrainedTypeShape for Fingerprint {
    const IRI: &'static str = "https://hologram.uor.foundation/type/fingerprint";
    const SITE_COUNT: usize = 32;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
