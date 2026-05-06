//! `HologramHostTypes` (spec III.1).

use uor_foundation::HostTypes;

/// Hologram's `HostTypes` impl. Identical layout to `DefaultHostTypes`,
/// distinct as a marker for downstream substitution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostTypes;

impl HostTypes for HologramHostTypes {
    type Decimal = f64;
    type HostString = str;
    type WitnessBytes = [u8];
    const EMPTY_DECIMAL: f64 = 0.0;
    const EMPTY_HOST_STRING: &'static str = "";
    const EMPTY_WITNESS_BYTES: &'static [u8] = &[];
}
