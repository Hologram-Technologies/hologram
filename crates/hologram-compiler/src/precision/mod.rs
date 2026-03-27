//! Observable-guided quantum-level selection.
//!
//! Uses the Q0 stratum (Hamming weight) and curvature (carry-depth) observables —
//! defined in hologram-core but unused during execution — to select the minimum
//! ring precision tier (Q0/Q1/Q2) required for each byte-domain node.
//!
//! UOR grounding: stratum and curvature are computable entirely from the node's
//! 256-byte output LUT at compile time. Low stratum + low curvature → Q0 sufficient
//! (carry-simple). High stratum or curvature → promote to Q1 or Q2.
//!
//! This is the mechanism that connects UOR observable tables (previously
//! informationally inert) to actual dispatch decisions.

pub mod pass;
pub use pass::promote_prim_ring_levels;

use hologram_core::lut::q0::{CURVATURE_Q0, STRATUM_Q0};
use hologram_core::op::RingLevel;
use hologram_core::view::ElementWiseView;

/// Mean stratum (Hamming weight) of a 256-entry output distribution.
///
/// Low value (< 4.0) ↔ carry-simple (few bits set per output byte) ↔ Q0 sufficient.
/// High value (> 6.0) ↔ carry-complex ↔ Q1/Q2 required.
///
/// Uses STRATUM_Q0 lookup table from hologram-core.
#[inline]
#[must_use]
pub fn mean_stratum_q0(table: &[u8; 256]) -> f32 {
    table
        .iter()
        .map(|&x| STRATUM_Q0[x as usize] as f32)
        .sum::<f32>()
        / 256.0
}

/// Mean curvature (carry-depth) of a 256-entry output distribution.
///
/// Curvature measures how many bits change between consecutive values.
/// Low value (< 1.5) → carry transitions are cheap → Q0 sufficient.
/// High value → deep carry chains → Q1 recommended.
///
/// Uses CURVATURE_Q0 lookup table from hologram-core.
#[inline]
#[must_use]
pub fn mean_curvature_q0(table: &[u8; 256]) -> f32 {
    table
        .iter()
        .map(|&x| CURVATURE_Q0[x as usize] as f32)
        .sum::<f32>()
        / 256.0
}

/// Stratum threshold above which Q1 ring precision is required.
pub const STRATUM_Q1_THRESHOLD: f32 = 4.0;
/// Stratum threshold above which Q2 ring precision is required.
pub const STRATUM_Q2_THRESHOLD: f32 = 6.0;
/// Curvature threshold above which Q1 ring precision is required.
pub const CURVATURE_Q1_THRESHOLD: f32 = 1.5;

/// Select the minimum sufficient `RingLevel` for a byte-domain node with the given LUT table.
///
/// Decision:
/// - mean_stratum > Q2_THRESHOLD → Q2 (high carry complexity, >6 bits/output average)
/// - mean_stratum > Q1_THRESHOLD OR mean_curvature > Q1_THRESHOLD → Q1
/// - otherwise → Q0 (carry-simple, L1-cacheable 256-entry LUT sufficient)
///
/// Called at compile time (tape build) — never at inference time.
#[must_use]
pub fn select_ring_level(table: &[u8; 256]) -> RingLevel {
    let s = mean_stratum_q0(table);
    let c = mean_curvature_q0(table);
    if s > STRATUM_Q2_THRESHOLD {
        RingLevel::Q2
    } else if s > STRATUM_Q1_THRESHOLD || c > CURVATURE_Q1_THRESHOLD {
        RingLevel::Q1
    } else {
        RingLevel::Q0
    }
}

/// Select the minimum sufficient `RingLevel` for a graph node given its `ElementWiseView`.
///
/// Delegates to `select_ring_level` via the view's table reference.
#[must_use]
pub fn select_ring_level_for_view(view: &ElementWiseView) -> RingLevel {
    select_ring_level(view.table())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_same_table(val: u8) -> [u8; 256] {
        [val; 256]
    }

    fn identity_table() -> [u8; 256] {
        let mut t = [0u8; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = i as u8;
        }
        t
    }

    fn ramp_table() -> [u8; 256] {
        // Output spans all 256 values → high stratum (~4 bits average for uniform)
        identity_table()
    }

    #[test]
    fn constant_zero_table_is_q0() {
        // Table mapping everything to 0: stratum=0, curvature=1 → Q0.
        let table = all_same_table(0);
        assert_eq!(select_ring_level(&table), RingLevel::Q0);
    }

    #[test]
    fn identity_table_promotes() {
        // Identity table: uniform distribution over all 256 bytes.
        // Mean stratum ≈ 4.0 (average Hamming weight of 0..255).
        let table = identity_table();
        let s = mean_stratum_q0(&table);
        assert!(
            s > 3.5 && s < 4.5,
            "mean stratum of identity table ≈ 4: got {s}"
        );
    }

    #[test]
    fn all_ff_table_is_q2() {
        // Table mapping everything to 0xFF: stratum=8 (max) → Q2.
        let table = all_same_table(0xFF);
        assert_eq!(select_ring_level(&table), RingLevel::Q2);
    }

    #[test]
    fn low_stratum_table_is_q0() {
        // Table mapping everything to 0x00: stratum=0, curvature=1 (0→1: 1 bit changes).
        let table = all_same_table(0x00);
        assert_eq!(mean_stratum_q0(&table), 0.0);
        assert_eq!(mean_curvature_q0(&table), 1.0);
        assert_eq!(select_ring_level(&table), RingLevel::Q0);
    }

    #[test]
    fn relu_like_table_is_q0() {
        // Relu: for values 128..255 (MSB set = negative in signed), output is 0.
        // For values 0..127, output is identity.
        // Low average stratum.
        let mut table = [0u8; 256];
        for (i, v) in table[..128].iter_mut().enumerate() {
            *v = i as u8;
        }
        let s = mean_stratum_q0(&table);
        // Half of outputs are 0 (stratum 0), half are identity (~4 bits average)
        // → overall ~2.0, which is below Q1 threshold.
        assert!(
            s < STRATUM_Q1_THRESHOLD,
            "relu-like should be Q0: stratum={s}"
        );
        assert_eq!(select_ring_level(&table), RingLevel::Q0);
    }

    #[test]
    fn mean_stratum_uniform_approx_4() {
        // Uniform distribution 0..=255: expected mean Hamming weight ≈ 4.0.
        let table = ramp_table();
        let s = mean_stratum_q0(&table);
        assert!((s - 4.0).abs() < 0.1, "uniform mean stratum ≈ 4.0: got {s}");
    }
}
