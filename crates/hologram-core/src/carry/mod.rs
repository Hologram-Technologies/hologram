//! Carry-preserving precision lifting — DC_5 protocol.
//!
//! Formalizes exact (lossless) transitions between quantum levels:
//! Q0 (Z/256Z) → Q1 (Z/65536Z) → Q2 (Z/2^24Z).
//!
//! **DC_5 (carry decomposition)**: A value in Z/2^nZ can be exactly lifted to
//! Z/2^(n+8)Z via zero-extension. Zero-extension never introduces carry
//! (the high bits are all zero), so the lifted value represents the same
//! ring element in the larger ring with zero carry flux.
//!
//! **CF_3/CF_4 (curvature flux)**: Carry flux is non-decreasing along any
//! computation path. Once a carry is introduced (by a non-trivial operation),
//! it cannot be reduced by subsequent operations in the same ring.
//! The `CurvatureFlux` struct tracks cumulative carry at each level.
//!
//! The unified `lift`/`lower` functions handle cross-level composition for
//! any pair of quantum levels.

use crate::op::{QuantumLevelExt, RingLevel};
use uor_foundation::QuantumLevel;

// ── Unified lifting functions ─────────────────────────────────────────────────

/// Lift a value from a lower to higher quantum level via zero-extension (DC_5).
#[inline(always)]
pub const fn lift(val: u64, _from: QuantumLevel, _to: QuantumLevel) -> u64 {
    val
}

/// Lower a value from a higher to lower quantum level.
/// Returns Err(val) if precision would be lost.
#[inline]
pub fn lower(val: u64, _from: QuantumLevel, to: QuantumLevel) -> Result<u64, u64> {
    let bits = (to.byte_width() as u32) * 8;
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    if val & !mask == 0 {
        Ok(val)
    } else {
        Err(val)
    }
}

// ── CurvatureFlux ─────────────────────────────────────────────────────────────

/// Cumulative carry flux across a sequence of ring operations.
///
/// CF_3/CF_4: carry flux is non-decreasing along any computation path.
/// Tracks total accumulated carry and the maximum byte width at which
/// carry has been observed.
/// The `required_level()` method returns the minimum Q-level needed to
/// avoid overflow given the accumulated carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CurvatureFlux {
    /// Total accumulated carry bits across all levels.
    pub carry: u64,
    /// Maximum byte width at which carry has been observed.
    pub max_carry_width: u8,
}

impl CurvatureFlux {
    /// Zero flux — no carry accumulated yet.
    pub const ZERO: Self = Self {
        carry: 0,
        max_carry_width: 0,
    };

    /// Accumulate carry from applying an op with the given curvature at the given level.
    ///
    /// CF_3: once carry is accumulated, `required_level()` can only increase
    /// (carry flux is non-decreasing).
    #[inline]
    pub fn accumulate(&mut self, curvature: u8, level: RingLevel) {
        self.carry += curvature as u64;
        if curvature > 0 {
            let w = level.byte_width();
            if w > self.max_carry_width {
                self.max_carry_width = w;
            }
        }
    }

    /// Accumulate carry at a dynamic quantum level.
    #[inline]
    pub fn accumulate_at(&mut self, curvature: u8, level: uor_foundation::QuantumLevel) {
        self.carry += curvature as u64;
        if curvature > 0 {
            let w = level.byte_width();
            if w > self.max_carry_width {
                self.max_carry_width = w;
            }
        }
    }

    /// Minimum Q-level required to avoid carry overflow given accumulated flux.
    ///
    /// Decision thresholds:
    /// - max_carry_width >= 4 → Q3 (octonion-level carry, 32-bit range needed)
    /// - max_carry_width >= 3 → Q2 (deep carry chain, Q1 range insufficient)
    /// - max_carry_width >= 2 OR carry > 8 → Q1 (Q0 range saturated)
    /// - otherwise → Q0 (carry-simple, 8-bit range sufficient)
    #[inline]
    #[must_use]
    pub fn required_level(self) -> RingLevel {
        if self.max_carry_width >= 4 {
            RingLevel::Q3
        } else if self.max_carry_width >= 3 {
            RingLevel::Q2
        } else if self.max_carry_width >= 2 || self.carry > 8 {
            RingLevel::Q1
        } else {
            RingLevel::Q0
        }
    }

    /// Reset all carry to zero. Used at frame boundaries in streaming workloads.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::ZERO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lift_lower_round_trip_q0_q1() {
        // All 256 Q0 values round-trip through Q1 losslessly.
        for x in 0u8..=255 {
            let lifted = lift(x as u64, QuantumLevel::Q0, QuantumLevel::Q1);
            let lowered =
                lower(lifted, QuantumLevel::Q1, QuantumLevel::Q0).expect("round-trip must succeed");
            assert_eq!(lowered as u8, x, "round-trip failed at {x}");
        }
    }

    #[test]
    fn lower_fails_for_high_byte() {
        assert!(lower(0x0100, QuantumLevel::Q1, QuantumLevel::Q0).is_err());
        assert!(lower(0xFF00, QuantumLevel::Q1, QuantumLevel::Q0).is_err());
        assert_eq!(
            lower(0x0100, QuantumLevel::Q1, QuantumLevel::Q0).unwrap_err(),
            0x0100
        );
    }

    #[test]
    fn lift_lower_round_trip_q1_q2() {
        for x in [0u16, 1, 127, 255, 256, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF] {
            let lifted = lift(x as u64, QuantumLevel::Q1, QuantumLevel::Q2);
            let lowered =
                lower(lifted, QuantumLevel::Q2, QuantumLevel::Q1).expect("round-trip must succeed");
            assert_eq!(lowered as u16, x, "Q1→Q2 round-trip failed at {x}");
        }
    }

    #[test]
    fn lower_q2_to_q1_fails_for_high_bits() {
        assert!(lower(0x00_010000, QuantumLevel::Q2, QuantumLevel::Q1).is_err());
        assert!(lower(0x00FF_0000, QuantumLevel::Q2, QuantumLevel::Q1).is_err());
    }

    #[test]
    fn lift_carry_is_zero() {
        // DC_5: zero-extension generates zero carry (lift is identity on u64).
        for x in 0u8..=255 {
            let lifted = lift(x as u64, QuantumLevel::Q0, QuantumLevel::Q1);
            assert_eq!(lifted, x as u64, "lift not zero-extending at {x}");
        }
    }

    #[test]
    fn curvature_flux_zero_state() {
        assert_eq!(CurvatureFlux::ZERO.required_level(), RingLevel::Q0);
    }

    #[test]
    fn curvature_flux_tracks_accumulation() {
        let mut flux = CurvatureFlux::ZERO;
        assert_eq!(flux.required_level(), RingLevel::Q0);
        // Accumulate 8 bits at Q0 — still within Q0 range.
        for _ in 0..8 {
            flux.accumulate(1, RingLevel::Q0);
        }
        assert_eq!(flux.required_level(), RingLevel::Q0);
        // One more bit → promote to Q1.
        flux.accumulate(1, RingLevel::Q0);
        assert_eq!(flux.required_level(), RingLevel::Q1);
    }

    #[test]
    fn curvature_flux_q1_carry_promotes_to_q1() {
        let mut flux = CurvatureFlux::ZERO;
        flux.accumulate(1, RingLevel::Q1);
        assert_eq!(flux.required_level(), RingLevel::Q1);
    }

    #[test]
    fn curvature_flux_q2_carry_promotes_to_q2() {
        let mut flux = CurvatureFlux::ZERO;
        flux.accumulate(1, RingLevel::Q2);
        assert_eq!(flux.required_level(), RingLevel::Q2);
    }

    #[test]
    fn lift_q0_to_q2_is_exact() {
        for x in [0u8, 1, 127, 255] {
            let q2 = lift(x as u64, QuantumLevel::Q0, QuantumLevel::Q2);
            assert_eq!(q2, x as u64, "lift Q0→Q2 not zero-extending at {x}");
        }
    }
}
