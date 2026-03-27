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
//! These lifting operations are the formal basis for cross-level composition:
//! a Q0 result can be an exact input to a Q1 operation via `lift_q0_to_q1`.

use crate::op::RingLevel;

// ── Exact lifting functions ───────────────────────────────────────────────────

/// Lift a Q0 value (u8) to Q1 (u16) via zero-extension.
///
/// DC_5: zero-extension is exact. The Q1 value represents the same ring
/// element as the Q0 value. `lift_carry_flux(x) = 0` for all x.
#[inline(always)]
#[must_use]
pub const fn lift_q0_to_q1(val: u8) -> u16 {
    val as u16
}

/// Lower a Q1 value (u16) to Q0 (u8) if it fits (high byte = 0).
///
/// Returns `Ok(lo_byte)` if the Q1 value is representable in Q0.
/// Returns `Err(val)` if the high byte is non-zero (precision would be lost).
#[inline]
pub const fn lower_q1_to_q0(val: u16) -> Result<u8, u16> {
    if val & 0xFF00 == 0 {
        Ok(val as u8)
    } else {
        Err(val)
    }
}

/// Lift a Q1 value (u16) to Q2 (u32, low 24 bits) via zero-extension.
///
/// DC_5: zero-extension is exact. The Q2 value represents the same ring
/// element as the Q1 value in Z/2^24Z.
#[inline(always)]
#[must_use]
pub const fn lift_q1_to_q2(val: u16) -> u32 {
    val as u32
}

/// Lower a Q2 value (u32, low 24 bits) to Q1 (u16) if it fits (bits 16-23 = 0).
///
/// Returns `Ok(lo_word)` if the Q2 value is representable in Q1.
/// Returns `Err(val)` if bits 16-23 are non-zero (precision would be lost).
#[inline]
pub const fn lower_q2_to_q1(val: u32) -> Result<u16, u32> {
    if val & 0x00FF_0000 == 0 {
        Ok(val as u16)
    } else {
        Err(val)
    }
}

/// Lift a Q0 value directly to Q2 via two zero-extension steps.
#[inline(always)]
#[must_use]
pub const fn lift_q0_to_q2(val: u8) -> u32 {
    val as u32
}

/// Carry flux introduced by lifting: always 0 for zero-extension (DC_5).
///
/// Zero-extension never generates carry — it only adds zero bits above.
/// This is the formal proof that all lifting operations are exact.
#[inline(always)]
#[must_use]
pub const fn lift_carry_flux(_val: u8) -> u8 {
    0
}

// ── CurvatureFlux ─────────────────────────────────────────────────────────────

/// Cumulative carry flux across a sequence of ring operations.
///
/// CF_3/CF_4: carry flux is non-decreasing along any computation path.
/// Tracks how much carry has accumulated at each ring level.
/// The `required_level()` method returns the minimum Q-level needed to
/// avoid overflow given the accumulated carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CurvatureFlux {
    /// Bits of carry generated at Q0 level.
    pub q0_carry: u32,
    /// Bits of carry generated at Q1 level.
    pub q1_carry: u32,
    /// Bits of carry generated at Q2 level.
    pub q2_carry: u32,
    /// Bits of carry generated at Q3 level.
    pub q3_carry: u32,
}

impl CurvatureFlux {
    /// Zero flux — no carry accumulated yet.
    pub const ZERO: Self = Self {
        q0_carry: 0,
        q1_carry: 0,
        q2_carry: 0,
        q3_carry: 0,
    };

    /// Accumulate carry from applying an op with the given curvature at the given level.
    ///
    /// CF_3: once carry is accumulated, `required_level()` can only increase
    /// (carry flux is non-decreasing).
    #[inline]
    pub fn accumulate(&mut self, curvature: u8, level: RingLevel) {
        match level {
            RingLevel::Q0 => self.q0_carry += curvature as u32,
            RingLevel::Q1 => self.q1_carry += curvature as u32,
            RingLevel::Q2 => self.q2_carry += curvature as u32,
            RingLevel::Q3 => self.q3_carry += curvature as u32,
        }
    }

    /// Minimum Q-level required to avoid carry overflow given accumulated flux.
    ///
    /// Decision thresholds:
    /// - q3_carry > 0 → Q3 (octonion-level carry, 32-bit range needed)
    /// - q2_carry > 0 → Q2 (deep carry chain, Q1 range insufficient)
    /// - q1_carry > 0 OR q0_carry > 8 → Q1 (Q0 range saturated)
    /// - otherwise → Q0 (carry-simple, 8-bit range sufficient)
    #[inline]
    #[must_use]
    pub fn required_level(self) -> RingLevel {
        if self.q3_carry > 0 {
            RingLevel::Q3
        } else if self.q2_carry > 0 {
            RingLevel::Q2
        } else if self.q1_carry > 0 || self.q0_carry > 8 {
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
    fn lift_q0_to_q1_exhaustive() {
        // All 256 Q0 values round-trip through Q1 losslessly.
        for x in 0u8..=255 {
            let lifted = lift_q0_to_q1(x);
            let lowered = lower_q1_to_q0(lifted).expect("round-trip must succeed");
            assert_eq!(lowered, x, "round-trip failed at {x}");
        }
    }

    #[test]
    fn lower_q1_to_q0_fails_for_high_byte() {
        assert!(lower_q1_to_q0(0x0100).is_err()); // bit 8 set
        assert!(lower_q1_to_q0(0xFF00).is_err());
        assert!(lower_q1_to_q0(0x0100).unwrap_err() == 0x0100);
    }

    #[test]
    fn lift_q1_to_q2_spot_checks() {
        for x in [0u16, 1, 127, 255, 256, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF] {
            let lifted = lift_q1_to_q2(x);
            let lowered = lower_q2_to_q1(lifted).expect("round-trip must succeed");
            assert_eq!(lowered, x, "Q1→Q2 round-trip failed at {x}");
        }
    }

    #[test]
    fn lower_q2_to_q1_fails_for_high_bits() {
        assert!(lower_q2_to_q1(0x00_010000).is_err()); // bit 16 set
        assert!(lower_q2_to_q1(0x00FF_0000).is_err());
    }

    #[test]
    fn lift_carry_flux_is_zero_for_all_bytes() {
        // DC_5: zero-extension generates zero carry.
        for x in 0u8..=255 {
            assert_eq!(lift_carry_flux(x), 0, "carry flux non-zero at {x}");
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
            let q2 = lift_q0_to_q2(x);
            assert_eq!(q2, x as u32, "lift_q0_to_q2 not zero-extending at {x}");
        }
    }
}
