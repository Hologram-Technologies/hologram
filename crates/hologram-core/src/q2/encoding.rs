//! Q2 (24-bit) encoding pipeline — continuous ↔ Z/2^24 Z.
//!
//! Four encodings matching the Q0/Q1 pattern:
//! - Unsigned: [0.0, 1.0] ↔ [0, 2^24 - 1]
//! - Signed:   [-1.0, 1.0] ↔ [0, 2^24 - 1]  (zero at 2^23)
//! - Angle:    [0.0, 2π) ↔ [0, 2^24 - 1]
//! - Raw:      identity truncation f64 → u32 (low 24 bits)

const MAX_Q2: f64 = 16_777_215.0; // 2^24 - 1

/// Encoding trait for Z/2^24 Z ↔ continuous domain.
pub trait Encoding24 {
    /// Embed a continuous value into [0, 2^24 - 1].
    fn embed(&self, val: f64) -> u32;
    /// Lift a Z/2^24 Z value to continuous domain.
    fn lift(&self, raw: u32) -> f64;
    /// Human-readable name.
    fn name(&self) -> &'static str;
}

/// [0.0, 1.0] ↔ [0, 2^24 - 1] (unsigned linear).
pub struct UnsignedEncoding24;

impl Encoding24 for UnsignedEncoding24 {
    #[inline]
    fn embed(&self, val: f64) -> u32 {
        libm::round(val.clamp(0.0, 1.0) * MAX_Q2) as u32 & 0x00FF_FFFF
    }
    #[inline]
    fn lift(&self, raw: u32) -> f64 {
        (raw & 0x00FF_FFFF) as f64 / MAX_Q2
    }
    fn name(&self) -> &'static str {
        "unsigned24"
    }
}

/// [-1.0, 1.0] ↔ [0, 2^24 - 1] (signed, zero at 2^23).
pub struct SignedEncoding24;

impl Encoding24 for SignedEncoding24 {
    #[inline]
    fn embed(&self, val: f64) -> u32 {
        let half = MAX_Q2 / 2.0;
        libm::round(val.clamp(-1.0, 1.0) * half + half) as u32 & 0x00FF_FFFF
    }
    #[inline]
    fn lift(&self, raw: u32) -> f64 {
        let half = MAX_Q2 / 2.0;
        ((raw & 0x00FF_FFFF) as f64 - half) / half
    }
    fn name(&self) -> &'static str {
        "signed24"
    }
}

/// [0.0, 2π) ↔ [0, 2^24 - 1].
pub struct AngleEncoding24;

impl Encoding24 for AngleEncoding24 {
    #[inline]
    fn embed(&self, val: f64) -> u32 {
        use core::f64::consts::TAU;
        let norm = libm::fmod(val, TAU);
        let norm = if norm < 0.0 { norm + TAU } else { norm } / TAU;
        libm::round(norm * MAX_Q2) as u32 & 0x00FF_FFFF
    }
    #[inline]
    fn lift(&self, raw: u32) -> f64 {
        use core::f64::consts::TAU;
        (raw & 0x00FF_FFFF) as f64 / MAX_Q2 * TAU
    }
    fn name(&self) -> &'static str {
        "angle24"
    }
}

/// Identity: truncate f64 to low 24 bits.
pub struct RawEncoding24;

impl Encoding24 for RawEncoding24 {
    #[inline]
    fn embed(&self, val: f64) -> u32 {
        (val as u32) & 0x00FF_FFFF
    }
    #[inline]
    fn lift(&self, raw: u32) -> f64 {
        (raw & 0x00FF_FFFF) as f64
    }
    fn name(&self) -> &'static str {
        "raw24"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_round_trip() {
        let enc = UnsignedEncoding24;
        for val in [0.0f64, 0.5, 1.0] {
            let raw = enc.embed(val);
            let back = enc.lift(raw);
            assert!(
                (val - back).abs() < 1e-6,
                "unsigned round-trip failed at {val}: got {back}"
            );
        }
    }

    #[test]
    fn signed_zero_at_midpoint() {
        let enc = SignedEncoding24;
        let mid = enc.embed(0.0);
        assert_eq!(mid, 0x00_800000, "signed zero should map to 2^23");
        let back = enc.lift(mid);
        assert!(back.abs() < 1e-6);
    }

    #[test]
    fn angle_full_cycle() {
        use core::f64::consts::TAU;
        let enc = AngleEncoding24;
        assert_eq!(enc.embed(0.0), 0);
        let back = enc.lift(enc.embed(TAU / 4.0));
        assert!((back - TAU / 4.0).abs() < 1e-4);
    }

    #[test]
    fn raw_identity() {
        let enc = RawEncoding24;
        assert_eq!(enc.embed(42.0), 42);
        assert!((enc.lift(42) - 42.0).abs() < 1e-10);
    }
}
