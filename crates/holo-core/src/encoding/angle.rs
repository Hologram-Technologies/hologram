//! AngleEncoding: maps [0, 2pi) ↔ [0, 255].

use super::Encoding;
use core::f64::consts::TAU;

/// Maps angles in [0, 2pi) to bytes [0, 255].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AngleEncoding;

impl Encoding for AngleEncoding {
    #[inline]
    fn embed(&self, value: f64) -> u8 {
        let normalized = value.rem_euclid(TAU) / TAU;
        (normalized * 256.0 + 0.5) as u8
    }

    #[inline]
    fn lift(&self, byte: u8) -> f64 {
        (byte as f64 / 256.0) * TAU
    }

    fn name(&self) -> &'static str {
        "angle"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_zero() {
        assert_eq!(AngleEncoding.embed(0.0), 0);
    }

    #[test]
    fn embed_pi() {
        let byte = AngleEncoding.embed(core::f64::consts::PI);
        assert_eq!(byte, 128);
    }

    #[test]
    fn lift_zero() {
        let val = AngleEncoding.lift(0);
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn round_trip() {
        for i in 0..=255u8 {
            let lifted = AngleEncoding.lift(i);
            let re = AngleEncoding.embed(lifted);
            assert_eq!(re, i, "round trip failed for byte {i}");
        }
    }

    #[test]
    fn wraps_around() {
        let byte = AngleEncoding.embed(TAU);
        assert_eq!(byte, 0);
    }
}
