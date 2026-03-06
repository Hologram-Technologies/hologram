//! Encoding16: embed continuous values into 16-bit word space and lift back.
//!
//! The pi-F-lambda pipeline at Q1:
//! - **pi (embed)**: continuous → u16
//! - **F**: O(1) u16→u16 LUT lookup
//! - **lambda (lift)**: u16 → continuous

/// Trait for embedding continuous values into the word ring and lifting back.
pub trait Encoding16 {
    /// Embed a continuous f64 value into a u16.
    fn embed(&self, value: f64) -> u16;

    /// Lift a u16 back to a continuous f64 value.
    fn lift(&self, word: u16) -> f64;

    /// Human-readable name of this encoding.
    fn name(&self) -> &'static str;
}

/// Angle encoding: maps [0, 2*pi) to [0, 65535].
#[derive(Debug, Clone, Copy)]
pub struct AngleEncoding16;

impl Encoding16 for AngleEncoding16 {
    #[inline]
    fn embed(&self, value: f64) -> u16 {
        let tau = core::f64::consts::TAU;
        let normalized = ((value % tau) + tau) % tau;
        let scaled = normalized / tau * 65536.0;
        if scaled >= 65536.0 {
            65535
        } else {
            scaled as u16
        }
    }

    #[inline]
    fn lift(&self, word: u16) -> f64 {
        word as f64 * core::f64::consts::TAU / 65536.0
    }

    fn name(&self) -> &'static str {
        "angle16"
    }
}

/// Signed encoding: maps [-1.0, 1.0] to [0, 65535].
#[derive(Debug, Clone, Copy)]
pub struct SignedEncoding16;

impl Encoding16 for SignedEncoding16 {
    #[inline]
    fn embed(&self, value: f64) -> u16 {
        let clamped = value.clamp(-1.0, 1.0);
        ((clamped + 1.0) * 0.5 * 65535.0) as u16
    }

    #[inline]
    fn lift(&self, word: u16) -> f64 {
        word as f64 / 65535.0 * 2.0 - 1.0
    }

    fn name(&self) -> &'static str {
        "signed16"
    }
}

/// Unsigned encoding: maps [0.0, 1.0] to [0, 65535].
#[derive(Debug, Clone, Copy)]
pub struct UnsignedEncoding16;

impl Encoding16 for UnsignedEncoding16 {
    #[inline]
    fn embed(&self, value: f64) -> u16 {
        let clamped = value.clamp(0.0, 1.0);
        (clamped * 65535.0) as u16
    }

    #[inline]
    fn lift(&self, word: u16) -> f64 {
        word as f64 / 65535.0
    }

    fn name(&self) -> &'static str {
        "unsigned16"
    }
}

/// Raw encoding: identity mapping (truncates f64 to u16).
#[derive(Debug, Clone, Copy)]
pub struct RawEncoding16;

impl Encoding16 for RawEncoding16 {
    #[inline]
    fn embed(&self, value: f64) -> u16 {
        if value < 0.0 {
            0
        } else if value > 65535.0 {
            65535
        } else {
            value as u16
        }
    }

    #[inline]
    fn lift(&self, word: u16) -> f64 {
        word as f64
    }

    fn name(&self) -> &'static str {
        "raw16"
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    // --- AngleEncoding16 ---

    #[test]
    fn angle_embed_zero() {
        let enc = AngleEncoding16;
        assert_eq!(enc.embed(0.0), 0);
    }

    #[test]
    fn angle_embed_pi() {
        let enc = AngleEncoding16;
        let half = enc.embed(core::f64::consts::PI);
        assert!((32700..=32900).contains(&half), "pi → {half}");
    }

    #[test]
    fn angle_lift_zero() {
        let enc = AngleEncoding16;
        let v = enc.lift(0);
        assert!((v - 0.0).abs() < 1e-6);
    }

    #[test]
    fn angle_round_trip() {
        let enc = AngleEncoding16;
        let original = 1.5;
        let word = enc.embed(original);
        let recovered = enc.lift(word);
        assert!((original - recovered).abs() < 0.001);
    }

    #[test]
    fn angle_wraps_around() {
        let enc = AngleEncoding16;
        let tau = core::f64::consts::TAU;
        let a = enc.embed(0.5);
        let b = enc.embed(0.5 + tau);
        assert!((a as i32 - b as i32).unsigned_abs() <= 1);
    }

    // --- SignedEncoding16 ---

    #[test]
    fn signed_embed_endpoints() {
        let enc = SignedEncoding16;
        assert_eq!(enc.embed(-1.0), 0);
        assert_eq!(enc.embed(1.0), 65535);
    }

    #[test]
    fn signed_lift_endpoints() {
        let enc = SignedEncoding16;
        assert!((enc.lift(0) - (-1.0)).abs() < 1e-6);
        assert!((enc.lift(65535) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn signed_embed_zero() {
        let enc = SignedEncoding16;
        let mid = enc.embed(0.0);
        assert!((32700..=32900).contains(&mid), "0.0 → {mid}");
    }

    #[test]
    fn signed_monotonic() {
        let enc = SignedEncoding16;
        let steps: Vec<u16> = (-100..=100).map(|i| enc.embed(i as f64 / 100.0)).collect();
        for w in steps.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn signed_clamps() {
        let enc = SignedEncoding16;
        assert_eq!(enc.embed(-5.0), 0);
        assert_eq!(enc.embed(5.0), 65535);
    }

    // --- UnsignedEncoding16 ---

    #[test]
    fn unsigned_embed_endpoints() {
        let enc = UnsignedEncoding16;
        assert_eq!(enc.embed(0.0), 0);
        assert_eq!(enc.embed(1.0), 65535);
    }

    #[test]
    fn unsigned_lift_endpoints() {
        let enc = UnsignedEncoding16;
        assert!((enc.lift(0) - 0.0).abs() < 1e-6);
        assert!((enc.lift(65535) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unsigned_monotonic() {
        let enc = UnsignedEncoding16;
        let steps: Vec<u16> = (0..=100).map(|i| enc.embed(i as f64 / 100.0)).collect();
        for w in steps.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn unsigned_clamps() {
        let enc = UnsignedEncoding16;
        assert_eq!(enc.embed(-1.0), 0);
        assert_eq!(enc.embed(2.0), 65535);
    }

    #[test]
    fn unsigned_round_trip() {
        let enc = UnsignedEncoding16;
        let original = 0.5;
        let word = enc.embed(original);
        let recovered = enc.lift(word);
        assert!((original - recovered).abs() < 0.001);
    }

    // --- RawEncoding16 ---

    #[test]
    fn raw_identity() {
        let enc = RawEncoding16;
        for i in (0u32..=65535).step_by(1000) {
            assert_eq!(enc.embed(i as f64), i as u16);
            assert_eq!(enc.lift(i as u16), i as f64);
        }
    }

    #[test]
    fn raw_clamps() {
        let enc = RawEncoding16;
        assert_eq!(enc.embed(-1.0), 0);
        assert_eq!(enc.embed(100000.0), 65535);
    }
}
