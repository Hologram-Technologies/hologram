//! UnsignedEncoding: maps [0, 1] ↔ [0, 255].

use super::Encoding;

/// Maps unsigned values in [0.0, 1.0] to bytes [0, 255].
///
/// 0.0 → 0, 1.0 → 255.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnsignedEncoding;

impl Encoding for UnsignedEncoding {
    #[inline]
    fn embed(&self, value: f64) -> u8 {
        let clamped = value.clamp(0.0, 1.0);
        (clamped * 255.0) as u8
    }

    #[inline]
    fn lift(&self, byte: u8) -> f64 {
        byte as f64 / 255.0
    }

    fn name(&self) -> &'static str {
        "unsigned"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_endpoints() {
        assert_eq!(UnsignedEncoding.embed(0.0), 0);
        assert_eq!(UnsignedEncoding.embed(1.0), 255);
    }

    #[test]
    fn lift_endpoints() {
        assert!((UnsignedEncoding.lift(0) - 0.0).abs() < 1e-10);
        assert!((UnsignedEncoding.lift(255) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn round_trip_approximate() {
        for i in 0..=255u8 {
            let lifted = UnsignedEncoding.lift(i);
            let re = UnsignedEncoding.embed(lifted);
            assert!(
                (re as i16 - i as i16).unsigned_abs() <= 1,
                "round trip drift > 1 for byte {i}: got {re}"
            );
        }
    }

    #[test]
    fn clamps() {
        assert_eq!(UnsignedEncoding.embed(-0.5), 0);
        assert_eq!(UnsignedEncoding.embed(1.5), 255);
    }

    #[test]
    fn monotonic() {
        for i in 0..255u8 {
            assert!(UnsignedEncoding.lift(i) < UnsignedEncoding.lift(i + 1));
        }
    }
}
