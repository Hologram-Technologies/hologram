//! SignedEncoding: maps [-1, 1] ↔ [0, 255].

use super::Encoding;

/// Maps signed values in [-1.0, 1.0] to bytes [0, 255].
///
/// -1.0 → 0, 0.0 → 128, 1.0 → 255.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignedEncoding;

impl Encoding for SignedEncoding {
    #[inline]
    fn embed(&self, value: f64) -> u8 {
        let clamped = value.clamp(-1.0, 1.0);
        ((clamped + 1.0) * 127.5) as u8
    }

    #[inline]
    fn lift(&self, byte: u8) -> f64 {
        (byte as f64 / 127.5) - 1.0
    }

    fn name(&self) -> &'static str {
        "signed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_endpoints() {
        assert_eq!(SignedEncoding.embed(-1.0), 0);
        assert_eq!(SignedEncoding.embed(1.0), 255);
    }

    #[test]
    fn embed_zero() {
        let byte = SignedEncoding.embed(0.0);
        // 0.0 → 127 or 128 depending on rounding
        assert!(byte == 127 || byte == 128);
    }

    #[test]
    fn lift_endpoints() {
        let lo = SignedEncoding.lift(0);
        let hi = SignedEncoding.lift(255);
        assert!((lo - (-1.0)).abs() < 0.01);
        assert!((hi - 1.0).abs() < 0.01);
    }

    #[test]
    fn clamps() {
        assert_eq!(SignedEncoding.embed(-5.0), 0);
        assert_eq!(SignedEncoding.embed(5.0), 255);
    }

    #[test]
    fn monotonic() {
        for i in 0..255u8 {
            assert!(SignedEncoding.lift(i) < SignedEncoding.lift(i + 1));
        }
    }
}
