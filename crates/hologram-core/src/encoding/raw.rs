//! RawEncoding: byte as-is (identity mapping).

use super::Encoding;

/// Identity encoding: byte value maps directly to/from f64.
///
/// embed truncates to u8 range, lift casts to f64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawEncoding;

impl Encoding for RawEncoding {
    #[inline]
    fn embed(&self, value: f64) -> u8 {
        value.clamp(0.0, 255.0) as u8
    }

    #[inline]
    fn lift(&self, byte: u8) -> f64 {
        byte as f64
    }

    fn name(&self) -> &'static str {
        "raw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        for i in 0..=255u8 {
            assert_eq!(RawEncoding.embed(i as f64), i);
            assert_eq!(RawEncoding.lift(i), i as f64);
        }
    }

    #[test]
    fn clamps() {
        assert_eq!(RawEncoding.embed(-10.0), 0);
        assert_eq!(RawEncoding.embed(300.0), 255);
    }

    #[test]
    fn round_trip() {
        for i in 0..=255u8 {
            assert_eq!(RawEncoding.embed(RawEncoding.lift(i)), i);
        }
    }
}
