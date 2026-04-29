//! Grounding implementations for Q0–Q3 ring types.
//!
//! Each grounding maps external `&[u8]` data into a `GroundedCoord` at the
//! appropriate quantum level. Per PRISM Section 1.3, these implement the
//! `source` boundary mapping.
//!
//! All groundings are O(1), zero-allocation, `#[inline]`.
//!
//! Per Plan 074, uor-foundation 0.3.0 reshaped the [`Grounding`] trait to
//! require a sealed-trait `program()` decomposition. Hologram's only
//! external consumer ([`ground_at_level`]) just needs the dispatch
//! function — so these are now plain inherent methods rather than full
//! `Grounding` impls. If hologram later needs the trait-driven machinery
//! (verified combinator decomposition, etc.) we can re-implement it then.

use crate::op::RingLevel;
use uor_foundation::enforcement::GroundedCoord;

/// Q0 grounding: external byte → Z/256Z via `GroundedCoord::w8`.
pub struct ByteGrounding;

impl ByteGrounding {
    #[inline]
    pub fn ground(&self, external: &[u8]) -> Option<GroundedCoord> {
        external.first().map(|&b| GroundedCoord::w8(b))
    }
}

/// Q1 grounding: external 2 bytes (LE) → Z/65536Z via `GroundedCoord::w16`.
pub struct WordGrounding;

impl WordGrounding {
    #[inline]
    pub fn ground(&self, external: &[u8]) -> Option<GroundedCoord> {
        if external.len() < 2 {
            return None;
        }
        Some(GroundedCoord::w16(u16::from_le_bytes([
            external[0],
            external[1],
        ])))
    }
}

/// Q2 grounding: external 3 bytes (LE, zero-padded to u32) → Z/16777216Z.
pub struct TripleGrounding;

impl TripleGrounding {
    #[inline]
    pub fn ground(&self, external: &[u8]) -> Option<GroundedCoord> {
        if external.len() < 3 {
            return None;
        }
        let v = u32::from_le_bytes([external[0], external[1], external[2], 0]);
        Some(GroundedCoord::w24(v & 0x00FF_FFFF))
    }
}

/// Q3 grounding: external 4 bytes (LE) → Z/2^32Z via `GroundedCoord::w32`.
pub struct QuadGrounding;

impl QuadGrounding {
    #[inline]
    pub fn ground(&self, external: &[u8]) -> Option<GroundedCoord> {
        if external.len() < 4 {
            return None;
        }
        Some(GroundedCoord::w32(u32::from_le_bytes([
            external[0],
            external[1],
            external[2],
            external[3],
        ])))
    }
}

/// Dispatch grounding by quantum level. O(1) enum match, no vtable.
#[inline]
pub fn ground_at_level(level: RingLevel, external: &[u8]) -> Option<GroundedCoord> {
    match level {
        RingLevel::Q0 => ByteGrounding.ground(external),
        RingLevel::Q1 => WordGrounding.ground(external),
        RingLevel::Q2 => TripleGrounding.ground(external),
        RingLevel::Q3 => QuadGrounding.ground(external),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grounding_q0_exhaustive() {
        for b in 0..=255u8 {
            let coord = ByteGrounding.ground(&[b]).unwrap();
            assert_eq!(coord, GroundedCoord::w8(b));
        }
    }

    #[test]
    fn grounding_q1_le_order() {
        let coord = WordGrounding.ground(&[0x34, 0x12]).unwrap();
        assert_eq!(coord, GroundedCoord::w16(0x1234));
    }

    #[test]
    fn grounding_q2_masks_to_24_bits() {
        let coord = TripleGrounding.ground(&[0xFF, 0xFF, 0xFF]).unwrap();
        assert_eq!(coord, GroundedCoord::w24(0x00FF_FFFF));
    }

    #[test]
    fn grounding_q3_le_order() {
        let coord = QuadGrounding.ground(&[0x78, 0x56, 0x34, 0x12]).unwrap();
        assert_eq!(coord, GroundedCoord::w32(0x12345678));
    }

    #[test]
    fn grounding_rejects_short_input() {
        assert!(WordGrounding.ground(&[0x42]).is_none());
        assert!(TripleGrounding.ground(&[0x42, 0x43]).is_none());
        assert!(QuadGrounding.ground(&[0x42, 0x43, 0x44]).is_none());
    }

    #[test]
    fn ground_at_level_dispatches_correctly() {
        assert!(ground_at_level(RingLevel::Q0, &[42]).is_some());
        assert!(ground_at_level(RingLevel::Q1, &[1, 2]).is_some());
        assert!(ground_at_level(RingLevel::Q2, &[1, 2, 3]).is_some());
        assert!(ground_at_level(RingLevel::Q3, &[1, 2, 3, 4]).is_some());
    }

    #[test]
    fn grounding_performance() {
        let data = [0x42u8; 4];
        let start = std::time::Instant::now();
        for _ in 0..10_000_000 {
            let _ = ground_at_level(RingLevel::Q0, &data);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10M groundings took {}ms (target < 50ms)",
            elapsed.as_millis()
        );
    }
}
