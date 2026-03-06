//! Q1 (16-bit) observable functions.
//!
//! Stratum and curvature delegate to Q0 tables via byte decomposition.
//! Domain, rank, torus, and orbit use direct computation (no LUT needed).

use crate::lut::q0;

/// Stratum (Hamming weight) for Q1: popcount of 16-bit value.
///
/// Decomposed via two Q0 lookups: `stratum_q0(hi) + stratum_q0(lo)`.
#[inline]
pub const fn stratum_q1(value: u16) -> u8 {
    q0::stratum_q1(value)
}

/// Curvature for Q1: `hamming(value, value + 1)`.
///
/// Decomposed via XOR + stratum.
#[inline]
pub const fn curvature_q1(value: u16) -> u8 {
    q0::curvature_q1(value)
}

/// Domain (mod 3) for Q1.
#[inline]
pub const fn domain_q1(value: u16) -> u16 {
    value % 3
}

/// Rank (div 3) for Q1.
#[inline]
pub const fn rank_q1(value: u16) -> u16 {
    value / 3
}

/// Torus page for Q1: `value / 16`.
///
/// Q1 torus uses 16-element pages (double Q0's 8-element pages).
#[inline]
pub const fn torus_page_q1(value: u16) -> u16 {
    value / 16
}

/// Torus offset for Q1: `value % 16`.
#[inline]
pub const fn torus_offset_q1(value: u16) -> u16 {
    value % 16
}

/// Orbit class for Q1: `value / 16`.
#[inline]
pub const fn orbit_class_q1(value: u16) -> u16 {
    value / 16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratum_matches_popcount() {
        for i in 0u16..=255 {
            assert_eq!(stratum_q1(i), i.count_ones() as u8);
        }
        assert_eq!(stratum_q1(0), 0);
        assert_eq!(stratum_q1(0xFFFF), 16);
        assert_eq!(stratum_q1(0x00FF), 8);
        assert_eq!(stratum_q1(0xFF00), 8);
        assert_eq!(stratum_q1(0x5555), 8);
        assert_eq!(stratum_q1(0xAAAA), 8);
        assert_eq!(stratum_q1(1), 1);
        assert_eq!(stratum_q1(0x8000), 1);
    }

    #[test]
    fn stratum_sampled_exhaustive() {
        // Sample every 256th value for broader coverage
        for i in (0u32..=65535).step_by(256) {
            let v = i as u16;
            assert_eq!(stratum_q1(v), v.count_ones() as u8);
        }
    }

    #[test]
    fn curvature_matches_hamming() {
        for i in 0u16..=1023 {
            let expected = (i ^ i.wrapping_add(1)).count_ones() as u8;
            assert_eq!(curvature_q1(i), expected);
        }
        assert_eq!(curvature_q1(0xFFFF), 16);
        assert_eq!(curvature_q1(0), 1);
    }

    #[test]
    fn curvature_sampled() {
        for i in (0u32..=65535).step_by(256) {
            let v = i as u16;
            let expected = (v ^ v.wrapping_add(1)).count_ones() as u8;
            assert_eq!(curvature_q1(v), expected);
        }
    }

    #[test]
    fn domain_matches_mod3() {
        for i in 0u16..=1000 {
            assert_eq!(domain_q1(i), i % 3);
        }
        assert_eq!(domain_q1(65535), 65535 % 3);
        assert_eq!(domain_q1(0), 0);
    }

    #[test]
    fn rank_matches_div3() {
        for i in 0u16..=1000 {
            assert_eq!(rank_q1(i), i / 3);
        }
        assert_eq!(rank_q1(65535), 65535 / 3);
    }

    #[test]
    fn torus_page() {
        assert_eq!(torus_page_q1(0), 0);
        assert_eq!(torus_page_q1(15), 0);
        assert_eq!(torus_page_q1(16), 1);
        assert_eq!(torus_page_q1(32), 2);
        assert_eq!(torus_page_q1(65535), 65535 / 16);
    }

    #[test]
    fn torus_offset() {
        assert_eq!(torus_offset_q1(0), 0);
        assert_eq!(torus_offset_q1(15), 15);
        assert_eq!(torus_offset_q1(16), 0);
        assert_eq!(torus_offset_q1(17), 1);
        assert_eq!(torus_offset_q1(65535), 65535 % 16);
    }

    #[test]
    fn orbit_class() {
        for i in 0u16..=255 {
            assert_eq!(orbit_class_q1(i), i / 16);
        }
        assert_eq!(orbit_class_q1(65535), 65535 / 16);
    }

    #[test]
    fn torus_page_offset_reconstruct() {
        // page * 16 + offset should reconstruct the value
        for i in (0u32..=65535).step_by(137) {
            let v = i as u16;
            assert_eq!(torus_page_q1(v) * 16 + torus_offset_q1(v), v);
        }
    }
}
