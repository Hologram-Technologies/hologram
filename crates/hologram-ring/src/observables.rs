//! Observable algebra: stratum, curvature, rank, domain.
//!
//! Each function compiles to 1-3 ALU instructions at any quantum level.

use crate::word::RingWord;

/// Stratum (Hamming weight / popcount) of a ring element.
#[inline]
#[must_use]
pub fn stratum<W: RingWord>(x: W) -> u32 {
    x.count_ones()
}

/// Curvature: Hamming distance between x and x+1.
#[inline]
#[must_use]
pub fn curvature<W: RingWord>(x: W) -> u32 {
    (x ^ x.wrapping_add(W::ONE)).count_ones()
}

/// Rank: number of trailing zeros (2-adic valuation).
#[inline]
#[must_use]
pub fn rank<W: RingWord>(x: W) -> u32 {
    x.trailing_zeros()
}

/// Domain: number of leading zeros.
#[inline]
#[must_use]
pub fn domain<W: RingWord>(x: W) -> u32 {
    x.leading_zeros()
}
