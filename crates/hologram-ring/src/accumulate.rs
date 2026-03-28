//! Accumulation: the fundamental fused multiply-add pattern.
//!
//! `acc = add(acc, mul(a, b))` — two ALU instructions at any quantum level.

use crate::word::RingWord;

/// Ring-native accumulation: `acc + a * b` (wrapping).
///
/// This is the fundamental multiply-accumulate pattern. Every matmul,
/// dot product, convolution, and attention computation decomposes into
/// iterated application of this function.
#[inline]
#[must_use]
pub fn accumulate<W: RingWord>(acc: W, a: W, b: W) -> W {
    acc.wrapping_add(a.wrapping_mul(b))
}
