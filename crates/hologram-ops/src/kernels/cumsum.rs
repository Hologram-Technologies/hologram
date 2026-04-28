//! Canonical `CumSum` op — cumulative sum along the last axis.
//!
//! Reference behaviour: treats the input as rows of `size` elements
//! and writes the running sum into the output. The `axis` attribute is
//! pinned to "last axis" for the canonical reference; non-last-axis
//! cumsum can be expressed by a `Transpose` → `CumSum` → `Transpose`
//! rewrite at the planner level.

use crate::attrs::CumSumAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `cumsum`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CumSumCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (same length as input).
    pub output: SlotSpan,
    /// Length of the cumulative axis (the last axis in the canonical layer).
    pub size: usize,
}

/// Marker struct for the canonical `cum_sum` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CumSum(pub CumSumAttrs);

impl Op for CumSum {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "cum_sum"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
}

/// Forward: `out[r, j] = sum(input[r, 0..=j])`.
#[inline]
pub fn cumsum(storage: &mut [f32], call: &CumSumCall) {
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.input.len % size, 0);
    debug_assert_eq!(call.output.len, call.input.len);
    let rows = call.input.len / size;
    for r in 0..rows {
        let in_off = call.input.offset + r * size;
        let out_off = call.output.offset + r * size;
        let mut acc = 0.0_f32;
        for j in 0..size {
            acc += storage[in_off + j];
            storage[out_off + j] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumsum_writes_running_total_per_row() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let call = CumSumCall {
            input: SlotSpan { offset: 0, len: 6 },
            output: SlotSpan { offset: 6, len: 6 },
            size: 3,
        };
        cumsum(&mut s, &call);
        assert_eq!(&s[6..12], &[1.0, 3.0, 6.0, 10.0, 30.0, 60.0]);
    }
}
