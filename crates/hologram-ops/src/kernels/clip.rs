//! Canonical `Clip` op — semantic identity, executable form, and CPU
//! reference kernel.
//!
//! Semantics: `out[i] = clamp(input[i], min, max)`. The bounds live on
//! `ClipAttrs` as `f32::to_bits()` so the attrs struct stays
//! `Copy + Eq + Hash`.

use crate::attrs::ClipAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for the `clip` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (same length as input).
    pub output: SlotSpan,
    /// Minimum bound, encoded as `f32::to_bits()`.
    pub min_bits: u32,
    /// Maximum bound, encoded as `f32::to_bits()`.
    pub max_bits: u32,
}

/// Marker struct for the canonical `clip` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Clip(pub ClipAttrs);

impl Op for Clip {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "clip"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
}

/// Forward: `out[i] = clamp(input[i], min, max)`.
#[inline]
pub fn clip(storage: &mut [f32], call: &ClipCall) {
    let n = call.input.len;
    debug_assert_eq!(call.output.len, n);
    let min = f32::from_bits(call.min_bits);
    let max = f32::from_bits(call.max_bits);
    for i in 0..n {
        let v = storage[call.input.offset + i];
        storage[call.output.offset + i] = v.clamp(min, max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_clamps_to_bounds() {
        let mut s = [
            -2.0_f32, -1.0, 0.0, 1.0, 2.0, // input
            0.0, 0.0, 0.0, 0.0, 0.0, // output
        ];
        let call = ClipCall {
            input: SlotSpan { offset: 0, len: 5 },
            output: SlotSpan { offset: 5, len: 5 },
            min_bits: (-1.0_f32).to_bits(),
            max_bits: 1.0_f32.to_bits(),
        };
        clip(&mut s, &call);
        assert_eq!(&s[5..10], &[-1.0, -1.0, 0.0, 1.0, 1.0]);
    }
}
