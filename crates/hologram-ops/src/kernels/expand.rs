//! Canonical `Expand` op (broadcast to target shape) — semantic
//! identity, executable form, and CPU reference kernel.
//!
//! Reference: each output position is computed by mapping back to the
//! corresponding input index, where any input dim that's `1` (or
//! shorter than the target dim) wraps via modulo. Standard NumPy
//! broadcast semantics.

use crate::attrs::ExpandAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `expand`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Source shape (first `ndim` entries valid).
    pub input_dims: [u32; 8],
    /// Target shape (first `ndim` entries valid).
    pub target_dims: [u32; 8],
    /// Number of valid dimensions (must match between source and target).
    pub ndim: u8,
}

/// Marker struct for the canonical `expand` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Expand(pub ExpandAttrs);

impl Op for Expand {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "expand"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Shape
    }
}

/// Forward: broadcast `input` to `target_dims`.
pub fn expand(storage: &mut [f32], call: &ExpandCall) {
    let nd = call.ndim as usize;
    debug_assert!(nd <= 8);
    let in_dims: [usize; 8] = std::array::from_fn(|i| call.input_dims[i] as usize);
    let out_dims: [usize; 8] = std::array::from_fn(|i| call.target_dims[i] as usize);

    // Strides for input (broadcast: dims of length 1 contribute stride 0).
    let mut in_strides = [0_usize; 8];
    let mut s = 1_usize;
    for d in (0..nd).rev() {
        in_strides[d] = if in_dims[d] == 1 { 0 } else { s };
        s *= in_dims[d];
    }
    // Strides for output (row-major).
    let mut out_strides = [1_usize; 8];
    for d in (0..nd.saturating_sub(1)).rev() {
        out_strides[d] = out_strides[d + 1] * out_dims[d + 1];
    }
    let total = (0..nd).map(|i| out_dims[i]).product::<usize>().max(1);

    let mut idx = [0_usize; 8];
    for _ in 0..total {
        let mut in_off = call.input.offset;
        let mut out_off = call.output.offset;
        for d in 0..nd {
            in_off += idx[d] * in_strides[d];
            out_off += idx[d] * out_strides[d];
        }
        storage[out_off] = storage[in_off];
        let mut d = nd;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            if idx[d] < out_dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_broadcasts_singleton_axis() {
        // Input shape [1, 3] = [1, 2, 3] → expand to [4, 3].
        let mut s = vec![0.0_f32; 3 + 12];
        s[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        let call = ExpandCall {
            input: SlotSpan { offset: 0, len: 3 },
            output: SlotSpan { offset: 3, len: 12 },
            input_dims: [1, 3, 0, 0, 0, 0, 0, 0],
            target_dims: [4, 3, 0, 0, 0, 0, 0, 0],
            ndim: 2,
        };
        expand(&mut s, &call);
        for r in 0..4 {
            assert_eq!(&s[3 + r * 3..3 + r * 3 + 3], &[1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn expand_identity_dims_is_copy() {
        let mut s = vec![0.0_f32; 4 + 4];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = ExpandCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 4 },
            input_dims: [2, 2, 0, 0, 0, 0, 0, 0],
            target_dims: [2, 2, 0, 0, 0, 0, 0, 0],
            ndim: 2,
        };
        expand(&mut s, &call);
        assert_eq!(&s[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }
}
