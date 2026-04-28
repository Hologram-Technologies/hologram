//! Canonical shape-rewriting ops (`Transpose`, `Slice`, `Concat`) —
//! semantic identity, executable form, and CPU reference kernels.
//!
//! These kernels operate on contiguous row-major storage. Shape-aware
//! logic is intentionally simple (last-axis only for `slice` and
//! `concat`, up to 4-D for `transpose`) — production-grade variants
//! are a backend concern.

use crate::attrs::{ConcatAttrs, SliceAttrs, TransposeAttrs};
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical `transpose` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Transpose(pub TransposeAttrs);

impl Op for Transpose {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "transpose"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Layout
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::TransposeBackward)
    }
}

/// Marker struct for the canonical `slice` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slice(pub SliceAttrs);

impl Op for Slice {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "slice"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Shape
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::SliceBackward)
    }
}

/// Marker struct for the canonical `concat` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Concat(pub ConcatAttrs);

impl Op for Concat {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "concat"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Shape
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ConcatBackward)
    }
}

/// Pre-resolved arguments for a physical transpose (up to 4-D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransposeCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Source dimensions (first `ndim` entries are valid).
    pub input_dims: [u32; 4],
    /// Permutation (first `ndim` entries are valid).
    pub perm: [u8; 4],
    /// Number of valid dimensions.
    pub ndim: u8,
}

/// Pre-resolved arguments for a last-axis contiguous slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Length of the sliced axis (last axis) in the input.
    pub axis_size: u32,
    /// Inclusive start index along the axis.
    pub start: u32,
    /// Exclusive end index along the axis.
    pub end: u32,
}

/// Pre-resolved arguments for a last-axis concat of two operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcatCall {
    /// Operand A span.
    pub a: SlotSpan,
    /// Operand B span.
    pub b: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Length of the concat axis in A.
    pub size_a: u32,
    /// Length of the concat axis in B.
    pub size_b: u32,
}

/// Forward: physical n-D transpose (n ≤ 4).
pub fn transpose(storage: &mut [f32], call: &TransposeCall) {
    let nd = call.ndim as usize;
    debug_assert!(nd <= 4);
    let in_dims: [usize; 4] = [
        call.input_dims[0] as usize,
        call.input_dims[1] as usize,
        call.input_dims[2] as usize,
        call.input_dims[3] as usize,
    ];
    let perm: [usize; 4] = [
        call.perm[0] as usize,
        call.perm[1] as usize,
        call.perm[2] as usize,
        call.perm[3] as usize,
    ];
    let mut in_strides = [1_usize; 4];
    for i in (0..nd.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
    }
    let total = (0..nd).map(|i| in_dims[i]).product::<usize>().max(1);
    let mut out_dims = [1_usize; 4];
    for i in 0..nd {
        out_dims[i] = in_dims[perm[i]];
    }
    let mut out_strides = [1_usize; 4];
    for i in (0..nd.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }
    let mut idx = [0_usize; 4];
    for _ in 0..total {
        let mut in_off = call.input.offset;
        let mut out_off = call.output.offset;
        for d in 0..nd {
            in_off += idx[d] * in_strides[perm[d]];
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

/// Forward: last-axis contiguous slice. Output rows are length `end-start`.
#[inline]
pub fn slice(storage: &mut [f32], call: &SliceCall) {
    let axis = call.axis_size as usize;
    debug_assert!(axis > 0);
    let start = call.start as usize;
    let end = call.end as usize;
    debug_assert!(end >= start && end <= axis);
    let out_row = end - start;
    debug_assert_eq!(call.input.len % axis, 0);
    debug_assert_eq!(call.output.len % out_row.max(1), 0);
    let rows = call.input.len / axis;
    for r in 0..rows {
        let src = call.input.offset + r * axis + start;
        let dst = call.output.offset + r * out_row;
        for i in 0..out_row {
            storage[dst + i] = storage[src + i];
        }
    }
}

/// Forward: last-axis concat of two operands.
#[inline]
pub fn concat(storage: &mut [f32], call: &ConcatCall) {
    let sa = call.size_a as usize;
    let sb = call.size_b as usize;
    debug_assert!(sa > 0 && sb > 0);
    debug_assert_eq!(call.a.len % sa, 0);
    debug_assert_eq!(call.b.len % sb, 0);
    let rows_a = call.a.len / sa;
    let rows_b = call.b.len / sb;
    debug_assert_eq!(rows_a, rows_b);
    let row_out = sa + sb;
    debug_assert_eq!(call.output.len, rows_a * row_out);
    for r in 0..rows_a {
        let dst = call.output.offset + r * row_out;
        for i in 0..sa {
            storage[dst + i] = storage[call.a.offset + r * sa + i];
        }
        for i in 0..sb {
            storage[dst + sa + i] = storage[call.b.offset + r * sb + i];
        }
    }
}

/// Pre-resolved arguments for `Transpose` backward — apply the
/// inverse of the forward permutation to `dC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransposeGradCall {
    /// Upstream gradient `dC` (shape = forward output).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (shape = forward input).
    pub da: SlotSpan,
    /// Source dimensions of forward input (first `ndim` valid).
    pub input_dims: [u32; 4],
    /// Inverse permutation (first `ndim` valid).
    pub inv_perm: [u8; 4],
    /// Number of valid dimensions.
    pub ndim: u8,
}

/// Pre-resolved arguments for `Slice` backward — scatter `dC` into
/// the slice region of `dA`. Outside-slice positions in `dA` are
/// untouched (already accumulated by other paths or zero-initialised).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceGradCall {
    /// Upstream gradient `dC` (rows of length `end - start`).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (rows of length `axis_size`).
    pub da: SlotSpan,
    /// Length of the sliced axis in the forward input.
    pub axis_size: u32,
    /// Inclusive start index along the axis.
    pub start: u32,
    /// Exclusive end index along the axis.
    pub end: u32,
}

/// Pre-resolved arguments for `Concat` backward — split `dC` along
/// the last axis into the two input grads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcatGradCall {
    /// Upstream gradient `dC` (rows of length `size_a + size_b`).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (rows of length `size_a`).
    pub da: SlotSpan,
    /// Gradient slot for `B` (rows of length `size_b`).
    pub db: SlotSpan,
    /// Length of the concat axis in `A`.
    pub size_a: u32,
    /// Length of the concat axis in `B`.
    pub size_b: u32,
}

/// Backward: apply the inverse permutation to `dC`. No-op if `da` is
/// empty.
pub fn transpose_grad(storage: &mut [f32], call: &TransposeGradCall) {
    if call.da.len == 0 {
        return;
    }
    let nd = call.ndim as usize;
    debug_assert!(nd <= 4);
    let in_dims: [usize; 4] = [
        call.input_dims[0] as usize,
        call.input_dims[1] as usize,
        call.input_dims[2] as usize,
        call.input_dims[3] as usize,
    ];
    let inv: [usize; 4] = [
        call.inv_perm[0] as usize,
        call.inv_perm[1] as usize,
        call.inv_perm[2] as usize,
        call.inv_perm[3] as usize,
    ];
    let mut in_strides = [1_usize; 4];
    for i in (0..nd.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
    }
    let mut out_dims = [1_usize; 4];
    for i in 0..nd {
        // Forward output dim d came from input dim perm[d] = inv⁻¹[d].
        // Equivalently, input dim k goes to output dim inv[k] under
        // the inverse permutation we're applying.
        out_dims[inv[i]] = in_dims[i];
    }
    let mut out_strides = [1_usize; 4];
    for i in (0..nd.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }
    let total = (0..nd).map(|i| in_dims[i]).product::<usize>().max(1);
    // Walk dA in row-major over the input shape; the dC source
    // multi-index is `inv` applied to the input index.
    let mut idx = [0_usize; 4];
    for _ in 0..total {
        let mut da_off = call.da.offset;
        let mut dc_off = call.dc.offset;
        for d in 0..nd {
            da_off += idx[d] * in_strides[d];
            dc_off += idx[d] * out_strides[inv[d]];
        }
        storage[da_off] += storage[dc_off];
        let mut d = nd;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            if idx[d] < in_dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// Backward: scatter `dC` into `dA[r, start..end]`.
#[inline]
pub fn slice_grad(storage: &mut [f32], call: &SliceGradCall) {
    if call.da.len == 0 {
        return;
    }
    let axis = call.axis_size as usize;
    let start = call.start as usize;
    let end = call.end as usize;
    let out_row = end - start;
    let rows = call.da.len / axis;
    debug_assert_eq!(call.dc.len, rows * out_row);
    for r in 0..rows {
        let src = call.dc.offset + r * out_row;
        let dst = call.da.offset + r * axis + start;
        for j in 0..out_row {
            storage[dst + j] += storage[src + j];
        }
    }
}

/// Backward: split `dC` rows into `dA` (first `size_a` cols) and
/// `dB` (next `size_b` cols). Empty grad spans skip their side.
#[inline]
pub fn concat_grad(storage: &mut [f32], call: &ConcatGradCall) {
    let sa = call.size_a as usize;
    let sb = call.size_b as usize;
    let row_in = sa + sb;
    debug_assert!(sa > 0 && sb > 0);
    debug_assert_eq!(call.dc.len % row_in, 0);
    let rows = call.dc.len / row_in;
    if call.da.len > 0 {
        debug_assert_eq!(call.da.len, rows * sa);
        for r in 0..rows {
            let src = call.dc.offset + r * row_in;
            let dst = call.da.offset + r * sa;
            for j in 0..sa {
                storage[dst + j] += storage[src + j];
            }
        }
    }
    if call.db.len > 0 {
        debug_assert_eq!(call.db.len, rows * sb);
        for r in 0..rows {
            let src = call.dc.offset + r * row_in + sa;
            let dst = call.db.offset + r * sb;
            for j in 0..sb {
                storage[dst + j] += storage[src + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    #[test]
    fn transpose_2d_swaps_rows_and_columns() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let call = TransposeCall {
            input: span(0, 6),
            output: span(6, 6),
            input_dims: [2, 3, 0, 0],
            perm: [1, 0, 0, 0],
            ndim: 2,
        };
        transpose(&mut s, &call);
        assert_eq!(&s[6..12], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_4d_with_identity_perm_is_copy() {
        let mut s = vec![0.0_f32; 48];
        for (i, slot) in s.iter_mut().take(24).enumerate() {
            *slot = i as f32;
        }
        let call = TransposeCall {
            input: span(0, 24),
            output: span(24, 24),
            input_dims: [2, 2, 2, 3],
            perm: [0, 1, 2, 3],
            ndim: 4,
        };
        transpose(&mut s, &call);
        for i in 0..24 {
            assert_eq!(s[24 + i], i as f32);
        }
    }

    #[test]
    fn slice_takes_inner_window_of_each_row() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let call = SliceCall {
            input: span(0, 8),
            output: span(8, 4),
            axis_size: 4,
            start: 1,
            end: 3,
        };
        slice(&mut s, &call);
        assert_eq!(&s[8..12], &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn concat_joins_rows_along_last_axis() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 4.0, 9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let call = ConcatCall {
            a: span(0, 4),
            b: span(4, 2),
            output: span(6, 6),
            size_a: 2,
            size_b: 1,
        };
        concat(&mut s, &call);
        assert_eq!(&s[6..12], &[1.0, 2.0, 9.0, 3.0, 4.0, 8.0]);
    }
}
