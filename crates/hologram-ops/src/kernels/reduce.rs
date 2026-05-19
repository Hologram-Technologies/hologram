//! Canonical row-wise reductions (`ReduceSum`, `ReduceMean`,
//! `ReduceMax`, `ReduceMin`, `ReduceProd`) — semantic identity,
//! executable form, and CPU reference kernels.
//!
//! Each reduction collapses the last axis of length `size` to a
//! single output element per row. Output length is therefore
//! `input.len / size`. Like `softmax`, the canonical layer assumes
//! the reduced axis is the last one.

use crate::attrs::ReduceAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Pre-resolved arguments for a row-wise reduction kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (length = `input.len / size`).
    pub output: SlotSpan,
    /// Length of the reduced (last) axis.
    pub size: usize,
}

/// Pre-resolved arguments for `ReduceSum` / `ReduceMean` backward.
///
/// Broadcasts `dC` (one element per row) back across the reduced
/// axis: `dA[r, j] += dC[r]` for sum, `dA[r, j] += dC[r] / size`
/// for mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceGradCall {
    /// Upstream gradient (length = `rows`).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (length = `rows * size`).
    pub da: SlotSpan,
    /// Length of the reduced (last) axis in the forward input.
    pub size: usize,
}

/// Identity tag for `ReduceSum` / `ReduceMean` backward
/// (shared `ReduceGradCall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceGradKind {
    /// `dA[r, j] += dC[r]`.
    Sum,
    /// `dA[r, j] += dC[r] / size`.
    Mean,
}

/// Identity tag for the canonical reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    /// Sum of row elements.
    Sum,
    /// Arithmetic mean of row elements.
    Mean,
    /// Row maximum.
    Max,
    /// Row minimum.
    Min,
    /// Product of row elements.
    Prod,
}

// All reduce ops carry backward rules and are declared explicitly.
/// Marker struct for the canonical `reduce_sum` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceSum(pub ReduceAttrs);

impl Op for ReduceSum {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "reduce_sum"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ReduceSumBackward)
    }
}

/// Marker struct for the canonical `reduce_mean` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceMean(pub ReduceAttrs);

impl Op for ReduceMean {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "reduce_mean"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ReduceMeanBackward)
    }
}

// `ReduceMax` / `ReduceMin` carry backward rules — declared explicitly.
/// Marker struct for the canonical `reduce_max` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceMax(pub ReduceAttrs);

impl Op for ReduceMax {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "reduce_max"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ReduceMaxBackward)
    }
}

/// Marker struct for the canonical `reduce_min` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceMin(pub ReduceAttrs);

impl Op for ReduceMin {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "reduce_min"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ReduceMinBackward)
    }
}

/// Marker struct for the canonical `reduce_prod` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceProd(pub ReduceAttrs);

impl Op for ReduceProd {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "reduce_prod"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ReduceProdBackward)
    }
}

/// Pre-resolved arguments for `ReduceMax` / `ReduceMin` backward.
///
/// Routes `dC[r]` to whichever input row entry equals the row
/// extremum (recomputed from the forward input). Ties: first
/// occurrence wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceArgGradCall {
    /// Forward input (length = `da.len`).
    pub a: SlotSpan,
    /// Upstream gradient `dC` (length = `rows`).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (length = `rows * size`).
    pub da: SlotSpan,
    /// Length of the reduced (last) axis in the forward input.
    pub size: usize,
}

/// Identity tag for `ReduceMax` / `ReduceMin` backward (shared call).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceArgGradKind {
    /// Route `dC` to `argmax`.
    Max,
    /// Route `dC` to `argmin`.
    Min,
}

/// Pre-resolved arguments for `ReduceProd` backward.
///
/// `dA[r,j] += dC[r] * Π_{k≠j} A[r,k]`. The reference kernel handles
/// zeros explicitly: when the row has no zeros, it uses
/// `dC[r] * out[r] / A[r,j]`; when exactly one zero at position `k`,
/// only `dA[r,k]` gets `dC[r] * Π_{i≠k} A[r,i]`; with ≥2 zeros every
/// partial is 0 by definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceProdGradCall {
    /// Forward input (length = `da.len`).
    pub a: SlotSpan,
    /// Forward output (length = `rows`).
    pub out: SlotSpan,
    /// Upstream gradient `dC` (length = `rows`).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (length = `rows * size`).
    pub da: SlotSpan,
    /// Length of the reduced (last) axis in the forward input.
    pub size: usize,
}

/// Backward: zero-aware reduce-prod gradient. No-op if `da.len == 0`.
pub fn reduce_prod_grad(storage: &mut [f32], call: &ReduceProdGradCall) {
    if call.da.len == 0 {
        return;
    }
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.da.len % size, 0);
    debug_assert_eq!(call.a.len, call.da.len);
    let rows = call.da.len / size;
    debug_assert_eq!(call.dc.len, rows);
    debug_assert_eq!(call.out.len, rows);
    for r in 0..rows {
        let a_off = call.a.offset + r * size;
        let da_off = call.da.offset + r * size;
        let dc = storage[call.dc.offset + r];
        let mut zero_count = 0_usize;
        let mut zero_idx = 0_usize;
        let mut prod_nonzero = 1.0_f32;
        for j in 0..size {
            let v = storage[a_off + j];
            if v == 0.0 {
                zero_count += 1;
                zero_idx = j;
            } else {
                prod_nonzero *= v;
            }
        }
        match zero_count {
            0 => {
                let out_r = storage[call.out.offset + r];
                for j in 0..size {
                    storage[da_off + j] += dc * out_r / storage[a_off + j];
                }
            }
            1 => {
                storage[da_off + zero_idx] += dc * prod_nonzero;
            }
            _ => {} // all partials are 0
        }
    }
}

/// Backward: route `dC` to the row argmax / argmin position. No-op if
/// `da.len == 0`.
pub fn dispatch_arg_grad(storage: &mut [f32], call: &ReduceArgGradCall, kind: ReduceArgGradKind) {
    if call.da.len == 0 {
        return;
    }
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.da.len % size, 0);
    debug_assert_eq!(call.a.len, call.da.len);
    let rows = call.da.len / size;
    debug_assert_eq!(call.dc.len, rows);
    for r in 0..rows {
        let a_off = call.a.offset + r * size;
        let mut best = storage[a_off];
        let mut idx = 0_usize;
        for j in 1..size {
            let v = storage[a_off + j];
            let pick = match kind {
                ReduceArgGradKind::Max => v > best,
                ReduceArgGradKind::Min => v < best,
            };
            if pick {
                best = v;
                idx = j;
            }
        }
        let da_off = call.da.offset + r * size;
        storage[da_off + idx] += storage[call.dc.offset + r];
    }
}

/// Apply the row-wise reduction identified by `kind`.
#[inline]
pub fn dispatch(storage: &mut [f32], call: &ReduceCall, kind: ReduceKind) {
    match kind {
        ReduceKind::Sum => apply(storage, call, 0.0, |acc, x| acc + x, identity),
        ReduceKind::Mean => {
            let n = call.size as f32;
            apply(storage, call, 0.0, |acc, x| acc + x, move |sum| sum / n)
        }
        ReduceKind::Max => apply(storage, call, f32::NEG_INFINITY, f32::max, identity),
        ReduceKind::Min => apply(storage, call, f32::INFINITY, f32::min, identity),
        ReduceKind::Prod => apply(storage, call, 1.0, |acc, x| acc * x, identity),
    }
}

#[inline]
fn identity(x: f32) -> f32 {
    x
}

#[inline]
fn apply<R, F>(storage: &mut [f32], call: &ReduceCall, init: f32, fold: R, finish: F)
where
    R: Fn(f32, f32) -> f32,
    F: Fn(f32) -> f32,
{
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.input.len % size, 0);
    let rows = call.input.len / size;
    debug_assert_eq!(call.output.len, rows);
    for r in 0..rows {
        let off = call.input.offset + r * size;
        let mut acc = init;
        for i in 0..size {
            acc = fold(acc, storage[off + i]);
        }
        storage[call.output.offset + r] = finish(acc);
    }
}

/// Backward: broadcast `dC` back across the reduced axis. No-op if
/// `da.len == 0`.
#[inline]
pub fn dispatch_grad(storage: &mut [f32], call: &ReduceGradCall, kind: ReduceGradKind) {
    if call.da.len == 0 {
        return;
    }
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.da.len % size, 0);
    let rows = call.da.len / size;
    debug_assert_eq!(call.dc.len, rows);
    let scale = match kind {
        ReduceGradKind::Sum => 1.0_f32,
        ReduceGradKind::Mean => 1.0 / size as f32,
    };
    for r in 0..rows {
        let dc = storage[call.dc.offset + r] * scale;
        let off = call.da.offset + r * size;
        for j in 0..size {
            storage[off + j] += dc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kind: ReduceKind, input: &[f32], size: usize) -> Vec<f32> {
        let n = input.len();
        let rows = n / size;
        let mut s = vec![0.0_f32; n + rows];
        s[..n].copy_from_slice(input);
        let call = ReduceCall {
            input: SlotSpan { offset: 0, len: n },
            output: SlotSpan {
                offset: n,
                len: rows,
            },
            size,
        };
        dispatch(&mut s, &call, kind);
        s[n..].to_vec()
    }

    #[test]
    fn reduce_sum_collapses_each_row() {
        assert_eq!(run(ReduceKind::Sum, &[1.0, 2.0, 3.0, 4.0], 2), &[3.0, 7.0]);
    }

    #[test]
    fn reduce_mean_divides_by_size() {
        assert_eq!(run(ReduceKind::Mean, &[2.0, 4.0, 6.0, 8.0], 2), &[3.0, 7.0]);
    }

    #[test]
    fn reduce_max_min_select_extrema() {
        assert_eq!(run(ReduceKind::Max, &[1.0, 5.0, 3.0, 2.0], 2), &[5.0, 3.0]);
        assert_eq!(run(ReduceKind::Min, &[1.0, 5.0, 3.0, 2.0], 2), &[1.0, 2.0]);
    }

    #[test]
    fn reduce_prod_grad_no_zeros_uses_out_over_a() {
        let n = 4;
        let rows = 2;
        let size = 2;
        let mut s = vec![0.0_f32; n + rows + rows + n];
        s[..n].copy_from_slice(&[2.0, 3.0, 4.0, 5.0]); // a
        s[n] = 6.0; // out row 0
        s[n + 1] = 20.0; // out row 1
        s[n + rows] = 1.0; // dC row 0
        s[n + rows + 1] = 1.0; // dC row 1
        let call = ReduceProdGradCall {
            a: SlotSpan { offset: 0, len: n },
            out: SlotSpan {
                offset: n,
                len: rows,
            },
            dc: SlotSpan {
                offset: n + rows,
                len: rows,
            },
            da: SlotSpan {
                offset: n + 2 * rows,
                len: n,
            },
            size,
        };
        reduce_prod_grad(&mut s, &call);
        let da = &s[n + 2 * rows..];
        // dA[r,j] = dC * out / A[r,j]. Row 0: 6/2=3, 6/3=2. Row 1: 20/4=5, 20/5=4.
        assert_eq!(da, &[3.0, 2.0, 5.0, 4.0]);
    }

    #[test]
    fn reduce_prod_grad_one_zero_routes_to_zero_index() {
        let n = 3;
        let rows = 1;
        let size = 3;
        let mut s = vec![0.0_f32; n + rows + rows + n];
        s[..n].copy_from_slice(&[2.0, 0.0, 4.0]);
        s[n] = 0.0; // out
        s[n + rows] = 1.0; // dC
        let call = ReduceProdGradCall {
            a: SlotSpan { offset: 0, len: n },
            out: SlotSpan {
                offset: n,
                len: rows,
            },
            dc: SlotSpan {
                offset: n + rows,
                len: rows,
            },
            da: SlotSpan {
                offset: n + 2 * rows,
                len: n,
            },
            size,
        };
        reduce_prod_grad(&mut s, &call);
        let da = &s[n + 2 * rows..];
        // Only dA[0,1] = 2*4 = 8 (product of non-zeros). Others 0.
        assert_eq!(da, &[0.0, 8.0, 0.0]);
    }

    #[test]
    fn reduce_arg_grad_routes_to_argmax_first_occurrence() {
        let n = 6;
        let rows = 2;
        let size = 3;
        let mut s = vec![0.0_f32; n + rows + n];
        s[..n].copy_from_slice(&[1.0, 5.0, 3.0, 4.0, 4.0, 2.0]); // ties in row 1 → first wins
        s[n] = 10.0;
        s[n + 1] = 7.0;
        let call = ReduceArgGradCall {
            a: SlotSpan { offset: 0, len: n },
            dc: SlotSpan {
                offset: n,
                len: rows,
            },
            da: SlotSpan {
                offset: n + rows,
                len: n,
            },
            size,
        };
        dispatch_arg_grad(&mut s, &call, ReduceArgGradKind::Max);
        // Row 0 argmax is index 1; row 1 argmax is index 0 (first 4).
        let da = &s[n + rows..];
        assert_eq!(da, &[0.0, 10.0, 0.0, 7.0, 0.0, 0.0]);
    }

    #[test]
    fn reduce_arg_grad_routes_to_argmin() {
        let n = 4;
        let rows = 2;
        let size = 2;
        let mut s = vec![0.0_f32; n + rows + n];
        s[..n].copy_from_slice(&[1.0, 5.0, 3.0, 2.0]);
        s[n] = 4.0;
        s[n + 1] = 3.0;
        let call = ReduceArgGradCall {
            a: SlotSpan { offset: 0, len: n },
            dc: SlotSpan {
                offset: n,
                len: rows,
            },
            da: SlotSpan {
                offset: n + rows,
                len: n,
            },
            size,
        };
        dispatch_arg_grad(&mut s, &call, ReduceArgGradKind::Min);
        let da = &s[n + rows..];
        assert_eq!(da, &[4.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn reduce_prod_multiplies() {
        assert_eq!(
            run(ReduceKind::Prod, &[2.0, 3.0, 4.0, 5.0], 2),
            &[6.0, 20.0]
        );
    }
}
