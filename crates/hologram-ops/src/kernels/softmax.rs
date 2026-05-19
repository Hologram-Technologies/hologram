//! Canonical `Softmax` and `LogSoftmax` ops — semantic identity,
//! executable form, and CPU reference kernels.
//!
//! Numerically stable: subtract row max, exponentiate, normalise. Both
//! variants share the same row scan so they live together.

use crate::attrs::SoftmaxAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical `softmax` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Softmax(pub SoftmaxAttrs);

impl Op for Softmax {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "softmax"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::SoftmaxBackward)
    }
}

/// Marker struct for the canonical `log_softmax` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogSoftmax(pub SoftmaxAttrs);

impl Op for LogSoftmax {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "log_softmax"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Reduction
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::LogSoftmaxBackward)
    }
}

/// Pre-resolved arguments for the softmax-family kernels.
///
/// `size` is the length of the normalised axis (the last axis); the
/// kernel iterates over `input.len / size` rows of length `size` each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (same length as input).
    pub output: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: usize,
}

/// Pre-resolved arguments for the softmax-family backward kernels.
///
/// `output` holds the forward result (softmax probabilities or
/// log-softmax log-probabilities); the kernel reads it row-wise to
/// build `dA`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxGradCall {
    /// Forward output span (length = `da.len`).
    pub output: SlotSpan,
    /// Upstream gradient `dC` (length = `da.len`).
    pub dc: SlotSpan,
    /// Gradient slot for `A` (length = `output.len`).
    pub da: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: usize,
}

/// Identity tag selecting `Softmax` vs `LogSoftmax` backward
/// (shared `SoftmaxGradCall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoftmaxGradKind {
    /// `dA[r,j] = out[r,j] * (dC[r,j] - Σ_k dC[r,k] * out[r,k])`.
    Softmax,
    /// `dA[r,j] = dC[r,j] - exp(out[r,j]) * Σ_k dC[r,k]`.
    LogSoftmax,
}

/// Backward: produce `dA` for either softmax variant. Accumulates into
/// `da` (no zero-init). No-op if `da.len == 0`.
pub fn dispatch_grad(storage: &mut [f32], call: &SoftmaxGradCall, kind: SoftmaxGradKind) {
    if call.da.len == 0 {
        return;
    }
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.da.len % size, 0);
    debug_assert_eq!(call.output.len, call.da.len);
    debug_assert_eq!(call.dc.len, call.da.len);
    let rows = call.da.len / size;
    for r in 0..rows {
        let row = r * size;
        let out_off = call.output.offset + row;
        let dc_off = call.dc.offset + row;
        let da_off = call.da.offset + row;
        match kind {
            SoftmaxGradKind::Softmax => {
                let mut dot = 0.0_f32;
                for j in 0..size {
                    dot += storage[dc_off + j] * storage[out_off + j];
                }
                for j in 0..size {
                    let o = storage[out_off + j];
                    storage[da_off + j] += o * (storage[dc_off + j] - dot);
                }
            }
            SoftmaxGradKind::LogSoftmax => {
                let mut dc_sum = 0.0_f32;
                for j in 0..size {
                    dc_sum += storage[dc_off + j];
                }
                for j in 0..size {
                    let p = libm::expf(storage[out_off + j]);
                    storage[da_off + j] += storage[dc_off + j] - p * dc_sum;
                }
            }
        }
    }
}

/// Forward: `out = softmax(input)` along the last axis.
#[inline]
pub fn softmax(storage: &mut [f32], call: &SoftmaxCall) {
    apply_rowwise(storage, call, false);
}

/// Forward: `out = log_softmax(input)` along the last axis.
#[inline]
pub fn log_softmax(storage: &mut [f32], call: &SoftmaxCall) {
    apply_rowwise(storage, call, true);
}

#[inline]
fn apply_rowwise(storage: &mut [f32], call: &SoftmaxCall, log_form: bool) {
    let size = call.size;
    debug_assert!(size > 0);
    debug_assert_eq!(call.input.len % size, 0);
    debug_assert_eq!(call.output.len, call.input.len);
    let rows = call.input.len / size;
    for r in 0..rows {
        let row_in_off = call.input.offset + r * size;
        let row_out_off = call.output.offset + r * size;
        let max = row_max(storage, row_in_off, size);
        let log_z = write_exp_minus_max(storage, row_in_off, row_out_off, size, max);
        finalise_row(storage, row_out_off, size, log_z, log_form);
    }
}

#[inline]
fn row_max(storage: &[f32], offset: usize, size: usize) -> f32 {
    let mut m = storage[offset];
    for i in 1..size {
        let v = storage[offset + i];
        if v > m {
            m = v;
        }
    }
    m
}

#[inline]
fn write_exp_minus_max(
    storage: &mut [f32],
    in_off: usize,
    out_off: usize,
    size: usize,
    max: f32,
) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..size {
        let e = libm::expf(storage[in_off + i] - max);
        storage[out_off + i] = e;
        sum += e;
    }
    libm::logf(sum)
}

#[inline]
fn finalise_row(storage: &mut [f32], out_off: usize, size: usize, log_z: f32, log_form: bool) {
    if log_form {
        for i in 0..size {
            let lv = libm::logf(storage[out_off + i]);
            storage[out_off + i] = lv - log_z;
        }
    } else {
        let z = libm::expf(log_z);
        for i in 0..size {
            storage[out_off + i] /= z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_softmax(input: &[f32], size: usize) -> Vec<f32> {
        let n = input.len();
        let mut s = vec![0.0_f32; n * 2];
        s[..n].copy_from_slice(input);
        let call = SoftmaxCall {
            input: SlotSpan { offset: 0, len: n },
            output: SlotSpan { offset: n, len: n },
            size,
        };
        softmax(&mut s, &call);
        s[n..].to_vec()
    }

    fn run_log_softmax(input: &[f32], size: usize) -> Vec<f32> {
        let n = input.len();
        let mut s = vec![0.0_f32; n * 2];
        s[..n].copy_from_slice(input);
        let call = SoftmaxCall {
            input: SlotSpan { offset: 0, len: n },
            output: SlotSpan { offset: n, len: n },
            size,
        };
        log_softmax(&mut s, &call);
        s[n..].to_vec()
    }

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{} != {}", a, b);
    }

    #[test]
    fn softmax_normalises_each_row_to_unit_sum() {
        let out = run_softmax(&[1.0, 2.0, 3.0, 4.0], 2);
        approx_eq(out[0] + out[1], 1.0);
        approx_eq(out[2] + out[3], 1.0);
    }

    #[test]
    fn softmax_uniform_input_is_uniform_output() {
        let out = run_softmax(&[5.0, 5.0, 5.0, 5.0], 4);
        for v in &out {
            approx_eq(*v, 0.25);
        }
    }

    #[test]
    fn softmax_grad_matches_jacobian_sum() {
        // For y = softmax(x), dx_i = y_i (dy_i - sum_j(dy_j y_j)).
        // Pick concrete values and verify against the closed form.
        let n = 3;
        let mut s = vec![0.0_f32; 3 * n];
        s[..n].copy_from_slice(&[0.2_f32, 0.5, 0.3]); // already softmax-like
        s[n..2 * n].copy_from_slice(&[1.0, -1.0, 2.0]); // dC
        let call = SoftmaxGradCall {
            output: SlotSpan { offset: 0, len: n },
            dc: SlotSpan { offset: n, len: n },
            da: SlotSpan {
                offset: 2 * n,
                len: n,
            },
            size: n,
        };
        dispatch_grad(&mut s, &call, SoftmaxGradKind::Softmax);
        let dot: f32 = (0..n).map(|i| s[i] * s[n + i]).sum();
        for i in 0..n {
            let expected = s[i] * (s[n + i] - dot);
            approx_eq(s[2 * n + i], expected);
        }
    }

    #[test]
    fn log_softmax_grad_subtracts_exp_out_dc_sum() {
        let n = 3;
        let mut s = vec![0.0_f32; 3 * n];
        s[..n].copy_from_slice(&[-1.0_f32, -0.5, -2.0]); // log-softmax-like
        s[n..2 * n].copy_from_slice(&[0.1, -0.2, 0.4]); // dC
        let call = SoftmaxGradCall {
            output: SlotSpan { offset: 0, len: n },
            dc: SlotSpan { offset: n, len: n },
            da: SlotSpan {
                offset: 2 * n,
                len: n,
            },
            size: n,
        };
        dispatch_grad(&mut s, &call, SoftmaxGradKind::LogSoftmax);
        let dc_sum: f32 = (0..n).map(|i| s[n + i]).sum();
        for i in 0..n {
            let expected = s[n + i] - libm::expf(s[i]) * dc_sum;
            approx_eq(s[2 * n + i], expected);
        }
    }

    #[test]
    fn log_softmax_rows_sum_to_log_of_one_after_exp() {
        let out = run_log_softmax(&[1.0, 2.0, 3.0], 3);
        let sum_exp: f32 = out.iter().map(|x| libm::expf(*x)).sum();
        approx_eq(sum_exp, 1.0);
    }
}
