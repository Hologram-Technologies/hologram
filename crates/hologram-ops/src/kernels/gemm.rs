//! Canonical `Gemm` op (`Y = alpha * (A @ B) + beta * C`) — semantic
//! identity, executable form, and CPU reference kernel.
//!
//! `Gemm` is the parameterised matmul: optional transpose flags on
//! both inputs, scalar `alpha`/`beta`, and an optional `C` bias term
//! that's broadcast-added. Distinct from `MatMul` (which is the
//! simple A @ B case used in transformer weight projections).
//! Quantised B (`quant_b` on `FloatOp::Gemm`) is *not* in canonical:
//! quantisation is an execution adapter (ADR-048).

use crate::attrs::GemmAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `gemm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmCall {
    /// Operand A. Shape is `[m, k]` if `!trans_a`, else `[k, m]`.
    pub a: SlotSpan,
    /// Operand B. Shape is `[k, n]` if `!trans_b`, else `[n, k]`.
    pub b: SlotSpan,
    /// Bias `C`. Length must be `m * n` (per-element) or `n`
    /// (per-column broadcast). Empty span means no bias.
    pub c: SlotSpan,
    /// Output `Y` (`[m, n]`).
    pub y: SlotSpan,
    /// Rows of `Y`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `Y`.
    pub n: usize,
    /// Whether `A` is provided transposed.
    pub trans_a: bool,
    /// Whether `B` is provided transposed.
    pub trans_b: bool,
    /// Scalar applied to `A @ B`, encoded as `f32::to_bits()`.
    pub alpha_bits: u32,
    /// Scalar applied to `C` before adding, encoded as `f32::to_bits()`.
    pub beta_bits: u32,
}

/// Marker struct for the canonical `gemm` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gemm(pub GemmAttrs);

impl Op for Gemm {
    #[inline]
    fn arity(self) -> u8 {
        // 2 if no bias, 3 with bias. The chain captures whichever
        // shape the user wired; the kernel handles both via the
        // empty-span sentinel on `c`.
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "gemm"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::LinearAlgebra
    }
}

/// Forward: `Y = alpha * op(A) @ op(B) + beta * C`.
pub fn gemm(storage: &mut [f32], call: &GemmCall) {
    let alpha = f32::from_bits(call.alpha_bits);
    let beta = f32::from_bits(call.beta_bits);
    let bias_present = call.c.len > 0;

    for i in 0..call.m {
        for j in 0..call.n {
            let mut acc = 0.0_f32;
            for p in 0..call.k {
                let av = if call.trans_a {
                    storage[call.a.offset + p * call.m + i]
                } else {
                    storage[call.a.offset + i * call.k + p]
                };
                let bv = if call.trans_b {
                    storage[call.b.offset + j * call.k + p]
                } else {
                    storage[call.b.offset + p * call.n + j]
                };
                acc += av * bv;
            }
            let mut y = alpha * acc;
            if bias_present {
                let c_val = if call.c.len == call.n {
                    // Per-column broadcast.
                    storage[call.c.offset + j]
                } else {
                    // Per-element bias `[m, n]`.
                    storage[call.c.offset + i * call.n + j]
                };
                y += beta * c_val;
            }
            storage[call.y.offset + i * call.n + j] = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_identity_alpha_no_bias_matches_matmul() {
        // A 2×3, B 3×2 — same vectors as the MatMul reference test.
        let mut s = vec![0.0_f32; 6 + 6 + 4];
        s[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s[6..12].copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let call = GemmCall {
            a: SlotSpan { offset: 0, len: 6 },
            b: SlotSpan { offset: 6, len: 6 },
            c: SlotSpan::empty(0),
            y: SlotSpan { offset: 12, len: 4 },
            m: 2,
            k: 3,
            n: 2,
            trans_a: false,
            trans_b: false,
            alpha_bits: 1.0_f32.to_bits(),
            beta_bits: 0_u32,
        };
        gemm(&mut s, &call);
        assert_eq!(&s[12..16], &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn gemm_with_bias_and_alpha_beta_scales() {
        // A 1×2 = [1, 2], B 2×1 = [3, 4]; A@B = 11.
        // alpha=2, beta=3, C=[10] → y = 2*11 + 3*10 = 52.
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 0.0];
        let call = GemmCall {
            a: SlotSpan { offset: 0, len: 2 },
            b: SlotSpan { offset: 2, len: 2 },
            c: SlotSpan { offset: 4, len: 1 },
            y: SlotSpan { offset: 5, len: 1 },
            m: 1,
            k: 2,
            n: 1,
            trans_a: false,
            trans_b: false,
            alpha_bits: 2.0_f32.to_bits(),
            beta_bits: 3.0_f32.to_bits(),
        };
        gemm(&mut s, &call);
        assert_eq!(s[5], 52.0);
    }

    #[test]
    fn gemm_with_trans_b_uses_transposed_layout() {
        // A 1×2 = [1, 2], B^T 1×2 = [3, 4] (so B is 2×1 = [3; 4]); A@B = 1*3 + 2*4 = 11.
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 0.0];
        let call = GemmCall {
            a: SlotSpan { offset: 0, len: 2 },
            b: SlotSpan { offset: 2, len: 2 },
            c: SlotSpan::empty(0),
            y: SlotSpan { offset: 4, len: 1 },
            m: 1,
            k: 2,
            n: 1,
            trans_a: false,
            trans_b: true,
            alpha_bits: 1.0_f32.to_bits(),
            beta_bits: 0_u32,
        };
        gemm(&mut s, &call);
        assert_eq!(s[4], 11.0);
    }
}
