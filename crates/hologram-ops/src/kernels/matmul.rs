//! Canonical `MatMul` op — semantic identity, executable form, and CPU
//! reference kernels (forward + 2 backwards).
//!
//! Reference (correctness-only) implementation. Tiling, vectorisation,
//! and BLAS dispatch are concerns for backend-specialised executors
//! that consume the same `KernelCall` form.

use crate::attrs::MatMulAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical `matmul` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMul(pub MatMulAttrs);

impl Op for MatMul {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "matmul"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::LinearAlgebra
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::MatMulBackward)
    }
}

/// Pre-resolved arguments for forward matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulCall {
    /// Left operand `A` (`[m, k]`).
    pub a: SlotSpan,
    /// Right operand `B` (`[k, n]`).
    pub b: SlotSpan,
    /// Output `C` (`[m, n]`).
    pub c: SlotSpan,
    /// Rows of `A` and `C`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `B` and `C`.
    pub n: usize,
}

/// Pre-resolved arguments for `dA += dC @ Bᵀ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulGradACall {
    /// Upstream gradient `dC` (`[m, n]`).
    pub dc: SlotSpan,
    /// Forward `B` (`[k, n]`).
    pub b: SlotSpan,
    /// Gradient slot for `A` (`[m, k]`, accumulated).
    pub da: SlotSpan,
    /// Rows of `A`/`C`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `B`/`C`.
    pub n: usize,
}

/// Pre-resolved arguments for `dB += Aᵀ @ dC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulGradBCall {
    /// Forward `A` (`[m, k]`).
    pub a: SlotSpan,
    /// Upstream gradient `dC` (`[m, n]`).
    pub dc: SlotSpan,
    /// Gradient slot for `B` (`[k, n]`, accumulated).
    pub db: SlotSpan,
    /// Rows of `A`/`C`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `B`/`C`.
    pub n: usize,
}

/// Forward: `C = A @ B` (`A:[m,k]`, `B:[k,n]`, `C:[m,n]`, row-major).
#[inline]
pub fn matmul(storage: &mut [f32], call: &MatMulCall) {
    debug_assert_eq!(call.a.len, call.m * call.k);
    debug_assert_eq!(call.b.len, call.k * call.n);
    debug_assert_eq!(call.c.len, call.m * call.n);
    for i in 0..call.m {
        for j in 0..call.n {
            storage[call.c.offset + i * call.n + j] = dot_a_b(storage, call, i, j);
        }
    }
}

#[inline]
fn dot_a_b(storage: &[f32], call: &MatMulCall, i: usize, j: usize) -> f32 {
    let mut acc = 0.0_f32;
    for p in 0..call.k {
        let av = storage[call.a.offset + i * call.k + p];
        let bv = storage[call.b.offset + p * call.n + j];
        acc += av * bv;
    }
    acc
}

/// Backward w.r.t. A: `dA += dC @ Bᵀ` (`dA:[m,k]`).
#[inline]
pub fn matmul_grad_a(storage: &mut [f32], call: &MatMulGradACall) {
    if call.da.len == 0 {
        return;
    }
    debug_assert_eq!(call.da.len, call.m * call.k);
    for i in 0..call.m {
        for p in 0..call.k {
            let acc = dot_dc_bt(storage, call, i, p);
            storage[call.da.offset + i * call.k + p] += acc;
        }
    }
}

#[inline]
fn dot_dc_bt(storage: &[f32], call: &MatMulGradACall, i: usize, p: usize) -> f32 {
    let mut acc = 0.0_f32;
    for j in 0..call.n {
        let dc = storage[call.dc.offset + i * call.n + j];
        let bv = storage[call.b.offset + p * call.n + j];
        acc += dc * bv;
    }
    acc
}

/// Backward w.r.t. B: `dB += Aᵀ @ dC` (`dB:[k,n]`).
#[inline]
pub fn matmul_grad_b(storage: &mut [f32], call: &MatMulGradBCall) {
    if call.db.len == 0 {
        return;
    }
    debug_assert_eq!(call.db.len, call.k * call.n);
    for p in 0..call.k {
        for j in 0..call.n {
            let acc = dot_at_dc(storage, call, p, j);
            storage[call.db.offset + p * call.n + j] += acc;
        }
    }
}

#[inline]
fn dot_at_dc(storage: &[f32], call: &MatMulGradBCall, p: usize, j: usize) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..call.m {
        let av = storage[call.a.offset + i * call.k + p];
        let dc = storage[call.dc.offset + i * call.n + j];
        acc += av * dc;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    #[test]
    fn op_trait_matmul_declares_backward() {
        let mm = MatMul(MatMulAttrs { m: 2, k: 3, n: 4 });
        assert_eq!(mm.arity(), 2);
        assert_eq!(mm.name(), "matmul");
        assert_eq!(mm.category(), OpCategory::LinearAlgebra);
        assert_eq!(mm.backward(), Some(BackwardRule::MatMulBackward));
    }

    #[test]
    fn op_trait_signature_is_consistent_with_category() {
        let sig = MatMul(MatMulAttrs { m: 1, k: 1, n: 1 }).signature();
        assert_eq!(sig.arity, 2);
        assert_eq!(sig.outputs, 1);
        assert_eq!(sig.category, OpCategory::LinearAlgebra);
        assert!(sig.differentiable);
        assert!(!sig.layout_only);
    }

    #[test]
    fn matmul_2x3_times_3x2_matches_reference() {
        let mut s = vec![0.0_f32; 6 + 6 + 4];
        s[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s[6..12].copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let call = MatMulCall {
            a: span(0, 6),
            b: span(6, 6),
            c: span(12, 4),
            m: 2,
            k: 3,
            n: 2,
        };
        matmul(&mut s, &call);
        assert_eq!(&s[12..16], &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn matmul_grad_a_accumulates_dc_b_transpose() {
        let mut s = vec![0.0_f32; 6 + 4 + 6];
        s[..6].copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        s[6..10].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = MatMulGradACall {
            dc: span(6, 4),
            b: span(0, 6),
            da: span(10, 6),
            m: 2,
            k: 3,
            n: 2,
        };
        matmul_grad_a(&mut s, &call);
        assert_eq!(&s[10..16], &[17.0, 23.0, 29.0, 39.0, 53.0, 67.0]);
    }

    #[test]
    fn matmul_grad_b_accumulates_a_transpose_dc() {
        let mut s = vec![0.0_f32; 6 + 4 + 6];
        s[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s[6..10].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = MatMulGradBCall {
            a: span(0, 6),
            dc: span(6, 4),
            db: span(10, 6),
            m: 2,
            k: 3,
            n: 2,
        };
        matmul_grad_b(&mut s, &call);
        assert_eq!(&s[10..16], &[13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
    }
}
