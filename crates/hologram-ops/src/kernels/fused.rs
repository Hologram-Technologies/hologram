//! Canonical fused ops — semantic identity, executable form, and CPU
//! reference kernels.
//!
//! Currently: `FusedSwiGlu` (`out = silu(gate) * up`). Other fused
//! variants (norm + activation, matmul + bias + activation, …) live in
//! the legacy `GraphOp::Fused*` path until they are migrated.

use super::binary::BinaryCall;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the fused SiLU-gated `swiglu` op.
///
/// Semantics: `out = silu(gate) * up`, where `gate` and `up` are two
/// separate input tensors of equal length. The legacy
/// `FloatOp::FusedSwiGLU` is also a binary op; the conformance suite
/// in `hologram-exec` pins this definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FusedSwiGlu;

impl Op for FusedSwiGlu {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "fused_swiglu"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Fused
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::FusedSwiGluBackward)
    }
}

/// Pre-resolved arguments for `FusedSwiGlu` backward.
///
/// `out = silu(gate) * up`. `d_gate += dC * up * silu'(gate)` and
/// `d_up += dC * silu(gate)`. Recomputes `silu(gate)` and the SiLU
/// derivative on the fly from the forward `gate` input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedSwiGluGradCall {
    /// Forward `gate` input.
    pub gate: SlotSpan,
    /// Forward `up` input.
    pub up: SlotSpan,
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `gate`.
    pub d_gate: SlotSpan,
    /// Gradient slot for `up`.
    pub d_up: SlotSpan,
}

/// Backward of `fused_swiglu`. Accumulates into `d_gate` and `d_up`.
pub fn fused_swiglu_grad(storage: &mut [f32], call: &FusedSwiGluGradCall) {
    let n = call.dc.len;
    debug_assert_eq!(call.gate.len, n);
    debug_assert_eq!(call.up.len, n);
    for i in 0..n {
        let gate = storage[call.gate.offset + i];
        let up = storage[call.up.offset + i];
        let dc = storage[call.dc.offset + i];
        let s = 1.0 / (1.0 + libm::expf(-gate));
        let silu = gate * s;
        let dsilu = s * (1.0 + gate * (1.0 - s));
        if call.d_gate.len > 0 {
            storage[call.d_gate.offset + i] += dc * up * dsilu;
        }
        if call.d_up.len > 0 {
            storage[call.d_up.offset + i] += dc * silu;
        }
    }
}

/// Forward: `out = silu(gate) * up`, elementwise.
///
/// `call.a` is `gate`, `call.b` is `up`. The two inputs and the output
/// must have equal length (planner guarantees this).
#[inline]
pub fn fused_swiglu(storage: &mut [f32], call: &BinaryCall) {
    let n = call.a.len;
    debug_assert_eq!(call.b.len, n);
    debug_assert_eq!(call.c.len, n);
    for i in 0..n {
        let gate = storage[call.a.offset + i];
        let up = storage[call.b.offset + i];
        let silu = gate / (1.0 + libm::expf(-gate));
        storage[call.c.offset + i] = silu * up;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SlotSpan;

    #[test]
    fn fused_swiglu_matches_silu_times_up() {
        let mut s = [0.0_f32, 1.0, -1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0];
        let call = BinaryCall {
            a: SlotSpan { offset: 0, len: 3 },
            b: SlotSpan { offset: 3, len: 3 },
            c: SlotSpan { offset: 6, len: 3 },
        };
        fused_swiglu(&mut s, &call);
        let silu = |x: f32| x / (1.0 + libm::expf(-x));
        let expected = [silu(0.0) * 2.0, silu(1.0) * 3.0, silu(-1.0) * 4.0];
        for (got, want) in s[6..9].iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5);
        }
    }

    #[test]
    fn fused_swiglu_grad_matches_finite_difference() {
        let gate = [0.5_f32, -0.7, 1.2];
        let up = [-0.3_f32, 0.8, 1.5];
        let dc = [1.0_f32, -0.5, 0.2];
        let n = 3;
        let mut s = vec![0.0_f32; n * 5];
        s[..n].copy_from_slice(&gate);
        s[n..2 * n].copy_from_slice(&up);
        s[2 * n..3 * n].copy_from_slice(&dc);
        let call = FusedSwiGluGradCall {
            gate: SlotSpan { offset: 0, len: n },
            up: SlotSpan { offset: n, len: n },
            dc: SlotSpan {
                offset: 2 * n,
                len: n,
            },
            d_gate: SlotSpan {
                offset: 3 * n,
                len: n,
            },
            d_up: SlotSpan {
                offset: 4 * n,
                len: n,
            },
        };
        fused_swiglu_grad(&mut s, &call);
        let silu = |x: f32| x / (1.0 + libm::expf(-x));
        let f = |g: &[f32], u: &[f32]| -> f32 { (0..n).map(|i| silu(g[i]) * u[i] * dc[i]).sum() };
        let h = 1e-3_f32;
        for i in 0..n {
            let mut gp = gate;
            gp[i] += h;
            let mut gm = gate;
            gm[i] -= h;
            let fd = (f(&gp, &up) - f(&gm, &up)) / (2.0 * h);
            assert!(
                (s[3 * n + i] - fd).abs() < 1e-2,
                "d_gate[{}]: got {}, fd {}",
                i,
                s[3 * n + i],
                fd
            );
            let mut ump = up;
            ump[i] += h;
            let mut umm = up;
            umm[i] -= h;
            let fd = (f(&gate, &ump) - f(&gate, &umm)) / (2.0 * h);
            assert!(
                (s[4 * n + i] - fd).abs() < 1e-2,
                "d_up[{}]: got {}, fd {}",
                i,
                s[4 * n + i],
                fd
            );
        }
    }
}
