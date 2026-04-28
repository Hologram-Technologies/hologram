//! Canonical `Add` op — semantic identity, executable form, and CPU
//! reference kernel.

use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical elementwise `add` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Add;

impl Op for Add {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "add"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::AddBackward)
    }
}

/// Pre-resolved arguments for forward elementwise add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddCall {
    /// Operand A.
    pub a: SlotSpan,
    /// Operand B.
    pub b: SlotSpan,
    /// Output C.
    pub c: SlotSpan,
}

/// Pre-resolved arguments for backward elementwise add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddGradCall {
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated).
    pub da: SlotSpan,
    /// Gradient slot for `B` (accumulated).
    pub db: SlotSpan,
}

/// Forward: `c = a + b`, elementwise.
///
/// Spans are guaranteed by the planner to be equal-length and disjoint.
#[inline]
pub fn add(storage: &mut [f32], call: &AddCall) {
    let n = call.a.len;
    debug_assert_eq!(call.b.len, n);
    debug_assert_eq!(call.c.len, n);
    for i in 0..n {
        let av = storage[call.a.offset + i];
        let bv = storage[call.b.offset + i];
        storage[call.c.offset + i] = av + bv;
    }
}

/// Backward: `da += dc`, `db += dc`.
///
/// Empty-length grad slots are no-ops, which lets the planner skip
/// emitting per-input branches.
#[inline]
pub fn add_grad(storage: &mut [f32], call: &AddGradCall) {
    let n = call.dc.len;
    accumulate_into(storage, call.da, call.dc, n);
    accumulate_into(storage, call.db, call.dc, n);
}

#[inline]
fn accumulate_into(storage: &mut [f32], dst: SlotSpan, src: SlotSpan, n: usize) {
    if dst.len == 0 {
        return;
    }
    debug_assert_eq!(dst.len, n);
    for i in 0..n {
        let s = storage[src.offset + i];
        storage[dst.offset + i] += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_trait_add_declares_backward() {
        assert_eq!(Add.arity(), 2);
        assert_eq!(Add.name(), "add");
        assert_eq!(Add.category(), OpCategory::Elementwise);
        assert_eq!(Add.backward(), Some(BackwardRule::AddBackward));
        assert!(Add.differentiable());
    }

    #[test]
    fn add_writes_elementwise_sum() {
        let mut s = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0];
        let call = AddCall {
            a: SlotSpan { offset: 0, len: 3 },
            b: SlotSpan { offset: 3, len: 3 },
            c: SlotSpan { offset: 6, len: 3 },
        };
        add(&mut s, &call);
        assert_eq!(&s[6..9], &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_grad_accumulates_into_both_inputs() {
        let mut s = vec![0.0_f32; 9];
        s[6] = 1.0;
        s[7] = 2.0;
        s[8] = 3.0;
        let call = AddGradCall {
            dc: SlotSpan { offset: 6, len: 3 },
            da: SlotSpan { offset: 0, len: 3 },
            db: SlotSpan { offset: 3, len: 3 },
        };
        add_grad(&mut s, &call);
        assert_eq!(&s[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&s[3..6], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_grad_is_no_op_when_grad_slot_empty() {
        let mut s = vec![0.0_f32; 6];
        s[3] = 7.0;
        let call = AddGradCall {
            dc: SlotSpan { offset: 3, len: 3 },
            da: SlotSpan { offset: 0, len: 0 },
            db: SlotSpan { offset: 0, len: 3 },
        };
        add_grad(&mut s, &call);
        assert_eq!(&s[0..3], &[7.0, 0.0, 0.0]);
    }
}
