//! Canonical `Where` ternary select op — semantic identity,
//! executable form, and CPU reference kernel.
//!
//! Semantics: `out[i] = condition[i] ? x[i] : y[i]`. f32 truthiness:
//! `0.0` is false, anything else is true.

use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for the `where` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhereCall {
    /// Condition input (f32 truthiness).
    pub condition: SlotSpan,
    /// `x` value taken when the condition is true.
    pub x: SlotSpan,
    /// `y` value taken when the condition is false.
    pub y: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
}

/// Marker struct for the canonical `where` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Where;

impl Op for Where {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "where"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
}

/// Forward: `out[i] = condition[i] != 0 ? x[i] : y[i]`.
#[inline]
pub fn r#where(storage: &mut [f32], call: &WhereCall) {
    let n = call.condition.len;
    debug_assert_eq!(call.x.len, n);
    debug_assert_eq!(call.y.len, n);
    debug_assert_eq!(call.output.len, n);
    for i in 0..n {
        let c = storage[call.condition.offset + i];
        let v = if c != 0.0 {
            storage[call.x.offset + i]
        } else {
            storage[call.y.offset + i]
        };
        storage[call.output.offset + i] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn where_picks_x_when_condition_truthy_else_y() {
        let mut s = [
            1.0_f32, 0.0, 1.0, 0.0, // condition
            10.0, 20.0, 30.0, 40.0, // x
            -1.0, -2.0, -3.0, -4.0, // y
            0.0, 0.0, 0.0, 0.0, // output
        ];
        let call = WhereCall {
            condition: SlotSpan { offset: 0, len: 4 },
            x: SlotSpan { offset: 4, len: 4 },
            y: SlotSpan { offset: 8, len: 4 },
            output: SlotSpan { offset: 12, len: 4 },
        };
        r#where(&mut s, &call);
        assert_eq!(&s[12..16], &[10.0, -2.0, 30.0, -4.0]);
    }
}
