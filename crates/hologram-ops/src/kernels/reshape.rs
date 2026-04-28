//! Canonical `Reshape` op — semantic identity, executable form, and
//! CPU reference kernel (in-storage copy).
//!
//! `Reshape` is metadata-only at the semantic level, but the chain may
//! still place input and output on different `SlotSpan`s. The reference
//! kernel is therefore an in-storage copy; a future planner pass can
//! collapse adjacent reshapes by aliasing spans, eliminating the copy.

use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Marker struct for the canonical metadata-only `reshape` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reshape;

impl Op for Reshape {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "reshape"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Layout
    }
}

/// Pre-resolved arguments for the reshape kernel (pure copy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReshapeCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (same length as input).
    pub output: SlotSpan,
}

/// Forward: `out = in`, byte-for-byte.
#[inline]
pub fn reshape(storage: &mut [f32], call: &ReshapeCall) {
    let n = call.input.len;
    debug_assert_eq!(call.output.len, n);
    if call.input.offset == call.output.offset {
        return;
    }
    for i in 0..n {
        storage[call.output.offset + i] = storage[call.input.offset + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_trait_reshape_is_layout_only() {
        assert!(Reshape.layout_only());
        assert_eq!(Reshape.category(), OpCategory::Layout);
    }

    #[test]
    fn reshape_copies_input_to_output() {
        let mut s = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let call = ReshapeCall {
            input: SlotSpan { offset: 0, len: 3 },
            output: SlotSpan { offset: 3, len: 3 },
        };
        reshape(&mut s, &call);
        assert_eq!(&s[3..6], &[1.0, 2.0, 3.0]);
    }
}
