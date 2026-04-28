//! Canonical `Sub`, `Mul`, `Div` ops — semantic identity, executable
//! form, and CPU reference kernels.
//!
//! Backward rules for these are not yet wired through the planner.

use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical elementwise `sub` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sub;

impl Op for Sub {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "sub"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::SubBackward)
    }
}

/// Marker struct for the canonical elementwise `mul` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mul;

impl Op for Mul {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "mul"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::MulBackward)
    }
}

/// Marker struct for the canonical elementwise `div` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Div;

impl Op for Div {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "div"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::DivBackward)
    }
}

/// Internal helper: declare a binary elementwise op marker struct
/// + `Op` impl with `Elementwise` category.
macro_rules! elementwise_binary_op {
    ($struct_name:ident, $name:literal) => {
        #[doc = concat!("Marker struct for the canonical `", $name, "` op.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $struct_name;

        impl Op for $struct_name {
            #[inline]
            fn arity(self) -> u8 {
                2
            }
            #[inline]
            fn name(self) -> &'static str {
                $name
            }
            #[inline]
            fn category(self) -> OpCategory {
                OpCategory::Elementwise
            }
        }
    };
}

elementwise_binary_op!(Mod, "mod");

// `Pow` carries a backward rule, so it's hand-impl'd outside the macro.
/// Marker struct for the canonical elementwise `pow` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pow;

impl Op for Pow {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "pow"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::PowBackward)
    }
}

// `Min` / `Max` carry a backward rule, so they're declared with an
// explicit impl rather than via `elementwise_binary_op!`.
/// Marker struct for the canonical elementwise `min` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Min;

impl Op for Min {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "min"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::MinBackward)
    }
}

/// Marker struct for the canonical elementwise `max` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Max;

impl Op for Max {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "max"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::MaxBackward)
    }
}
elementwise_binary_op!(Equal, "equal");
elementwise_binary_op!(Less, "less");
elementwise_binary_op!(LessOrEqual, "less_or_equal");
elementwise_binary_op!(Greater, "greater");
elementwise_binary_op!(GreaterOrEqual, "greater_or_equal");
elementwise_binary_op!(And, "and");
elementwise_binary_op!(Or, "or");
elementwise_binary_op!(Xor, "xor");

/// Forward: `c = a^b`, elementwise (uses `libm::powf`).
#[inline]
pub fn pow(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, libm::powf);
}

/// Forward: `c = a mod b`, elementwise (IEEE remainder via `libm::fmodf`).
#[inline]
pub fn modulo(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, libm::fmodf);
}

/// Forward: `c = min(a, b)`, elementwise (`-0.0` handling matches IEEE).
#[inline]
pub fn min(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, f32::min);
}

/// Forward: `c = max(a, b)`, elementwise.
#[inline]
pub fn max(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, f32::max);
}

/// Forward: `c = (a == b) ? 1.0 : 0.0`, elementwise.
#[inline]
pub fn equal(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| if a == b { 1.0 } else { 0.0 });
}

/// Forward: `c = (a < b) ? 1.0 : 0.0`.
#[inline]
pub fn less(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| if a < b { 1.0 } else { 0.0 });
}

/// Forward: `c = (a <= b) ? 1.0 : 0.0`.
#[inline]
pub fn less_or_equal(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| if a <= b { 1.0 } else { 0.0 });
}

/// Forward: `c = (a > b) ? 1.0 : 0.0`.
#[inline]
pub fn greater(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| if a > b { 1.0 } else { 0.0 });
}

/// Forward: `c = (a >= b) ? 1.0 : 0.0`.
#[inline]
pub fn greater_or_equal(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| if a >= b { 1.0 } else { 0.0 });
}

/// Forward: logical AND on f32 truthiness (`0.0` is false, anything else true).
#[inline]
pub fn and(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(
        storage,
        call,
        |a, b| {
            if a != 0.0 && b != 0.0 {
                1.0
            } else {
                0.0
            }
        },
    );
}

/// Forward: logical OR on f32 truthiness.
#[inline]
pub fn or(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(
        storage,
        call,
        |a, b| {
            if a != 0.0 || b != 0.0 {
                1.0
            } else {
                0.0
            }
        },
    );
}

/// Backward: `dA += dC`, `dB += -dC`.
#[inline]
pub fn sub_grad(storage: &mut [f32], call: &SubGradCall) {
    let n = call.dc.len;
    if call.da.len > 0 {
        debug_assert_eq!(call.da.len, n);
        for i in 0..n {
            storage[call.da.offset + i] += storage[call.dc.offset + i];
        }
    }
    if call.db.len > 0 {
        debug_assert_eq!(call.db.len, n);
        for i in 0..n {
            storage[call.db.offset + i] -= storage[call.dc.offset + i];
        }
    }
}

/// Backward: `dA += dC * B`, `dB += dC * A`.
#[inline]
pub fn mul_grad(storage: &mut [f32], call: &MulGradCall) {
    let n = call.dc.len;
    debug_assert_eq!(call.a.len, n);
    debug_assert_eq!(call.b.len, n);
    if call.da.len > 0 {
        debug_assert_eq!(call.da.len, n);
        for i in 0..n {
            storage[call.da.offset + i] += storage[call.dc.offset + i] * storage[call.b.offset + i];
        }
    }
    if call.db.len > 0 {
        debug_assert_eq!(call.db.len, n);
        for i in 0..n {
            storage[call.db.offset + i] += storage[call.dc.offset + i] * storage[call.a.offset + i];
        }
    }
}

/// Backward: `dA += dC / B`, `dB += -dC * A / B²`.
#[inline]
pub fn div_grad(storage: &mut [f32], call: &DivGradCall) {
    let n = call.dc.len;
    debug_assert_eq!(call.a.len, n);
    debug_assert_eq!(call.b.len, n);
    if call.da.len > 0 {
        debug_assert_eq!(call.da.len, n);
        for i in 0..n {
            storage[call.da.offset + i] += storage[call.dc.offset + i] / storage[call.b.offset + i];
        }
    }
    if call.db.len > 0 {
        debug_assert_eq!(call.db.len, n);
        for i in 0..n {
            let b_val = storage[call.b.offset + i];
            storage[call.db.offset + i] -=
                storage[call.dc.offset + i] * storage[call.a.offset + i] / (b_val * b_val);
        }
    }
}

/// Backward: `dA += dC·B·out/A`, `dB += dC·out·ln(A)`. No-op for
/// any empty grad span.
#[inline]
pub fn pow_grad(storage: &mut [f32], call: &PowGradCall) {
    let n = call.dc.len;
    debug_assert_eq!(call.a.len, n);
    debug_assert_eq!(call.b.len, n);
    debug_assert_eq!(call.out.len, n);
    if call.da.len > 0 {
        debug_assert_eq!(call.da.len, n);
        for i in 0..n {
            let av = storage[call.a.offset + i];
            let bv = storage[call.b.offset + i];
            let out = storage[call.out.offset + i];
            // d/da a^b = b · a^(b-1) = b · (a^b)/a, valid for a ≠ 0.
            let da = if av != 0.0 { bv * out / av } else { 0.0 };
            storage[call.da.offset + i] += storage[call.dc.offset + i] * da;
        }
    }
    if call.db.len > 0 {
        debug_assert_eq!(call.db.len, n);
        for i in 0..n {
            let av = storage[call.a.offset + i];
            let out = storage[call.out.offset + i];
            // d/db a^b = a^b · ln(a), valid for a > 0.
            let db = if av > 0.0 { out * libm::logf(av) } else { 0.0 };
            storage[call.db.offset + i] += storage[call.dc.offset + i] * db;
        }
    }
}

/// Backward: `Min`/`Max` route the gradient to whichever input
/// elementwise-"wins" (ties go to A). `dA[i] += dC[i]` where A wins,
/// `dB[i] += dC[i]` where B wins. Empty grad spans are no-ops.
#[inline]
pub fn min_max_grad(storage: &mut [f32], call: &MinMaxGradCall, kind: MinMaxGradKind) {
    let n = call.dc.len;
    debug_assert_eq!(call.a.len, n);
    debug_assert_eq!(call.b.len, n);
    let want_da = call.da.len > 0;
    let want_db = call.db.len > 0;
    if want_da {
        debug_assert_eq!(call.da.len, n);
    }
    if want_db {
        debug_assert_eq!(call.db.len, n);
    }
    for i in 0..n {
        let av = storage[call.a.offset + i];
        let bv = storage[call.b.offset + i];
        let dc = storage[call.dc.offset + i];
        let a_wins = match kind {
            MinMaxGradKind::Min => av <= bv,
            MinMaxGradKind::Max => av >= bv,
        };
        if a_wins {
            if want_da {
                storage[call.da.offset + i] += dc;
            }
        } else if want_db {
            storage[call.db.offset + i] += dc;
        }
    }
}

/// Forward: logical XOR on f32 truthiness.
#[inline]
pub fn xor(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| {
        if (a != 0.0) ^ (b != 0.0) {
            1.0
        } else {
            0.0
        }
    });
}

/// Pre-resolved arguments for a forward elementwise binary kernel.
///
/// Shared by `Sub`, `Mul`, `Div`, and `FusedSwiGlu`. The shape contract
/// is equal-length spans; only the per-element operation differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryCall {
    /// Operand A.
    pub a: SlotSpan,
    /// Operand B.
    pub b: SlotSpan,
    /// Output C.
    pub c: SlotSpan,
}

/// Pre-resolved arguments for backward `Sub`.
///
/// `dA += dC`, `dB += -dC`. Empty grad spans are treated as no-ops so
/// the planner can skip per-input emission for non-`requires_grad`
/// tensors (same convention as `AddGradCall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubGradCall {
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated).
    pub da: SlotSpan,
    /// Gradient slot for `B` (accumulated, sign-flipped).
    pub db: SlotSpan,
}

/// Pre-resolved arguments for backward `Mul`.
///
/// `dA += dC * B`, `dB += dC * A`. Both forward inputs are needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulGradCall {
    /// Forward `A`.
    pub a: SlotSpan,
    /// Forward `B`.
    pub b: SlotSpan,
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated).
    pub da: SlotSpan,
    /// Gradient slot for `B` (accumulated).
    pub db: SlotSpan,
}

/// Pre-resolved arguments for backward `Div`.
///
/// `dA += dC / B`, `dB += -dC * A / B²`. Both forward inputs needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivGradCall {
    /// Forward `A`.
    pub a: SlotSpan,
    /// Forward `B`.
    pub b: SlotSpan,
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated).
    pub da: SlotSpan,
    /// Gradient slot for `B` (accumulated).
    pub db: SlotSpan,
}

/// Pre-resolved arguments for `Min` / `Max` backward.
///
/// Routes the gradient to whichever input "won" elementwise: for
/// `Min`, `dA[i] += dC[i]` where `A[i] <= B[i]` (and `dB[i] += dC[i]`
/// otherwise); `Max` flips the comparison. Equal-valued positions
/// route to A by convention (matches the typical
/// PyTorch / TensorFlow tie-break).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinMaxGradCall {
    /// Forward `A`.
    pub a: SlotSpan,
    /// Forward `B`.
    pub b: SlotSpan,
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated).
    pub da: SlotSpan,
    /// Gradient slot for `B` (accumulated).
    pub db: SlotSpan,
}

/// Identity tag for `Min` vs `Max` backward (shared `MinMaxGradCall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MinMaxGradKind {
    /// Backward of `Min`.
    Min,
    /// Backward of `Max`.
    Max,
}

/// Pre-resolved arguments for `Pow` backward.
///
/// `dA += dC · B · A^(B-1) = dC · B · out / A` (when `A ≠ 0`),
/// `dB += dC · out · ln(A)` (when `A > 0`).
///
/// At `A = 0` or `A < 0` the partials may be undefined or NaN; the
/// reference kernel mirrors `libm::powf` semantics — non-finite
/// outputs propagate. Real models avoid those regions by
/// construction (positive bases, integer exponents handled
/// elsewhere).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowGradCall {
    /// Forward `A`.
    pub a: SlotSpan,
    /// Forward `B`.
    pub b: SlotSpan,
    /// Forward output `out = A^B`.
    pub out: SlotSpan,
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A`.
    pub da: SlotSpan,
    /// Gradient slot for `B`.
    pub db: SlotSpan,
}

/// Forward: `c = a - b`, elementwise.
#[inline]
pub fn sub(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| a - b);
}

/// Forward: `c = a * b`, elementwise.
#[inline]
pub fn mul(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| a * b);
}

/// Forward: `c = a / b`, elementwise.
#[inline]
pub fn div(storage: &mut [f32], call: &BinaryCall) {
    apply_binary(storage, call, |a, b| a / b);
}

#[inline]
fn apply_binary<F: Fn(f32, f32) -> f32>(storage: &mut [f32], call: &BinaryCall, f: F) {
    let n = call.a.len;
    debug_assert_eq!(call.b.len, n);
    debug_assert_eq!(call.c.len, n);
    for i in 0..n {
        let av = storage[call.a.offset + i];
        let bv = storage[call.b.offset + i];
        storage[call.c.offset + i] = f(av, bv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    #[test]
    fn sub_writes_elementwise_difference() {
        let mut s = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let call = BinaryCall {
            a: span(0, 3),
            b: span(3, 3),
            c: span(6, 3),
        };
        sub(&mut s, &call);
        assert_eq!(&s[6..9], &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn mul_writes_elementwise_product() {
        let mut s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0];
        let call = BinaryCall {
            a: span(0, 3),
            b: span(3, 3),
            c: span(6, 3),
        };
        mul(&mut s, &call);
        assert_eq!(&s[6..9], &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn div_writes_elementwise_quotient() {
        let mut s = [10.0, 20.0, 30.0, 2.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        let call = BinaryCall {
            a: span(0, 3),
            b: span(3, 3),
            c: span(6, 3),
        };
        div(&mut s, &call);
        assert_eq!(&s[6..9], &[5.0, 5.0, 6.0]);
    }
}
