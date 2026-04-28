//! Canonical unary ops — 18 marker structs sharing `UnaryCall` and a
//! `UnaryKind`-dispatched kernel.
//!
//! All 18 ops live in this file because they share `UnaryCall`, a
//! single dispatch function, and identical `Op` trait facts modulo
//! their name. Backward rules are not yet wired through the planner;
//! adding one means a `KernelCall::UnaryGrad` variant plus a
//! `BackwardRule` arm.

use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical elementwise `neg` op.
///
/// Defined outside `elementwise_unary_op!` because it carries a
/// non-default `backward` impl (`NegBackward`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Neg;

impl Op for Neg {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "neg"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::NegBackward)
    }
}

/// Internal helper: declare a differentiable unary op marker struct
/// + `Op` impl with a custom `backward` rule.
macro_rules! differentiable_unary_op {
    ($struct_name:ident, $name:literal, $rule:ident) => {
        #[doc = concat!("Marker struct for the canonical `", $name, "` op.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $struct_name;

        impl Op for $struct_name {
            #[inline]
            fn arity(self) -> u8 {
                1
            }
            #[inline]
            fn name(self) -> &'static str {
                $name
            }
            #[inline]
            fn category(self) -> OpCategory {
                OpCategory::Elementwise
            }
            #[inline]
            fn backward(self) -> Option<BackwardRule> {
                Some(BackwardRule::$rule)
            }
        }
    };
}

differentiable_unary_op!(Relu, "relu", ReluBackward);
differentiable_unary_op!(Sigmoid, "sigmoid", SigmoidBackward);
differentiable_unary_op!(Tanh, "tanh", TanhBackward);
differentiable_unary_op!(Exp, "exp", ExpBackward);
differentiable_unary_op!(Log, "log", LogBackward);

macro_rules! elementwise_unary_op {
    ($struct_name:ident, $name:literal) => {
        #[doc = concat!("Marker struct for the canonical `", $name, "` op.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $struct_name;

        impl Op for $struct_name {
            #[inline]
            fn arity(self) -> u8 {
                1
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

differentiable_unary_op!(Gelu, "gelu", GeluBackward);
differentiable_unary_op!(Silu, "silu", SiluBackward);
differentiable_unary_op!(Sqrt, "sqrt", SqrtBackward);
differentiable_unary_op!(Abs, "abs", AbsBackward);
differentiable_unary_op!(Reciprocal, "reciprocal", ReciprocalBackward);
elementwise_unary_op!(Cos, "cos");
elementwise_unary_op!(Sin, "sin");
elementwise_unary_op!(Sign, "sign");
elementwise_unary_op!(Floor, "floor");
elementwise_unary_op!(Ceil, "ceil");
elementwise_unary_op!(Round, "round");
elementwise_unary_op!(Erf, "erf");
elementwise_unary_op!(Not, "not");
elementwise_unary_op!(IsNaN, "is_nan");

/// Pre-resolved arguments for a forward elementwise unary kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (same length as input).
    pub output: SlotSpan,
}

/// Pre-resolved arguments for backward `Neg`.
///
/// `dA += -dC`. Empty `da` span is a no-op (same convention as
/// `AddGradCall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegGradCall {
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated, sign-flipped).
    pub da: SlotSpan,
}

/// Pre-resolved arguments for elementwise unary backward kernels
/// (`Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log`).
///
/// `source` is the forward span the derivative needs:
///
/// - `ReluBackward`, `LogBackward`: forward **input** `A`.
/// - `SigmoidBackward`, `TanhBackward`, `ExpBackward`: forward
///   **output** (cheaper than recomputing).
///
/// The planner picks the right span based on the kernel kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryGradCall {
    /// Forward input or output, depending on `UnaryGradKind`.
    pub source: SlotSpan,
    /// Upstream gradient `dC`.
    pub dc: SlotSpan,
    /// Gradient slot for `A` (accumulated).
    pub da: SlotSpan,
}

/// Identity tag for elementwise unary backward kernels sharing
/// `UnaryGradCall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryGradKind {
    /// `dA += dC * (input > 0 ? 1 : 0)`. `source` = forward input.
    Relu,
    /// `dA += dC * out * (1 - out)`. `source` = forward output.
    Sigmoid,
    /// `dA += dC * (1 - out²)`. `source` = forward output.
    Tanh,
    /// `dA += dC * out` (since d/dx exp = exp). `source` = forward output.
    Exp,
    /// `dA += dC / input`. `source` = forward input.
    Log,
    /// `dA += dC / (2 * sqrt(input)) = dC / (2 * out)`. `source` = forward output.
    Sqrt,
    /// `dA += dC * sign(input)`. `source` = forward input.
    Abs,
    /// `dA += -dC * out²`. `source` = forward output.
    Reciprocal,
    /// Tanh-approximation GELU derivative. `source` = forward input.
    Gelu,
    /// SiLU derivative `σ(x)·(1 + x·(1 - σ(x)))`. `source` = forward input.
    Silu,
}

/// Identity tag for elementwise unary kernels.
///
/// The dispatch surface for the 18 canonical unary ops. The dispatcher
/// matches on `UnaryKind` to pick the per-element function — still all
/// enum dispatch, no virtual call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryKind {
    /// `-x`
    Neg,
    /// `max(x, 0)`
    Relu,
    /// Gaussian Error Linear Unit (tanh approximation).
    Gelu,
    /// `x * sigmoid(x)`
    Silu,
    /// `tanh(x)`
    Tanh,
    /// `1 / (1 + e^-x)`
    Sigmoid,
    /// `e^x`
    Exp,
    /// `ln(x)`
    Log,
    /// `√x`
    Sqrt,
    /// `|x|`
    Abs,
    /// `1 / x`
    Reciprocal,
    /// `cos(x)`
    Cos,
    /// `sin(x)`
    Sin,
    /// `sign(x)` — `-1` / `0` / `1`.
    Sign,
    /// `⌊x⌋`
    Floor,
    /// `⌈x⌉`
    Ceil,
    /// Round half-away-from-zero.
    Round,
    /// Error function.
    Erf,
    /// Logical NOT on f32 truthiness (`0.0` is false; result is
    /// `1.0` for false input, `0.0` otherwise).
    Not,
    /// `1.0` if `x.is_nan()`, otherwise `0.0`.
    IsNaN,
}

/// Apply the unary op identified by `kind` to every element of `call.input`.
#[inline]
pub fn dispatch(storage: &mut [f32], call: &UnaryCall, kind: UnaryKind) {
    match kind {
        UnaryKind::Neg => apply(storage, call, |x| -x),
        UnaryKind::Relu => apply(storage, call, |x| if x > 0.0 { x } else { 0.0 }),
        UnaryKind::Gelu => apply(storage, call, gelu),
        UnaryKind::Silu => apply(storage, call, |x| x * sigmoid(x)),
        UnaryKind::Tanh => apply(storage, call, libm::tanhf),
        UnaryKind::Sigmoid => apply(storage, call, sigmoid),
        UnaryKind::Exp => apply(storage, call, libm::expf),
        UnaryKind::Log => apply(storage, call, libm::logf),
        UnaryKind::Sqrt => apply(storage, call, libm::sqrtf),
        UnaryKind::Abs => apply(storage, call, libm::fabsf),
        UnaryKind::Reciprocal => apply(storage, call, |x| 1.0 / x),
        UnaryKind::Cos => apply(storage, call, libm::cosf),
        UnaryKind::Sin => apply(storage, call, libm::sinf),
        UnaryKind::Sign => apply(storage, call, sign),
        UnaryKind::Floor => apply(storage, call, libm::floorf),
        UnaryKind::Ceil => apply(storage, call, libm::ceilf),
        UnaryKind::Round => apply(storage, call, libm::roundf),
        UnaryKind::Erf => apply(storage, call, libm::erff),
        UnaryKind::Not => apply(storage, call, |x| if x == 0.0 { 1.0 } else { 0.0 }),
        UnaryKind::IsNaN => apply(storage, call, |x| if x.is_nan() { 1.0 } else { 0.0 }),
    }
}

#[inline]
fn apply<F: Fn(f32) -> f32>(storage: &mut [f32], call: &UnaryCall, f: F) {
    let n = call.input.len;
    debug_assert_eq!(call.output.len, n);
    for i in 0..n {
        let x = storage[call.input.offset + i];
        storage[call.output.offset + i] = f(x);
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// Tanh-approximation GELU: `0.5x * (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
#[inline]
fn gelu(x: f32) -> f32 {
    const COEFF: f32 = 0.044_715;
    const SQRT_2_OVER_PI: f32 = 0.797_884_5;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + libm::tanhf(inner))
}

/// Apply the unary backward identified by `kind`. Each formula
/// accumulates into `da` (no-op if `da.len == 0`).
#[inline]
pub fn dispatch_grad(storage: &mut [f32], call: &UnaryGradCall, kind: UnaryGradKind) {
    if call.da.len == 0 {
        return;
    }
    let n = call.dc.len;
    debug_assert_eq!(call.source.len, n);
    debug_assert_eq!(call.da.len, n);
    match kind {
        UnaryGradKind::Relu => {
            for i in 0..n {
                let mask = if storage[call.source.offset + i] > 0.0 {
                    1.0
                } else {
                    0.0
                };
                storage[call.da.offset + i] += storage[call.dc.offset + i] * mask;
            }
        }
        UnaryGradKind::Sigmoid => {
            for i in 0..n {
                let s = storage[call.source.offset + i];
                storage[call.da.offset + i] += storage[call.dc.offset + i] * s * (1.0 - s);
            }
        }
        UnaryGradKind::Tanh => {
            for i in 0..n {
                let t = storage[call.source.offset + i];
                storage[call.da.offset + i] += storage[call.dc.offset + i] * (1.0 - t * t);
            }
        }
        UnaryGradKind::Exp => {
            for i in 0..n {
                storage[call.da.offset + i] +=
                    storage[call.dc.offset + i] * storage[call.source.offset + i];
            }
        }
        UnaryGradKind::Log => {
            for i in 0..n {
                storage[call.da.offset + i] +=
                    storage[call.dc.offset + i] / storage[call.source.offset + i];
            }
        }
        UnaryGradKind::Sqrt => {
            for i in 0..n {
                let out = storage[call.source.offset + i];
                storage[call.da.offset + i] += storage[call.dc.offset + i] * 0.5 / out;
            }
        }
        UnaryGradKind::Abs => {
            for i in 0..n {
                let x = storage[call.source.offset + i];
                let s = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                storage[call.da.offset + i] += storage[call.dc.offset + i] * s;
            }
        }
        UnaryGradKind::Reciprocal => {
            for i in 0..n {
                let out = storage[call.source.offset + i];
                storage[call.da.offset + i] -= storage[call.dc.offset + i] * out * out;
            }
        }
        UnaryGradKind::Gelu => {
            // d/dx [0.5 x (1 + tanh(inner))] where
            //   inner = √(2/π) · (x + 0.044715·x³)
            // = 0.5 · (1 + tanh(inner)) + 0.5·x·(1 - tanh²(inner))·d_inner
            //   d_inner = √(2/π) · (1 + 3·0.044715·x²)
            const COEFF: f32 = 0.044_715;
            const SQRT_2_OVER_PI: f32 = 0.797_884_5;
            for i in 0..n {
                let x = storage[call.source.offset + i];
                let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                let t = libm::tanhf(inner);
                let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x);
                let dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner;
                storage[call.da.offset + i] += storage[call.dc.offset + i] * dgelu;
            }
        }
        UnaryGradKind::Silu => {
            // silu(x) = x·σ(x); d/dx = σ(x) + x·σ(x)·(1 - σ(x))
            //                       = σ(x) · (1 + x · (1 - σ(x)))
            for i in 0..n {
                let x = storage[call.source.offset + i];
                let s = 1.0 / (1.0 + libm::expf(-x));
                let dsilu = s * (1.0 + x * (1.0 - s));
                storage[call.da.offset + i] += storage[call.dc.offset + i] * dsilu;
            }
        }
    }
}

/// Backward: `dA += -dC`.
#[inline]
pub fn neg_grad(storage: &mut [f32], call: &NegGradCall) {
    if call.da.len == 0 {
        return;
    }
    let n = call.dc.len;
    debug_assert_eq!(call.da.len, n);
    for i in 0..n {
        storage[call.da.offset + i] -= storage[call.dc.offset + i];
    }
}

#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kind: UnaryKind, inputs: &[f32]) -> Vec<f32> {
        let n = inputs.len();
        let mut s = vec![0.0_f32; n * 2];
        s[..n].copy_from_slice(inputs);
        let call = UnaryCall {
            input: SlotSpan { offset: 0, len: n },
            output: SlotSpan { offset: n, len: n },
        };
        dispatch(&mut s, &call, kind);
        s[n..].to_vec()
    }

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{} != {}", a, b);
    }

    #[test]
    fn op_trait_relu_carries_relu_backward() {
        assert_eq!(Relu.arity(), 1);
        assert_eq!(Relu.name(), "relu");
        assert_eq!(Relu.category(), OpCategory::Elementwise);
        assert_eq!(Relu.backward(), Some(BackwardRule::ReluBackward));
        assert!(Relu.differentiable());
    }

    #[test]
    fn op_trait_gelu_carries_gelu_backward() {
        assert_eq!(Gelu.arity(), 1);
        assert_eq!(Gelu.backward(), Some(BackwardRule::GeluBackward));
        assert!(Gelu.differentiable());
    }

    #[test]
    fn neg_relu_abs_match_simple_definitions() {
        assert_eq!(run(UnaryKind::Neg, &[1.0, -2.0, 0.0]), &[-1.0, 2.0, -0.0]);
        assert_eq!(run(UnaryKind::Relu, &[1.0, -2.0, 0.0]), &[1.0, 0.0, 0.0]);
        assert_eq!(run(UnaryKind::Abs, &[1.0, -2.0, 0.0]), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn sign_floor_ceil_round_match_definition() {
        assert_eq!(run(UnaryKind::Sign, &[3.0, -2.0, 0.0]), &[1.0, -1.0, 0.0]);
        assert_eq!(run(UnaryKind::Floor, &[1.7, -1.7]), &[1.0, -2.0]);
        assert_eq!(run(UnaryKind::Ceil, &[1.2, -1.2]), &[2.0, -1.0]);
        assert_eq!(run(UnaryKind::Round, &[1.4, 1.5, -1.5]), &[1.0, 2.0, -2.0]);
    }

    #[test]
    fn sigmoid_silu_gelu_are_finite() {
        let xs = [-3.0_f32, -1.0, 0.0, 1.0, 3.0];
        for kind in [UnaryKind::Sigmoid, UnaryKind::Silu, UnaryKind::Gelu] {
            for &x in &xs {
                let y = run(kind, &[x])[0];
                assert!(y.is_finite(), "{:?}({}) = {}", kind, x, y);
            }
        }
        approx_eq(run(UnaryKind::Sigmoid, &[0.0])[0], 0.5);
    }

    #[test]
    fn exp_log_sqrt_reciprocal_match_definition() {
        approx_eq(run(UnaryKind::Exp, &[0.0])[0], 1.0);
        approx_eq(run(UnaryKind::Log, &[1.0])[0], 0.0);
        approx_eq(run(UnaryKind::Sqrt, &[4.0])[0], 2.0);
        approx_eq(run(UnaryKind::Reciprocal, &[2.0])[0], 0.5);
    }

    #[test]
    fn trig_and_erf_match_libm() {
        approx_eq(run(UnaryKind::Cos, &[0.0])[0], 1.0);
        approx_eq(run(UnaryKind::Sin, &[0.0])[0], 0.0);
        approx_eq(run(UnaryKind::Erf, &[0.0])[0], 0.0);
        approx_eq(run(UnaryKind::Tanh, &[0.0])[0], 0.0);
    }
}
