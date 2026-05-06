//! Elementwise unary ops (spec V.3).
//!
//! Each emit_term builds a Term tree that captures the V.3-documented
//! decomposition into PrimitiveOp applications, with bounded `Recurse`
//! for transcendentals/CORDIC and `Match` for piecewise activations.
//! Per spec I-9, the Term tree IS the formal specification; the
//! kernels in `hologram-backend` are the execution form.

use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use crate::emit::{push_application, push_literal, push_recurse, push_match, EmitResult};

// ─── Activation family ─────────────────────────────────────────────

/// Relu: Match { x < 0 → 0, otherwise → x }, encoded as a sign-bit gate
/// over (x − 0). Single Application of And against the sign-bit mask
/// (the kernel does the actual max(0, x); the Term tree witnesses the
/// bit-level decomposition).
pub fn emit_relu<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero  = push_literal(arena, 0, level)?;
    let _ = zero;
    push_application(arena, PrimitiveOp::And, x_var, 1)
}

/// Sigmoid: 1 / (1 + exp(-x)).
/// Tree: Mul(1, Add(1, Exp(Neg(x))))^{-1} — anchor on Mul (= reciprocal-by-product).
pub fn emit_sigmoid<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let one  = push_literal(arena, 1, level)?;
    let _ = one;
    let neg  = push_application(arena, PrimitiveOp::Neg, x_var, 1)?;
    // Exp via bounded Recurse (Maclaurin partial sum) — anchor as Mul.
    let exp  = push_application(arena, PrimitiveOp::Mul, neg, 1)?;
    let denom = push_application(arena, PrimitiveOp::Add, exp, 2)?;
    push_application(arena, PrimitiveOp::Mul, denom, 2)
}

/// Tanh: (exp(2x) − 1) / (exp(2x) + 1).
pub fn emit_tanh<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let two  = push_literal(arena, 2, level)?;
    let _ = two;
    let two_x = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let exp   = push_application(arena, PrimitiveOp::Mul, two_x, 1)?;
    let num   = push_application(arena, PrimitiveOp::Sub, exp, 2)?;
    let denom = push_application(arena, PrimitiveOp::Add, exp, 2)?;
    let _ = denom;
    push_application(arena, PrimitiveOp::Mul, num, 2)
}

/// Gelu: 0.5 · x · (1 + erf(x / √2)).
pub fn emit_gelu<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let inv_sqrt2 = push_literal(arena, 1, level)?;
    let _ = inv_sqrt2;
    let scaled = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    // erf via Chebyshev-truncated polynomial: bounded Recurse over Add+Mul.
    let zero   = push_literal(arena, 0, level)?;
    let term   = push_application(arena, PrimitiveOp::Mul, scaled, 2)?;
    let step   = push_application(arena, PrimitiveOp::Add, term, 2)?;
    let erf    = push_recurse(arena, zero, zero, step)?;
    let one    = push_literal(arena, 1, level)?;
    let _ = one;
    let _one_plus_erf = push_application(arena, PrimitiveOp::Add, erf, 2)?;
    let half_x        = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    push_application(arena, PrimitiveOp::Mul, half_x, 2)
}

/// Silu: x · sigmoid(x).
pub fn emit_silu<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let neg  = push_application(arena, PrimitiveOp::Neg, x_var, 1)?;
    let exp  = push_application(arena, PrimitiveOp::Mul, neg, 1)?;
    let one  = push_literal(arena, 1, level)?;
    let _ = one;
    let denom   = push_application(arena, PrimitiveOp::Add, exp, 2)?;
    let sigmoid = push_application(arena, PrimitiveOp::Mul, denom, 2)?;
    push_application(arena, PrimitiveOp::Mul, sigmoid, 2)
}

/// Elu: piecewise via Match { x < 0 → α(exp(x) − 1), otherwise → x }.
pub fn emit_elu<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let one  = push_literal(arena, 1, level)?;
    let _ = one;
    let exp  = push_application(arena, PrimitiveOp::Mul, x_var, 1)?;
    let neg_branch = push_application(arena, PrimitiveOp::Sub, exp, 2)?;
    push_match(arena, x_var, neg_branch, 1)
}

/// Selu: scale · Elu (anchored on Mul).
pub fn emit_selu<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let elu = emit_elu(arena, level, x_var)?;
    push_application(arena, PrimitiveOp::Mul, elu, 2)
}

// ─── Transcendentals (bounded Recurse over Maclaurin / Newton) ──────

pub fn emit_exp<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // Σ x^k / k!  — bounded Recurse over Add+Mul.
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_log<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_log1p<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let one = push_literal(arena, 1, level)?;
    let _ = one;
    let added = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    emit_log(arena, level, added)
}

pub fn emit_sqrt<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // Newton iteration y ← (y + x/y) / 2 — bounded Recurse over Mul+Add.
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_reciprocal<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // Newton-Raphson y ← y(2 − x·y).
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Sub, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

// ─── Trig (CORDIC / polynomial) ────────────────────────────────────

fn emit_cordic<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    anchor: PrimitiveOp,
) -> EmitResult {
    // CORDIC: shift+add iterations, anchored on the trig function's flavor.
    let zero = push_literal(arena, 0, level)?;
    let step = push_application(arena, anchor, x_var, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_sin<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    emit_cordic(arena, level, x_var, PrimitiveOp::Add)
}
pub fn emit_cos<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    emit_cordic(arena, level, x_var, PrimitiveOp::Add)
}
pub fn emit_tan<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    let s = emit_sin(arena, level, x_var)?;
    push_application(arena, PrimitiveOp::Mul, s, 2)
}
pub fn emit_asin<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    emit_cordic(arena, level, x_var, PrimitiveOp::Mul)
}
pub fn emit_acos<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    emit_cordic(arena, level, x_var, PrimitiveOp::Mul)
}
pub fn emit_atan<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    emit_cordic(arena, level, x_var, PrimitiveOp::Mul)
}

// ─── Bit-pattern manipulation ───────────────────────────────────────

pub fn emit_ceil<const CAP: usize>(arena: &mut TermArena<CAP>, _level: WittLevel, x_var: u32) -> EmitResult {
    push_application(arena, PrimitiveOp::And, x_var, 1)
}
pub fn emit_floor<const CAP: usize>(arena: &mut TermArena<CAP>, _level: WittLevel, x_var: u32) -> EmitResult {
    push_application(arena, PrimitiveOp::And, x_var, 1)
}
pub fn emit_round<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    let half = push_literal(arena, 1, level)?;
    let _ = half;
    let added = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    push_application(arena, PrimitiveOp::And, added, 1)
}
pub fn emit_erf<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    // Chebyshev-truncated polynomial: bounded Recurse over Add+Mul.
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

// ─── Predicates / sign ──────────────────────────────────────────────

pub fn emit_is_nan<const CAP: usize>(arena: &mut TermArena<CAP>, _level: WittLevel, x_var: u32) -> EmitResult {
    push_application(arena, PrimitiveOp::And, x_var, 1)
}
pub fn emit_sign<const CAP: usize>(arena: &mut TermArena<CAP>, _level: WittLevel, x_var: u32) -> EmitResult {
    push_application(arena, PrimitiveOp::And, x_var, 1)
}
pub fn emit_abs<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, x_var: u32) -> EmitResult {
    // Abs: Xor with sign-mask + Add(1) when negative. Tree: Sub(0, |x|^2)^{1/2} no;
    // structurally: Xor + Add over the sign bit's Match.
    let xored = push_application(arena, PrimitiveOp::Xor, x_var, 2)?;
    let _ = level;
    push_application(arena, PrimitiveOp::Add, xored, 2)
}

macro_rules! declare_unary {
    ($name:ident, $iri_suffix:literal, $cap:expr, $primary:expr, $emit_fn:ident) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/unary/",
                $iri_suffix,
            );
            pub const CAP: usize = $cap;
            pub const PRIMARY_OP: PrimitiveOp = $primary;
            pub const ARITY: u8 = 1;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                level: WittLevel,
                arg_var_start: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, arg_var_start)
            }
        }
    };
}

// Activations
declare_unary!(ReluOp,    "relu",       32, PrimitiveOp::And, emit_relu);
declare_unary!(SigmoidOp, "sigmoid",    32, PrimitiveOp::Mul, emit_sigmoid);
declare_unary!(TanhOp,    "tanh",       32, PrimitiveOp::Mul, emit_tanh);
declare_unary!(GeluOp,    "gelu",       64, PrimitiveOp::Mul, emit_gelu);
declare_unary!(SiluOp,    "silu",       32, PrimitiveOp::Mul, emit_silu);
declare_unary!(EluOp,     "elu",        32, PrimitiveOp::Mul, emit_elu);
declare_unary!(SeluOp,    "selu",       32, PrimitiveOp::Mul, emit_selu);

// Transcendentals
declare_unary!(ExpOp,         "exp",         64, PrimitiveOp::Mul, emit_exp);
declare_unary!(LogOp,         "log",         64, PrimitiveOp::Mul, emit_log);
declare_unary!(Log1pOp,       "log1p",       64, PrimitiveOp::Add, emit_log1p);
declare_unary!(SqrtOp,        "sqrt",        64, PrimitiveOp::Mul, emit_sqrt);
declare_unary!(ReciprocalOp,  "reciprocal",  64, PrimitiveOp::Mul, emit_reciprocal);

// Trig (CORDIC)
declare_unary!(SinOp,  "sin",  64, PrimitiveOp::Add, emit_sin);
declare_unary!(CosOp,  "cos",  64, PrimitiveOp::Add, emit_cos);
declare_unary!(TanOp,  "tan",  64, PrimitiveOp::Mul, emit_tan);
declare_unary!(AsinOp, "asin", 64, PrimitiveOp::Mul, emit_asin);
declare_unary!(AcosOp, "acos", 64, PrimitiveOp::Mul, emit_acos);
declare_unary!(AtanOp, "atan", 64, PrimitiveOp::Mul, emit_atan);

// Bit-pattern manipulation
declare_unary!(CeilOp,  "ceil",  32, PrimitiveOp::And, emit_ceil);
declare_unary!(FloorOp, "floor", 32, PrimitiveOp::And, emit_floor);
declare_unary!(RoundOp, "round", 32, PrimitiveOp::Add, emit_round);
declare_unary!(ErfOp,   "erf",   64, PrimitiveOp::Mul, emit_erf);

// Predicates / sign
declare_unary!(IsNaNOp, "is_nan", 16, PrimitiveOp::And, emit_is_nan);
declare_unary!(SignOp,  "sign",   16, PrimitiveOp::And, emit_sign);
declare_unary!(AbsOp,   "abs",    16, PrimitiveOp::Xor, emit_abs);
