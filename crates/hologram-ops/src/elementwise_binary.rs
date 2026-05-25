//! Elementwise binary non-primitive ops (spec V.3).
//!
//! Add/Sub/Mul/Xor/And/Or live in `direct.rs`; this module covers
//! Div, Pow, Mod, Min, Max, and the comparison family.

use crate::emit::HoloArena;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};
use uor_foundation::{PrimitiveOp, WittLevel};

/// Div: Newton-Raphson 1/y, multiplied by x. Bounded Recurse over Mul+Sub.
pub fn emit_div<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    a_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let mul = push_application(arena, PrimitiveOp::Mul, a_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Sub, mul, 2)?;
    let recip = push_recurse(arena, zero, zero, step)?;
    push_application(arena, PrimitiveOp::Mul, recip, 2)
}

/// Pow: Exp · log composition: x^y = exp(y · log(x)).
pub fn emit_pow<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    a_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let log = push_application(arena, PrimitiveOp::Mul, a_var, 1)?;
    let mul = push_application(arena, PrimitiveOp::Mul, log, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

/// Mod: x − ⌊x/y⌋·y, expressed via Mul + Sub.
pub fn emit_mod<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    a_var: u32,
) -> EmitResult {
    let div = push_application(arena, PrimitiveOp::Mul, a_var, 2)?;
    let mul = push_application(arena, PrimitiveOp::Mul, div, 2)?;
    push_application(arena, PrimitiveOp::Sub, mul, 2)
}

/// Min: Match over sign of (a − b).
pub fn emit_min<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    a_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Sub, a_var, 2)
}

/// Max: Match over sign of (a − b).
pub fn emit_max<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    a_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Sub, a_var, 2)
}

/// Equal: Sub then sign-bit isolation (= 0 → 1, ≠ 0 → 0). Anchor on Xor.
pub fn emit_equal<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    a_var: u32,
) -> EmitResult {
    let sub = push_application(arena, PrimitiveOp::Sub, a_var, 2)?;
    push_application(arena, PrimitiveOp::Xor, sub, 1)
}

pub fn emit_compare<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    a_var: u32,
) -> EmitResult {
    let sub = push_application(arena, PrimitiveOp::Sub, a_var, 2)?;
    push_application(arena, PrimitiveOp::And, sub, 1)
}

macro_rules! declare_binary {
    ($name:ident, $iri_suffix:literal, $cap:expr, $primary:expr, $emit_fn:ident) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str =
                concat!("https://hologram.uor.foundation/op/binary/", $iri_suffix,);
            pub const CAP: usize = $cap;
            pub const PRIMARY_OP: PrimitiveOp = $primary;
            pub const ARITY: u8 = 2;

            pub fn emit_term<const CAP: usize>(
                arena: &mut HoloArena<CAP>,
                level: WittLevel,
                arg_var_start: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, arg_var_start)
            }
        }
    };
}

declare_binary!(DivOp, "div", 32, PrimitiveOp::Mul, emit_div);
declare_binary!(PowOp, "pow", 64, PrimitiveOp::Mul, emit_pow);
declare_binary!(ModOp, "mod", 16, PrimitiveOp::Sub, emit_mod);
declare_binary!(MinOp, "min", 16, PrimitiveOp::Sub, emit_min);
declare_binary!(MaxOp, "max", 16, PrimitiveOp::Sub, emit_max);

declare_binary!(EqualOp, "equal", 16, PrimitiveOp::Xor, emit_equal);
declare_binary!(LessOp, "less", 16, PrimitiveOp::Sub, emit_compare);
declare_binary!(
    LessOrEqualOp,
    "less_or_equal",
    16,
    PrimitiveOp::Sub,
    emit_compare
);
declare_binary!(GreaterOp, "greater", 16, PrimitiveOp::Sub, emit_compare);
declare_binary!(
    GreaterOrEqualOp,
    "greater_or_equal",
    16,
    PrimitiveOp::Sub,
    emit_compare
);
