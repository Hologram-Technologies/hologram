//! Activation+reduce ops (spec V.3): Softmax, LogSoftmax.
//!
//! V.3 (Softmax):
//!   ReduceMax → Sub → Exp → ReduceSum → Div
//! V.3 (LogSoftmax):
//!   Softmax → Log

use crate::emit::{push_application, push_literal, push_recurse, EmitResult};
use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;
use uor_foundation::{PrimitiveOp, WittLevel};

pub fn emit_softmax<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // ReduceMax (anchored Sub-step), Sub for stability, Exp (Mul anchor),
    // ReduceSum (Add-step), Div (Mul anchor).
    let zero = push_literal(arena, 0, level)?;
    let max_step = push_application(arena, PrimitiveOp::Sub, x_var, 2)?;
    let max = push_recurse(arena, zero, zero, max_step)?;
    let centered = push_application(arena, PrimitiveOp::Sub, max, 2)?;
    let exp = push_application(arena, PrimitiveOp::Mul, centered, 2)?;
    let sum_step = push_application(arena, PrimitiveOp::Add, exp, 2)?;
    let sum = push_recurse(arena, zero, zero, sum_step)?;
    push_application(arena, PrimitiveOp::Mul, sum, 2)
}

pub fn emit_log_softmax<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let softmax = emit_softmax(arena, level, x_var)?;
    push_application(arena, PrimitiveOp::Mul, softmax, 1)
}

macro_rules! declare_actred {
    ($name:ident, $iri_suffix:literal, $emit_fn:ident) => {
        pub struct $name<S, Axis, D, B>(PhantomData<(S, Axis, D, B)>)
        where
            S: ConstrainedTypeShape,
            Axis: ConstrainedTypeShape,
            D: ConstrainedTypeShape,
            B: HostBounds;

        impl<S, Axis, D, B> Default for $name<S, Axis, D, B>
        where
            S: ConstrainedTypeShape,
            Axis: ConstrainedTypeShape,
            D: ConstrainedTypeShape,
            B: HostBounds,
        {
            fn default() -> Self {
                Self(PhantomData)
            }
        }

        impl<S, Axis, D, B> $name<S, Axis, D, B>
        where
            S: ConstrainedTypeShape,
            Axis: ConstrainedTypeShape,
            D: ConstrainedTypeShape,
            B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/activation_reduce/",
                $iri_suffix,
            );
            pub const CAP: usize = 32;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                level: WittLevel,
                x_var: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, x_var)
            }
        }
    };
}

declare_actred!(SoftmaxOp, "softmax", emit_softmax);
declare_actred!(LogSoftmaxOp, "log_softmax", emit_log_softmax);
