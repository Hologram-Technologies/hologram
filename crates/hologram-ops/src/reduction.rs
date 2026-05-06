//! Reduction ops (spec V.3): single Recurse over the reduction axes.
//!
//! V.3:
//!   ReduceSum  : Recurse, step = Add
//!   ReduceMean : ReduceSum + Mul (1/count)
//!   ReduceProd : Recurse, step = Mul
//!   ReduceMin  : Recurse, step = Match (a < b ? a : b)  — encoded as Sub-anchor
//!   ReduceMax  : Recurse, step = Match (a > b ? a : b)  — encoded as Sub-anchor

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

fn emit_reduction_body<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    step_op: PrimitiveOp,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let step = push_application(arena, step_op, x_var, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_reduce_sum<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    emit_reduction_body(arena, level, x_var, PrimitiveOp::Add)
}

pub fn emit_reduce_mean<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // Mean = Sum then Mul(1/count). The reciprocal is a precomputed Binding;
    // the tree shape is Recurse(Add) followed by Mul.
    let zero = push_literal(arena, 0, level)?;
    let step = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    let sum  = push_recurse(arena, zero, zero, step)?;
    push_application(arena, PrimitiveOp::Mul, sum, 2)
}

pub fn emit_reduce_prod<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    emit_reduction_body(arena, level, x_var, PrimitiveOp::Mul)
}

pub fn emit_reduce_min<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    emit_reduction_body(arena, level, x_var, PrimitiveOp::Sub)
}

pub fn emit_reduce_max<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    emit_reduction_body(arena, level, x_var, PrimitiveOp::Sub)
}

macro_rules! declare_reduction {
    ($name:ident, $iri_suffix:literal, $step_op:expr, $emit_fn:ident) => {
        pub struct $name<S, Axes, D, B>(PhantomData<(S, Axes, D, B)>)
        where
            S: ConstrainedTypeShape, Axes: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds;

        impl<S, Axes, D, B> Default for $name<S, Axes, D, B>
        where
            S: ConstrainedTypeShape, Axes: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<S, Axes, D, B> $name<S, Axes, D, B>
        where
            S: ConstrainedTypeShape, Axes: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/reduction/",
                $iri_suffix,
            );
            pub const CAP: usize = 16;
            pub const STEP_OP: PrimitiveOp = $step_op;

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

declare_reduction!(ReduceSumOp,  "reduce_sum",  PrimitiveOp::Add, emit_reduce_sum);
declare_reduction!(ReduceMeanOp, "reduce_mean", PrimitiveOp::Add, emit_reduce_mean);
declare_reduction!(ReduceProdOp, "reduce_prod", PrimitiveOp::Mul, emit_reduce_prod);
declare_reduction!(ReduceMinOp,  "reduce_min",  PrimitiveOp::Sub, emit_reduce_min);
declare_reduction!(ReduceMaxOp,  "reduce_max",  PrimitiveOp::Sub, emit_reduce_max);
