//! Pooling ops (spec V.3): MaxPool2d, AvgPool2d, GlobalAvgPool.
//!
//! V.3:
//!   MaxPool2d : windowed Recurse, step = Match (max) — anchored on Sub
//!   AvgPool2d : windowed Recurse, step = Add; final Mul (1/window)
//!   GlobalAvgPool : ReduceMeanOp over spatial axes

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

pub fn emit_max_pool_2d<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero  = push_literal(arena, 0, level)?;
    let step  = push_application(arena, PrimitiveOp::Sub, x_var, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_avg_pool_2d<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let step = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    let sum  = push_recurse(arena, zero, zero, step)?;
    push_application(arena, PrimitiveOp::Mul, sum, 2)
}

pub fn emit_global_avg_pool<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    emit_avg_pool_2d(arena, level, x_var)
}

macro_rules! declare_pool {
    ($name:ident, $iri_suffix:literal, $step_op:expr, $emit_fn:ident) => {
        pub struct $name<X, K, S, D, B>(PhantomData<(X, K, S, D, B)>)
        where
            X: ConstrainedTypeShape, K: ConstrainedTypeShape, S: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds;

        impl<X, K, S, D, B> Default for $name<X, K, S, D, B>
        where
            X: ConstrainedTypeShape, K: ConstrainedTypeShape, S: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<X, K, S, D, B> $name<X, K, S, D, B>
        where
            X: ConstrainedTypeShape, K: ConstrainedTypeShape, S: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/pooling/",
                $iri_suffix,
            );
            pub const CAP: usize = 32;
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

declare_pool!(MaxPool2dOp, "max_pool_2d", PrimitiveOp::Sub, emit_max_pool_2d);
declare_pool!(AvgPool2dOp, "avg_pool_2d", PrimitiveOp::Add, emit_avg_pool_2d);

pub struct GlobalAvgPoolOp<S, D, B>(PhantomData<(S, D, B)>)
where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds;

impl<S, D, B> Default for GlobalAvgPoolOp<S, D, B>
where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
{ fn default() -> Self { Self(PhantomData) } }

impl<S, D, B> GlobalAvgPoolOp<S, D, B>
where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/pooling/global_avg_pool";
    pub const CAP: usize = 32;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        x_var: u32,
    ) -> EmitResult {
        emit_global_avg_pool(arena, level, x_var)
    }
}
