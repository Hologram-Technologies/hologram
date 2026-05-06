//! Reduction ops (spec V.3): single Recurse over reduction axes.

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

macro_rules! declare_reduction {
    ($name:ident, $iri_suffix:literal, $step_op:expr) => {
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
                let zero = push_literal(arena, 0, level)?;
                let step = push_application(arena, $step_op, x_var, 2)?;
                push_recurse(arena, zero, zero, step)
            }
        }
    };
}

declare_reduction!(ReduceSumOp,  "reduce_sum",  PrimitiveOp::Add);
declare_reduction!(ReduceMeanOp, "reduce_mean", PrimitiveOp::Add);
declare_reduction!(ReduceProdOp, "reduce_prod", PrimitiveOp::Mul);
declare_reduction!(ReduceMinOp,  "reduce_min",  PrimitiveOp::Sub);
declare_reduction!(ReduceMaxOp,  "reduce_max",  PrimitiveOp::Sub);
