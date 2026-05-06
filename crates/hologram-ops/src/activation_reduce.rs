//! Activation+reduce ops (spec V.3): Softmax, LogSoftmax.

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, EmitResult};

macro_rules! declare_actred {
    ($name:ident, $iri_suffix:literal) => {
        pub struct $name<S, Axis, D, B>(PhantomData<(S, Axis, D, B)>)
        where
            S: ConstrainedTypeShape, Axis: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds;

        impl<S, Axis, D, B> Default for $name<S, Axis, D, B>
        where
            S: ConstrainedTypeShape, Axis: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<S, Axis, D, B> $name<S, Axis, D, B>
        where
            S: ConstrainedTypeShape, Axis: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/activation_reduce/",
                $iri_suffix,
            );
            pub const CAP: usize = 32;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                x_var: u32,
            ) -> EmitResult {
                push_application(arena, PrimitiveOp::Mul, x_var, 1)
            }
        }
    };
}

declare_actred!(SoftmaxOp,    "softmax");
declare_actred!(LogSoftmaxOp, "log_softmax");
