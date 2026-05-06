//! Convolution ops (spec V.3): Conv2d, ConvTranspose2d.

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

macro_rules! declare_conv {
    ($name:ident, $iri_suffix:literal) => {
        pub struct $name<X, W, P, S, D, B>(PhantomData<(X, W, P, S, D, B)>)
        where
            X: ConstrainedTypeShape, W: ConstrainedTypeShape,
            P: ConstrainedTypeShape, S: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds;

        impl<X, W, P, S, D, B> Default for $name<X, W, P, S, D, B>
        where
            X: ConstrainedTypeShape, W: ConstrainedTypeShape,
            P: ConstrainedTypeShape, S: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        {
            fn default() -> Self { Self(PhantomData) }
        }

        impl<X, W, P, S, D, B> $name<X, W, P, S, D, B>
        where
            X: ConstrainedTypeShape, W: ConstrainedTypeShape,
            P: ConstrainedTypeShape, S: ConstrainedTypeShape,
            D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/conv/",
                $iri_suffix,
            );
            pub const CAP: usize = 64;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                level: WittLevel,
                x_var: u32,
                _w_var: u32,
            ) -> EmitResult {
                let zero = push_literal(arena, 0, level)?;
                let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
                let add  = push_application(arena, PrimitiveOp::Add, mul, 2)?;
                push_recurse(arena, zero, mul, add)
            }
        }
    };
}

declare_conv!(Conv2dOp, "conv2d");
declare_conv!(ConvTranspose2dOp, "conv_transpose_2d");
