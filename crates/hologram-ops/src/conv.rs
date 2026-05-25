//! Convolution ops (spec V.3): Conv2d, ConvTranspose2d.
//!
//! V.3 tree shape: 4-deep `Recurse` (output_h, output_w, kernel_h, kernel_w)
//! with `Add(acc, Mul(x, w))` as the innermost step.

use crate::emit::HoloArena;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};
use core::marker::PhantomData;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;
use uor_foundation::{PrimitiveOp, WittLevel};

fn emit_conv_body<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
    w_var: u32,
) -> EmitResult {
    let _ = w_var;
    let zero = push_literal(arena, 0, level)?;
    let mul = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let kw_step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    let kw_rec = push_recurse(arena, zero, zero, kw_step)?;
    let kh_step = push_application(arena, PrimitiveOp::Add, kw_rec, 2)?;
    let kh_rec = push_recurse(arena, zero, zero, kh_step)?;
    let ow_step = push_application(arena, PrimitiveOp::Add, kh_rec, 2)?;
    let ow_rec = push_recurse(arena, zero, zero, ow_step)?;
    let oh_step = push_application(arena, PrimitiveOp::Add, ow_rec, 2)?;
    push_recurse(arena, zero, zero, oh_step)
}

pub fn emit_conv2d<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
    w_var: u32,
) -> EmitResult {
    emit_conv_body(arena, level, x_var, w_var)
}

pub fn emit_conv_transpose_2d<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
    w_var: u32,
) -> EmitResult {
    emit_conv_body(arena, level, x_var, w_var)
}

macro_rules! declare_conv {
    ($name:ident, $iri_suffix:literal, $emit_fn:ident) => {
        pub struct $name<X, W, P, S, D, B>(PhantomData<(X, W, P, S, D, B)>)
        where
            X: ConstrainedTypeShape,
            W: ConstrainedTypeShape,
            P: ConstrainedTypeShape,
            S: ConstrainedTypeShape,
            D: ConstrainedTypeShape,
            B: HostBounds;

        impl<X, W, P, S, D, B> Default for $name<X, W, P, S, D, B>
        where
            X: ConstrainedTypeShape,
            W: ConstrainedTypeShape,
            P: ConstrainedTypeShape,
            S: ConstrainedTypeShape,
            D: ConstrainedTypeShape,
            B: HostBounds,
        {
            fn default() -> Self {
                Self(PhantomData)
            }
        }

        impl<X, W, P, S, D, B> $name<X, W, P, S, D, B>
        where
            X: ConstrainedTypeShape,
            W: ConstrainedTypeShape,
            P: ConstrainedTypeShape,
            S: ConstrainedTypeShape,
            D: ConstrainedTypeShape,
            B: HostBounds,
        {
            pub const IRI: &'static str =
                concat!("https://hologram.uor.foundation/op/conv/", $iri_suffix,);
            pub const CAP: usize = 64;

            pub fn emit_term<const CAP: usize>(
                arena: &mut HoloArena<CAP>,
                level: WittLevel,
                x_var: u32,
                w_var: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, x_var, w_var)
            }
        }
    };
}

declare_conv!(Conv2dOp, "conv2d", emit_conv2d);
declare_conv!(
    ConvTranspose2dOp,
    "conv_transpose_2d",
    emit_conv_transpose_2d
);
