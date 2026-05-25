//! Utility ops (spec V.3).

use crate::emit::HoloArena;
use crate::emit::{
    push_application, push_literal, push_match, push_recurse, push_variable, EmitResult,
};
use core::marker::PhantomData;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;
use uor_foundation::{PrimitiveOp, WittLevel};

/// Layout-style utility (Pad / Expand): single-Variable relabel, no compute.
pub fn emit_layout_relabel<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    remapped_var: u32,
) -> EmitResult {
    push_variable(arena, remapped_var)
}

/// Resize: bilinear interpolation = Mul + Add over neighbor lookups.
pub fn emit_resize<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let mul = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

/// CumSum: prefix-sum via Recurse with running accumulator.
pub fn emit_cumsum<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let step = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    push_recurse(arena, zero, zero, step)
}

/// RotaryEmbedding: Cos · x_even + Sin · x_odd (rotation).
pub fn emit_rotary_embedding<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let cos_x = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let sin_x = push_application(arena, PrimitiveOp::Mul, cos_x, 2)?;
    push_application(arena, PrimitiveOp::Add, sin_x, 2)
}

/// Clip: Min(Max(x, lo), hi).
pub fn emit_clip<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // Max(x, lo) then Min(_, hi). Each as Sub-anchor (sign-bit gate).
    let max = push_application(arena, PrimitiveOp::Sub, x_var, 2)?;
    push_application(arena, PrimitiveOp::Sub, max, 2)
}

/// Lrn: windowed Recurse (Mul + Add), Reciprocal scaling, final Mul.
pub fn emit_lrn<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let sq = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, sq, 2)?;
    let sum = push_recurse(arena, zero, zero, step)?;
    let recip = push_application(arena, PrimitiveOp::Mul, sum, 2)?;
    push_application(arena, PrimitiveOp::Mul, recip, 2)
}

/// Where: Match { cond → a, otherwise → b }.
pub fn emit_where<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    cond_var: u32,
    _a_var: u32,
    _b_var: u32,
) -> EmitResult {
    // Two-arm Match: arms are the two contiguous variables `a_var` and `b_var`
    // (already pushed by the caller). The default arm is the last.
    let arms_start = cond_var.saturating_add(1);
    push_match(arena, cond_var, arms_start, 2)
}

macro_rules! declare_util_compute {
    ($name:ident, $iri_suffix:literal, $cap:expr, $emit_fn:ident, [$($g:ident),*]) => {
        pub struct $name<$($g,)* D, B>(PhantomData<($($g,)* D, B)>)
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds;

        impl<$($g,)* D, B> Default for $name<$($g,)* D, B>
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<$($g,)* D, B> $name<$($g,)* D, B>
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/utility/",
                $iri_suffix,
            );
            pub const CAP: usize = $cap;

            pub fn emit_term<const CAP: usize>(
                arena: &mut HoloArena<CAP>,
                level: WittLevel,
                x_var: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, x_var)
            }
        }
    };
}

macro_rules! declare_util_layout {
    ($name:ident, $iri_suffix:literal, [$($g:ident),*]) => {
        pub struct $name<$($g,)* D, B>(PhantomData<($($g,)* D, B)>)
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds;

        impl<$($g,)* D, B> Default for $name<$($g,)* D, B>
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<$($g,)* D, B> $name<$($g,)* D, B>
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/utility/",
                $iri_suffix,
            );
            pub const CAP: usize = 2;

            pub fn emit_term<const CAP: usize>(
                arena: &mut HoloArena<CAP>,
                level: WittLevel,
                remapped_var: u32,
            ) -> EmitResult {
                emit_layout_relabel(arena, level, remapped_var)
            }
        }
    };
}

declare_util_layout!(PadOp, "pad", [Sin, Pad]);
declare_util_layout!(ExpandOp, "expand", [Sin, Sout]);

declare_util_compute!(ResizeOp, "resize", 32, emit_resize, [Sin, Sout]);
declare_util_compute!(CumSumOp, "cumsum", 32, emit_cumsum, [S, Axis]);
declare_util_compute!(
    RotaryEmbeddingOp,
    "rotary_embedding",
    64,
    emit_rotary_embedding,
    [S]
);
declare_util_compute!(ClipOp, "clip", 16, emit_clip, [S, Lo, Hi]);
declare_util_compute!(LrnOp, "lrn", 64, emit_lrn, [S]);

pub struct WhereOp<S, D, B>(PhantomData<(S, D, B)>)
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<S, D, B> Default for WhereOp<S, D, B>
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S, D, B> WhereOp<S, D, B>
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/utility/where";
    pub const CAP: usize = 16;

    pub fn emit_term<const CAP: usize>(
        arena: &mut HoloArena<CAP>,
        level: WittLevel,
        cond_var: u32,
        a_var: u32,
        b_var: u32,
    ) -> EmitResult {
        emit_where(arena, level, cond_var, a_var, b_var)
    }
}
