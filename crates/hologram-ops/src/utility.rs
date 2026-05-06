//! Utility ops (spec V.3).

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_variable, EmitResult};

macro_rules! declare_util_compute {
    ($name:ident, $iri_suffix:literal, $cap:expr, $primary:expr, [$($g:ident),*]) => {
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
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                x_var: u32,
            ) -> EmitResult {
                push_application(arena, $primary, x_var, 1)
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
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                remapped_var: u32,
            ) -> EmitResult {
                push_variable(arena, remapped_var)
            }
        }
    };
}

declare_util_layout!(PadOp,    "pad",    [Sin, Pad]);
declare_util_layout!(ExpandOp, "expand", [Sin, Sout]);

declare_util_compute!(ResizeOp,          "resize",           32, PrimitiveOp::Add,  [Sin, Sout]);
declare_util_compute!(CumSumOp,          "cumsum",           32, PrimitiveOp::Add,  [S, Axis]);
declare_util_compute!(RotaryEmbeddingOp, "rotary_embedding", 64, PrimitiveOp::Mul,  [S]);
declare_util_compute!(ClipOp,            "clip",             16, PrimitiveOp::And,  [S, Lo, Hi]);
declare_util_compute!(LrnOp,             "lrn",              64, PrimitiveOp::Mul,  [S]);
declare_util_compute!(WhereOp,           "where",            16, PrimitiveOp::Or,   [S]);
