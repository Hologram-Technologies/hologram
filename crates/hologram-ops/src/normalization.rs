//! Normalization ops (spec V.3).

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, EmitResult};

macro_rules! declare_norm {
    ($name:ident, $iri_suffix:literal) => {
        pub struct $name<S, D, B>(PhantomData<(S, D, B)>)
        where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds;

        impl<S, D, B> Default for $name<S, D, B>
        where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<S, D, B> $name<S, D, B>
        where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/normalization/",
                $iri_suffix,
            );
            pub const CAP: usize = 64;

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

declare_norm!(LayerNormOp,    "layer_norm");
declare_norm!(RmsNormOp,      "rms_norm");
declare_norm!(GroupNormOp,    "group_norm");
declare_norm!(InstanceNormOp, "instance_norm");
declare_norm!(AddRmsNormOp,   "add_rms_norm");
