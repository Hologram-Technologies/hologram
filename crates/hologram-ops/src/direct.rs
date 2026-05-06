//! Direct `PrimitiveOp` wrappers (spec V.3).
//!
//! These are not compositions — each marker is a single
//! `Term::Application { operator: PrimitiveOp::*, args }` tree.
//! They exist as named markers so the catalog is uniform.

use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use crate::emit::{push_application, EmitResult};

/// IRI prefix for direct ops.
const IRI_PREFIX: &str = "https://hologram.uor.foundation/op/direct/";

macro_rules! declare_direct {
    ($name:ident, $iri_suffix:literal, $prim:expr, $arity:expr) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/direct/",
                $iri_suffix,
            );
            /// Term arena CAP per spec V.5: 4 for direct wrappers.
            pub const CAP: usize = 4;
            pub const PRIMITIVE: PrimitiveOp = $prim;
            pub const ARITY: u8 = $arity;

            /// Emit the canonical Term tree: `Application(prim, [var_0, ...])`.
            /// `arg_var_indices` are the arena indices of the already-pushed
            /// `Term::Variable` argument terms.
            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                arg_var_start: u32,
            ) -> EmitResult {
                push_application(arena, $prim, arg_var_start, $arity as u32)
            }
        }

        // Suppress unused-warning on IRI_PREFIX in this module's macro context.
        const _: &str = IRI_PREFIX;
    };
}

declare_direct!(NegOp,  "neg",  PrimitiveOp::Neg,  1);
declare_direct!(BnotOp, "bnot", PrimitiveOp::Bnot, 1);
declare_direct!(SuccOp, "succ", PrimitiveOp::Succ, 1);
declare_direct!(PredOp, "pred", PrimitiveOp::Pred, 1);
declare_direct!(AddOp,  "add",  PrimitiveOp::Add,  2);
declare_direct!(SubOp,  "sub",  PrimitiveOp::Sub,  2);
declare_direct!(MulOp,  "mul",  PrimitiveOp::Mul,  2);
declare_direct!(XorOp,  "xor",  PrimitiveOp::Xor,  2);
declare_direct!(AndOp,  "and",  PrimitiveOp::And,  2);
declare_direct!(OrOp,   "or",   PrimitiveOp::Or,   2);
