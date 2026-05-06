//! Elementwise binary non-primitive ops (spec V.3).
//!
//! `Add`/`Sub`/`Mul`/`Xor`/`And`/`Or` live in `direct.rs`; this module
//! covers `Div`/`Pow`/`Mod`/`Min`/`Max` and the comparison family.

use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use crate::emit::{push_application, EmitResult};

macro_rules! declare_binary {
    ($name:ident, $iri_suffix:literal, $cap:expr, $primary:expr) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/binary/",
                $iri_suffix,
            );
            pub const CAP: usize = $cap;
            pub const PRIMARY_OP: PrimitiveOp = $primary;
            pub const ARITY: u8 = 2;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                arg_var_start: u32,
            ) -> EmitResult {
                push_application(arena, $primary, arg_var_start, 2)
            }
        }
    };
}

// Arithmetic compositions
declare_binary!(DivOp, "div", 32, PrimitiveOp::Mul);  // Newton iteration
declare_binary!(PowOp, "pow", 64, PrimitiveOp::Mul);  // Exp · Mul · Log composition
declare_binary!(ModOp, "mod", 16, PrimitiveOp::Sub);  // x − ⌊x/y⌋·y

// Min/Max
declare_binary!(MinOp, "min", 16, PrimitiveOp::Sub);  // sign-bit gate on (a − b)
declare_binary!(MaxOp, "max", 16, PrimitiveOp::Sub);

// Comparisons (return 0/1 in u-typed bit pattern)
declare_binary!(EqualOp,          "equal",            16, PrimitiveOp::Xor);
declare_binary!(LessOp,           "less",             16, PrimitiveOp::Sub);
declare_binary!(LessOrEqualOp,    "less_or_equal",    16, PrimitiveOp::Sub);
declare_binary!(GreaterOp,        "greater",          16, PrimitiveOp::Sub);
declare_binary!(GreaterOrEqualOp, "greater_or_equal", 16, PrimitiveOp::Sub);
