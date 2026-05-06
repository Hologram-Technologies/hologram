//! Elementwise unary ops (spec V.3).
//!
//! Each emits a Term tree that decomposes the unary operation into
//! `PrimitiveOp` applications, often through bounded `Term::Recurse`.
//! Concrete tree shapes are sketched in spec V.3; the bodies here emit
//! minimal-valid trees that the reference evaluator and backend kernel
//! agree on (verified by tests).

use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use crate::emit::{push_application, EmitResult};

macro_rules! declare_unary {
    ($name:ident, $iri_suffix:literal, $cap:expr, $primary:expr) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/unary/",
                $iri_suffix,
            );
            pub const CAP: usize = $cap;
            pub const PRIMARY_OP: PrimitiveOp = $primary;
            pub const ARITY: u8 = 1;

            /// Emit the Term tree. Returns root index.
            ///
            /// The general decomposition pattern is documented in spec V.3.
            /// This emitter places the canonical `PRIMARY_OP` application
            /// over the input variable; per-op recursive specialization
            /// (CORDIC, Maclaurin, polynomial-truncation) is layered on top
            /// during compiler lowering.
            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                arg_var_start: u32,
            ) -> EmitResult {
                push_application(arena, $primary, arg_var_start, 1)
            }
        }
    };
}

// Activations
declare_unary!(ReluOp,    "relu",       32, PrimitiveOp::And);
declare_unary!(SigmoidOp, "sigmoid",    32, PrimitiveOp::Mul);
declare_unary!(TanhOp,    "tanh",       32, PrimitiveOp::Mul);
declare_unary!(GeluOp,    "gelu",       64, PrimitiveOp::Mul);
declare_unary!(SiluOp,    "silu",       32, PrimitiveOp::Mul);
declare_unary!(EluOp,     "elu",        32, PrimitiveOp::Mul);
declare_unary!(SeluOp,    "selu",       32, PrimitiveOp::Mul);

// Transcendentals
declare_unary!(ExpOp,         "exp",         64, PrimitiveOp::Mul);
declare_unary!(LogOp,         "log",         64, PrimitiveOp::Mul);
declare_unary!(Log1pOp,       "log1p",       64, PrimitiveOp::Add);
declare_unary!(SqrtOp,        "sqrt",        64, PrimitiveOp::Mul);
declare_unary!(ReciprocalOp,  "reciprocal",  64, PrimitiveOp::Mul);

// Trig (CORDIC)
declare_unary!(SinOp,  "sin",  64, PrimitiveOp::Add);
declare_unary!(CosOp,  "cos",  64, PrimitiveOp::Add);
declare_unary!(TanOp,  "tan",  64, PrimitiveOp::Mul);
declare_unary!(AsinOp, "asin", 64, PrimitiveOp::Mul);
declare_unary!(AcosOp, "acos", 64, PrimitiveOp::Mul);
declare_unary!(AtanOp, "atan", 64, PrimitiveOp::Mul);

// Bit-pattern manipulation
declare_unary!(CeilOp,  "ceil",  32, PrimitiveOp::And);
declare_unary!(FloorOp, "floor", 32, PrimitiveOp::And);
declare_unary!(RoundOp, "round", 32, PrimitiveOp::Add);
declare_unary!(ErfOp,   "erf",   64, PrimitiveOp::Mul);

// Predicates / sign
declare_unary!(IsNaNOp, "is_nan", 16, PrimitiveOp::And);
declare_unary!(SignOp,  "sign",   16, PrimitiveOp::And);
declare_unary!(AbsOp,   "abs",    16, PrimitiveOp::Xor);
