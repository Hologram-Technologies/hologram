//! Op types: PrimOp (10 primitives), LutOp (21+ activation functions),
//! and FloatOp (typed f32 tensor operations for AI inference).

mod float_op;
mod lut_op;
mod prim;
mod shape_spec;

pub use float_op::{bits_to_f32, f32_to_bits, FloatDType, FloatOp, OpCategory};
pub use lut_op::LutOp;
pub use prim::PrimOp;
pub use shape_spec::{ShapeDim, ShapeSpec};

/// Ring quantum level for ring-arithmetic execution.
///
/// Selects which ring Z/2^nZ to operate in:
/// - Q0: Z/256Z (8-bit)
/// - Q1: Z/65536Z (16-bit)
/// - Q2: Z/2^24Z (24-bit)
/// - Q3: Z/2^32Z (32-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[repr(u8)]
pub enum RingLevel {
    Q0 = 0,
    Q1 = 1,
    Q2 = 2,
    Q3 = 3,
}

/// Unified operation enum for all byte-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub enum Op {
    /// One of the 10 UOR primitive operations.
    Prim(PrimOp),
    /// An activation/scientific function via LUT.
    Lut(LutOp),
    /// A typed f32 tensor operation for AI inference.
    Float(FloatOp),
}

impl Op {
    /// Arity of this operation (1 = unary, 2 = binary).
    #[inline]
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Prim(p) => p.arity(),
            Self::Lut(_) => 1,
            Self::Float(f) => f.arity(),
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Prim(p) => p.name(),
            Self::Lut(l) => l.name(),
            Self::Float(f) => f.name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_arity() {
        assert_eq!(Op::Prim(PrimOp::Neg).arity(), 1);
        assert_eq!(Op::Prim(PrimOp::Add).arity(), 2);
        assert_eq!(Op::Lut(LutOp::Sigmoid).arity(), 1);
    }

    #[test]
    fn op_name() {
        assert_eq!(Op::Prim(PrimOp::Neg).name(), "neg");
        assert_eq!(Op::Lut(LutOp::Relu).name(), "relu");
    }
}
