//! Op types: PrimOp (10 primitives) and LutOp (21+ activation functions).

mod lut_op;
mod prim;

pub use lut_op::LutOp;
pub use prim::PrimOp;

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
}

impl Op {
    /// Arity of this operation (1 = unary, 2 = binary).
    #[inline]
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Prim(p) => p.arity(),
            Self::Lut(_) => 1,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Prim(p) => p.name(),
            Self::Lut(l) => l.name(),
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
