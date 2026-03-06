//! PrimOp: the 10 UOR primitive operations, mirroring uor-foundation PrimitiveOp.

use crate::lut::arith;

/// The 10 primitive operations on Z/256Z.
///
/// Mirrors `uor_foundation::enums::PrimitiveOp` but adds
/// LUT-backed apply methods for O(1) execution.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub enum PrimOp {
    /// neg(x) = (-x) mod 256
    Neg,
    /// bnot(x) = 255 ^ x
    Bnot,
    /// succ(x) = (x + 1) mod 256
    Succ,
    /// pred(x) = (x - 1) mod 256
    Pred,
    /// add(x, y) = (x + y) mod 256
    Add,
    /// sub(x, y) = (x - y) mod 256
    Sub,
    /// mul(x, y) = (x * y) mod 256
    Mul,
    /// xor(x, y) = x ^ y
    Xor,
    /// and(x, y) = x & y
    And,
    /// or(x, y) = x | y
    Or,
}

impl PrimOp {
    /// Arity: 1 for unary, 2 for binary.
    #[inline]
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Neg | Self::Bnot | Self::Succ | Self::Pred => 1,
            Self::Add | Self::Sub | Self::Mul => 2,
            Self::Xor | Self::And | Self::Or => 2,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Bnot => "bnot",
            Self::Succ => "succ",
            Self::Pred => "pred",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Xor => "xor",
            Self::And => "and",
            Self::Or => "or",
        }
    }

    /// Apply a unary primitive operation.
    #[inline]
    #[must_use]
    pub const fn apply_unary(&self, x: u8) -> u8 {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::Succ => x.wrapping_add(1),
            Self::Pred => x.wrapping_sub(1),
            _ => 0, // binary ops: caller error
        }
    }

    /// Apply a binary primitive operation (via LUT).
    #[inline]
    #[must_use]
    pub fn apply_binary(&self, x: u8, y: u8) -> u8 {
        match self {
            Self::Add => arith::add_q0(x, y),
            Self::Sub => arith::sub_q0(x, y),
            Self::Mul => arith::mul_q0(x, y),
            Self::Xor => x ^ y,
            Self::And => x & y,
            Self::Or => x | y,
            _ => 0, // unary ops: caller error
        }
    }

    /// Convert to uor-foundation PrimitiveOp.
    #[inline]
    #[must_use]
    pub const fn to_foundation(&self) -> uor_foundation::enums::PrimitiveOp {
        match self {
            Self::Neg => uor_foundation::enums::PrimitiveOp::Neg,
            Self::Bnot => uor_foundation::enums::PrimitiveOp::Bnot,
            Self::Succ => uor_foundation::enums::PrimitiveOp::Succ,
            Self::Pred => uor_foundation::enums::PrimitiveOp::Pred,
            Self::Add => uor_foundation::enums::PrimitiveOp::Add,
            Self::Sub => uor_foundation::enums::PrimitiveOp::Sub,
            Self::Mul => uor_foundation::enums::PrimitiveOp::Mul,
            Self::Xor => uor_foundation::enums::PrimitiveOp::Xor,
            Self::And => uor_foundation::enums::PrimitiveOp::And,
            Self::Or => uor_foundation::enums::PrimitiveOp::Or,
        }
    }

    /// Convert from uor-foundation PrimitiveOp.
    #[inline]
    #[must_use]
    pub const fn from_foundation(op: uor_foundation::enums::PrimitiveOp) -> Self {
        match op {
            uor_foundation::enums::PrimitiveOp::Neg => Self::Neg,
            uor_foundation::enums::PrimitiveOp::Bnot => Self::Bnot,
            uor_foundation::enums::PrimitiveOp::Succ => Self::Succ,
            uor_foundation::enums::PrimitiveOp::Pred => Self::Pred,
            uor_foundation::enums::PrimitiveOp::Add => Self::Add,
            uor_foundation::enums::PrimitiveOp::Sub => Self::Sub,
            uor_foundation::enums::PrimitiveOp::Mul => Self::Mul,
            uor_foundation::enums::PrimitiveOp::Xor => Self::Xor,
            uor_foundation::enums::PrimitiveOp::And => Self::And,
            uor_foundation::enums::PrimitiveOp::Or => Self::Or,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_neg() {
        assert_eq!(PrimOp::Neg.apply_unary(0), 0);
        assert_eq!(PrimOp::Neg.apply_unary(1), 255);
        assert_eq!(PrimOp::Neg.apply_unary(128), 128);
    }

    #[test]
    fn unary_bnot() {
        assert_eq!(PrimOp::Bnot.apply_unary(0), 255);
        assert_eq!(PrimOp::Bnot.apply_unary(0xAA), 0x55);
    }

    #[test]
    fn unary_succ_pred() {
        for i in 0..=255u8 {
            let s = PrimOp::Succ.apply_unary(i);
            let p = PrimOp::Pred.apply_unary(s);
            assert_eq!(p, i);
        }
    }

    #[test]
    fn binary_add() {
        assert_eq!(PrimOp::Add.apply_binary(100, 200), 44);
    }

    #[test]
    fn binary_xor() {
        assert_eq!(PrimOp::Xor.apply_binary(0xFF, 0x0F), 0xF0);
    }

    #[test]
    fn foundation_round_trip() {
        let ops = [
            PrimOp::Neg,
            PrimOp::Bnot,
            PrimOp::Succ,
            PrimOp::Pred,
            PrimOp::Add,
            PrimOp::Sub,
            PrimOp::Mul,
            PrimOp::Xor,
            PrimOp::And,
            PrimOp::Or,
        ];
        for op in ops {
            let f = op.to_foundation();
            let back = PrimOp::from_foundation(f);
            assert_eq!(back, op);
        }
    }
}
