//! PrimOp: the 10 UOR primitive operations, mirroring uor-foundation PrimitiveOp.

use crate::lut::arith;

/// The 10 primitive operations on Z/256Z.
///
/// Mirrors `uor_foundation::enums::PrimitiveOp` but adds
/// LUT-backed apply methods for O(1) execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
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

    /// Returns true if this binary operation is commutative: op(x,y) = op(y,x).
    ///
    /// `Sub` returns false — operand order is semantically significant.
    /// Callers (e.g. CSE) use this to decide whether to sort operands
    /// before forming a signature.
    #[inline]
    #[must_use]
    pub const fn is_commutative_binary(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Xor | Self::And | Self::Or
        )
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

    /// Apply a unary primitive operation in Z/65536Z (Q1 ring).
    ///
    /// No float conversion — exact ring arithmetic on u16 values.
    #[inline]
    #[must_use]
    pub const fn apply_unary_q1(self, x: u16) -> u16 {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::Succ => x.wrapping_add(1),
            Self::Pred => x.wrapping_sub(1),
            _ => 0, // binary ops: caller error
        }
    }

    /// Apply a binary primitive operation in Z/65536Z (Q1 ring).
    ///
    /// No float conversion — uses add_q1/mul_q1 native wrapping ops.
    #[inline]
    #[must_use]
    pub fn apply_binary_q1(self, a: u16, b: u16) -> u16 {
        match self {
            Self::Add => crate::q1::arith::add_q1(a, b),
            Self::Sub => crate::q1::arith::sub_q1(a, b),
            Self::Mul => crate::q1::arith::mul_q1(a, b),
            Self::Xor => a ^ b,
            Self::And => a & b,
            Self::Or => a | b,
            _ => 0, // unary ops: caller error
        }
    }

    /// Apply a unary primitive operation in Z/2^24 Z (Q2 ring).
    #[inline(always)]
    #[must_use]
    pub const fn apply_unary_q2(self, x: u32) -> u32 {
        match self {
            Self::Neg => crate::q2::arith::neg_q2(x),
            Self::Bnot => crate::q2::arith::bnot_q2(x),
            Self::Succ => crate::q2::arith::succ_q2(x),
            Self::Pred => crate::q2::arith::pred_q2(x),
            _ => 0, // binary ops: caller error
        }
    }

    /// Apply a binary primitive operation in Z/2^24 Z (Q2 ring).
    #[inline(always)]
    #[must_use]
    pub fn apply_binary_q2(self, a: u32, b: u32) -> u32 {
        match self {
            Self::Add => crate::q2::arith::add_q2(a, b),
            Self::Sub => crate::q2::arith::sub_q2(a, b),
            Self::Mul => crate::q2::arith::mul_q2(a, b),
            Self::Xor => (a ^ b) & 0x00FF_FFFF,
            Self::And => a & b & 0x00FF_FFFF,
            Self::Or => (a | b) & 0x00FF_FFFF,
            _ => 0, // unary ops: caller error
        }
    }

    /// Apply a unary primitive operation in Z/2^32 Z (Q3 ring).
    #[inline(always)]
    #[must_use]
    pub const fn apply_unary_q3(self, x: u32) -> u32 {
        match self {
            Self::Neg => crate::quantum::q3_neg(x),
            Self::Bnot => !x,
            Self::Succ => x.wrapping_add(1),
            Self::Pred => x.wrapping_sub(1),
            _ => 0, // binary ops: caller error
        }
    }

    /// Apply a binary primitive operation in Z/2^32 Z (Q3 ring).
    #[inline(always)]
    #[must_use]
    pub fn apply_binary_q3(self, a: u32, b: u32) -> u32 {
        match self {
            Self::Add => crate::quantum::q3_add(a, b),
            Self::Sub => crate::quantum::q3_sub(a, b),
            Self::Mul => crate::quantum::q3_mul(a, b),
            Self::Xor => a ^ b,
            Self::And => a & b,
            Self::Or => a | b,
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

use uor_foundation::enums::GeometricCharacter;

static PRIM_NEG: PrimOp = PrimOp::Neg;
static PRIM_BNOT: PrimOp = PrimOp::Bnot;
static PRIM_SUCC: PrimOp = PrimOp::Succ;
static PRIM_PRED: PrimOp = PrimOp::Pred;
static PRIM_ADD: PrimOp = PrimOp::Add;
static PRIM_SUB: PrimOp = PrimOp::Sub;
static PRIM_MUL: PrimOp = PrimOp::Mul;
static PRIM_XOR: PrimOp = PrimOp::Xor;
static PRIM_AND: PrimOp = PrimOp::And;
static PRIM_OR: PrimOp = PrimOp::Or;

impl uor_foundation::kernel::op::Operation<crate::HoloPrimitives> for PrimOp {
    #[inline]
    fn arity(&self) -> u64 {
        (*self).arity() as u64
    }

    #[inline]
    fn has_geometric_character(&self) -> GeometricCharacter {
        self.to_foundation().has_geometric_character()
    }

    type OperationTarget = PrimOp;

    #[inline]
    fn inverse(&self) -> &Self::OperationTarget {
        match self {
            Self::Neg => &PRIM_NEG,
            Self::Bnot => &PRIM_BNOT,
            Self::Succ => &PRIM_PRED,
            Self::Pred => &PRIM_SUCC,
            Self::Add => &PRIM_SUB,
            Self::Sub => &PRIM_ADD,
            Self::Mul => &PRIM_MUL,
            Self::Xor => &PRIM_XOR,
            Self::And => &PRIM_AND,
            Self::Or => &PRIM_OR,
        }
    }

    #[inline]
    fn composed_of(&self) -> &str {
        self.name()
    }
}

impl uor_foundation::kernel::op::BinaryOp<crate::HoloPrimitives> for PrimOp {
    #[inline]
    fn commutative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Xor | Self::And | Self::Or
        )
    }
    #[inline]
    fn associative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Xor | Self::And | Self::Or
        )
    }
    #[inline]
    fn identity(&self) -> i64 {
        match self {
            Self::Add | Self::Sub => 0,
            Self::Mul => 1,
            Self::Xor | Self::Or => 0,
            Self::And => 255,
            _ => 0,
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
    fn unary_q1_neg() {
        assert_eq!(PrimOp::Neg.apply_unary_q1(0), 0);
        assert_eq!(PrimOp::Neg.apply_unary_q1(1), 65535);
        assert_eq!(PrimOp::Neg.apply_unary_q1(32768), 32768);
    }

    #[test]
    fn unary_q1_bnot() {
        assert_eq!(PrimOp::Bnot.apply_unary_q1(0), 65535);
        assert_eq!(PrimOp::Bnot.apply_unary_q1(0x00FF), 0xFF00);
    }

    #[test]
    fn binary_q1_add_wrapping() {
        assert_eq!(PrimOp::Add.apply_binary_q1(65535, 1), 0);
        assert_eq!(PrimOp::Add.apply_binary_q1(100, 200), 300);
    }

    #[test]
    fn binary_q1_mul_wrapping() {
        assert_eq!(PrimOp::Mul.apply_binary_q1(256, 256), 0); // 65536 mod 65536
        assert_eq!(PrimOp::Mul.apply_binary_q1(3, 3), 9);
    }

    #[test]
    fn unary_q2_neg_involution() {
        for x in [0u32, 1, 255, 0xFFFF, 0xFFFFFF] {
            let neg = PrimOp::Neg.apply_unary_q2(x);
            assert_eq!(PrimOp::Neg.apply_unary_q2(neg), x & 0x00FF_FFFF);
        }
    }

    #[test]
    fn binary_q2_add_inverse() {
        for a in [0u32, 1, 255, 65535, 0xFFFFFF] {
            let neg_a = PrimOp::Neg.apply_unary_q2(a);
            assert_eq!(PrimOp::Add.apply_binary_q2(a, neg_a), 0);
        }
    }

    #[test]
    fn unary_q3_neg_involution() {
        for x in [0u32, 1, 127, u32::MAX / 2, u32::MAX] {
            assert_eq!(PrimOp::Neg.apply_unary_q3(PrimOp::Neg.apply_unary_q3(x)), x);
        }
    }

    #[test]
    fn binary_q3_add_wrapping() {
        assert_eq!(PrimOp::Add.apply_binary_q3(u32::MAX, 1), 0);
        assert_eq!(PrimOp::Sub.apply_binary_q3(0, 1), u32::MAX);
        assert_eq!(PrimOp::Mul.apply_binary_q3(2, 3), 6);
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

    #[test]
    fn binary_op_commutativity_flags() {
        use uor_foundation::kernel::op::BinaryOp;
        assert!(PrimOp::Add.commutative());
        assert!(PrimOp::Mul.commutative());
        assert!(PrimOp::Xor.commutative());
        assert!(PrimOp::And.commutative());
        assert!(PrimOp::Or.commutative());
        assert!(!PrimOp::Sub.commutative());
        assert!(PrimOp::Xor.associative());
        assert!(!PrimOp::Sub.associative());
    }
}
