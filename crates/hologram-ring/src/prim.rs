//! PrimOp: the 10 UOR primitive operations, generic over RingWord.
//!
//! Every method compiles to 1-2 ALU instructions at any quantum level.
//! No LUT tables. No external calls.

use crate::word::RingWord;
use crate::PrismPrimitives;
use uor_foundation::enums::GeometricCharacter;

/// The 10 primitive operations on Z/(2^n)Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimOp {
    /// neg(x) = (-x) mod 2^n
    Neg,
    /// bnot(x) = bitwise NOT
    Bnot,
    /// succ(x) = (x + 1) mod 2^n
    Succ,
    /// pred(x) = (x - 1) mod 2^n
    Pred,
    /// add(x, y) = (x + y) mod 2^n
    Add,
    /// sub(x, y) = (x - y) mod 2^n
    Sub,
    /// mul(x, y) = (x * y) mod 2^n
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
            Self::Add | Self::Sub | Self::Mul | Self::Xor | Self::And | Self::Or => 2,
        }
    }

    /// Returns true if this binary operation is commutative.
    #[inline]
    #[must_use]
    pub const fn is_commutative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Xor | Self::And | Self::Or
        )
    }

    /// Returns true if this binary operation is associative.
    #[inline]
    #[must_use]
    pub const fn is_associative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Xor | Self::And | Self::Or
        )
    }

    /// Human-readable name.
    #[inline]
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

    /// Apply a unary primitive operation. Generic over any RingWord.
    #[inline]
    #[must_use]
    pub fn apply_unary<W: RingWord>(&self, x: W) -> W {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::Succ => x.wrapping_add(W::ONE),
            Self::Pred => x.wrapping_sub(W::ONE),
            _ => W::ZERO, // binary ops: caller error
        }
    }

    /// Apply a binary primitive operation. Generic over any RingWord.
    #[inline]
    #[must_use]
    pub fn apply_binary<W: RingWord>(&self, a: W, b: W) -> W {
        match self {
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Xor => a ^ b,
            Self::And => a & b,
            Self::Or => a | b,
            _ => W::ZERO, // unary ops: caller error
        }
    }
}

// ── UOR Operation + BinaryOp traits ──────────────────────────────────────

static PRIMOP_STATICS: [PrimOp; 10] = [
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

impl uor_foundation::kernel::op::Operation<PrismPrimitives> for PrimOp {
    fn arity(&self) -> u64 {
        PrimOp::arity(self) as u64
    }

    fn has_geometric_character(&self) -> GeometricCharacter {
        match self {
            Self::Neg => GeometricCharacter::RingReflection,
            Self::Bnot => GeometricCharacter::HypercubeReflection,
            _ => GeometricCharacter::RingReflection, // default
        }
    }

    type OperationTarget = PrimOp;

    fn inverse(&self) -> &Self::OperationTarget {
        &PRIMOP_STATICS[*self as usize]
    }

    fn composed_of(&self) -> &str {
        self.name()
    }

    /// Per the 0.3.0 ontology: every PrimOp participates in the
    /// Z/(2^n)Z ring-arithmetic vocabulary. Drives Lean RingOp class
    /// generation in UOR/Enforcement.lean.
    fn is_ring_op(&self) -> bool {
        true
    }
}

impl uor_foundation::kernel::op::BinaryOp<PrismPrimitives> for PrimOp {
    fn commutative(&self) -> bool {
        self.is_commutative()
    }

    fn associative(&self) -> bool {
        self.is_associative()
    }

    fn identity(&self) -> i64 {
        match self {
            Self::Add | Self::Sub | Self::Xor | Self::Or => 0,
            Self::Mul => 1,
            Self::And => -1, // MAX as signed
            _ => 0,
        }
    }
}
