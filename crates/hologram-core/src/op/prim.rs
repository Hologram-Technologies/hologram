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

    // ── Unified ring arithmetic (dynamic precision) ──────────────────────────

    /// Apply a unary ring operation at any byte width (1–8).
    ///
    /// The mask `(1u64 << (byte_width * 8)) - 1` truncates to the ring modulus
    /// Z/(2^(byte_width*8))Z. For byte_width == 8 (Q7), mask is u64::MAX.
    ///
    /// This is the single fast path for all quantum levels Q0–Q7.
    #[inline(always)]
    #[must_use]
    pub fn apply_unary_u64(self, x: u64, byte_width: u8) -> u64 {
        let bits = (byte_width as u32) * 8;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let r = match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::Succ => x.wrapping_add(1),
            Self::Pred => x.wrapping_sub(1),
            _ => 0,
        };
        r & mask
    }

    /// Apply a binary ring operation at any byte width (1–8).
    ///
    /// Uses native wrapping arithmetic on u64. For byte_width < 8, the result
    /// is masked to the ring modulus. This produces identical results to the
    /// per-level functions (which also use wrapping arithmetic + mask).
    #[inline(always)]
    #[must_use]
    pub fn apply_binary_u64(self, a: u64, b: u64, byte_width: u8) -> u64 {
        let bits = (byte_width as u32) * 8;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let r = match self {
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Xor => a ^ b,
            Self::And => a & b,
            Self::Or => a | b,
            _ => 0,
        };
        r & mask
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

    #[inline]
    fn is_ring_op(&self) -> bool {
        true
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
        assert_eq!(PrimOp::Neg.apply_unary_u64(0, 2) as u16, 0);
        assert_eq!(PrimOp::Neg.apply_unary_u64(1, 2) as u16, 65535);
        assert_eq!(PrimOp::Neg.apply_unary_u64(32768, 2) as u16, 32768);
    }

    #[test]
    fn unary_q1_bnot() {
        assert_eq!(PrimOp::Bnot.apply_unary_u64(0, 2) as u16, 65535);
        assert_eq!(PrimOp::Bnot.apply_unary_u64(0x00FF, 2) as u16, 0xFF00);
    }

    #[test]
    fn binary_q1_add_wrapping() {
        assert_eq!(PrimOp::Add.apply_binary_u64(65535, 1, 2) as u16, 0);
        assert_eq!(PrimOp::Add.apply_binary_u64(100, 200, 2) as u16, 300);
    }

    #[test]
    fn binary_q1_mul_wrapping() {
        assert_eq!(PrimOp::Mul.apply_binary_u64(256, 256, 2) as u16, 0); // 65536 mod 65536
        assert_eq!(PrimOp::Mul.apply_binary_u64(3, 3, 2) as u16, 9);
    }

    #[test]
    fn unary_q2_neg_involution() {
        for x in [0u64, 1, 255, 0xFFFF, 0xFFFFFF] {
            let neg = PrimOp::Neg.apply_unary_u64(x, 3);
            assert_eq!(PrimOp::Neg.apply_unary_u64(neg, 3), x & 0x00FF_FFFF);
        }
    }

    #[test]
    fn binary_q2_add_inverse() {
        for a in [0u64, 1, 255, 65535, 0xFFFFFF] {
            let neg_a = PrimOp::Neg.apply_unary_u64(a, 3);
            assert_eq!(PrimOp::Add.apply_binary_u64(a, neg_a, 3), 0);
        }
    }

    #[test]
    fn unary_q3_neg_involution() {
        for x in [0u64, 1, 127, (u32::MAX / 2) as u64, u32::MAX as u64] {
            assert_eq!(
                PrimOp::Neg.apply_unary_u64(PrimOp::Neg.apply_unary_u64(x, 4), 4),
                x
            );
        }
    }

    #[test]
    fn binary_q3_add_wrapping() {
        assert_eq!(
            PrimOp::Add.apply_binary_u64(u32::MAX as u64, 1, 4) as u32,
            0
        );
        assert_eq!(PrimOp::Sub.apply_binary_u64(0, 1, 4) as u32, u32::MAX);
        assert_eq!(PrimOp::Mul.apply_binary_u64(2, 3, 4) as u32, 6);
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

    // ── Unified u64 arithmetic conformance ──────────────────────────────

    #[test]
    fn u64_unary_matches_q0_exhaustive() {
        for x in 0..=255u8 {
            let xw = x as u64;
            assert_eq!(
                PrimOp::Neg.apply_unary_u64(xw, 1) as u8,
                PrimOp::Neg.apply_unary(x)
            );
            assert_eq!(
                PrimOp::Bnot.apply_unary_u64(xw, 1) as u8,
                PrimOp::Bnot.apply_unary(x)
            );
            assert_eq!(
                PrimOp::Succ.apply_unary_u64(xw, 1) as u8,
                PrimOp::Succ.apply_unary(x)
            );
            assert_eq!(
                PrimOp::Pred.apply_unary_u64(xw, 1) as u8,
                PrimOp::Pred.apply_unary(x)
            );
        }
    }

    #[test]
    fn u64_binary_add_matches_q0_exhaustive() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(
                    PrimOp::Add.apply_binary_u64(a as u64, b as u64, 1) as u8,
                    PrimOp::Add.apply_binary(a, b),
                    "Add({a}, {b}) mismatch"
                );
            }
        }
    }

    #[test]
    fn u64_binary_all_ops_q0_spot_check() {
        let pairs = [
            (0u8, 0u8),
            (1, 1),
            (127, 128),
            (200, 100),
            (255, 255),
            (255, 1),
        ];
        for (a, b) in pairs {
            assert_eq!(
                PrimOp::Sub.apply_binary_u64(a as u64, b as u64, 1) as u8,
                PrimOp::Sub.apply_binary(a, b)
            );
            assert_eq!(
                PrimOp::Mul.apply_binary_u64(a as u64, b as u64, 1) as u8,
                PrimOp::Mul.apply_binary(a, b)
            );
            assert_eq!(
                PrimOp::Xor.apply_binary_u64(a as u64, b as u64, 1) as u8,
                PrimOp::Xor.apply_binary(a, b)
            );
            assert_eq!(
                PrimOp::And.apply_binary_u64(a as u64, b as u64, 1) as u8,
                PrimOp::And.apply_binary(a, b)
            );
            assert_eq!(
                PrimOp::Or.apply_binary_u64(a as u64, b as u64, 1) as u8,
                PrimOp::Or.apply_binary(a, b)
            );
        }
    }

    #[test]
    fn u64_binary_q1_spot_check() {
        let pairs = [
            (0u16, 0u16),
            (1, 1),
            (255, 256),
            (60000, 10000),
            (u16::MAX, 1),
        ];
        for (a, b) in pairs {
            // Add wraps at 16 bits
            let sum = PrimOp::Add.apply_binary_u64(a as u64, b as u64, 2) as u16;
            assert_eq!(sum, a.wrapping_add(b), "Q1 Add({a}, {b}) mismatch");
        }
    }

    #[test]
    fn u64_binary_q3_spot_check() {
        let pairs = [(0u32, 0u32), (1, 1), (u32::MAX, 1), (u32::MAX, u32::MAX)];
        for (a, b) in pairs {
            let sum = PrimOp::Add.apply_binary_u64(a as u64, b as u64, 4) as u32;
            assert_eq!(sum, a.wrapping_add(b), "Q3 Add({a}, {b}) mismatch");
        }
    }

    #[test]
    fn u64_critical_identity_all_widths() {
        // neg(bnot(x)) = succ(x) for all x in Z/(2^(w*8))Z
        for width in 1..=8u8 {
            let max_val = if width >= 8 {
                255u64
            } else {
                (1u64 << (width as u64 * 8)) - 1
            };
            // Test boundary values
            for x in [0u64, 1, max_val / 2, max_val - 1, max_val] {
                let lhs =
                    PrimOp::Neg.apply_unary_u64(PrimOp::Bnot.apply_unary_u64(x, width), width);
                let rhs = PrimOp::Succ.apply_unary_u64(x, width);
                assert_eq!(lhs, rhs, "critical identity failed at width={width}, x={x}");
            }
        }
        // Exhaustive at width=1 (Q0)
        for x in 0..=255u64 {
            let lhs = PrimOp::Neg.apply_unary_u64(PrimOp::Bnot.apply_unary_u64(x, 1), 1);
            let rhs = PrimOp::Succ.apply_unary_u64(x, 1);
            assert_eq!(lhs, rhs, "critical identity Q0 exhaustive failed at x={x}");
        }
    }

    #[test]
    fn u64_ring_closure_width8() {
        // Q7 (64-bit): verify wrapping arithmetic at u64 boundaries
        let max = u64::MAX;
        assert_eq!(PrimOp::Add.apply_binary_u64(max, 1, 8), 0);
        assert_eq!(PrimOp::Sub.apply_binary_u64(0, 1, 8), max);
        assert_eq!(PrimOp::Neg.apply_unary_u64(1, 8), max);
        assert_eq!(PrimOp::Bnot.apply_unary_u64(0, 8), max);
    }

    #[test]
    fn u64_performance() {
        let start = std::time::Instant::now();
        let mut acc = 0u64;
        for i in 0..1_000_000u64 {
            acc = PrimOp::Add.apply_binary_u64(acc, i, 4);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "1M apply_binary_u64 calls took {}ms (target < 50ms)",
            elapsed.as_millis()
        );
        let _ = acc; // prevent optimization
    }
}
