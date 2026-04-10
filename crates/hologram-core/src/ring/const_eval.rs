//! Compile-time ring evaluation via uor-foundation enforcement layer.
//!
//! Re-exports `const fn` evaluators and provides hologram-native wrappers
//! dispatched by `RingLevel`. All functions are `#[inline]` and zero-allocation.

pub use hologram_foundation::enforcement::{
    const_ring_eval_q0, const_ring_eval_q1, const_ring_eval_q3, const_ring_eval_q7,
    const_ring_eval_unary_q0, const_ring_eval_unary_q1, const_ring_eval_unary_q3,
    const_ring_eval_unary_q7,
};

use crate::op::RingLevel;
use hologram_foundation::enums::PrimitiveOp;

/// Evaluate a binary ring operation dispatched by level.
/// O(1), zero allocation.
#[inline]
pub fn eval_binary(level: RingLevel, op: PrimitiveOp, a: u64, b: u64) -> u64 {
    match level {
        RingLevel::Q0 => const_ring_eval_q0(op, a as u8, b as u8) as u64,
        RingLevel::Q1 => const_ring_eval_q1(op, a as u16, b as u16) as u64,
        RingLevel::Q2 => {
            // Q2 uses Q3 evaluator with 24-bit mask
            (const_ring_eval_q3(op, (a as u32) & 0x00FF_FFFF, (b as u32) & 0x00FF_FFFF) as u64)
                & 0x00FF_FFFF
        }
        RingLevel::Q3 => const_ring_eval_q3(op, a as u32, b as u32) as u64,
    }
}

/// Evaluate a unary ring operation dispatched by level.
/// O(1), zero allocation.
#[inline]
pub fn eval_unary(level: RingLevel, op: PrimitiveOp, a: u64) -> u64 {
    match level {
        RingLevel::Q0 => const_ring_eval_unary_q0(op, a as u8) as u64,
        RingLevel::Q1 => const_ring_eval_unary_q1(op, a as u16) as u64,
        RingLevel::Q2 => {
            (const_ring_eval_unary_q3(op, (a as u32) & 0x00FF_FFFF) as u64) & 0x00FF_FFFF
        }
        RingLevel::Q3 => const_ring_eval_unary_q3(op, a as u32) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_eval_q0_add_exhaustive() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(
                    const_ring_eval_q0(PrimitiveOp::Add, a, b),
                    a.wrapping_add(b),
                    "Q0 Add({a}, {b}) mismatch"
                );
            }
        }
    }

    #[test]
    fn const_eval_q0_mul_exhaustive() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(
                    const_ring_eval_q0(PrimitiveOp::Mul, a, b),
                    a.wrapping_mul(b),
                );
            }
        }
    }

    #[test]
    fn const_eval_q0_all_binary_ops_exhaustive() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(
                    const_ring_eval_q0(PrimitiveOp::Sub, a, b),
                    a.wrapping_sub(b)
                );
                assert_eq!(const_ring_eval_q0(PrimitiveOp::Xor, a, b), a ^ b);
                assert_eq!(const_ring_eval_q0(PrimitiveOp::And, a, b), a & b);
                assert_eq!(const_ring_eval_q0(PrimitiveOp::Or, a, b), a | b);
            }
        }
    }

    #[test]
    fn const_eval_q0_unary_exhaustive() {
        for a in 0..=255u8 {
            assert_eq!(
                const_ring_eval_unary_q0(PrimitiveOp::Neg, a),
                a.wrapping_neg()
            );
            assert_eq!(const_ring_eval_unary_q0(PrimitiveOp::Bnot, a), !a);
            assert_eq!(
                const_ring_eval_unary_q0(PrimitiveOp::Succ, a),
                a.wrapping_add(1)
            );
            assert_eq!(
                const_ring_eval_unary_q0(PrimitiveOp::Pred, a),
                a.wrapping_sub(1)
            );
        }
    }

    #[test]
    fn const_eval_critical_identity_q0_exhaustive() {
        // neg(bnot(x)) = succ(x) for all x in Z/256Z
        for x in 0..=255u8 {
            let lhs = const_ring_eval_unary_q0(
                PrimitiveOp::Neg,
                const_ring_eval_unary_q0(PrimitiveOp::Bnot, x),
            );
            let rhs = const_ring_eval_unary_q0(PrimitiveOp::Succ, x);
            assert_eq!(lhs, rhs, "critical identity failed at x={x}");
        }
    }

    #[test]
    fn const_eval_q3_spot_check() {
        assert_eq!(const_ring_eval_q3(PrimitiveOp::Add, 0, 0), 0);
        assert_eq!(const_ring_eval_q3(PrimitiveOp::Add, u32::MAX, 1), 0);
        assert_eq!(
            const_ring_eval_q3(PrimitiveOp::Mul, 2, u32::MAX),
            u32::MAX - 1
        );
        assert_eq!(const_ring_eval_unary_q3(PrimitiveOp::Neg, 0), 0);
        assert_eq!(const_ring_eval_unary_q3(PrimitiveOp::Neg, 1), u32::MAX);
    }

    #[test]
    fn eval_binary_dispatches_correctly() {
        assert_eq!(eval_binary(RingLevel::Q0, PrimitiveOp::Add, 200, 100), 44); // 300 mod 256
        assert_eq!(
            eval_binary(RingLevel::Q1, PrimitiveOp::Add, 60000, 10000),
            4464
        ); // 70000 mod 65536
        assert_eq!(
            eval_binary(RingLevel::Q3, PrimitiveOp::Add, u32::MAX as u64, 1),
            0
        );
    }

    #[test]
    fn eval_unary_dispatches_correctly() {
        assert_eq!(eval_unary(RingLevel::Q0, PrimitiveOp::Neg, 1), 255);
        assert_eq!(eval_unary(RingLevel::Q1, PrimitiveOp::Neg, 1), 65535);
        assert_eq!(
            eval_unary(RingLevel::Q3, PrimitiveOp::Neg, 1),
            u32::MAX as u64
        );
    }

    #[test]
    fn const_eval_performance() {
        let start = std::time::Instant::now();
        let mut acc = 0u8;
        for i in 0..1_000_000u32 {
            acc = const_ring_eval_q0(PrimitiveOp::Add, acc, (i & 0xFF) as u8);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "1M const evals took {}ms (target < 50ms)",
            elapsed.as_millis()
        );
        let _ = acc;
    }
}
