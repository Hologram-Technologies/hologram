//! Phase 0 scaffold test: verify that prism-ring compiles and
//! all placeholder types exist with correct structure.

use hologram_ring::*;

// ── RingWord trait exists and is implemented for all carrier types ─────────

#[test]
fn ring_word_u8_exists() {
    let x: u8 = 42;
    assert_eq!(<u8 as RingWord>::ZERO, 0);
    assert_eq!(<u8 as RingWord>::ONE, 1);
    assert_eq!(<u8 as RingWord>::MAX, 255);
    assert_eq!(<u8 as RingWord>::BITS, 8);
    assert_eq!(x.wrapping_add(1), 43);
}

#[test]
fn ring_word_u16_exists() {
    assert_eq!(<u16 as RingWord>::BITS, 16);
    assert_eq!(<u16 as RingWord>::MAX, 65535);
}

#[test]
fn ring_word_u32_exists() {
    assert_eq!(<u32 as RingWord>::BITS, 32);
    assert_eq!(<u32 as RingWord>::MAX, u32::MAX);
}

#[test]
fn ring_word_u64_exists() {
    assert_eq!(<u64 as RingWord>::BITS, 64);
    assert_eq!(<u64 as RingWord>::MAX, u64::MAX);
}

#[test]
fn ring_word_u128_exists() {
    assert_eq!(<u128 as RingWord>::BITS, 128);
    assert_eq!(<u128 as RingWord>::MAX, u128::MAX);
}

// ── WittLevelMarker trait exists with correct constants ──────────────────────

#[test]
fn quantum_level_q0() {
    assert_eq!(W8::BITS, 8);
    assert_eq!(W8::INDEX, 0);
    assert_eq!(<W8 as WittLevelMarker>::BITS, 8 * (W8::INDEX + 1));
    // W8 is a ZST
    assert_eq!(core::mem::size_of::<W8>(), 0);
}

#[test]
fn quantum_level_q1() {
    assert_eq!(W16::BITS, 16);
    assert_eq!(W16::INDEX, 1);
    assert_eq!(<W16 as WittLevelMarker>::BITS, 8 * (W16::INDEX + 1));
    assert_eq!(core::mem::size_of::<W16>(), 0);
}

#[test]
fn quantum_level_q3() {
    assert_eq!(W32::BITS, 32);
    assert_eq!(W32::INDEX, 3);
    assert_eq!(<W32 as WittLevelMarker>::BITS, 8 * (W32::INDEX + 1));
    assert_eq!(core::mem::size_of::<W32>(), 0);
}

#[test]
fn quantum_level_q7() {
    assert_eq!(W64::BITS, 64);
    assert_eq!(W64::INDEX, 7);
    assert_eq!(<W64 as WittLevelMarker>::BITS, 8 * (W64::INDEX + 1));
    assert_eq!(core::mem::size_of::<W64>(), 0);
}

#[test]
fn quantum_level_q15() {
    assert_eq!(W128::BITS, 128);
    assert_eq!(W128::INDEX, 15);
    assert_eq!(<W128 as WittLevelMarker>::BITS, 8 * (W128::INDEX + 1));
    assert_eq!(core::mem::size_of::<W128>(), 0);
}

// ── Word type matches level ──────────────────────────────────────────────

#[test]
fn word_type_matches_level() {
    assert_eq!(<<W8 as WittLevelMarker>::Word as RingWord>::BITS, W8::BITS);
    assert_eq!(
        <<W16 as WittLevelMarker>::Word as RingWord>::BITS,
        W16::BITS
    );
    assert_eq!(
        <<W32 as WittLevelMarker>::Word as RingWord>::BITS,
        W32::BITS
    );
    assert_eq!(
        <<W64 as WittLevelMarker>::Word as RingWord>::BITS,
        W64::BITS
    );
    assert_eq!(
        <<W128 as WittLevelMarker>::Word as RingWord>::BITS,
        W128::BITS
    );
}

// ── PrimOp exists with all 10 variants ───────────────────────────────────

#[test]
fn primop_variants_exist() {
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
    assert_eq!(ops.len(), 10);
    // Unary
    assert_eq!(PrimOp::Neg.arity(), 1);
    assert_eq!(PrimOp::Bnot.arity(), 1);
    assert_eq!(PrimOp::Succ.arity(), 1);
    assert_eq!(PrimOp::Pred.arity(), 1);
    // Binary
    assert_eq!(PrimOp::Add.arity(), 2);
    assert_eq!(PrimOp::Sub.arity(), 2);
    assert_eq!(PrimOp::Mul.arity(), 2);
    assert_eq!(PrimOp::Xor.arity(), 2);
    assert_eq!(PrimOp::And.arity(), 2);
    assert_eq!(PrimOp::Or.arity(), 2);
}

// ── PrimOp apply works generically ──────────────────────────────────────

#[test]
fn primop_apply_generic() {
    // W8 (u8)
    assert_eq!(PrimOp::Neg.apply_unary(1u8), 255);
    assert_eq!(PrimOp::Add.apply_binary(100u8, 200u8), 44); // wrapping
                                                            // W32 (u32)
    assert_eq!(PrimOp::Neg.apply_unary(1u32), u32::MAX);
    assert_eq!(PrimOp::Mul.apply_binary(3u32, 5u32), 15);
    // W64 (u64)
    assert_eq!(PrimOp::Succ.apply_unary(u64::MAX), 0);
}

// ── Critical identity: neg(bnot(x)) == succ(x) ─────────────────────────

#[test]
fn critical_identity_q0_exhaustive() {
    for x in 0u8..=255 {
        let neg_bnot = PrimOp::Neg.apply_unary(PrimOp::Bnot.apply_unary(x));
        let succ = PrimOp::Succ.apply_unary(x);
        assert_eq!(neg_bnot, succ, "Critical identity failed at W8 x={x}");
    }
}

#[test]
fn critical_identity_q7_sampled() {
    let vals: &[u64] = &[0, 1, 127, 255, 0xFFFF, 0xFFFF_FFFF, u64::MAX / 2, u64::MAX];
    for &x in vals {
        let neg_bnot = PrimOp::Neg.apply_unary(PrimOp::Bnot.apply_unary(x));
        let succ = PrimOp::Succ.apply_unary(x);
        assert_eq!(neg_bnot, succ, "Critical identity failed at W64 x={x:#x}");
    }
}

// ── Involution exists ───────────────────────────────────────────────────

#[test]
fn involution_exists() {
    let neg: Involution<W8> = Involution::Neg;
    let bnot: Involution<W8> = Involution::Bnot;
    // Self-inverse
    for x in 0u8..=255 {
        assert_eq!(neg.apply(neg.apply(x)), x);
        assert_eq!(bnot.apply(bnot.apply(x)), x);
    }
}

// ── Observables exist ───────────────────────────────────────────────────

#[test]
fn observables_exist() {
    assert_eq!(stratum(0u8), 0);
    assert_eq!(stratum(255u8), 8);
    assert_eq!(curvature(0u8), 1); // 0 ^ 1 = 1, popcount = 1
    assert_eq!(rank(0u8), 8); // trailing_zeros of 0 is BITS
    assert_eq!(domain(0u8), 8); // leading_zeros of 0 is BITS
}

// ── Accumulate exists ───────────────────────────────────────────────────

#[test]
fn accumulate_exists() {
    assert_eq!(accumulate(0u8, 3, 5), 15);
    assert_eq!(accumulate(10u32, 3, 5), 25);
}

// ── ActivationOp enum exists ────────────────────────────────────────────

#[test]
fn activation_op_exists() {
    use hologram_ring::activation::ActivationOp;
    let _ = ActivationOp::Relu;
    let _ = ActivationOp::Sigmoid;
    let _ = ActivationOp::Gelu;
    let _ = ActivationOp::Silu;
    let _ = ActivationOp::Tanh;
}

// ── PrismPrimitives implements hologram_foundation::Primitives ───────────────

#[test]
fn prism_primitives_exists() {
    use hologram_ring::PrismPrimitives;
    // This test just verifies the type exists and implements the trait.
    // The trait bound is checked at compile time.
    fn assert_primitives<T: hologram_foundation::Primitives>() {}
    assert_primitives::<PrismPrimitives>();
}
