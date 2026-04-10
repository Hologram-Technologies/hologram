//! PrimOp conformance tests.
//!
//! Tests the 10 primitive operations at every quantum level.
//! Critical identity: neg(bnot(x)) == succ(x) — the UOR fundamental relation.

use hologram_ring::{PrimOp, RingWord, WittLevelMarker, W128, W16, W32, W64, W8};

// ── Generic test harness ─────────────────────────────────────────────────

fn sample_values<W: RingWord>() -> Vec<W> {
    let mut vals = vec![W::ZERO, W::ONE, W::MAX];
    for &v in &[
        2u64,
        3,
        7,
        15,
        42,
        127,
        128,
        255,
        256,
        1000,
        0xFFFF,
        0xDEAD_BEEF,
    ] {
        vals.push(W::from_u64(v));
    }
    vals
}

fn assert_primop_at_level<Q: WittLevelMarker>()
where
    Q::Word: core::fmt::Debug,
{
    let vals = sample_values::<Q::Word>();

    // Unary ops
    for &x in &vals {
        // Neg: neg(neg(x)) == x
        assert_eq!(
            PrimOp::Neg.apply_unary(PrimOp::Neg.apply_unary(x)),
            x,
            "neg involution at {x:?}"
        );
        // Bnot: bnot(bnot(x)) == x
        assert_eq!(
            PrimOp::Bnot.apply_unary(PrimOp::Bnot.apply_unary(x)),
            x,
            "bnot involution at {x:?}"
        );
        // Succ/Pred inverse: pred(succ(x)) == x
        assert_eq!(
            PrimOp::Pred.apply_unary(PrimOp::Succ.apply_unary(x)),
            x,
            "succ/pred inverse at {x:?}"
        );
        // Pred/Succ inverse: succ(pred(x)) == x
        assert_eq!(
            PrimOp::Succ.apply_unary(PrimOp::Pred.apply_unary(x)),
            x,
            "pred/succ inverse at {x:?}"
        );
        // Critical identity: neg(bnot(x)) == succ(x)
        assert_eq!(
            PrimOp::Neg.apply_unary(PrimOp::Bnot.apply_unary(x)),
            PrimOp::Succ.apply_unary(x),
            "critical identity at {x:?}"
        );
    }

    // Binary ops
    for &a in &vals {
        for &b in &vals {
            // Add commutativity
            assert_eq!(
                PrimOp::Add.apply_binary(a, b),
                PrimOp::Add.apply_binary(b, a),
                "add commutativity"
            );
            // Mul commutativity
            assert_eq!(
                PrimOp::Mul.apply_binary(a, b),
                PrimOp::Mul.apply_binary(b, a),
                "mul commutativity"
            );
            // Sub/Add inverse: (a + b) - b == a
            assert_eq!(
                PrimOp::Sub.apply_binary(PrimOp::Add.apply_binary(a, b), b),
                a,
                "sub/add inverse"
            );
            // Xor commutativity
            assert_eq!(
                PrimOp::Xor.apply_binary(a, b),
                PrimOp::Xor.apply_binary(b, a),
            );
            // And commutativity
            assert_eq!(
                PrimOp::And.apply_binary(a, b),
                PrimOp::And.apply_binary(b, a),
            );
            // Or commutativity
            assert_eq!(PrimOp::Or.apply_binary(a, b), PrimOp::Or.apply_binary(b, a),);

            // Identity elements
            assert_eq!(
                PrimOp::Add.apply_binary(a, Q::Word::ZERO),
                a,
                "add identity"
            );
            assert_eq!(PrimOp::Mul.apply_binary(a, Q::Word::ONE), a, "mul identity");
            assert_eq!(
                PrimOp::Xor.apply_binary(a, Q::Word::ZERO),
                a,
                "xor identity"
            );
            assert_eq!(PrimOp::Or.apply_binary(a, Q::Word::ZERO), a, "or identity");
            assert_eq!(PrimOp::And.apply_binary(a, Q::Word::MAX), a, "and identity");

            // Associativity
            for &c in &vals {
                assert_eq!(
                    PrimOp::Add.apply_binary(PrimOp::Add.apply_binary(a, b), c),
                    PrimOp::Add.apply_binary(a, PrimOp::Add.apply_binary(b, c)),
                    "add associativity"
                );
                assert_eq!(
                    PrimOp::Mul.apply_binary(PrimOp::Mul.apply_binary(a, b), c),
                    PrimOp::Mul.apply_binary(a, PrimOp::Mul.apply_binary(b, c)),
                    "mul associativity"
                );
                // Distributivity
                assert_eq!(
                    PrimOp::Mul.apply_binary(a, PrimOp::Add.apply_binary(b, c)),
                    PrimOp::Add.apply_binary(
                        PrimOp::Mul.apply_binary(a, b),
                        PrimOp::Mul.apply_binary(a, c)
                    ),
                    "distributivity"
                );
            }
        }
    }
}

// ── Per-level tests ──────────────────────────────────────────────────────

#[test]
fn primop_q0() {
    assert_primop_at_level::<W8>();
}

#[test]
fn primop_q1() {
    assert_primop_at_level::<W16>();
}

#[test]
fn primop_q3() {
    assert_primop_at_level::<W32>();
}

#[test]
fn primop_q7() {
    assert_primop_at_level::<W64>();
}

#[test]
fn primop_q15() {
    assert_primop_at_level::<W128>();
}

// ── Critical identity exhaustive at W8 ──────────────────────────────────

#[test]
fn critical_identity_q0_exhaustive() {
    for x in 0u8..=255 {
        let neg_bnot = PrimOp::Neg.apply_unary(PrimOp::Bnot.apply_unary(x));
        let succ = PrimOp::Succ.apply_unary(x);
        assert_eq!(neg_bnot, succ, "critical identity at W8 x={x}");
    }
}

// ── Known-answer vectors ─────────────────────────────────────────────────

#[test]
fn known_answers_q0() {
    // Neg
    assert_eq!(PrimOp::Neg.apply_unary(0u8), 0);
    assert_eq!(PrimOp::Neg.apply_unary(1u8), 255);
    assert_eq!(PrimOp::Neg.apply_unary(128u8), 128);
    // Bnot
    assert_eq!(PrimOp::Bnot.apply_unary(0u8), 255);
    assert_eq!(PrimOp::Bnot.apply_unary(255u8), 0);
    assert_eq!(PrimOp::Bnot.apply_unary(0xAAu8), 0x55);
    // Succ
    assert_eq!(PrimOp::Succ.apply_unary(0u8), 1);
    assert_eq!(PrimOp::Succ.apply_unary(255u8), 0);
    // Pred
    assert_eq!(PrimOp::Pred.apply_unary(0u8), 255);
    assert_eq!(PrimOp::Pred.apply_unary(1u8), 0);
    // Add
    assert_eq!(PrimOp::Add.apply_binary(100u8, 200u8), 44); // wrapping
    assert_eq!(PrimOp::Add.apply_binary(1u8, 255u8), 0);
    // Sub
    assert_eq!(PrimOp::Sub.apply_binary(0u8, 1u8), 255);
    assert_eq!(PrimOp::Sub.apply_binary(100u8, 50u8), 50);
    // Mul
    assert_eq!(PrimOp::Mul.apply_binary(3u8, 5u8), 15);
    assert_eq!(PrimOp::Mul.apply_binary(16u8, 16u8), 0); // 256 mod 256
                                                         // Xor
    assert_eq!(PrimOp::Xor.apply_binary(0xAAu8, 0x55u8), 0xFF);
    assert_eq!(PrimOp::Xor.apply_binary(0xFFu8, 0xFFu8), 0x00);
    // And
    assert_eq!(PrimOp::And.apply_binary(0xAAu8, 0x55u8), 0x00);
    assert_eq!(PrimOp::And.apply_binary(0xFFu8, 0xAAu8), 0xAA);
    // Or
    assert_eq!(PrimOp::Or.apply_binary(0xAAu8, 0x55u8), 0xFF);
    assert_eq!(PrimOp::Or.apply_binary(0x00u8, 0xAAu8), 0xAA);
}

#[test]
fn known_answers_q3() {
    assert_eq!(PrimOp::Neg.apply_unary(1u32), u32::MAX);
    assert_eq!(PrimOp::Neg.apply_unary(0u32), 0);
    assert_eq!(PrimOp::Bnot.apply_unary(0u32), u32::MAX);
    assert_eq!(PrimOp::Add.apply_binary(u32::MAX, 1u32), 0);
    assert_eq!(PrimOp::Mul.apply_binary(0x10000u32, 0x10000u32), 0); // 2^32 mod 2^32
}

#[test]
fn known_answers_q7() {
    assert_eq!(PrimOp::Neg.apply_unary(1u64), u64::MAX);
    assert_eq!(PrimOp::Succ.apply_unary(u64::MAX), 0);
    assert_eq!(PrimOp::Pred.apply_unary(0u64), u64::MAX);
}

// ── Arity and metadata ──────────────────────────────────────────────────

#[test]
fn arity_correct() {
    assert_eq!(PrimOp::Neg.arity(), 1);
    assert_eq!(PrimOp::Bnot.arity(), 1);
    assert_eq!(PrimOp::Succ.arity(), 1);
    assert_eq!(PrimOp::Pred.arity(), 1);
    assert_eq!(PrimOp::Add.arity(), 2);
    assert_eq!(PrimOp::Sub.arity(), 2);
    assert_eq!(PrimOp::Mul.arity(), 2);
    assert_eq!(PrimOp::Xor.arity(), 2);
    assert_eq!(PrimOp::And.arity(), 2);
    assert_eq!(PrimOp::Or.arity(), 2);
}

#[test]
fn commutativity_correct() {
    assert!(PrimOp::Add.is_commutative());
    assert!(PrimOp::Mul.is_commutative());
    assert!(PrimOp::Xor.is_commutative());
    assert!(PrimOp::And.is_commutative());
    assert!(PrimOp::Or.is_commutative());
    assert!(!PrimOp::Sub.is_commutative());
}

#[test]
fn associativity_correct() {
    assert!(PrimOp::Add.is_associative());
    assert!(PrimOp::Mul.is_associative());
    assert!(PrimOp::Xor.is_associative());
    assert!(PrimOp::And.is_associative());
    assert!(PrimOp::Or.is_associative());
    assert!(!PrimOp::Sub.is_associative());
}

// ── Cross-level embedding ───────────────────────────────────────────────

#[test]
fn cross_level_bitwise_consistency() {
    // For bitwise ops (Xor, And, Or, Bnot), zero-extending a W8 value to W16
    // and applying the op should give the same lower 8 bits.
    for x in 0u8..=255 {
        let x16 = x as u16;
        // Bnot: lower 8 bits of bnot(x16) should be bnot(x8)
        let bnot8 = PrimOp::Bnot.apply_unary(x);
        let bnot16 = PrimOp::Bnot.apply_unary(x16);
        assert_eq!(bnot16 as u8, bnot8, "bnot cross-level at x={x}");
    }
    for x in (0u8..=255).step_by(17) {
        for y in (0u8..=255).step_by(19) {
            let (x16, y16) = (x as u16, y as u16);
            // Xor
            assert_eq!(
                PrimOp::Xor.apply_binary(x16, y16) as u8,
                PrimOp::Xor.apply_binary(x, y)
            );
            // And
            assert_eq!(
                PrimOp::And.apply_binary(x16, y16) as u8,
                PrimOp::And.apply_binary(x, y)
            );
            // Or
            assert_eq!(
                PrimOp::Or.apply_binary(x16, y16) as u8,
                PrimOp::Or.apply_binary(x, y)
            );
        }
    }
}
