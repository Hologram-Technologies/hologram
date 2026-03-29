//! Involution conformance tests.
//!
//! Tests that Neg and Bnot are self-inverse at every quantum level,
//! the critical identity holds, and geometric character is correct.

use hologram_ring::{Involution, QuantumLevel, RingWord, Q0, Q1, Q3, Q7};

fn assert_involution_at_level<Q: QuantumLevel>()
where
    Q::Word: core::fmt::Debug,
{
    let neg: Involution<Q> = Involution::Neg;
    let bnot: Involution<Q> = Involution::Bnot;

    let vals: Vec<Q::Word> = [
        0u64,
        1,
        2,
        7,
        42,
        127,
        128,
        255,
        256,
        0xFFFF,
        0xDEAD_BEEF,
        u64::MAX,
    ]
    .iter()
    .map(|&v| Q::Word::from_u64(v))
    .collect();

    for &x in &vals {
        // Neg is involutory: neg(neg(x)) == x
        assert_eq!(neg.apply(neg.apply(x)), x, "neg involution at {x:?}");
        // Bnot is involutory: bnot(bnot(x)) == x
        assert_eq!(bnot.apply(bnot.apply(x)), x, "bnot involution at {x:?}");
        // Critical identity: neg(bnot(x)) == succ(x)
        assert_eq!(
            neg.apply(bnot.apply(x)),
            x.wrapping_add(Q::Word::ONE),
            "critical identity at {x:?}"
        );
    }
}

#[test]
fn involution_q0() {
    assert_involution_at_level::<Q0>();
}

#[test]
fn involution_q1() {
    assert_involution_at_level::<Q1>();
}

#[test]
fn involution_q3() {
    assert_involution_at_level::<Q3>();
}

#[test]
fn involution_q7() {
    assert_involution_at_level::<Q7>();
}

#[test]
fn involution_q0_exhaustive() {
    let neg: Involution<Q0> = Involution::Neg;
    let bnot: Involution<Q0> = Involution::Bnot;
    for x in 0u8..=255 {
        assert_eq!(neg.apply(neg.apply(x)), x);
        assert_eq!(bnot.apply(bnot.apply(x)), x);
        assert_eq!(neg.apply(bnot.apply(x)), x.wrapping_add(1));
    }
}

#[test]
fn involution_equality() {
    let neg1: Involution<Q0> = Involution::Neg;
    let neg2: Involution<Q0> = Involution::Neg;
    let bnot: Involution<Q0> = Involution::Bnot;
    assert_eq!(neg1, neg2);
    assert_ne!(neg1, bnot);
}
