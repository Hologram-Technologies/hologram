//! W64 (u64) activation conformance tests.
//!
//! Verifies all 21 activations work correctly at the maximum practical
//! quantum level (64-bit ring Z/2^64Z).

use hologram_ring::activation::ActivationOp;
use hologram_ring::{PrimOp, W64};

const ALL_ACTIVATIONS: &[ActivationOp] = &[
    ActivationOp::Relu,
    ActivationOp::Abs,
    ActivationOp::Square,
    ActivationOp::Cube,
    ActivationOp::Sigmoid,
    ActivationOp::Tanh,
    ActivationOp::Gelu,
    ActivationOp::Silu,
    ActivationOp::Exp,
    ActivationOp::Exp2,
    ActivationOp::Exp10,
    ActivationOp::Log,
    ActivationOp::Log2,
    ActivationOp::Log10,
    ActivationOp::Sin,
    ActivationOp::Cos,
    ActivationOp::Tan,
    ActivationOp::Asin,
    ActivationOp::Acos,
    ActivationOp::Atan,
    ActivationOp::Sqrt,
];

#[test]
fn all_21_activations_no_panic_q7() {
    let test_vals: Vec<u64> = vec![
        0,
        1,
        42,
        255,
        0xFFFF,
        0xFFFF_FFFF,
        u64::MAX / 8,
        u64::MAX / 4,
        u64::MAX / 2,
        u64::MAX / 4 * 3,
        u64::MAX - 1,
        u64::MAX,
    ];
    for &act in ALL_ACTIVATIONS {
        for &x in &test_vals {
            let _ = act.apply::<W64>(x); // must not panic
        }
    }
}

#[test]
fn simple_activations_q7_known_answers() {
    // Square(5) = 25
    assert_eq!(ActivationOp::Square.apply::<W64>(5u64), 25);
    // Cube(3) = 27
    assert_eq!(ActivationOp::Cube.apply::<W64>(3u64), 27);
    // Relu(42) = 42 (positive, below half)
    assert_eq!(ActivationOp::Relu.apply::<W64>(42u64), 42);
    // Relu(negative) = 0
    assert_eq!(ActivationOp::Relu.apply::<W64>(u64::MAX), 0);
    // Abs(42) = 42
    assert_eq!(ActivationOp::Abs.apply::<W64>(42u64), 42);
}

#[test]
fn sqrt_q7_known_answers() {
    // isqrt(0) = 0, isqrt(1) = 1
    assert_eq!(ActivationOp::Sqrt.apply::<W64>(0u64), 0);
    assert_eq!(ActivationOp::Sqrt.apply::<W64>(1u64), 1);
    // isqrt(4) = 2, isqrt(9) = 3, isqrt(100) = 10
    assert_eq!(ActivationOp::Sqrt.apply::<W64>(4u64), 2);
    assert_eq!(ActivationOp::Sqrt.apply::<W64>(9u64), 3);
    assert_eq!(ActivationOp::Sqrt.apply::<W64>(100u64), 10);
    // isqrt(u32::MAX) ~ 65535
    let result = ActivationOp::Sqrt.apply::<W64>(u32::MAX as u64);
    assert!(
        (result as i64 - 65535).abs() <= 1,
        "isqrt(u32::MAX) = {result}, expected ~65535"
    );
}

#[test]
fn sigmoid_monotonic_q7() {
    let mut prev = ActivationOp::Sigmoid.apply::<W64>(0u64);
    for i in 1..=100u64 {
        let x = u64::MAX / 100 * i;
        let cur = ActivationOp::Sigmoid.apply::<W64>(x);
        assert!(
            cur >= prev,
            "sigmoid W64 not monotonic at step {i}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

#[test]
fn critical_identity_q7() {
    let vals: &[u64] = &[0, 1, 127, 255, 0xFFFF, 0xFFFF_FFFF, u64::MAX / 2, u64::MAX];
    for &x in vals {
        let neg_bnot = PrimOp::Neg.apply_unary(PrimOp::Bnot.apply_unary(x));
        let succ = PrimOp::Succ.apply_unary(x);
        assert_eq!(neg_bnot, succ, "critical identity at W64 x={x:#x}");
    }
}

#[test]
fn non_identity_q7() {
    // Each activation must NOT be identity for at least one value
    let test_vals: Vec<u64> = vec![1, 42, 0xFFFF, u64::MAX / 2, u64::MAX - 1];
    for &act in ALL_ACTIVATIONS {
        let mut found = false;
        for &x in &test_vals {
            if act.apply::<W64>(x) != x {
                found = true;
                break;
            }
        }
        assert!(found, "{act:?} is identity at W64 — must be ring-native");
    }
}
