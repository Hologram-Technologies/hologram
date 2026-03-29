//! Activation conformance tests.
//!
//! Tests that every activation is ring-closed (takes W, returns W),
//! decomposition witness exists, and structural properties hold.

use hologram_ring::activation::ActivationOp;
use hologram_ring::{QuantumLevel, RingWord, Q0, Q1, Q3};

// ── Ring closure: apply takes W, returns W ───────────────────────────────

fn assert_ring_closure<Q: QuantumLevel>()
where
    Q::Word: core::fmt::Debug,
{
    let vals: Vec<Q::Word> = [0u64, 1, 42, 127, 128, 200, 255]
        .iter()
        .map(|&v| Q::Word::from_u64(v))
        .collect();

    let simple_ops = [
        ActivationOp::Relu,
        ActivationOp::Abs,
        ActivationOp::Square,
        ActivationOp::Cube,
        ActivationOp::Sigmoid,
        ActivationOp::Tanh,
    ];

    for op in &simple_ops {
        for &x in &vals {
            let _ = op.apply::<Q>(x); // must not panic; result is Q::Word
        }
    }
}

#[test]
fn ring_closure_q0() {
    assert_ring_closure::<Q0>();
}

#[test]
fn ring_closure_q1() {
    assert_ring_closure::<Q1>();
}

#[test]
fn ring_closure_q3() {
    assert_ring_closure::<Q3>();
}

// ── Square properties ────────────────────────────────────────────────────

#[test]
fn square_known_answers() {
    assert_eq!(ActivationOp::Square.apply::<Q0>(0u8), 0);
    assert_eq!(ActivationOp::Square.apply::<Q0>(1u8), 1);
    assert_eq!(ActivationOp::Square.apply::<Q0>(2u8), 4);
    assert_eq!(ActivationOp::Square.apply::<Q0>(3u8), 9);
    assert_eq!(ActivationOp::Square.apply::<Q0>(15u8), 225);
    // Wrapping: 16*16 = 256 mod 256 = 0
    assert_eq!(ActivationOp::Square.apply::<Q0>(16u8), 0);
}

#[test]
fn square_even_function() {
    // square(neg(x)) == square(x) at Q0
    for x in 0u8..=255 {
        let sq_x = ActivationOp::Square.apply::<Q0>(x);
        let sq_neg = ActivationOp::Square.apply::<Q0>(x.wrapping_neg());
        assert_eq!(sq_x, sq_neg, "square not even at x={x}");
    }
}

#[test]
fn square_q3() {
    assert_eq!(ActivationOp::Square.apply::<Q3>(5u32), 25);
    assert_eq!(ActivationOp::Square.apply::<Q3>(0u32), 0);
    assert_eq!(ActivationOp::Square.apply::<Q3>(1u32), 1);
}

// ── Cube properties ──────────────────────────────────────────────────────

#[test]
fn cube_known_answers() {
    assert_eq!(ActivationOp::Cube.apply::<Q0>(0u8), 0);
    assert_eq!(ActivationOp::Cube.apply::<Q0>(1u8), 1);
    assert_eq!(ActivationOp::Cube.apply::<Q0>(2u8), 8);
    assert_eq!(ActivationOp::Cube.apply::<Q0>(3u8), 27);
    assert_eq!(ActivationOp::Cube.apply::<Q0>(5u8), 125);
}

#[test]
fn cube_q3() {
    assert_eq!(ActivationOp::Cube.apply::<Q3>(10u32), 1000);
}

// ── Relu properties ──────────────────────────────────────────────────────

#[test]
fn relu_zero_at_zero() {
    assert_eq!(ActivationOp::Relu.apply::<Q0>(0u8), 0);
    assert_eq!(ActivationOp::Relu.apply::<Q3>(0u32), 0);
}

#[test]
fn relu_positive_passthrough_q0() {
    // Values 0-127 are "positive" (MSB=0), should pass through
    for x in 0u8..128 {
        assert_eq!(
            ActivationOp::Relu.apply::<Q0>(x),
            x,
            "relu should pass through positive x={x}"
        );
    }
}

#[test]
fn relu_negative_zeroed_q0() {
    // Values 128-255 are "negative" (MSB=1), should be zeroed
    for x in 128u8..=255 {
        assert_eq!(
            ActivationOp::Relu.apply::<Q0>(x),
            0,
            "relu should zero negative x={x}"
        );
    }
}

// ── Abs properties ───────────────────────────────────────────────────────

#[test]
fn abs_idempotent_q0() {
    // abs(abs(x)) == abs(x)
    for x in 0u8..=255 {
        let a = ActivationOp::Abs.apply::<Q0>(x);
        let aa = ActivationOp::Abs.apply::<Q0>(a);
        assert_eq!(a, aa, "abs not idempotent at x={x}");
    }
}

#[test]
fn abs_positive_passthrough_q0() {
    // Positive values (0-127) pass through
    for x in 0u8..128 {
        assert_eq!(ActivationOp::Abs.apply::<Q0>(x), x, "abs positive at x={x}");
    }
}

#[test]
fn abs_negative_negated_q0() {
    // Negative values (128-255) get negated
    for x in 129u8..=255 {
        // neg(x) for x in 129..255 gives values 1..127
        assert_eq!(
            ActivationOp::Abs.apply::<Q0>(x),
            x.wrapping_neg(),
            "abs negative at x={x}"
        );
    }
}

// ── Sigmoid properties ───────────────────────────────────────────────────

#[test]
fn sigmoid_monotonic_q0() {
    let mut prev = ActivationOp::Sigmoid.apply::<Q0>(0u8);
    for x in 1u8..=255 {
        let cur = ActivationOp::Sigmoid.apply::<Q0>(x);
        assert!(
            cur >= prev,
            "sigmoid not monotonic at Q0 x={x}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

#[test]
fn sigmoid_bounded_q0() {
    for x in 0u8..=255 {
        let y = ActivationOp::Sigmoid.apply::<Q0>(x);
        // Output should be in [0, 255]
        let _ = y; // u8 is always <= 255
    }
}

#[test]
fn sigmoid_boundary_q0() {
    // At x=0 (most negative input), sigmoid should be near 0
    let lo = ActivationOp::Sigmoid.apply::<Q0>(0u8);
    assert!(lo < 32, "sigmoid(0) should be near 0, got {lo}");
    // At x=255 (most positive input), sigmoid should be near 255
    let hi = ActivationOp::Sigmoid.apply::<Q0>(255u8);
    assert!(hi > 224, "sigmoid(255) should be near 255, got {hi}");
}

// ── Tanh properties ──────────────────────────────────────────────────────

#[test]
fn tanh_monotonic_q0() {
    let mut prev = ActivationOp::Tanh.apply::<Q0>(0u8);
    for x in 1u8..=255 {
        let cur = ActivationOp::Tanh.apply::<Q0>(x);
        assert!(
            cur >= prev,
            "tanh not monotonic at Q0 x={x}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

#[test]
fn tanh_bounded_q0() {
    for x in 0u8..=255 {
        let y = ActivationOp::Tanh.apply::<Q0>(x);
        let _ = y; // u8 is always <= 255
    }
}

// ── Decomposition witness exists ─────────────────────────────────────────

#[test]
fn decompose_returns_primops() {
    // Simple activations have non-empty decompositions
    assert!(!ActivationOp::Square.decompose().is_empty());
    assert!(!ActivationOp::Cube.decompose().is_empty());
    assert!(!ActivationOp::Relu.decompose().is_empty());
    assert!(!ActivationOp::Abs.decompose().is_empty());
}
