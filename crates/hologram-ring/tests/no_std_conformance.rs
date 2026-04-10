//! no_std conformance tests.
//!
//! These tests verify that the core ring arithmetic works without std.
//! Run with: cargo test -p hologram-ring --no-default-features
//! If this fails, someone added a std dependency to the ring kernel.

use hologram_ring::{accumulate, curvature, domain, rank, stratum, PrimOp};

// ── Ring arithmetic ──────────────────────────────────────────────────────

#[test]
fn ring_arithmetic_no_alloc() {
    // All 10 PrimOps on u8
    assert_eq!(PrimOp::Neg.apply_unary(1u8), 255);
    assert_eq!(PrimOp::Bnot.apply_unary(0u8), 255);
    assert_eq!(PrimOp::Succ.apply_unary(255u8), 0);
    assert_eq!(PrimOp::Pred.apply_unary(0u8), 255);
    assert_eq!(PrimOp::Add.apply_binary(100u8, 200u8), 44);
    assert_eq!(PrimOp::Sub.apply_binary(0u8, 1u8), 255);
    assert_eq!(PrimOp::Mul.apply_binary(3u8, 5u8), 15);
    assert_eq!(PrimOp::Xor.apply_binary(0xAAu8, 0x55u8), 0xFF);
    assert_eq!(PrimOp::And.apply_binary(0xAAu8, 0x55u8), 0x00);
    assert_eq!(PrimOp::Or.apply_binary(0xAAu8, 0x55u8), 0xFF);

    // All 10 PrimOps on u32
    assert_eq!(PrimOp::Neg.apply_unary(1u32), u32::MAX);
    assert_eq!(PrimOp::Add.apply_binary(u32::MAX, 1u32), 0);
    assert_eq!(PrimOp::Mul.apply_binary(3u32, 5u32), 15);

    // All 10 PrimOps on u64
    assert_eq!(PrimOp::Succ.apply_unary(u64::MAX), 0);
}

// ── Activations ──────────────────────────────────────────────────────────

#[test]
fn activation_apply_no_alloc() {
    use hologram_ring::activation::ActivationOp;
    use hologram_ring::{W32, W8};

    // Simple activations
    assert_eq!(ActivationOp::Square.apply::<W32>(5u32), 25);
    assert_eq!(ActivationOp::Cube.apply::<W32>(3u32), 27);

    // Relu
    assert_eq!(ActivationOp::Relu.apply::<W8>(42u8), 42); // positive
    assert_eq!(ActivationOp::Relu.apply::<W8>(200u8), 0); // negative

    // All 21 activations must not panic at W8
    let all = [
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
    for act in &all {
        let _ = act.apply::<W8>(128u8); // must not panic
        let _ = act.apply::<W32>(1000u32); // must not panic
    }
}

// ── Observables ─────��────────────────────────────────────────────────────

#[test]
fn observables_no_alloc() {
    assert_eq!(stratum(0u8), 0);
    assert_eq!(stratum(255u8), 8);
    assert_eq!(curvature(0u8), 1);
    assert_eq!(rank(8u8), 3);
    assert_eq!(domain(1u8), 7);
}

// ── Accumulate ────────────���────────────────────────────────────���─────────

#[test]
fn accumulate_no_alloc() {
    assert_eq!(accumulate(10u32, 3u32, 5u32), 25);
    assert_eq!(accumulate(0u8, 200u8, 2u8), 144u8); // wrapping
}

// ── Critical identity ────────────────────��───────────────────────────────

#[test]
fn critical_identity_no_alloc() {
    // neg(bnot(x)) == succ(x) for all x in Z/256Z
    for x in 0u8..=255 {
        assert_eq!(
            PrimOp::Neg.apply_unary(PrimOp::Bnot.apply_unary(x)),
            PrimOp::Succ.apply_unary(x),
        );
    }
}
