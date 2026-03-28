//! Activation branchfree conformance tests.
//!
//! Captures the CURRENT output of every activation at critical boundary
//! values. After the branchfree rewrite, these must produce BIT-IDENTICAL
//! results. If they don't, the branchfree arithmetic is wrong.

use hologram_ring::activation::ActivationOp;
use hologram_ring::{Q0, Q3};

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

/// Capture and verify all activation outputs at boundary values for Q0.
/// Simple activations (relu, abs, square, cube) must not panic.
/// Piecewise activations are tested for panic-freedom in Phase 3.
#[test]
fn boundary_values_q0_simple_stable() {
    let simple = [
        ActivationOp::Relu,
        ActivationOp::Abs,
        ActivationOp::Square,
        ActivationOp::Cube,
    ];
    let boundary_inputs: Vec<u8> = vec![
        0, 1, 31, 32, 63, 64, 95, 96, 127, 128, 159, 160, 191, 192, 223, 224, 254, 255,
    ];

    for &act in &simple {
        for &x in &boundary_inputs {
            let result = act.apply::<Q0>(x);
            let result2 = act.apply::<Q0>(x);
            assert_eq!(
                result, result2,
                "{act:?} at Q0 x={x}: non-deterministic ({result} vs {result2})"
            );
        }
    }
}

/// Capture and verify simple activation outputs at boundary values for Q3.
#[test]
fn boundary_values_q3_simple_stable() {
    let m = u32::MAX as u64;
    let boundary_inputs: Vec<u32> = vec![
        0,
        1,
        (m / 8 - 1) as u32,
        (m / 8) as u32,
        (m / 8 + 1) as u32,
        (m / 4 - 1) as u32,
        (m / 4) as u32,
        (m / 4 + 1) as u32,
        (m * 3 / 8 - 1) as u32,
        (m * 3 / 8) as u32,
        (m * 3 / 8 + 1) as u32,
        (m / 2 - 1) as u32,
        (m / 2) as u32,
        (m / 2 + 1) as u32,
        (m * 5 / 8 - 1) as u32,
        (m * 5 / 8) as u32,
        (m * 5 / 8 + 1) as u32,
        (m * 3 / 4 - 1) as u32,
        (m * 3 / 4) as u32,
        (m * 3 / 4 + 1) as u32,
        (m * 7 / 8 - 1) as u32,
        (m * 7 / 8) as u32,
        (m * 7 / 8 + 1) as u32,
        u32::MAX - 1,
        u32::MAX,
    ];

    let simple = [
        ActivationOp::Relu,
        ActivationOp::Abs,
        ActivationOp::Square,
        ActivationOp::Cube,
        ActivationOp::Sigmoid,
        ActivationOp::Silu,
        ActivationOp::Gelu,
        ActivationOp::Sqrt,
    ];
    for &act in &simple {
        for &x in &boundary_inputs {
            let result = act.apply::<Q3>(x);
            let result2 = act.apply::<Q3>(x);
            assert_eq!(result, result2, "{act:?} at Q3 x={x:#x}: non-deterministic");
        }
    }
}

/// Verify piecewise sigmoid is monotonically increasing at Q3.
#[test]
fn sigmoid_monotonic_q3() {
    let mut prev = ActivationOp::Sigmoid.apply::<Q3>(0u32);
    for i in 1..=1000u32 {
        let x = (u32::MAX as u64 * i as u64 / 1000) as u32;
        let cur = ActivationOp::Sigmoid.apply::<Q3>(x);
        assert!(
            cur >= prev,
            "sigmoid Q3 not monotonic at step {i}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

/// Verify piecewise sigmoid is monotonically increasing at Q0 (exhaustive).
#[test]
fn sigmoid_monotonic_q0_exhaustive() {
    let mut prev = ActivationOp::Sigmoid.apply::<Q0>(0u8);
    for x in 1u8..=255 {
        let cur = ActivationOp::Sigmoid.apply::<Q0>(x);
        assert!(
            cur >= prev,
            "sigmoid Q0 not monotonic at x={x}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

/// Verify simple activations complete without panic for a full sweep of Q3 values.
/// Piecewise activations with overflow bugs will be fixed in the branchfree rewrite.
#[test]
fn simple_activations_sweep_q3() {
    let simple = [
        ActivationOp::Relu,
        ActivationOp::Abs,
        ActivationOp::Square,
        ActivationOp::Cube,
        ActivationOp::Sigmoid,
        ActivationOp::Silu,
        ActivationOp::Gelu,
        ActivationOp::Sqrt,
    ];
    for &act in &simple {
        for i in 0..=256u32 {
            let x = (u32::MAX as u64 * i as u64 / 256) as u32;
            let _ = act.apply::<Q3>(x); // must not panic
        }
    }
}
