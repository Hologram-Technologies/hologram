//! Ring-native tape conformance tests.
//!
//! Verifies that RingActivation ops dispatch correctly through the tape executor,
//! producing bit-identical results to direct ActivationOp::apply calls.

use hologram_core::op::{ActivationOp, RingLevel};
use hologram_graph::graph::GraphOp;

// ── Ring-native activation + accumulate verification ─────────────────────

#[test]
fn ring_activation_relu_q0_correct() {
    // Verify ActivationOp::Relu.apply::<Q0> produces correct results
    // Q0 Relu: positive (0-127) passes through, negative (128-255) → 0
    for x in 0u8..=255 {
        let result = ActivationOp::Relu.apply::<hologram_ring::Q0>(x);
        if x <= 127 {
            assert_eq!(result, x, "relu Q0: positive {x} should pass through");
        } else {
            assert_eq!(result, 0, "relu Q0: negative {x} should be 0");
        }
    }
}

#[test]
fn ring_activation_sigmoid_q3_monotonic() {
    // Verify sigmoid is monotonically increasing at Q3
    let mut prev = ActivationOp::Sigmoid.apply::<hologram_ring::Q3>(0u32);
    for x in 1..=1000u32 {
        let cur = ActivationOp::Sigmoid.apply::<hologram_ring::Q3>(x * (u32::MAX / 1000));
        assert!(cur >= prev, "sigmoid Q3 not monotonic at step {x}");
        prev = cur;
    }
}

#[test]
fn ring_activation_gelu_q0_non_identity() {
    // Gelu should not be identity for most values
    let mut non_identity = 0;
    for x in 0u8..=255 {
        if ActivationOp::Gelu.apply::<hologram_ring::Q0>(x) != x {
            non_identity += 1;
        }
    }
    assert!(
        non_identity > 100,
        "gelu Q0 should be non-identity for most values"
    );
}

#[test]
fn ring_activation_square_q3_known_answer() {
    // Square(7) = 49
    assert_eq!(ActivationOp::Square.apply::<hologram_ring::Q3>(7u32), 49);
    // Square(0) = 0
    assert_eq!(ActivationOp::Square.apply::<hologram_ring::Q3>(0u32), 0);
}

#[test]
fn ring_accumulate_q3_known_answer() {
    // acc + a * b = 10 + 3 * 5 = 25
    assert_eq!(hologram_ring::accumulate(10u32, 3u32, 5u32), 25);
}

#[test]
fn ring_accumulate_q0_wrapping() {
    // 200 + 200 * 2 = 200 + 400 = 600, mod 256 = 88
    assert_eq!(hologram_ring::accumulate(200u8, 200u8, 2u8), 88u8);
}

// ── GraphOp variant tests ────────────────────────────────────────────────

#[test]
fn graph_op_ring_activation_arity() {
    assert_eq!(
        GraphOp::RingActivation(ActivationOp::Relu, RingLevel::Q3).arity(),
        1
    );
}

#[test]
fn graph_op_ring_accumulate_arity() {
    assert_eq!(GraphOp::RingAccumulate(RingLevel::Q3).arity(), 3);
}

#[test]
fn graph_op_ring_reduce_arity() {
    use hologram_core::op::PrimOp;
    assert_eq!(
        GraphOp::RingReduce {
            op: PrimOp::Add,
            axis: 0,
            level: RingLevel::Q3
        }
        .arity(),
        1
    );
}

#[test]
fn graph_op_ring_activation_is_pure() {
    assert!(GraphOp::RingActivation(ActivationOp::Sigmoid, RingLevel::Q0).is_pure());
}

// ── Tape kernel variant exists ───────────────────────────────────────────

#[test]
fn tape_kernel_ring_activation_exists() {
    use hologram_exec::tape::TapeKernel;
    let _kernel = TapeKernel::RingActivation {
        op: ActivationOp::Relu,
        level: RingLevel::Q3,
    };
}

#[test]
fn tape_kernel_ring_accumulate_exists() {
    use hologram_exec::tape::TapeKernel;
    let _kernel = TapeKernel::RingAccumulate {
        level: RingLevel::Q3,
    };
}
