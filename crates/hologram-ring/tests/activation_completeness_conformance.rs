//! Activation completeness conformance tests.
//!
//! Every one of the 21 activations must be ring-native (not identity).
//! No fallbacks. The ring arithmetic IS the computation.

use hologram_ring::activation::ActivationOp;
use hologram_ring::{W32, W8};

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

// ── Non-identity at W8 ──────────────────────────────────────────────────

#[test]
fn all_activations_non_identity_q0() {
    for &act in ALL_ACTIVATIONS {
        let mut found_non_identity = false;
        for x in 1u8..255 {
            if act.apply::<W8>(x) != x {
                found_non_identity = true;
                break;
            }
        }
        assert!(
            found_non_identity,
            "{act:?} is identity at W8 — must be ring-native, not a fallback"
        );
    }
}

// ── Non-identity at W32 ──────────────────────────────────────────────────

#[test]
fn all_activations_non_identity_q3() {
    // Include both positive (small) and negative (large) values to catch
    // activations like Relu that are identity for positive inputs.
    let test_vals: Vec<u32> = vec![1, 2, 7, 42, 100, 0xFFFF, 0xDEAD, 0x8000_0000, 0xFFFF_FFFE];
    for &act in ALL_ACTIVATIONS {
        let mut found_non_identity = false;
        for &x in &test_vals {
            if act.apply::<W32>(x) != x {
                found_non_identity = true;
                break;
            }
        }
        assert!(
            found_non_identity,
            "{act:?} is identity at W32 — must be ring-native, not a fallback"
        );
    }
}

// ── Decompose is non-empty ──────────────────────────────────────────────

#[test]
fn all_activations_decompose_nonempty() {
    for &act in ALL_ACTIVATIONS {
        assert!(
            !act.decompose().is_empty(),
            "{act:?} has empty decomposition — must provide composedOf witness"
        );
    }
}

// ── Silu == x * sigmoid(x) ──────────────────────────────────────────────

#[test]
fn silu_equals_x_times_sigmoid_q3() {
    for x in [1u32, 42, 100, 0x7FFF, 0x7FFF_FFFF] {
        let silu_result = ActivationOp::Silu.apply::<W32>(x);
        let sig = ActivationOp::Sigmoid.apply::<W32>(x);
        let expected = x.wrapping_mul(sig);
        assert_eq!(
            silu_result, expected,
            "Silu({x}) should equal x * Sigmoid(x)"
        );
    }
}

// ── Gelu monotonic in positive half ──────────────────────────────────────

#[test]
fn gelu_monotonic_positive_q0() {
    // For x > 128 (positive half in unsigned encoding), Gelu should be non-decreasing
    let mut prev = ActivationOp::Gelu.apply::<W8>(128u8);
    for x in 129u8..=255 {
        let cur = ActivationOp::Gelu.apply::<W8>(x);
        assert!(
            cur >= prev,
            "Gelu not monotonic at W8 x={x}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

// ── Sqrt structural property ─────────────────────────────────────────────

#[test]
fn sqrt_small_values_q3() {
    // For small perfect squares, sqrt should produce the exact root
    for root in 0u32..=255 {
        let x = root.wrapping_mul(root);
        let result = ActivationOp::Sqrt.apply::<W32>(x);
        // Allow ±1 tolerance since ring sqrt may round
        let diff = result.abs_diff(root);
        assert!(
            diff <= 1,
            "Sqrt({x}) = {result}, expected ~{root}, diff={diff}"
        );
    }
}

// ── Exp is positive ──────────────────────────────────────────────────────

#[test]
fn exp_positive_q0() {
    // Exp output should be > 0 for inputs above the low saturation region.
    // Low saturation boundary is at MAX/4 = 63. Start above that.
    for x in 80u8..=192 {
        let result = ActivationOp::Exp.apply::<W8>(x);
        assert!(result > 0, "Exp({x}) should be positive, got 0");
    }
}

// ── Log monotonic ────────────────────────────────────────────────────────

#[test]
fn log_monotonic_q0() {
    // Log should be monotonically increasing for positive inputs
    let mut prev = ActivationOp::Log.apply::<W8>(1u8);
    for x in 2u8..=255 {
        let cur = ActivationOp::Log.apply::<W8>(x);
        assert!(
            cur >= prev,
            "Log not monotonic at W8 x={x}: {prev} -> {cur}"
        );
        prev = cur;
    }
}

// ── Sin bounded ──────────────────────────────────────────────────────────

#[test]
fn sin_bounded_q0() {
    for x in 0u8..=255 {
        let _ = ActivationOp::Sin.apply::<W8>(x);
        // Just verifying it doesn't panic — boundedness is implicit in u8
    }
}

// ── Cos bounded ──────────────────────────────────────────────────────────

#[test]
fn cos_bounded_q0() {
    for x in 0u8..=255 {
        let _ = ActivationOp::Cos.apply::<W8>(x);
    }
}
