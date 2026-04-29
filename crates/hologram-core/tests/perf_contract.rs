//! Performance contract tests.
//!
//! These tests measure execution time and FAIL if the performance contract
//! is violated. Thresholds include 5× CI headroom to avoid flakiness while
//! catching order-of-magnitude regressions.
//!
//! Every hot-path operation in hologram has a time budget encoded here.
//! If an implementation change causes a test to fail, the implementation
//! must be fixed — the budget is the contract.

#![cfg(feature = "std")]

use std::hint::black_box;
use std::time::Instant;

/// Assert that `op` completes `n` iterations in under `max_ms` milliseconds.
fn assert_throughput<F: FnMut()>(mut op: F, n: usize, max_ms: u64, label: &str) {
    // Warm up
    for _ in 0..100 {
        op();
    }
    let start = Instant::now();
    for _ in 0..n {
        op();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < max_ms as u128,
        "{label}: {n} ops took {}ms, budget {max_ms}ms",
        elapsed.as_millis()
    );
}

// ── Q0 Activation Lookup ────────────────────────────────────────────────────

#[test]
fn perf_q0_activation_lookup() {
    use hologram_core::view::ElementWiseView;
    let view = ElementWiseView::new(|x| x.wrapping_add(1)); // succ
    let mut val = 0u8;
    assert_throughput(
        || val = view.apply(black_box(val)),
        1_000_000,
        10, // 1M lookups in < 10ms = < 10ns/op with margin
        "Q0 activation lookup",
    );
    black_box(val);
}

// ── CurvatureFlux Query ─────────────────────────────────────────────────────

#[test]
fn perf_curvature_flux_query() {
    use hologram_core::carry::CurvatureFlux;
    use hologram_core::op::RingLevel;
    let mut flux = CurvatureFlux::ZERO;
    flux.accumulate(5, RingLevel::Q0);
    let mut level = RingLevel::Q0;
    assert_throughput(
        || level = black_box(flux).required_level(),
        10_000_000,
        50, // 10M queries in < 50ms = < 5ns/query
        "CurvatureFlux::required_level()",
    );
    black_box(level);
}

// ── Carry Lift ──────────────────────────────────────────────────────────────

#[test]
fn perf_carry_lift_q0_q1() {
    use hologram_core::carry::lift;
    use uor_foundation::WittLevel as QuantumLevel;
    let mut val = 0u64;
    assert_throughput(
        || val = lift(black_box(42u64), QuantumLevel::W8, QuantumLevel::W16),
        1_000_000,
        10, // 1M lifts in < 10ms
        "lift",
    );
    black_box(val);
}

// ── View apply_slice ────────────────────────────────────────────────────────

#[test]
fn perf_view_apply_slice_q0_64kb() {
    use hologram_core::view::ElementWiseView;
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    let mut buf = vec![0u8; 65536]; // 64KB
    assert_throughput(
        || view.apply_slice(black_box(&mut buf)),
        100,
        50, // 100 × 64KB in < 50ms = < 500µs each
        "view::apply_slice(64KB)",
    );
    black_box(&buf);
}

// ── Q3 Wrapping Arithmetic ──────────────────────────────────────────────────

#[test]
fn perf_q3_wrapping_arithmetic() {
    use hologram_core::quantum::{q3_add, q3_mul};
    let mut val = 1u32;
    assert_throughput(
        || {
            val = q3_add(black_box(val), black_box(7));
            val = q3_mul(black_box(val), black_box(3));
        },
        1_000_000,
        10, // 2M ops in < 10ms = native u32 speed
        "Q3 wrapping arithmetic",
    );
    black_box(val);
}

// ── CD Multiplication (Phase 2E) ────────────────────────────────────────────

#[test]
fn perf_cd_mul_q3() {
    use hologram_core::q3::arith::cd_mul;
    let mut val = 0x0102_0304u32;
    assert_throughput(
        || val = cd_mul(black_box(val), black_box(0x0506_0708)),
        1_000_000,
        50, // 1M cd_mul in < 50ms = < 50ns/op
        "cd_mul",
    );
    black_box(val);
}

// ── Associator Norm ─────────────────────────────────────────────────────────

#[test]
fn perf_associator_norm() {
    use hologram_core::q3::arith::associator_norm;
    let mut val = 0u8;
    assert_throughput(
        || val = associator_norm(black_box(0x0102), black_box(0x0304), black_box(0x0506)),
        100_000,
        50, // 100K evals in < 50ms = < 500ns/eval
        "associator_norm",
    );
    black_box(val);
}
