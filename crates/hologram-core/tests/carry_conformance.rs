//! Carry-preserving precision conformance tests.
//!
//! Encodes the DC_5 carry decomposition protocol and CF_3/CF_4 flux invariants.
//! These tests define the dynamic precision contract: CurvatureFlux must
//! select the correct ring level for any (model, actor) workload.

use hologram_core::carry::{
    lift_q0_to_q1, lift_q0_to_q2, lift_q1_to_q2, lower_q1_to_q0, lower_q2_to_q1, CurvatureFlux,
};
use hologram_core::op::RingLevel;

// ── DC_5: Exact Lift/Lower Round-Trips ──────────────────────────────────────

#[test]
fn lift_lower_round_trip_exhaustive_q0() {
    for x in 0u8..=255 {
        let q1 = lift_q0_to_q1(x);
        let back = lower_q1_to_q0(q1).expect("Q0→Q1→Q0 round-trip must succeed");
        assert_eq!(back, x, "round-trip failed at {x}");
    }
}

#[test]
fn lift_lower_round_trip_q1_q2() {
    for x in (0u16..=u16::MAX).step_by(257) {
        let q2 = lift_q1_to_q2(x);
        let back = lower_q2_to_q1(q2).expect("Q1→Q2→Q1 round-trip must succeed");
        assert_eq!(back, x, "round-trip failed at {x}");
    }
}

#[test]
fn lift_composition_exact() {
    for x in 0u8..=255 {
        let direct = lift_q0_to_q2(x);
        let stepped = lift_q1_to_q2(lift_q0_to_q1(x));
        assert_eq!(direct, stepped, "lift composition not exact at {x}");
    }
}

// ── CF_3/CF_4: Flux Monotonicity ────────────────────────────────────────────

#[test]
fn flux_monotonic() {
    let mut flux = CurvatureFlux::ZERO;
    let mut prev_level = flux.required_level();

    for _ in 0..100 {
        flux.accumulate(1, RingLevel::Q0);
        let level = flux.required_level();
        assert!(
            (level as u8) >= (prev_level as u8),
            "flux level decreased: {prev_level:?} → {level:?}"
        );
        prev_level = level;
    }
}

#[test]
fn flux_promotion_thresholds() {
    // Zero flux → Q0
    assert_eq!(CurvatureFlux::ZERO.required_level(), RingLevel::Q0);

    // 8 bits of Q0 carry → still Q0
    let mut flux = CurvatureFlux::ZERO;
    for _ in 0..8 {
        flux.accumulate(1, RingLevel::Q0);
    }
    assert_eq!(flux.required_level(), RingLevel::Q0);

    // 9 bits of Q0 carry → promotes to Q1
    flux.accumulate(1, RingLevel::Q0);
    assert_eq!(flux.required_level(), RingLevel::Q1);

    // Any Q1 carry → Q1
    let mut flux = CurvatureFlux::ZERO;
    flux.accumulate(1, RingLevel::Q1);
    assert_eq!(flux.required_level(), RingLevel::Q1);

    // Any Q2 carry → Q2
    let mut flux = CurvatureFlux::ZERO;
    flux.accumulate(1, RingLevel::Q2);
    assert_eq!(flux.required_level(), RingLevel::Q2);
}

// ── Zero-Alloc Per-Element Tracking ─────────────────────────────────────────

#[test]
fn flux_copy_and_size() {
    let flux = CurvatureFlux::ZERO;
    let copy = flux;
    assert_eq!(copy.required_level(), flux.required_level());
    assert!(
        core::mem::size_of::<CurvatureFlux>() <= 16,
        "CurvatureFlux too large for per-element tracking: {} bytes",
        core::mem::size_of::<CurvatureFlux>()
    );
}

// ── Model-Actor Headroom ────────────────────────────────────────────────────

#[test]
fn flux_model_actor_headroom_gpt2_small() {
    // GPT-2 small: D=768, L=12. Accumulate 768 units of curvature at Q0.
    let mut flux = CurvatureFlux::ZERO;
    for _ in 0..768 {
        flux.accumulate(1, RingLevel::Q0);
    }
    assert_ne!(
        flux.required_level(),
        RingLevel::Q0,
        "D=768 accumulation must promote past Q0"
    );
}

// ── Q3 Carry Tracking (Phase 2A) ───────────────────────────────────────────

// This test will FAIL until CurvatureFlux is extended with q3_carry field.
// Currently Q3 carry is lumped with Q2 in the accumulate() match arm.
#[test]
fn flux_extends_to_q3() {
    let mut flux = CurvatureFlux::ZERO;
    flux.accumulate(1, RingLevel::Q3);
    // After Phase 2A: Q3 carry must promote to Q3, not Q2
    assert_eq!(
        flux.required_level(),
        RingLevel::Q3,
        "Q3 carry must promote to Q3 (not Q2)"
    );
}

// ── Streaming Reset ─────────────────────────────────────────────────────────

#[test]
fn flux_streaming_reset() {
    let mut flux = CurvatureFlux::ZERO;
    flux.accumulate(100, RingLevel::Q2);
    assert_eq!(flux.required_level(), RingLevel::Q2);
    flux.reset();
    assert_eq!(flux.required_level(), RingLevel::Q0);
}
