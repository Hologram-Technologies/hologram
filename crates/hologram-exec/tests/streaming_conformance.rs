//! Streaming dynamic dispatch conformance tests.
//!
//! Encodes the contract for carry-driven precision at runtime:
//! - CurvatureFlux in TapeContext tracks carry across ring operations
//! - Flux can be reset at frame boundaries
//! - Precision promotion is automatic when carry accumulates

use hologram_core::carry::CurvatureFlux;
use hologram_core::op::RingLevel;
use hologram_exec::kv::weight_cache::WeightCache;
use hologram_exec::tape::TapeContext;
use std::cell::RefCell;

// ── TapeContext Flux Integration ────────────────────────────────────────────

#[test]
fn tape_context_has_flux() {
    let store = hologram_graph::constant::ConstantStore::new();
    let wc = RefCell::new(WeightCache::default());
    let ctx = TapeContext::new(&store, &[], &wc);
    // Flux starts at zero — Q0 level
    assert_eq!(ctx.flux.get().required_level(), RingLevel::Q0);
}

#[test]
fn tape_context_flux_reset() {
    let store = hologram_graph::constant::ConstantStore::new();
    let wc = RefCell::new(WeightCache::default());
    let ctx = TapeContext::new(&store, &[], &wc);

    // Accumulate carry
    let mut flux = ctx.flux.get();
    flux.accumulate(100, RingLevel::Q2);
    ctx.flux.set(flux);
    assert_eq!(ctx.flux.get().required_level(), RingLevel::Q2);

    // Reset at frame boundary
    ctx.reset_flux();
    assert_eq!(ctx.flux.get().required_level(), RingLevel::Q0);
}

#[test]
fn flux_is_zero_cost_copy() {
    // CurvatureFlux must be Copy for Cell<CurvatureFlux> to work
    let flux = CurvatureFlux::ZERO;
    let copy = flux;
    assert_eq!(flux, copy);

    // Must fit in 2 registers (16 bytes)
    assert!(
        core::mem::size_of::<CurvatureFlux>() <= 16,
        "CurvatureFlux too large: {} bytes",
        core::mem::size_of::<CurvatureFlux>()
    );
}

// ── Streaming Scenarios ─────────────────────────────────────────────────────

#[test]
fn streaming_steady_state_q0() {
    // Smooth input (zero curvature) — flux stays at Q0 forever
    let mut flux = CurvatureFlux::ZERO;
    for _ in 0..10_000 {
        flux.accumulate(0, RingLevel::Q0);
    }
    assert_eq!(
        flux.required_level(),
        RingLevel::Q0,
        "zero curvature must stay at Q0"
    );
}

#[test]
fn streaming_burst_promotes() {
    // High-curvature burst promotes to Q1
    let mut flux = CurvatureFlux::ZERO;
    // 9 bits of Q0 carry → promotes to Q1
    for _ in 0..9 {
        flux.accumulate(1, RingLevel::Q0);
    }
    assert_eq!(
        flux.required_level(),
        RingLevel::Q1,
        "9 bits Q0 carry must promote to Q1"
    );

    // Reset for next frame
    flux.reset();
    assert_eq!(flux.required_level(), RingLevel::Q0);
}

#[test]
fn streaming_q3_promotion() {
    let mut flux = CurvatureFlux::ZERO;
    flux.accumulate(1, RingLevel::Q3);
    assert_eq!(
        flux.required_level(),
        RingLevel::Q3,
        "Q3 carry must promote to Q3"
    );
}

// ── Performance Contract ────────────────────────────────────────────────────

#[cfg(feature = "std")]
#[test]
fn perf_flux_accumulate_throughput() {
    use std::hint::black_box;
    use std::time::Instant;

    let mut flux = CurvatureFlux::ZERO;
    let start = Instant::now();
    for _ in 0..10_000_000 {
        flux.accumulate(black_box(1), black_box(RingLevel::Q0));
        black_box(flux.required_level());
    }
    let elapsed = start.elapsed();
    black_box(flux);
    assert!(
        elapsed.as_millis() < 100,
        "10M accumulate+query took {}ms, budget 100ms (< 10ns/op)",
        elapsed.as_millis()
    );
}
