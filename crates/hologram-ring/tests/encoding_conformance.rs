//! Encoding conformance tests.
//!
//! Tests round-trip fidelity, monotonicity, range coverage, and
//! boundary exactness for all encodings at multiple quantum levels.

use hologram_ring::encoding::*;
use hologram_ring::{W16, W32, W8};

// ── Unsigned Encoding ────────────────────────────────────────────────────

#[test]
fn unsigned_round_trip_q0() {
    let enc = UnsignedEncoding::<W8>::new();
    // Check boundary values
    assert_eq!(enc.embed(0.0), 0u8);
    assert_eq!(enc.embed(1.0), 255u8);
    // Round-trip fidelity within quantization step (1/255)
    for i in 0..=10 {
        let v = i as f64 / 10.0;
        let rt = enc.lift(enc.embed(v));
        let step = 1.0 / 255.0;
        assert!(
            (rt - v).abs() < step + 1e-10,
            "unsigned W8 round-trip at {v}: got {rt}"
        );
    }
}

#[test]
fn unsigned_round_trip_q1() {
    let enc = UnsignedEncoding::<W16>::new();
    assert_eq!(enc.embed(0.0), 0u16);
    assert_eq!(enc.embed(1.0), 65535u16);
    for i in 0..=20 {
        let v = i as f64 / 20.0;
        let rt = enc.lift(enc.embed(v));
        let step = 1.0 / 65535.0;
        assert!(
            (rt - v).abs() < step + 1e-10,
            "unsigned W16 round-trip at {v}: got {rt}"
        );
    }
}

#[test]
fn unsigned_monotonic_q0() {
    let enc = UnsignedEncoding::<W8>::new();
    let mut prev = enc.embed(0.0);
    for i in 1..=100 {
        let v = i as f64 / 100.0;
        let cur = enc.embed(v);
        assert!(cur >= prev, "unsigned W8 not monotonic at {v}");
        prev = cur;
    }
}

#[test]
fn unsigned_range_coverage_q0() {
    let enc = UnsignedEncoding::<W8>::new();
    assert_eq!(enc.embed(0.0), 0u8);
    assert_eq!(enc.embed(1.0), 255u8);
}

#[test]
fn unsigned_clamp() {
    let enc = UnsignedEncoding::<W8>::new();
    assert_eq!(enc.embed(-1.0), 0u8);
    assert_eq!(enc.embed(2.0), 255u8);
}

// ── Signed Encoding ──────────────────────────────────────────────────────

#[test]
fn signed_round_trip_q0() {
    let enc = SignedEncoding::<W8>::new();
    assert_eq!(enc.embed(-1.0), 0u8);
    assert_eq!(enc.embed(1.0), 255u8);
    // Midpoint: 0.0 → ~128
    let mid = enc.embed(0.0);
    assert!(
        (mid as i16 - 128).unsigned_abs() <= 1,
        "signed W8 midpoint: got {mid}"
    );
    // Round-trip
    for i in 0..=20 {
        let v = (i as f64 / 10.0) - 1.0; // [-1.0, 1.0]
        let rt = enc.lift(enc.embed(v));
        let step = 2.0 / 255.0;
        assert!(
            (rt - v).abs() < step + 1e-10,
            "signed W8 round-trip at {v}: got {rt}"
        );
    }
}

#[test]
fn signed_monotonic_q0() {
    let enc = SignedEncoding::<W8>::new();
    let mut prev = enc.embed(-1.0);
    for i in 1..=100 {
        let v = (i as f64 / 50.0) - 1.0;
        let cur = enc.embed(v);
        assert!(cur >= prev, "signed W8 not monotonic at {v}");
        prev = cur;
    }
}

// ── Angle Encoding ───────────────────────────────────────────────────────

#[test]
fn angle_round_trip_q0() {
    let enc = AngleEncoding::<W8>::new();
    let two_pi = 2.0 * core::f64::consts::PI;
    // 0 → 0
    assert_eq!(enc.embed(0.0), 0u8);
    // Round-trip
    for i in 0..=16 {
        let v = i as f64 / 16.0 * two_pi * 0.99; // avoid exact 2π wrap
        let rt = enc.lift(enc.embed(v));
        let step = two_pi / 256.0;
        assert!(
            (rt - v).abs() < step + 1e-10,
            "angle W8 round-trip at {v}: got {rt}"
        );
    }
}

#[test]
fn angle_wraps_q0() {
    let enc = AngleEncoding::<W8>::new();
    let two_pi = 2.0 * core::f64::consts::PI;
    // 2π and 0 should map to the same value (modular)
    assert_eq!(enc.embed(0.0), enc.embed(two_pi));
    // Negative angles wrap
    let neg = enc.embed(-core::f64::consts::FRAC_PI_2);
    let pos = enc.embed(1.5 * core::f64::consts::PI);
    // Should be close (within 1 step)
    assert!(
        (neg as i16 - pos as i16).unsigned_abs() <= 1,
        "negative angle wrap: {neg} vs {pos}"
    );
}

// ── Raw Encoding ─────────────────────────────────────────────────────────

#[test]
fn raw_identity_q0() {
    let enc = RawEncoding::<W8>::new();
    for i in 0u8..=255 {
        assert_eq!(enc.embed(i as f64), i);
        assert_eq!(enc.lift(i), i as f64);
    }
}

#[test]
fn raw_clamp_q0() {
    let enc = RawEncoding::<W8>::new();
    assert_eq!(enc.embed(-1.0), 0u8);
    assert_eq!(enc.embed(300.0), 255u8);
}

// ── Names ────────────────────────────────────────────────────────────────

#[test]
fn encoding_names() {
    assert_eq!(UnsignedEncoding::<W8>::new().name(), "unsigned");
    assert_eq!(SignedEncoding::<W8>::new().name(), "signed");
    assert_eq!(AngleEncoding::<W8>::new().name(), "angle");
    assert_eq!(RawEncoding::<W8>::new().name(), "raw");
}

// ── Higher quantum levels ────────────────────────────────────────────────

#[test]
fn unsigned_q3_precision() {
    let enc = UnsignedEncoding::<W32>::new();
    assert_eq!(enc.embed(0.0), 0u32);
    assert_eq!(enc.embed(1.0), u32::MAX);
    // W32 has ~10^-9 quantization step
    let v = 0.5;
    let rt = enc.lift(enc.embed(v));
    assert!((rt - v).abs() < 1e-8, "W32 precision: got {rt}");
}

#[test]
fn signed_q1_precision() {
    let enc = SignedEncoding::<W16>::new();
    assert_eq!(enc.embed(-1.0), 0u16);
    assert_eq!(enc.embed(1.0), 65535u16);
    let v = 0.0;
    let rt = enc.lift(enc.embed(v));
    let step = 2.0 / 65535.0;
    assert!(
        (rt - v).abs() < step + 1e-10,
        "W16 signed precision: got {rt}"
    );
}
