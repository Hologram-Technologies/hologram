//! Q3 Octonion & Associator conformance tests.
//!
//! Encodes the algebraic contract for the Q3 (octonion) level:
//! - Cayley-Dickson product is non-commutative
//! - Octonion product is non-associative
//! - Associator is non-zero for imaginary embeddings, zero for real subalgebra
//! - OctonionRing reports correct algebraic properties

use hologram_core::q3::arith::{associator, associator_norm, cd_conj, cd_mul, commutator, oct_mul};

// ── Cayley-Dickson Product ──────────────────────────────────────────────────

#[test]
fn cd_mul_non_commutative() {
    let a = 0x0102_0304u32;
    let b = 0x0506_0708u32;
    assert_ne!(cd_mul(a, b), cd_mul(b, a), "cd_mul must be non-commutative");
}

#[test]
fn commutator_nonzero() {
    let a = 0x0102_0304u32;
    let b = 0x0506_0708u32;
    assert_ne!(commutator(a, b), 0, "commutator must be non-zero");
}

#[test]
fn cd_conj_involution() {
    for x in [0u32, 1, 0x0102_0304, 0xDEAD_BEEF, u32::MAX] {
        assert_eq!(
            cd_conj(cd_conj(x)),
            x,
            "cd_conj must be involutory at {x:#x}"
        );
    }
}

// ── Octonion Non-Associativity ──────────────────────────────────────────────

#[test]
fn oct_mul_non_associative() {
    let a = (0u32, 0x0102_0304);
    let b = (0u32, 0x0506_0708);
    let c = (0u32, 0x090A_0B0C);
    let lhs = oct_mul(oct_mul(a, b), c);
    let rhs = oct_mul(a, oct_mul(b, c));
    assert_ne!(lhs, rhs, "oct_mul must be non-associative");
}

#[test]
fn associator_nonzero_for_imaginary() {
    let (hi, lo) = associator(0x0102_0304, 0x0506_0708, 0x090A_0B0C);
    assert!(
        hi != 0 || lo != 0,
        "associator must be non-zero for imaginary embedding"
    );
}

#[test]
fn associator_zero_for_real_subalgebra() {
    // Embedding as (x, 0) — quaternion subalgebra. Must be associative.
    let a = (0x0001_0000u32, 0);
    let b = (0x0002_0000u32, 0);
    let c = (0x0003_0000u32, 0);
    let lhs = oct_mul(oct_mul(a, b), c);
    let rhs = oct_mul(a, oct_mul(b, c));
    assert_eq!(lhs, rhs, "real subalgebra must be associative");
}

#[test]
fn associator_norm_bounded() {
    let norm = associator_norm(0x0102_0304, 0x0506_0708, 0x090A_0B0C);
    assert!(norm <= 64, "norm must be <= 64");
    assert!(norm > 0, "norm must be > 0 for non-trivial inputs");
}

// ── OctonionRing Algebraic Properties ───────────────────────────────────────

#[test]
fn octonion_ring_properties() {
    use hologram_core::q3::OctonionRing;
    use uor_foundation::kernel::division::NormedDivisionAlgebra;

    let r = OctonionRing;
    assert_eq!(r.algebra_dimension(), 8);
    assert!(!r.is_commutative());
    assert!(!r.is_associative());
    assert_eq!(r.basis_elements(), "{1, e1, e2, e3, e4, e5, e6, e7}");
}

#[test]
fn octonion_ring_quantum_level() {
    use hologram_core::q3::OctonionRing;
    use uor_foundation::kernel::schema::Ring;
    use uor_foundation::WittLevel as QuantumLevel;

    assert_eq!(OctonionRing.at_witt_level(), QuantumLevel::W32);
    assert_eq!(OctonionRing.ring_witt_length(), 32);
    assert_eq!(OctonionRing.modulus(), 4_294_967_296);
}

// ── Cayley-Dickson Chain Complete ────────────────────────────────────────────

#[test]
fn cayley_dickson_chain_complete() {
    use hologram_core::q1::ring::WordRing;
    use hologram_core::q2::ring::TripleRing;
    use hologram_core::ring::ByteRing;
    use uor_foundation::kernel::division::{CayleyDicksonConstruction, NormedDivisionAlgebra};

    // Q0(1) → Q1(2) → Q2(4) → Q3(8)
    assert_eq!(ByteRing.cayley_dickson_source().algebra_dimension(), 1);
    assert_eq!(ByteRing.cayley_dickson_target().algebra_dimension(), 2);
    assert_eq!(WordRing.cayley_dickson_source().algebra_dimension(), 2);
    assert_eq!(WordRing.cayley_dickson_target().algebra_dimension(), 4);
    assert_eq!(TripleRing.cayley_dickson_source().algebra_dimension(), 4);
    assert_eq!(TripleRing.cayley_dickson_target().algebra_dimension(), 8);

    // Full doubling chain
    let dims = [1u64, 2, 4, 8];
    for w in dims.windows(2) {
        assert_eq!(w[1], w[0] * 2, "CD must double at each step");
    }
}

// ── QuadDatum ───────────────────────────────────────────────────────────────

#[test]
fn quad_datum_round_trip() {
    use hologram_core::q3::datum::QuadDatum;
    for x in [0u32, 1, 0xFF, 0xFFFF, 0xDEAD_BEEF, u32::MAX] {
        let d = QuadDatum::new(x);
        assert_eq!(d.value(), x);
        assert_eq!(d.spectrum().len(), 32);
    }
}

// ── Performance ─────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
#[test]
fn perf_cd_mul() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut val = 0x0102_0304u32;
    let start = Instant::now();
    for _ in 0..1_000_000 {
        val = cd_mul(black_box(val), black_box(0x0506_0708));
    }
    let elapsed = start.elapsed();
    black_box(val);
    assert!(
        elapsed.as_millis() < 50,
        "1M cd_mul took {}ms, budget 50ms",
        elapsed.as_millis()
    );
}

#[cfg(feature = "std")]
#[test]
fn perf_associator() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut val = 0u8;
    let start = Instant::now();
    for _ in 0..100_000 {
        val = associator_norm(black_box(0x0102), black_box(0x0304), black_box(0x0506));
    }
    let elapsed = start.elapsed();
    black_box(val);
    assert!(
        elapsed.as_millis() < 50,
        "100K associator_norm took {}ms, budget 50ms",
        elapsed.as_millis()
    );
}
