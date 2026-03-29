#![allow(clippy::eq_op)]
//! Ring word algebraic conformance tests.
//!
//! These tests encode the algebraic contract: the properties that every
//! RingWord implementation MUST satisfy. Every carrier type (u8–u128)
//! must uphold these invariants.
//!
//! Test tiers:
//! 1. Ring axioms: closure, associativity, commutativity, identity, inverse
//! 2. Distributivity: mul distributes over add
//! 3. Constants: ZERO, ONE, MAX
//! 4. Bit intrinsics: count_ones, leading_zeros, trailing_zeros

use hologram_ring::RingWord;

// ── Generic test harness ─────────────────────────────────────────────────

/// Sample values for a given RingWord type. Returns a representative set
/// including boundary values and mid-range values.
fn sample_values<W: RingWord>() -> Vec<W> {
    let mut vals = vec![W::ZERO, W::ONE, W::MAX];
    // Add some mid-range values via from_u64
    for &v in &[2u64, 3, 7, 15, 42, 127, 128, 255, 256, 1000, 0xFFFF, 0xDEAD] {
        vals.push(W::from_u64(v));
    }
    vals
}

fn assert_ring_axioms<W: RingWord + core::fmt::Debug>() {
    let vals = sample_values::<W>();

    for &a in &vals {
        // Additive identity
        assert_eq!(a.wrapping_add(W::ZERO), a, "add identity failed for {a:?}");
        // Multiplicative identity
        assert_eq!(a.wrapping_mul(W::ONE), a, "mul identity failed for {a:?}");
        // Additive inverse
        assert_eq!(
            a.wrapping_add(a.wrapping_neg()),
            W::ZERO,
            "add inverse failed for {a:?}"
        );
        // Neg is involutory
        assert_eq!(
            a.wrapping_neg().wrapping_neg(),
            a,
            "neg involution failed for {a:?}"
        );
        // Not is involutory
        assert_eq!(!!a, a, "not involution failed for {a:?}");

        for &b in &vals {
            // Add commutativity
            assert_eq!(
                a.wrapping_add(b),
                b.wrapping_add(a),
                "add commutativity failed for {a:?}, {b:?}"
            );
            // Mul commutativity
            assert_eq!(
                a.wrapping_mul(b),
                b.wrapping_mul(a),
                "mul commutativity failed for {a:?}, {b:?}"
            );
            // Sub/Add inverse
            assert_eq!(
                a.wrapping_add(b).wrapping_sub(b),
                a,
                "sub/add inverse failed for {a:?}, {b:?}"
            );
            // XOR commutativity
            assert_eq!(a ^ b, b ^ a, "xor commutativity failed");
            // AND commutativity
            assert_eq!(a & b, b & a, "and commutativity failed");
            // OR commutativity
            assert_eq!(a | b, b | a, "or commutativity failed");

            for &c in &vals {
                // Add associativity
                assert_eq!(
                    a.wrapping_add(b).wrapping_add(c),
                    a.wrapping_add(b.wrapping_add(c)),
                    "add associativity failed"
                );
                // Mul associativity
                assert_eq!(
                    a.wrapping_mul(b).wrapping_mul(c),
                    a.wrapping_mul(b.wrapping_mul(c)),
                    "mul associativity failed"
                );
                // Distributivity: a * (b + c) == a*b + a*c
                assert_eq!(
                    a.wrapping_mul(b.wrapping_add(c)),
                    a.wrapping_mul(b).wrapping_add(a.wrapping_mul(c)),
                    "distributivity failed"
                );
            }
        }
    }
}

// ── u8: exhaustive for critical properties ───────────────────────────────

#[test]
fn u8_ring_axioms() {
    assert_ring_axioms::<u8>();
}

#[test]
fn u8_add_associative_exhaustive() {
    for a in (0u8..=255).step_by(17) {
        for b in (0u8..=255).step_by(19) {
            for c in (0u8..=255).step_by(23) {
                let lhs = a.wrapping_add(b).wrapping_add(c);
                let rhs = a.wrapping_add(b.wrapping_add(c));
                assert_eq!(lhs, rhs, "u8 add not associative at ({a},{b},{c})");
            }
        }
    }
}

#[test]
fn u8_mul_associative_exhaustive() {
    for a in (0u8..=255).step_by(17) {
        for b in (0u8..=255).step_by(19) {
            for c in (0u8..=255).step_by(23) {
                let lhs = a.wrapping_mul(b).wrapping_mul(c);
                let rhs = a.wrapping_mul(b.wrapping_mul(c));
                assert_eq!(lhs, rhs, "u8 mul not associative at ({a},{b},{c})");
            }
        }
    }
}

#[test]
fn u8_additive_inverse_exhaustive() {
    for x in 0u8..=255 {
        assert_eq!(x.wrapping_add(x.wrapping_neg()), 0);
    }
}

#[test]
fn u8_identities_exhaustive() {
    for x in 0u8..=255 {
        assert_eq!(x.wrapping_add(0), x);
        assert_eq!(x.wrapping_mul(1), x);
    }
}

#[test]
fn u8_distributivity_exhaustive() {
    for a in (0u8..=255).step_by(17) {
        for b in (0u8..=255).step_by(19) {
            for c in (0u8..=255).step_by(23) {
                assert_eq!(
                    a.wrapping_mul(b.wrapping_add(c)),
                    a.wrapping_mul(b).wrapping_add(a.wrapping_mul(c)),
                    "u8 distributivity at ({a},{b},{c})"
                );
            }
        }
    }
}

// ── u16 ──────────────────────────────────────────────────────────────────

#[test]
fn u16_ring_axioms() {
    assert_ring_axioms::<u16>();
}

// ── u32 ──────────────────────────────────────────────────────────────────

#[test]
fn u32_ring_axioms() {
    assert_ring_axioms::<u32>();
}

// ── u64 ──────────────────────────────────────────────────────────────────

#[test]
fn u64_ring_axioms() {
    assert_ring_axioms::<u64>();
}

// ── u128 ─────────────────────────────────────────────────────────────────

#[test]
fn u128_ring_axioms() {
    assert_ring_axioms::<u128>();
}

// ── Constants ────────────────────────────────────────────────────────────

#[test]
fn constants_correct() {
    assert_eq!(<u8 as RingWord>::ZERO, 0u8);
    assert_eq!(<u8 as RingWord>::ONE, 1u8);
    assert_eq!(<u8 as RingWord>::MAX, 255u8);
    assert_eq!(<u8 as RingWord>::BITS, 8);

    assert_eq!(<u16 as RingWord>::ZERO, 0u16);
    assert_eq!(<u16 as RingWord>::ONE, 1u16);
    assert_eq!(<u16 as RingWord>::MAX, 65535u16);
    assert_eq!(<u16 as RingWord>::BITS, 16);

    assert_eq!(<u32 as RingWord>::ZERO, 0u32);
    assert_eq!(<u32 as RingWord>::ONE, 1u32);
    assert_eq!(<u32 as RingWord>::MAX, u32::MAX);
    assert_eq!(<u32 as RingWord>::BITS, 32);

    assert_eq!(<u64 as RingWord>::ZERO, 0u64);
    assert_eq!(<u64 as RingWord>::ONE, 1u64);
    assert_eq!(<u64 as RingWord>::MAX, u64::MAX);
    assert_eq!(<u64 as RingWord>::BITS, 64);

    assert_eq!(<u128 as RingWord>::ZERO, 0u128);
    assert_eq!(<u128 as RingWord>::ONE, 1u128);
    assert_eq!(<u128 as RingWord>::MAX, u128::MAX);
    assert_eq!(<u128 as RingWord>::BITS, 128);
}

// ── Bit intrinsics ──────────────────────────────────────────────────────

#[test]
fn bit_intrinsics_match_stdlib() {
    // u8
    assert_eq!(RingWord::count_ones(0b1010_1010u8), 4);
    assert_eq!(RingWord::count_ones(0u8), 0);
    assert_eq!(RingWord::count_ones(255u8), 8);
    assert_eq!(RingWord::leading_zeros(1u8), 7);
    assert_eq!(RingWord::leading_zeros(0u8), 8);
    assert_eq!(RingWord::trailing_zeros(0u8), 8);
    assert_eq!(RingWord::trailing_zeros(4u8), 2);

    // u32
    assert_eq!(RingWord::count_ones(0xDEAD_BEEFu32), 24);
    assert_eq!(RingWord::leading_zeros(1u32), 31);
    assert_eq!(RingWord::trailing_zeros(8u32), 3);

    // u64
    assert_eq!(RingWord::count_ones(u64::MAX), 64);
    assert_eq!(RingWord::leading_zeros(0u64), 64);
    assert_eq!(RingWord::trailing_zeros(1u64 << 63), 63);
}

// ── Conversion round-trips ──────────────────────────────────────────────

#[test]
fn from_to_u64_round_trip() {
    // Values that fit in all types
    for &v in &[0u64, 1, 42, 127, 255] {
        assert_eq!(<u8 as RingWord>::from_u64(v).to_u64(), v);
        assert_eq!(<u16 as RingWord>::from_u64(v).to_u64(), v);
        assert_eq!(<u32 as RingWord>::from_u64(v).to_u64(), v);
        assert_eq!(<u64 as RingWord>::from_u64(v).to_u64(), v);
        assert_eq!(<u128 as RingWord>::from_u64(v).to_u64(), v);
    }
    // Values that truncate for smaller types
    assert_eq!(<u8 as RingWord>::from_u64(256), 0u8); // wraps
    assert_eq!(<u8 as RingWord>::from_u64(257), 1u8); // wraps
}
