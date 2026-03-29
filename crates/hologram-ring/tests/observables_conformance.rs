//! Observable algebra conformance tests.
//!
//! Tests stratum, curvature, rank, domain at every quantum level.

use hologram_ring::{curvature, domain, rank, stratum, QuantumLevel, RingWord, Q0, Q1, Q3, Q7};

fn assert_observables_at_level<Q: QuantumLevel>()
where
    Q::Word: core::fmt::Debug,
{
    let zero = Q::Word::ZERO;
    let one = Q::Word::ONE;
    let max = Q::Word::MAX;

    // Stratum boundaries
    assert_eq!(stratum(zero), 0, "stratum(ZERO) must be 0");
    assert_eq!(stratum(max), Q::BITS, "stratum(MAX) must be BITS");
    assert_eq!(stratum(one), 1, "stratum(ONE) must be 1");

    // Stratum range
    let vals: Vec<Q::Word> = (0..64).map(Q::Word::from_u64).collect();
    for &x in &vals {
        let s = stratum(x);
        assert!(s <= Q::BITS, "stratum must be <= BITS");
    }

    // Curvature: always >= 1 (at least one bit flips when adding 1)
    for &x in &vals {
        let c = curvature(x);
        assert!(c >= 1, "curvature must be >= 1 for x={x:?}");
        assert!(c <= Q::BITS, "curvature must be <= BITS");
    }

    // Curvature at MAX: MAX + 1 = 0, so curvature(MAX) = popcount(MAX ^ 0) = BITS
    assert_eq!(curvature(max), Q::BITS, "curvature(MAX) must be BITS");
    // Curvature at 0: 0 + 1 = 1, so curvature(0) = popcount(0 ^ 1) = 1
    assert_eq!(curvature(zero), 1, "curvature(ZERO) must be 1");

    // Rank (trailing zeros)
    assert_eq!(rank(zero), Q::BITS, "rank(ZERO) must be BITS");
    assert_eq!(rank(one), 0, "rank(ONE) must be 0");

    // Domain (leading zeros)
    assert_eq!(domain(zero), Q::BITS, "domain(ZERO) must be BITS");
    assert_eq!(domain(max), 0, "domain(MAX) must be 0");
}

#[test]
fn observables_q0() {
    assert_observables_at_level::<Q0>();
}

#[test]
fn observables_q1() {
    assert_observables_at_level::<Q1>();
}

#[test]
fn observables_q3() {
    assert_observables_at_level::<Q3>();
}

#[test]
fn observables_q7() {
    assert_observables_at_level::<Q7>();
}

#[test]
fn stratum_is_popcount_q0_exhaustive() {
    for x in 0u8..=255 {
        assert_eq!(stratum(x), x.count_ones());
    }
}

#[test]
fn curvature_is_hamming_distance_q0_exhaustive() {
    for x in 0u8..=255 {
        assert_eq!(curvature(x), (x ^ x.wrapping_add(1)).count_ones());
    }
}

#[test]
fn rank_is_trailing_zeros_q0_exhaustive() {
    for x in 0u8..=255 {
        assert_eq!(rank(x), x.trailing_zeros());
    }
}

#[test]
fn domain_is_leading_zeros_q0_exhaustive() {
    for x in 0u8..=255 {
        assert_eq!(domain(x), x.leading_zeros());
    }
}
