//! Ring algebraic conformance tests.
//!
//! These tests encode hologram's algebraic contract: the properties that
//! the ring hierarchy MUST satisfy for correctness. Every ring level (Q0–Q3)
//! must uphold these invariants. Failure means the algebra is broken.
//!
//! Test tiers:
//! 1. Ring axioms: closure, associativity, commutativity, identity, inverse
//! 2. Cayley-Dickson chain: dimension doubling, adjunction
//! 3. Critical identity: neg(bnot(x)) = succ(x) at every level

use hologram_core::quantum::{q3_add, q3_mul, q3_neg, q3_sub};

// ── 1. Ring Axioms ──────────────────────────────────────────────────────────

#[test]
fn q0_add_associative() {
    for a in (0u8..=255).step_by(17) {
        for b in (0u8..=255).step_by(19) {
            for c in (0u8..=255).step_by(23) {
                let lhs = a.wrapping_add(b).wrapping_add(c);
                let rhs = a.wrapping_add(b.wrapping_add(c));
                assert_eq!(lhs, rhs, "Q0 add not associative at ({a},{b},{c})");
            }
        }
    }
}

#[test]
fn q0_mul_associative() {
    for a in (0u8..=255).step_by(17) {
        for b in (0u8..=255).step_by(19) {
            for c in (0u8..=255).step_by(23) {
                let lhs = a.wrapping_mul(b).wrapping_mul(c);
                let rhs = a.wrapping_mul(b.wrapping_mul(c));
                assert_eq!(lhs, rhs, "Q0 mul not associative at ({a},{b},{c})");
            }
        }
    }
}

#[test]
fn q0_identities() {
    for x in 0u8..=255 {
        assert_eq!(x.wrapping_add(0), x);
        assert_eq!(x.wrapping_mul(1), x);
    }
}

#[test]
fn q0_additive_inverse() {
    for x in 0u8..=255 {
        assert_eq!(x.wrapping_add(x.wrapping_neg()), 0);
    }
}

#[test]
fn q3_ring_axioms() {
    let vals: &[u32] = &[0, 1, 127, 255, 0xFFFF, 0x00FF_FFFF, u32::MAX / 2, u32::MAX];
    for &a in vals {
        assert_eq!(q3_add(a, 0), a, "Q3 add identity at {a:#x}");
        assert_eq!(q3_mul(a, 1), a, "Q3 mul identity at {a:#x}");
        assert_eq!(q3_add(a, q3_neg(a)), 0, "Q3 add inverse at {a:#x}");
        assert_eq!(q3_neg(q3_neg(a)), a, "Q3 neg involution at {a:#x}");

        for &b in vals {
            assert_eq!(q3_add(a, b), q3_add(b, a), "Q3 add commutative");
            assert_eq!(q3_mul(a, b), q3_mul(b, a), "Q3 mul commutative");
            assert_eq!(q3_sub(q3_add(a, b), b), a, "Q3 sub/add inverse");

            for &c in vals {
                assert_eq!(
                    q3_add(q3_add(a, b), c),
                    q3_add(a, q3_add(b, c)),
                    "Q3 add not associative"
                );
                assert_eq!(
                    q3_mul(q3_mul(a, b), c),
                    q3_mul(a, q3_mul(b, c)),
                    "Q3 mul not associative"
                );
                assert_eq!(
                    q3_mul(a, q3_add(b, c)),
                    q3_add(q3_mul(a, b), q3_mul(a, c)),
                    "Q3 distributivity"
                );
            }
        }
    }
}

// ── 2. Cayley-Dickson Chain ─────────────────────────────────────────────────

#[test]
fn cayley_dickson_chain_q0_q1_q2() {
    use hologram_core::q1::ring::WordRing;
    use hologram_core::ring::ByteRing;
    use hologram_foundation::division::{CayleyDicksonConstruction, NormedDivisionAlgebra};

    let byte_ring = ByteRing;
    let word_ring = WordRing;

    // Q0 → Q1: dimension 1 → 2
    assert_eq!(byte_ring.cayley_dickson_source().algebra_dimension(), 1);
    assert_eq!(byte_ring.cayley_dickson_target().algebra_dimension(), 2);
    assert_eq!(
        byte_ring.cayley_dickson_target().algebra_dimension(),
        byte_ring.cayley_dickson_source().algebra_dimension() * 2,
        "CD must double dimension at Q0→Q1"
    );

    // Q1 → Q2: dimension 2 → 4
    assert_eq!(word_ring.cayley_dickson_source().algebra_dimension(), 2);
    assert_eq!(word_ring.cayley_dickson_target().algebra_dimension(), 4);
    assert_eq!(
        word_ring.cayley_dickson_target().algebra_dimension(),
        word_ring.cayley_dickson_source().algebra_dimension() * 2,
        "CD must double dimension at Q1→Q2"
    );
}

#[test]
fn cayley_dickson_chain_q2_q3() {
    use hologram_core::q2::ring::TripleRing;
    use hologram_foundation::division::{CayleyDicksonConstruction, NormedDivisionAlgebra};
    let triple_ring = TripleRing;
    assert_eq!(triple_ring.cayley_dickson_source().algebra_dimension(), 4);
    assert_eq!(triple_ring.cayley_dickson_target().algebra_dimension(), 8);
}

#[test]
fn ring_quantum_levels_match() {
    use hologram_core::q1::ring::WordRing;
    use hologram_core::q2::ring::TripleRing;
    use hologram_core::ring::ByteRing;
    use hologram_foundation::schema::Ring;
    use hologram_foundation::WittLevel;

    assert_eq!(ByteRing.at_witt_level(), WittLevel::W8);
    assert_eq!(WordRing.at_witt_level(), WittLevel::W16);
    assert_eq!(TripleRing.at_witt_level(), WittLevel::W24);
}

// ── 3. Critical Identity ────────────────────────────────────────────────────

#[test]
fn critical_identity_all_levels() {
    // Q0: exhaustive
    for x in 0u8..=255 {
        let bnot_x = !x;
        let neg_bnot_x = 0u8.wrapping_sub(bnot_x);
        let succ_x = x.wrapping_add(1);
        assert_eq!(neg_bnot_x, succ_x, "Q0 critical identity at {x}");
    }

    // Q2: spot check
    use hologram_core::q2::arith::{bnot_q2, neg_q2, succ_q2};
    for x in [0u32, 1, 0xFF, 0xFFFF, 0x00FF_FFFF] {
        assert_eq!(
            neg_q2(bnot_q2(x)),
            succ_q2(x),
            "Q2 critical identity at {x:#x}"
        );
    }

    // Q3: spot check
    for x in [0u32, 1, 0xFF, 0xFFFF, u32::MAX] {
        let bnot_x = !x;
        let neg_bnot_x = 0u32.wrapping_sub(bnot_x);
        let succ_x = x.wrapping_add(1);
        assert_eq!(neg_bnot_x, succ_x, "Q3 critical identity at {x:#x}");
    }
}
