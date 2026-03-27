//! Cayley-Dickson arithmetic for Q3 — three-level tower.
//!
//! Level 1: Q1 conjugation — (re, im) → (re, -im) over u8 halves.
//! Level 2: Quaternion product — CD over (u16, u16) with Level 1 conjugation.
//!          Non-commutative, associative.
//! Level 3: Octonion product — CD over (u32, u32) with Level 2 product.
//!          Non-commutative, NON-associative.
//!
//! The non-associativity emerges at Level 3 because the base (Level 2) is
//! non-commutative: CD over a non-commutative algebra produces a non-associative one.

/// Q3 wrapping arithmetic re-exports from the quantum module.
pub use crate::quantum::{q3_add, q3_curvature, q3_mul, q3_neg, q3_stratum, q3_sub};

// ── Level 1: Q1 Conjugation ────────────────────────────────────────────────

/// Q1 conjugation: split u16 into (re: hi_u8, im: lo_u8), negate the imaginary part.
///
/// conj(a + bi) = a − bi in Z/256Z components.
#[inline]
#[must_use]
pub const fn conj_q1(x: u16) -> u16 {
    let re = (x >> 8) as u8;
    let im = (x & 0xFF) as u8;
    ((re as u16) << 8) | (im.wrapping_neg() as u16)
}

// ── Level 2: Quaternion Product ─────────────────────────────────────────────

/// Cayley-Dickson product at quaternion level.
///
/// Splits u32 into (a: hi_u16, b: lo_u16), uses the CD formula:
/// (a, b) × (c, d) = (a·c − conj(d)·b, d·a + b·conj(c))
///
/// Non-commutative (verified by conformance tests), associative.
#[inline]
#[must_use]
pub const fn cd_mul(x: u32, y: u32) -> u32 {
    let a = (x >> 16) as u16;
    let b = (x & 0xFFFF) as u16;
    let c = (y >> 16) as u16;
    let d = (y & 0xFFFF) as u16;

    let hi = a.wrapping_mul(c).wrapping_sub(conj_q1(d).wrapping_mul(b));
    let lo = d.wrapping_mul(a).wrapping_add(b.wrapping_mul(conj_q1(c)));

    ((hi as u32) << 16) | (lo as u32)
}

/// Cayley-Dickson conjugation at quaternion level.
///
/// conj(a, b) = (conj_q1(a), -b) where -b is wrapping negation in Z/65536Z.
#[inline]
#[must_use]
pub const fn cd_conj(x: u32) -> u32 {
    let a = (x >> 16) as u16;
    let b = (x & 0xFFFF) as u16;
    ((conj_q1(a) as u32) << 16) | (b.wrapping_neg() as u32)
}

/// Commutator [a,b] = cd_mul(a,b) − cd_mul(b,a).
///
/// Non-zero for most inputs (quaternion product is non-commutative).
#[inline]
#[must_use]
pub const fn commutator(a: u32, b: u32) -> u32 {
    cd_mul(a, b).wrapping_sub(cd_mul(b, a))
}

// ── Level 3: Octonion Product ───────────────────────────────────────────────

/// Octonion product on (u32, u32) pairs.
///
/// Uses the Cayley-Dickson formula over the non-commutative quaternion base:
/// (p, q) × (r, s) = (p·r − conj(s)·q, s·p + q·conj(r))
///
/// Non-commutative, NON-associative — the non-associativity emerges because
/// the base product (cd_mul) is non-commutative.
#[inline]
#[must_use]
pub const fn oct_mul(a: (u32, u32), b: (u32, u32)) -> (u32, u32) {
    let hi = cd_mul(a.0, b.0).wrapping_sub(cd_mul(cd_conj(b.1), a.1));
    let lo = cd_mul(b.1, a.0).wrapping_add(cd_mul(a.1, cd_conj(b.0)));
    (hi, lo)
}

/// Octonion associator: embed u32 values as purely imaginary octonions (0, x),
/// compute [(0,a), (0,b), (0,c)] = ((0,a)·(0,b))·(0,c) − (0,a)·((0,b)·(0,c)).
///
/// Returns (hi, lo) pair. Non-zero when the computation is genuinely
/// non-associative — measures how much evaluation order matters.
#[inline]
#[must_use]
pub const fn associator(a: u32, b: u32, c: u32) -> (u32, u32) {
    let oa = (0u32, a);
    let ob = (0u32, b);
    let oc = (0u32, c);
    let ab = oct_mul(oa, ob);
    let bc = oct_mul(ob, oc);
    let lhs = oct_mul(ab, oc);
    let rhs = oct_mul(oa, bc);
    (lhs.0.wrapping_sub(rhs.0), lhs.1.wrapping_sub(rhs.1))
}

/// Scalar associator norm: total popcount of both halves.
///
/// Range: 0 (fully associative for these inputs) to 64 (maximally non-associative).
/// Used by the precision system to gauge whether evaluation order matters.
#[inline]
#[must_use]
pub const fn associator_norm(a: u32, b: u32, c: u32) -> u8 {
    let (hi, lo) = associator(a, b, c);
    (hi.count_ones() + lo.count_ones()) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conj_q1_negates_imaginary() {
        // (re=0x01, im=0x02) → (0x01, 0xFE)
        assert_eq!(conj_q1(0x0102), 0x01FE);
        // Double conjugation is identity
        for x in [0u16, 0x0100, 0x00FF, 0x0102, 0xFFFF] {
            assert_eq!(conj_q1(conj_q1(x)), x, "conj involution at {x:#x}");
        }
    }

    #[test]
    fn cd_conj_involution() {
        for x in [0u32, 1, 0x0102_0304, 0xFFFF_FFFF] {
            assert_eq!(cd_conj(cd_conj(x)), x, "cd_conj involution at {x:#x}");
        }
    }

    #[test]
    fn cd_mul_non_commutative() {
        let a = 0x0102_0304u32;
        let b = 0x0506_0708u32;
        assert_ne!(
            cd_mul(a, b),
            cd_mul(b, a),
            "cd_mul should be non-commutative"
        );
    }

    #[test]
    fn commutator_nonzero() {
        let a = 0x0102_0304u32;
        let b = 0x0506_0708u32;
        assert_ne!(commutator(a, b), 0, "commutator should be non-zero");
    }

    #[test]
    fn cd_mul_zero_identity() {
        // Z/65536Z has zero divisors (256² ≡ 0), so the CD construction
        // doesn't have a standard identity. But cd_mul(0, x) = 0 for all x.
        for x in [0u32, 1, 0x0102_0304, 0xFFFF_FFFF] {
            assert_eq!(cd_mul(0, x), 0, "zero left-absorbs at {x:#x}");
        }
    }

    #[test]
    fn oct_mul_non_associative() {
        let a = (0u32, 0x0102_0304);
        let b = (0u32, 0x0506_0708);
        let c = (0u32, 0x090A_0B0C);
        let lhs = oct_mul(oct_mul(a, b), c);
        let rhs = oct_mul(a, oct_mul(b, c));
        assert_ne!(lhs, rhs, "oct_mul should be non-associative");
    }

    #[test]
    fn associator_nonzero_for_imaginary() {
        let (hi, lo) = associator(0x0102_0304, 0x0506_0708, 0x090A_0B0C);
        assert!(
            hi != 0 || lo != 0,
            "associator should be non-zero for imaginary embedding"
        );
    }

    #[test]
    fn associator_norm_bounded() {
        let norm = associator_norm(0x0102_0304, 0x0506_0708, 0x090A_0B0C);
        assert!(norm <= 64, "norm must be ≤ 64 (two u32 popcount)");
        assert!(norm > 0, "norm should be non-zero for non-trivial inputs");
    }

    #[test]
    fn associator_zero_for_real_embedding() {
        // Embedding as (x, 0) — real part only. The octonion product of
        // (x,0)*(y,0) = (cd_mul(x,y), 0), which is quaternion product.
        // Quaternion product IS associative, so associator should be zero.
        let a = (0x0001_0000u32, 0);
        let b = (0x0002_0000u32, 0);
        let c = (0x0003_0000u32, 0);
        let lhs = oct_mul(oct_mul(a, b), c);
        let rhs = oct_mul(a, oct_mul(b, c));
        assert_eq!(lhs, rhs, "real subalgebra must be associative");
    }
}
