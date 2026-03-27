//! Q2 (24-bit) arithmetic — direct wrapping operations masked to 24 bits.
//!
//! Z/2^24Z: modulus 16,777,216. All operations mask to 0x00FF_FFFF.
//! No LUT (a Q2 table would require ~50MB per function).
//! Native u32 wrapping ops are O(1) and single-cycle on modern CPUs.

/// Addition in Z/2^24 Z.
#[inline(always)]
#[must_use]
pub const fn add_q2(a: u32, b: u32) -> u32 {
    a.wrapping_add(b) & 0x00FF_FFFF
}

/// Subtraction in Z/2^24 Z.
#[inline(always)]
#[must_use]
pub const fn sub_q2(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b) & 0x00FF_FFFF
}

/// Multiplication in Z/2^24 Z.
#[inline(always)]
#[must_use]
pub const fn mul_q2(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b) & 0x00FF_FFFF
}

/// Exponentiation in Z/2^24 Z via repeated squaring.
#[inline]
#[must_use]
pub const fn pow_q2(base: u32, exp: u32) -> u32 {
    let mut result = 1u32;
    let mut b = base & 0x00FF_FFFF;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_q2(result, b);
        }
        b = mul_q2(b, b);
        e >>= 1;
    }
    result
}

/// Negation (additive inverse) in Z/2^24 Z: neg(x) = (2^24 - x) mod 2^24.
#[inline(always)]
#[must_use]
pub const fn neg_q2(a: u32) -> u32 {
    (0x01_000000u32.wrapping_sub(a)) & 0x00FF_FFFF
}

/// Bitwise NOT masked to 24 bits: bnot(x) = (2^24 - 1) ^ x.
#[inline(always)]
#[must_use]
pub const fn bnot_q2(a: u32) -> u32 {
    (!a) & 0x00FF_FFFF
}

/// Successor in Z/2^24 Z: succ(x) = (x + 1) mod 2^24.
#[inline(always)]
#[must_use]
pub const fn succ_q2(a: u32) -> u32 {
    a.wrapping_add(1) & 0x00FF_FFFF
}

/// Predecessor in Z/2^24 Z: pred(x) = (x - 1) mod 2^24.
#[inline(always)]
#[must_use]
pub const fn pred_q2(a: u32) -> u32 {
    a.wrapping_sub(1) & 0x00FF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_wrapping() {
        assert_eq!(add_q2(0x00FF_FFFF, 1), 0);
        assert_eq!(add_q2(100, 200), 300);
        assert_eq!(add_q2(0, 0), 0);
    }

    #[test]
    fn sub_wrapping() {
        assert_eq!(sub_q2(0, 1), 0x00FF_FFFF);
        assert_eq!(sub_q2(300, 200), 100);
    }

    #[test]
    fn mul_wrapping() {
        assert_eq!(mul_q2(0x01_0000, 0x01_0000), 0); // 2^16 * 2^16 = 2^32 mod 2^24 = 0
        assert_eq!(mul_q2(3, 4), 12);
    }

    #[test]
    fn neg_is_additive_inverse() {
        for x in [0u32, 1, 127, 255, 0xFFFF, 0x00FF_FFFF] {
            assert_eq!(add_q2(x, neg_q2(x)), 0, "neg inverse failed at {x:#x}");
        }
    }

    #[test]
    fn bnot_involution() {
        for x in [0u32, 1, 0xFF, 0xFFFF, 0x00FF_FFFF] {
            assert_eq!(
                bnot_q2(bnot_q2(x)),
                x & 0x00FF_FFFF,
                "bnot involution at {x:#x}"
            );
        }
    }

    #[test]
    fn critical_identity() {
        // UOR critical identity: neg(bnot(x)) = succ(x) in Z/2^24Z.
        for x in [
            0u32,
            1,
            127,
            128,
            255,
            256,
            0x7FFF,
            0x8000,
            0xFFFF,
            0x00FF_FFFF,
        ] {
            assert_eq!(
                neg_q2(bnot_q2(x)),
                succ_q2(x),
                "critical identity failed at {x:#x}"
            );
        }
    }

    #[test]
    fn ring_axioms() {
        let vals = [0u32, 1, 0xFF, 0xFFFF, 0x00FF_FFFF];
        for &a in &vals {
            for &b in &vals {
                assert_eq!(
                    add_q2(a, b),
                    add_q2(b, a),
                    "add commutativity at ({a:#x},{b:#x})"
                );
                assert_eq!(add_q2(a, 0), a & 0x00FF_FFFF, "add identity at {a:#x}");
                assert_eq!(
                    mul_q2(a, b),
                    mul_q2(b, a),
                    "mul commutativity at ({a:#x},{b:#x})"
                );
                assert_eq!(
                    mul_q2(a, 1) & 0x00FF_FFFF,
                    a & 0x00FF_FFFF,
                    "mul identity at {a:#x}"
                );
            }
        }
    }

    #[test]
    fn succ_pred_inverse() {
        for x in [0u32, 1, 0xFF, 0xFFFF, 0x00FF_FFFF] {
            assert_eq!(
                pred_q2(succ_q2(x)),
                x & 0x00FF_FFFF,
                "succ/pred inverse at {x:#x}"
            );
            assert_eq!(
                succ_q2(pred_q2(x)),
                x & 0x00FF_FFFF,
                "pred/succ inverse at {x:#x}"
            );
        }
    }

    #[test]
    fn pow_base_cases() {
        assert_eq!(pow_q2(0, 0), 1); // 0^0 = 1 by convention
        assert_eq!(pow_q2(0, 1), 0);
        assert_eq!(pow_q2(1, 0x00FF_FFFF), 1);
        assert_eq!(pow_q2(2, 0), 1);
        assert_eq!(pow_q2(2, 24), 0); // 2^24 mod 2^24 = 0
        assert_eq!(pow_q2(3, 2), 9);
    }
}
