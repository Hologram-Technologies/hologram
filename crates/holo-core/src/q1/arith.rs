//! Q1 (16-bit) arithmetic — direct wrapping operations.
//!
//! Unlike Q0 which uses precomputed 256×256 LUT tables (64 KB each),
//! Q1 arithmetic uses native wrapping operations directly. A Q1 LUT
//! would require 65536×65536 = 4 GB per table, which is infeasible.
//! Native wrapping ops are O(1) and single-cycle on modern CPUs.

/// Addition in Z/65536Z.
#[inline]
pub const fn add_q1(a: u16, b: u16) -> u16 {
    a.wrapping_add(b)
}

/// Subtraction in Z/65536Z.
#[inline]
pub const fn sub_q1(a: u16, b: u16) -> u16 {
    a.wrapping_sub(b)
}

/// Multiplication in Z/65536Z.
#[inline]
pub const fn mul_q1(a: u16, b: u16) -> u16 {
    a.wrapping_mul(b)
}

/// Exponentiation in Z/65536Z via repeated squaring.
#[inline]
pub const fn pow_q1(mut base: u16, mut exp: u16) -> u16 {
    let mut result: u16 = 1;
    // Use `base = base * base` mod 2^16 at each step
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exp >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_wrapping() {
        assert_eq!(add_q1(100, 200), 300);
        assert_eq!(add_q1(65535, 1), 0);
        assert_eq!(add_q1(65535, 65535), 65534);
        assert_eq!(add_q1(0, 0), 0);
        assert_eq!(add_q1(32768, 32768), 0);
    }

    #[test]
    fn add_commutative() {
        for a in (0u32..=65535).step_by(1000) {
            for b in (0u32..=65535).step_by(1000) {
                let (a, b) = (a as u16, b as u16);
                assert_eq!(add_q1(a, b), add_q1(b, a));
            }
        }
    }

    #[test]
    fn add_identity() {
        for a in (0u32..=65535).step_by(500) {
            let a = a as u16;
            assert_eq!(add_q1(a, 0), a);
        }
    }

    #[test]
    fn sub_inverse_of_add() {
        for a in (0u32..=65535).step_by(1000) {
            for b in (0u32..=65535).step_by(1000) {
                let (a, b) = (a as u16, b as u16);
                assert_eq!(sub_q1(add_q1(a, b), b), a);
            }
        }
    }

    #[test]
    fn sub_self_is_zero() {
        for a in (0u32..=65535).step_by(500) {
            let a = a as u16;
            assert_eq!(sub_q1(a, a), 0);
        }
    }

    #[test]
    fn mul_commutative() {
        for a in (0u32..=65535).step_by(2000) {
            for b in (0u32..=65535).step_by(2000) {
                let (a, b) = (a as u16, b as u16);
                assert_eq!(mul_q1(a, b), mul_q1(b, a));
            }
        }
    }

    #[test]
    fn mul_identity_and_zero() {
        for a in (0u32..=65535).step_by(500) {
            let a = a as u16;
            assert_eq!(mul_q1(a, 1), a);
            assert_eq!(mul_q1(a, 0), 0);
        }
    }

    #[test]
    fn mul_wrapping() {
        assert_eq!(mul_q1(256, 256), 0); // 65536 mod 65536
        assert_eq!(mul_q1(255, 2), 510);
        assert_eq!(mul_q1(65535, 65535), 1); // (-1)^2 mod 2^16
    }

    #[test]
    fn pow_base_cases() {
        assert_eq!(pow_q1(0, 0), 1); // 0^0 = 1 by convention
        assert_eq!(pow_q1(0, 1), 0);
        assert_eq!(pow_q1(1, 65535), 1);
        assert_eq!(pow_q1(2, 0), 1);
        assert_eq!(pow_q1(2, 1), 2);
        assert_eq!(pow_q1(2, 10), 1024);
        assert_eq!(pow_q1(2, 16), 0); // 65536 mod 65536
    }

    #[test]
    fn pow_small_values() {
        assert_eq!(pow_q1(3, 2), 9);
        assert_eq!(pow_q1(3, 3), 27);
        assert_eq!(pow_q1(10, 2), 100);
        assert_eq!(pow_q1(10, 3), 1000);
        assert_eq!(pow_q1(10, 4), 10000);
    }

    #[test]
    fn pow_wrapping() {
        // 256^2 = 65536 = 0 mod 2^16
        assert_eq!(pow_q1(256, 2), 0);
        // 65535^2 = (-1)^2 = 1 mod 2^16
        assert_eq!(pow_q1(65535, 2), 1);
    }
}
