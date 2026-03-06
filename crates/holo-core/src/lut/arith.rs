//! Q0 arithmetic tables (256x256 = 64 KB each) and GF polynomial tables.

/// Addition: `ADD_Q0[a][b]` = (a + b) mod 256.  64 KB.
pub static ADD_Q0: [[u8; 256]; 256] = {
    let mut t = [[0u8; 256]; 256];
    let mut a = 0u16;
    while a < 256 {
        let mut b = 0u16;
        while b < 256 {
            t[a as usize][b as usize] = (a as u8).wrapping_add(b as u8);
            b += 1;
        }
        a += 1;
    }
    t
};

/// Subtraction: `SUB_Q0[a][b]` = (a - b) mod 256.  64 KB.
pub static SUB_Q0: [[u8; 256]; 256] = {
    let mut t = [[0u8; 256]; 256];
    let mut a = 0u16;
    while a < 256 {
        let mut b = 0u16;
        while b < 256 {
            t[a as usize][b as usize] = (a as u8).wrapping_sub(b as u8);
            b += 1;
        }
        a += 1;
    }
    t
};

/// Multiplication: `MUL_Q0[a][b]` = (a * b) mod 256.  64 KB.
pub static MUL_Q0: [[u8; 256]; 256] = {
    let mut t = [[0u8; 256]; 256];
    let mut a = 0u16;
    while a < 256 {
        let mut b = 0u16;
        while b < 256 {
            t[a as usize][b as usize] = (a as u8).wrapping_mul(b as u8);
            b += 1;
        }
        a += 1;
    }
    t
};

/// Power: `POW_Q0[base][exp]` = (base ^ exp) mod 256.  64 KB.
pub static POW_Q0: [[u8; 256]; 256] = {
    let mut t = [[0u8; 256]; 256];
    let mut base = 0u16;
    while base < 256 {
        t[base as usize][0] = 1;
        let mut exp = 1u16;
        while exp < 256 {
            t[base as usize][exp as usize] =
                t[base as usize][(exp - 1) as usize].wrapping_mul(base as u8);
            exp += 1;
        }
        base += 1;
    }
    t
};

/// GF(2) polynomial multiplication: `GF2_MUL_Q0[a][b]`.  128 KB.
pub static GF2_MUL_Q0: [[u16; 256]; 256] = {
    let mut t = [[0u16; 256]; 256];
    let mut a = 0u16;
    while a < 256 {
        let mut b = 0u16;
        while b < 256 {
            let mut result: u16 = 0;
            let mut b_rem = b as u8;
            let mut shift = 0u32;
            while b_rem != 0 {
                if b_rem & 1 != 0 {
                    result ^= a << shift;
                }
                b_rem >>= 1;
                shift += 1;
            }
            t[a as usize][b as usize] = result;
            b += 1;
        }
        a += 1;
    }
    t
};

const GF3_MAX_COEFFS_U8: usize = 6;
const GF3_MAX_COEFFS_U16: usize = 11;
const GF3_POW3: [u16; 11] = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049];

/// GF(3) polynomial multiplication: `GF3_MUL_Q0[a][b]`.  128 KB.
#[allow(long_running_const_eval)]
pub static GF3_MUL_Q0: [[u16; 256]; 256] = {
    let mut t = [[0u16; 256]; 256];
    let mut a = 0u16;
    while a < 256 {
        let mut ca = [0u8; GF3_MAX_COEFFS_U8];
        let mut k = 0;
        while k < GF3_MAX_COEFFS_U8 {
            ca[k] = ((a / GF3_POW3[k]) % 3) as u8;
            k += 1;
        }
        let mut b = 0u16;
        while b < 256 {
            let mut cb = [0u8; GF3_MAX_COEFFS_U8];
            k = 0;
            while k < GF3_MAX_COEFFS_U8 {
                cb[k] = ((b / GF3_POW3[k]) % 3) as u8;
                k += 1;
            }
            let mut result = [0u8; GF3_MAX_COEFFS_U16];
            let mut i = 0;
            while i < GF3_MAX_COEFFS_U8 {
                if ca[i] != 0 {
                    let mut j = 0;
                    while j < GF3_MAX_COEFFS_U8 {
                        if cb[j] != 0 {
                            result[i + j] = (result[i + j] + ca[i] * cb[j]) % 3;
                        }
                        j += 1;
                    }
                }
                i += 1;
            }
            let mut encoded: u16 = 0;
            k = 0;
            while k < GF3_MAX_COEFFS_U16 {
                encoded += result[k] as u16 * GF3_POW3[k];
                k += 1;
            }
            t[a as usize][b as usize] = encoded;
            b += 1;
        }
        a += 1;
    }
    t
};

// ── Inline accessor functions ──────────────────────────────────

#[inline]
pub const fn add_q0(a: u8, b: u8) -> u8 {
    ADD_Q0[a as usize][b as usize]
}

#[inline]
pub const fn sub_q0(a: u8, b: u8) -> u8 {
    SUB_Q0[a as usize][b as usize]
}

#[inline]
pub const fn mul_q0(a: u8, b: u8) -> u8 {
    MUL_Q0[a as usize][b as usize]
}

#[inline]
pub const fn pow_q0(base: u8, exp: u8) -> u8 {
    POW_Q0[base as usize][exp as usize]
}

#[inline]
pub const fn gf2_mul_q0(a: u8, b: u8) -> u16 {
    GF2_MUL_Q0[a as usize][b as usize]
}

#[inline]
pub const fn gf3_mul_q0(a: u8, b: u8) -> u16 {
    GF3_MUL_Q0[a as usize][b as usize]
}

/// Wrapping byte-domain addition.
#[inline]
pub const fn byte_add(a: u8, b: u8) -> u8 {
    ADD_Q0[a as usize][b as usize]
}

/// Wrapping byte-domain subtraction.
#[inline]
pub const fn byte_sub(a: u8, b: u8) -> u8 {
    SUB_Q0[a as usize][b as usize]
}

/// Wrapping byte-domain multiplication.
#[inline]
pub const fn byte_mul(a: u8, b: u8) -> u8 {
    MUL_Q0[a as usize][b as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_wrapping() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(add_q0(a, b), a.wrapping_add(b));
            }
        }
    }

    #[test]
    fn sub_matches_wrapping() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(sub_q0(a, b), a.wrapping_sub(b));
            }
        }
    }

    #[test]
    fn mul_matches_wrapping() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(mul_q0(a, b), a.wrapping_mul(b));
            }
        }
    }

    #[test]
    fn pow_matches_iterative() {
        for base in 0..=255u8 {
            for exp in 0..=255u8 {
                let mut expected: u8 = 1;
                for _ in 0..exp {
                    expected = expected.wrapping_mul(base);
                }
                assert_eq!(pow_q0(base, exp), expected);
            }
        }
    }

    #[test]
    fn add_commutative() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(add_q0(a, b), add_q0(b, a));
            }
        }
    }

    #[test]
    fn mul_commutative() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(mul_q0(a, b), mul_q0(b, a));
            }
        }
    }

    #[test]
    fn add_sub_inverse() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(sub_q0(add_q0(a, b), b), a);
            }
        }
    }

    #[test]
    fn gf2_mul_commutative() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(gf2_mul_q0(a, b), gf2_mul_q0(b, a));
            }
        }
    }

    #[test]
    fn gf2_mul_identity_and_zero() {
        for a in 0..=255u8 {
            assert_eq!(gf2_mul_q0(a, 1), a as u16);
            assert_eq!(gf2_mul_q0(a, 0), 0);
        }
    }

    #[test]
    fn gf3_mul_commutative() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(gf3_mul_q0(a, b), gf3_mul_q0(b, a));
            }
        }
    }

    #[test]
    fn gf3_mul_identity_and_zero() {
        for a in 0..=255u8 {
            assert_eq!(gf3_mul_q0(a, 1), a as u16);
            assert_eq!(gf3_mul_q0(a, 0), 0);
        }
    }
}
