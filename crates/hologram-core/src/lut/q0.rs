//! Q0 (8-bit) unary observable tables and torus/orbit classification.
//!
//! Each table is 256 bytes, computed at compile time, fits in L1 cache.

/// Stratum (Hamming weight): `STRATUM_Q0[x]` = popcount(x).
pub static STRATUM_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = (i as u8).count_ones() as u8;
        i += 1;
    }
    t
};

/// Curvature (cascade length): `CURVATURE_Q0[x]` = hamming(x, x+1).
pub static CURVATURE_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = i as u8;
        t[i as usize] = (x ^ x.wrapping_add(1)).count_ones() as u8;
        i += 1;
    }
    t
};

/// Domain (mod 3): `DOMAIN_Q0[x]` = x % 3.
pub static DOMAIN_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = (i % 3) as u8;
        i += 1;
    }
    t
};

/// Rank (div 3): `RANK_Q0[x]` = x / 3.
pub static RANK_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = (i / 3) as u8;
        i += 1;
    }
    t
};

/// Torus page: `TORUS_PAGE_Q0[x]` = x / 8.
pub static TORUS_PAGE_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = (i / 8) as u8;
        i += 1;
    }
    t
};

/// Torus offset: `TORUS_OFFSET_Q0[x]` = x % 8.
pub static TORUS_OFFSET_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = (i % 8) as u8;
        i += 1;
    }
    t
};

/// Orbit class: `ORBIT_CLASS_Q0[x]` = x / 8.
pub static ORBIT_CLASS_Q0: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = (i / 8) as u8;
        i += 1;
    }
    t
};

// ── Inline accessor functions ──────────────────────────────────

#[inline]
pub const fn stratum_q0(value: u8) -> u8 {
    STRATUM_Q0[value as usize]
}

#[inline]
pub const fn curvature_q0(value: u8) -> u8 {
    CURVATURE_Q0[value as usize]
}

#[inline]
pub const fn domain_q0(value: u8) -> u8 {
    DOMAIN_Q0[value as usize]
}

#[inline]
pub const fn rank_q0(value: u8) -> u8 {
    RANK_Q0[value as usize]
}

#[inline]
pub const fn torus_page_q0(value: u8) -> u8 {
    TORUS_PAGE_Q0[value as usize]
}

#[inline]
pub const fn torus_offset_q0(value: u8) -> u8 {
    TORUS_OFFSET_Q0[value as usize]
}

#[inline]
pub const fn orbit_class_q0(value: u8) -> u8 {
    ORBIT_CLASS_Q0[value as usize]
}

// ── Q1 helpers (via two Q0 lookups) ────────────────────────────

/// Stratum for Q1 via two Q0 lookups.
#[inline]
pub const fn stratum_q1(value: u16) -> u8 {
    let high = (value >> 8) as u8;
    let low = value as u8;
    stratum_q0(high) + stratum_q0(low)
}

/// Curvature for Q1: hamming(value, value+1).
#[inline]
pub const fn curvature_q1(value: u16) -> u8 {
    let xor = value ^ value.wrapping_add(1);
    let high = (xor >> 8) as u8;
    let low = xor as u8;
    stratum_q0(high) + stratum_q0(low)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratum_matches_popcount() {
        for i in 0..=255u8 {
            assert_eq!(stratum_q0(i), i.count_ones() as u8);
        }
    }

    #[test]
    fn curvature_matches_hamming() {
        for i in 0..=255u8 {
            let expected = (i ^ i.wrapping_add(1)).count_ones() as u8;
            assert_eq!(curvature_q0(i), expected);
        }
    }

    #[test]
    fn domain_matches_mod3() {
        for i in 0..=255u8 {
            assert_eq!(domain_q0(i), i % 3);
        }
    }

    #[test]
    fn rank_matches_div3() {
        for i in 0..=255u8 {
            assert_eq!(rank_q0(i), i / 3);
        }
    }

    #[test]
    fn torus_page() {
        assert_eq!(torus_page_q0(0), 0);
        assert_eq!(torus_page_q0(7), 0);
        assert_eq!(torus_page_q0(8), 1);
        assert_eq!(torus_page_q0(255), 31);
    }

    #[test]
    fn orbit_class() {
        for i in 0..=255u8 {
            assert_eq!(orbit_class_q0(i), i / 8);
        }
    }

    #[test]
    fn mean_curvature() {
        let sum: u32 = (0..=255u8).map(|i| curvature_q0(i) as u32).sum();
        let mean = sum as f64 / 256.0;
        assert!((mean - 1.9921875).abs() < 0.0001);
    }

    #[test]
    fn stratum_q1_values() {
        assert_eq!(stratum_q1(0), 0);
        assert_eq!(stratum_q1(0xFFFF), 16);
        assert_eq!(stratum_q1(0x00FF), 8);
        assert_eq!(stratum_q1(0xFF00), 8);
        assert_eq!(stratum_q1(0x5555), 8);
    }

    #[test]
    fn curvature_q1_values() {
        assert_eq!(curvature_q1(0), 1);
        assert_eq!(curvature_q1(1), 2);
        assert_eq!(curvature_q1(0xFF), 9);
        assert_eq!(curvature_q1(0xFFFF), 16);
    }
}
