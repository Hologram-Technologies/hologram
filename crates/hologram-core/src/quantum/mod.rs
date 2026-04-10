//! Quantum level utilities and scaling strategy documentation.
//!
//! This module provides helpers for working across quantum levels and documents
//! the scaling strategy from Q0 through Q4+.
//!
//! # Quantum Levels
//!
//! | Level | Bits | Ring | States | Table Strategy |
//! |-------|------|------|--------|----------------|
//! | Q0 | 8 | Z/256Z | 256 | Full LUT (256B each) |
//! | Q1 | 16 | Z/65536Z | 65,536 | Full LUT (128KB each) |
//! | Q2 | 24 | Z/16777216Z | 16.7M | Hierarchical segmentation (~50MB each) |
//! | Q3 | 32 | Z/4294967296Z | 4.3B | Algorithmic only (17GB/table infeasible) |
//! | Q4+ | 40+ | Z/2^nZ | 2^n | Algorithmic with optional LRU cache |
//!
//! # Q0 (8-bit): Full Table
//!
//! All 256 entries fit in a single cache line (or two). Arithmetic uses
//! precomputed 256×256 = 64KB tables. Activations are 256B each.
//! Total memory: ~519KB. Fits in L1/L2 cache.
//!
//! # Q1 (16-bit): Full Table
//!
//! All 65,536 entries = 128KB per table. 21 activation tables = 2.7MB total.
//! Arithmetic uses native wrapping ops (a 65536×65536 table would be 4GB).
//! Total memory: ~2.7MB. Fits comfortably in L2/L3 cache.
//!
//! # Q2 (24-bit): Hierarchical Segmentation (Design Only)
//!
//! A full Q2 table would be 2^24 × 3 bytes = ~50MB per table. This is
//! borderline for L3 cache on modern CPUs (typically 6-32MB). Strategy:
//!
//! - **High-byte coarse table** (256 entries): Maps the top 8 bits to a
//!   segment index, providing a coarse approximation.
//! - **Q1 fine table** (65,536 entries per segment): For each coarse region,
//!   a Q1-sized table handles the lower 16 bits with higher precision.
//! - **Correction table** (optional): A small table for interpolation error.
//!
//! This gives O(2) lookup (coarse + fine) with ~50MB per activation total.
//! Implementation deferred to a future sprint when Q2 support is needed.
//!
//! # Q3 (32-bit): Algorithmic Computation (Design Only)
//!
//! A full Q3 table would be 2^32 × 4 bytes = 17GB — impossible to table.
//! Strategy: pure algorithmic computation with optional Q1 piecewise
//! approximation for hot paths.
//!
//! - Use native f32/f64 math for transcendental functions.
//! - Optional: segment the input range into Q1-sized blocks and use
//!   Q1 tables for the hot regions, falling back to computation for cold.
//! - Arithmetic: native u32 wrapping operations (already O(1)).
//!
//! # Q4+ (40-bit and beyond): Algorithmic Only (Design Only)
//!
//! uor-foundation defines Q0–Q3 as named constants on the open-world
//! `WittLevel` struct. Arbitrary levels can be constructed via
//! `WittLevel::new(k)` with bit width `8*(k+1)`. Future levels follow
//! the +8 bits/level pattern:
//!
//! - **Q4 (40-bit)**: 2^40 = 1.1 trillion states. Tables completely infeasible.
//! - **Strategy**: Algorithmic computation only, with optional LRU cache for
//!   frequently accessed values. Composition via function chaining rather than
//!   table fusion.

use hologram_foundation::WittLevel;

/// Bit width for a given quantum level: `8 * (k + 1)`.
#[inline]
#[must_use]
pub const fn quantum_bit_width(level: WittLevel) -> u64 {
    level.bits_width() as u64
}

/// Modulus (ring size) for a given quantum level: 2^bits.
#[inline]
#[must_use]
pub const fn quantum_modulus(level: WittLevel) -> u64 {
    match level.cycle_size() {
        Some(s) => s as u64,
        None => 0,
    }
}

/// Number of entries in a full table at this quantum level.
#[inline]
#[must_use]
pub const fn quantum_table_entries(level: WittLevel) -> u64 {
    quantum_modulus(level)
}

/// Whether a full activation table is feasible at this quantum level.
///
/// W8 and W16 tables fit in L2/L3 cache. W24 is borderline (~50MB).
/// W32 and above are infeasible (17GB+).
#[inline]
#[must_use]
pub const fn quantum_is_table_feasible(level: WittLevel) -> bool {
    // v0.2.0: use bit width directly. W8/W16 are feasible (≤16 bits).
    level.witt_length() <= 16
}

/// Memory per activation table in bytes at this quantum level.
///
/// `entries * bytes_per_entry` where bytes_per_entry = ceil(bits / 8).
#[inline]
#[must_use]
pub const fn quantum_table_size_bytes(level: WittLevel) -> u64 {
    let entries = quantum_table_entries(level);
    let bytes_per = quantum_bit_width(level).div_ceil(8);
    entries * bytes_per
}

// ── Q2 helpers ──────────────────────────────────────────────────

/// Stratum (popcount) for Q2 (24-bit) values.
#[inline]
#[must_use]
pub const fn q2_stratum(value: u32) -> u8 {
    (value & 0x00FF_FFFF).count_ones() as u8
}

/// Curvature for Q2: `hamming(value, value + 1)` masked to 24 bits.
#[inline]
#[must_use]
pub const fn q2_curvature(value: u32) -> u8 {
    let masked = value & 0x00FF_FFFF;
    let next = masked.wrapping_add(1) & 0x00FF_FFFF;
    (masked ^ next).count_ones() as u8
}

/// Addition in Z/2^24 Z.
#[inline]
#[must_use]
pub const fn q2_add(a: u32, b: u32) -> u32 {
    a.wrapping_add(b) & 0x00FF_FFFF
}

/// Subtraction in Z/2^24 Z.
#[inline]
#[must_use]
pub const fn q2_sub(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b) & 0x00FF_FFFF
}

/// Multiplication in Z/2^24 Z.
#[inline]
#[must_use]
pub const fn q2_mul(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b) & 0x00FF_FFFF
}

// ── Q3 helpers ──────────────────────────────────────────────────

/// Stratum (popcount) for Q3 (32-bit) values.
#[inline]
#[must_use]
pub const fn q3_stratum(value: u32) -> u8 {
    value.count_ones() as u8
}

/// Curvature for Q3: `hamming(value, value + 1)`.
#[inline]
#[must_use]
pub const fn q3_curvature(value: u32) -> u8 {
    (value ^ value.wrapping_add(1)).count_ones() as u8
}

/// Addition in Z/2^32 Z.
#[inline]
#[must_use]
pub const fn q3_add(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}

/// Subtraction in Z/2^32 Z.
#[inline]
#[must_use]
pub const fn q3_sub(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b)
}

/// Multiplication in Z/2^32 Z.
#[inline]
#[must_use]
pub const fn q3_mul(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b)
}

/// Negation (additive inverse) in Z/2^32 Z.
#[inline]
#[must_use]
pub const fn q3_neg(a: u32) -> u32 {
    a.wrapping_neg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_widths() {
        assert_eq!(quantum_bit_width(WittLevel::W8), 8);
        assert_eq!(quantum_bit_width(WittLevel::W16), 16);
        assert_eq!(quantum_bit_width(WittLevel::W24), 24);
        assert_eq!(quantum_bit_width(WittLevel::W32), 32);
    }

    #[test]
    fn moduli() {
        assert_eq!(quantum_modulus(WittLevel::W8), 256);
        assert_eq!(quantum_modulus(WittLevel::W16), 65536);
        assert_eq!(quantum_modulus(WittLevel::W24), 16_777_216);
        assert_eq!(quantum_modulus(WittLevel::W32), 4_294_967_296);
    }

    #[test]
    fn table_feasibility() {
        assert!(quantum_is_table_feasible(WittLevel::W8));
        assert!(quantum_is_table_feasible(WittLevel::W16));
        assert!(!quantum_is_table_feasible(WittLevel::W24));
        assert!(!quantum_is_table_feasible(WittLevel::W32));
    }

    #[test]
    fn table_sizes() {
        // Q0: 256 entries * 1 byte = 256
        assert_eq!(quantum_table_size_bytes(WittLevel::W8), 256);
        // Q1: 65536 entries * 2 bytes = 131072
        assert_eq!(quantum_table_size_bytes(WittLevel::W16), 131_072);
        // Q2: 16.7M entries * 3 bytes ≈ 50.3MB
        assert_eq!(quantum_table_size_bytes(WittLevel::W24), 50_331_648);
        // Q3: 4.3B entries * 4 bytes = 17.2GB
        assert_eq!(quantum_table_size_bytes(WittLevel::W32), 17_179_869_184);
    }

    #[test]
    fn q2_stratum_values() {
        assert_eq!(q2_stratum(0), 0);
        assert_eq!(q2_stratum(0x00FF_FFFF), 24);
        assert_eq!(q2_stratum(0xFF00_0000), 0); // high bits masked out
        assert_eq!(q2_stratum(0x00AA_AAAA), 12);
    }

    #[test]
    fn q2_curvature_values() {
        assert_eq!(q2_curvature(0), 1); // 0→1, 1 bit changes
        assert_eq!(q2_curvature(0x00FF_FFFF), 24); // all bits flip
    }

    #[test]
    fn q2_add_wraps() {
        assert_eq!(q2_add(0x00FF_FFFF, 1), 0);
        assert_eq!(q2_add(100, 200), 300);
    }

    #[test]
    fn q3_stratum_values() {
        assert_eq!(q3_stratum(0), 0);
        assert_eq!(q3_stratum(u32::MAX), 32);
        assert_eq!(q3_stratum(0xAAAA_AAAA), 16);
    }

    #[test]
    fn q3_curvature_values() {
        assert_eq!(q3_curvature(0), 1);
        assert_eq!(q3_curvature(u32::MAX), 32); // 0xFFFF_FFFF → 0, all bits flip
    }

    #[test]
    fn q3_add_wraps() {
        assert_eq!(q3_add(u32::MAX, 1), 0);
        assert_eq!(q3_add(100, 200), 300);
    }

    #[test]
    fn q3_sub_wrapping() {
        assert_eq!(q3_sub(0, 1), u32::MAX); // underflow wraps
        assert_eq!(q3_sub(100, 50), 50);
        assert_eq!(q3_sub(u32::MAX, u32::MAX), 0);
    }

    #[test]
    fn q3_mul_wrapping() {
        assert_eq!(q3_mul(2, 3), 6);
        assert_eq!(q3_mul(0, u32::MAX), 0);
        assert_eq!(q3_mul(u32::MAX, 2), u32::MAX - 1); // wrapping
    }

    #[test]
    fn q3_neg_involution() {
        for x in [0u32, 1, 127, u32::MAX / 2, u32::MAX] {
            assert_eq!(q3_neg(q3_neg(x)), x);
        }
    }

    #[test]
    fn q3_add_sub_inverse() {
        for a in [0u32, 1, 0xFFFF, u32::MAX] {
            assert_eq!(q3_sub(a, a), 0);
            assert_eq!(q3_add(a, q3_neg(a)), 0);
        }
    }

    #[test]
    fn table_entries_matches_modulus() {
        for level in [
            WittLevel::W8,
            WittLevel::W16,
            WittLevel::W24,
            WittLevel::W32,
        ] {
            assert_eq!(quantum_table_entries(level), quantum_modulus(level));
        }
    }
}
