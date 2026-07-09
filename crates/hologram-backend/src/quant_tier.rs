//! The quantized-weight **tier registry** — one declaration per encoding.
//!
//! Adding a tier used to mean editing ~8 independent `match` arms across six
//! files (dtype tables, `in_bytes`, `dtype_ok`, the fusion eligibility test, the
//! k-divisibility guard, the output-major repack, the kernel dispatch), several
//! of which had `_ =>` fallbacks that silently miscomputed rather than failed.
//! A tier is now a single [`QuantTier`] row; the compiler and the backend both
//! read it, so an unregistered tag is a typed `None`, never a wrong answer.
//!
//! ## The one shape that covers every tier
//!
//! A quantized `[k, n]` weight is a grid of **stored units**. One unit encodes
//! `group_dim` logical weights and occupies `unit_bits = group_dim ×
//! bits_per_weight` bits:
//!
//! | tier   | `group_dim` | `bits_per_weight` | `unit_bits` | grid          |
//! |--------|-------------|-------------------|-------------|---------------|
//! | `i8`   | 1           | 8                 | 8           | `[k, n]` bytes |
//! | `u8`   | 1           | 8                 | 8           | `[k, n]` bytes |
//! | `i4`   | 1           | 4                 | 4           | `[k, n]` nibbles |
//! | `e8cb` | 8           | 1                 | 8           | `[k/8, n]` index bytes |
//!
//! So the output-major repack — which used to be three hand-written branches —
//! is one transpose of a `[rows, n]` grid of `unit_bits`-wide cells, with
//! `rows = k / group_dim`. Only the cell width (4 or 8 bits) varies.

use alloc::vec;
use alloc::vec::Vec;
use hologram_types::DTypeId;

/// Everything the compiler and the backend need to know about one quantized
/// weight encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantTier {
    pub dtype: DTypeId,
    /// Logical weights encoded by a single stored unit. `1` for the scalar
    /// tiers; `8` for the E8 codebook, because E8 is an 8-dimensional lattice
    /// and one index names one lattice point (definitional, not a knob).
    pub group_dim: u32,
    /// Bits of storage per logical weight.
    pub bits_per_weight: u32,
    /// Decoding needs the model's codebook as a constant operand.
    pub needs_codebook: bool,
    /// The scalar (W8A32) dequant loop can read this tier directly, without
    /// extra operands.
    pub scalar_decodable: bool,
    /// A fused output-major W8A8 decode GEMV exists for this tier.
    pub omajor_fusable: bool,
}

impl QuantTier {
    /// Bits in one stored unit. Always 4 or 8 — the two cell widths the repack
    /// knows how to transpose.
    #[must_use]
    #[inline]
    pub const fn unit_bits(&self) -> u32 {
        self.group_dim * self.bits_per_weight
    }

    /// `k` must be a whole number of groups for this tier.
    #[must_use]
    #[inline]
    pub const fn divides_k(&self, k: usize) -> bool {
        k.is_multiple_of(self.group_dim as usize)
    }

    /// Rows of stored units in a `[k, n]` weight.
    #[must_use]
    #[inline]
    pub const fn rows(&self, k: usize) -> usize {
        k / self.group_dim as usize
    }

    /// Stored bytes for a `[k, n]` weight of this tier.
    #[must_use]
    pub const fn weight_bytes(&self, k: usize, n: usize) -> Option<usize> {
        match self.dtype.storage_bytes_u64((k as u64) * (n as u64)) {
            Some(b) => Some(b as usize),
            None => None,
        }
    }

    /// Transpose the input-major `[rows, n]` unit grid into the output-major
    /// `[n, rows]` layout the decode GEMV streams (each output column's units
    /// contiguous). Baked into the archive at compile time; zero runtime copy.
    ///
    /// `None` if `src` is not exactly the tier's `[k, n]` storage, or `k` is not
    /// a whole number of groups.
    #[must_use]
    pub fn omajor_repack(&self, src: &[u8], k: usize, n: usize) -> Option<Vec<u8>> {
        if !self.divides_k(k) || src.len() != self.weight_bytes(k, n)? {
            return None;
        }
        let rows = self.rows(k);
        match self.unit_bits() {
            8 => {
                let mut t = vec![0u8; rows * n];
                for r in 0..rows {
                    let row = &src[r * n..(r + 1) * n];
                    for (j, &cell) in row.iter().enumerate() {
                        t[j * rows + r] = cell;
                    }
                }
                Some(t)
            }
            4 => {
                // Nibble cells: unit `i` lives in the low nibble of byte `i/2`
                // when `i` is even, the high nibble otherwise (archive order).
                let mut t = vec![0u8; (rows * n).div_ceil(2)];
                for r in 0..rows {
                    for j in 0..n {
                        let s = r * n + j;
                        let byte = src[s >> 1];
                        let nib = if s & 1 == 0 { byte & 0x0F } else { byte >> 4 };
                        let d = j * rows + r;
                        t[d >> 1] |= if d & 1 == 0 { nib } else { nib << 4 };
                    }
                }
                Some(t)
            }
            _ => None,
        }
    }
}

/// The registry. One row per encoding; adding a tier is adding a row (plus its
/// kernel). Order is irrelevant — lookup is by dtype tag.
const TIERS: &[QuantTier] = &[
    QuantTier {
        dtype: DTypeId::I8,
        group_dim: 1,
        bits_per_weight: 8,
        needs_codebook: false,
        scalar_decodable: true,
        omajor_fusable: true,
    },
    QuantTier {
        dtype: DTypeId::U8,
        group_dim: 1,
        bits_per_weight: 8,
        needs_codebook: false,
        scalar_decodable: true,
        // Asymmetric-by-convention (ONNX default); the fused W8A8 GEMV assumes
        // a signed, symmetric weight, so u8 stays on the generic dequant path.
        omajor_fusable: false,
    },
    QuantTier {
        dtype: DTypeId::I4,
        group_dim: 1,
        bits_per_weight: 4,
        needs_codebook: false,
        scalar_decodable: true,
        omajor_fusable: true,
    },
    QuantTier {
        dtype: DTypeId::E8CB,
        group_dim: DTypeId::E8CB_GROUP_DIM,
        bits_per_weight: 1,
        needs_codebook: true,
        // Weights are codebook indices — meaningless read as raw bytes.
        scalar_decodable: false,
        omajor_fusable: true,
    },
];

/// The tier for a dtype tag, or `None` if the tag names no quantized weight
/// encoding this build understands. Callers must propagate the `None`.
#[must_use]
pub fn quant_tier(dtype: DTypeId) -> Option<&'static QuantTier> {
    let mut i = 0;
    while i < TIERS.len() {
        if TIERS[i].dtype.raw() == dtype.raw() {
            return Some(&TIERS[i]);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tier_has_a_four_or_eight_bit_unit() {
        for t in TIERS {
            assert!(
                t.unit_bits() == 4 || t.unit_bits() == 8,
                "{}: unit_bits {} — the repack only transposes 4/8-bit cells",
                t.dtype.name(),
                t.unit_bits()
            );
            // The registry must agree with the canonical storage formula.
            let (k, n) = (16usize, 3usize);
            let cells = t.rows(k) * n;
            let expect = match t.unit_bits() {
                8 => cells,
                _ => cells.div_ceil(2),
            };
            assert_eq!(t.weight_bytes(k, n), Some(expect), "{}", t.dtype.name());
        }
    }

    #[test]
    fn unknown_and_non_quant_tags_have_no_tier() {
        assert!(quant_tier(DTypeId::F32).is_none());
        assert!(quant_tier(DTypeId(200)).is_none());
        assert!(quant_tier(DTypeId::I8).is_some());
        assert!(quant_tier(DTypeId::E8CB).is_some());
    }

    /// The 8-bit repack is a plain `[rows, n] → [n, rows]` transpose.
    #[test]
    fn byte_repack_transposes_the_unit_grid() {
        let t = quant_tier(DTypeId::I8).unwrap();
        let (k, n) = (3usize, 2usize);
        // [k,n] row-major: rows r=0..3, cols j=0..2
        let src = vec![0u8, 1, 10, 11, 20, 21];
        let got = t.omajor_repack(&src, k, n).unwrap();
        // [n,k]: column 0 then column 1
        assert_eq!(got, vec![0, 10, 20, 1, 11, 21]);
    }

    /// E8CB has `group_dim = 8`, so its grid is `[k/8, n]` index bytes.
    #[test]
    fn codebook_repack_transposes_the_index_grid() {
        let t = quant_tier(DTypeId::E8CB).unwrap();
        let (k, n) = (16usize, 3usize); // rows = 2
        assert_eq!(t.rows(k), 2);
        assert_eq!(t.weight_bytes(k, n), Some(6));
        let src = vec![0u8, 1, 2, 10, 11, 12]; // [2,3]
        let got = t.omajor_repack(&src, k, n).unwrap();
        assert_eq!(got, vec![0, 10, 1, 11, 2, 12]); // [3,2]
    }

    /// The 4-bit repack moves nibbles, low-nibble-first, and round-trips.
    #[test]
    fn nibble_repack_round_trips() {
        let t = quant_tier(DTypeId::I4).unwrap();
        let (k, n) = (4usize, 2usize); // 8 cells -> 4 bytes
        let src: Vec<u8> = vec![0x10, 0x32, 0x54, 0x76]; // cells 0..7 = 0,1,2,...,7
        let got = t.omajor_repack(&src, k, n).unwrap();
        // Read cell `i` out of a packed buffer.
        let cell = |b: &[u8], i: usize| -> u8 {
            let byte = b[i >> 1];
            if i & 1 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            }
        };
        for r in 0..k {
            for j in 0..n {
                assert_eq!(cell(&src, r * n + j), cell(&got, j * k + r), "({r},{j})");
            }
        }
    }

    #[test]
    fn repack_rejects_bad_lengths_and_ragged_k() {
        let t = quant_tier(DTypeId::E8CB).unwrap();
        // k not a whole number of 8-element groups.
        assert!(t.omajor_repack(&[0; 6], 12, 3).is_none());
        // src length disagrees with the tier's storage formula.
        assert!(t.omajor_repack(&[0; 5], 16, 3).is_none());
    }
}
