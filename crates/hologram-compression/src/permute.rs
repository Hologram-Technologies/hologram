//! Bijective pre-transforms using ElementWiseView.
//!
//! Applying a bijective permutation before compression can reduce entropy
//! by reordering the byte space to better match the data's structure.
//! Since the permutation is invertible, the transform is lossless.

use hologram_core::view::ElementWiseView;

/// Known permutation IDs stored in the compressed block header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PermuteId {
    /// Identity (no transform).
    Identity = 0,
    /// Gray code: maps i → i ^ (i >> 1). Adjacent values differ by 1 bit.
    GrayCode = 1,
    /// Stratum sort: reorder bytes so low-stratum values come first.
    StratumSort = 2,
    /// Neg-complement: maps i → neg(bnot(i)) = i + 1 (successor permutation).
    NegComplement = 3,
}

impl PermuteId {
    /// Parse from a raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Identity),
            1 => Some(Self::GrayCode),
            2 => Some(Self::StratumSort),
            3 => Some(Self::NegComplement),
            _ => None,
        }
    }

    /// All available permutations for auto-selection.
    pub fn all() -> &'static [Self] {
        &[
            Self::Identity,
            Self::GrayCode,
            Self::StratumSort,
            Self::NegComplement,
        ]
    }
}

/// Build the forward ElementWiseView for a given permutation.
pub fn forward(id: PermuteId) -> ElementWiseView {
    match id {
        PermuteId::Identity => ElementWiseView::identity(),
        PermuteId::GrayCode => ElementWiseView::new(|x| x ^ (x >> 1)),
        PermuteId::StratumSort => build_stratum_sort(),
        PermuteId::NegComplement => ElementWiseView::new(|x| x.wrapping_add(1)),
    }
}

/// Build the inverse ElementWiseView for a given permutation.
pub fn inverse(id: PermuteId) -> ElementWiseView {
    // All our permutations are bijective, so inverse() always succeeds.
    forward(id)
        .inverse()
        .expect("all PermuteId permutations are bijective")
}

/// Build the stratum-sort permutation: reorder bytes by (stratum, value).
/// Bytes with low popcount come first, then higher.
fn build_stratum_sort() -> ElementWiseView {
    use hologram_core::lut::q0::stratum_q0;

    // Collect all 256 bytes sorted by (stratum, value).
    let mut sorted: [(u8, u8); 256] = [(0, 0); 256];
    for i in 0..=255u8 {
        sorted[i as usize] = (stratum_q0(i), i);
    }
    sorted.sort_by_key(|&(s, v)| (s, v));

    // Build the permutation table: original_byte → position_in_sorted_order.
    let mut table = [0u8; 256];
    for (pos, &(_s, val)) in sorted.iter().enumerate() {
        table[val as usize] = pos as u8;
    }
    ElementWiseView::from_table(table)
}

/// Auto-select the best permutation for given data by trial-encoding a sample.
///
/// Uses a simple entropy estimate: count distinct residuals after ring-diff
/// with each permutation applied. Lower distinct count → better compression.
pub fn auto_select(data: &[u8]) -> PermuteId {
    if data.is_empty() {
        return PermuteId::Identity;
    }

    // Sample up to 1024 bytes.
    let sample = if data.len() > 1024 {
        &data[..1024]
    } else {
        data
    };

    let mut best_id = PermuteId::Identity;
    let mut best_entropy = f64::MAX;

    for &id in PermuteId::all() {
        let view = forward(id);
        // Apply permutation to sample, then estimate entropy.
        let mut transformed = alloc::vec![0u8; sample.len()];
        view.apply_to(sample, &mut transformed);

        let entropy = byte_entropy(&transformed);
        if entropy < best_entropy {
            best_entropy = entropy;
            best_id = id;
        }
    }

    best_id
}

/// Estimate Shannon entropy of a byte stream (bits per byte).
fn byte_entropy(data: &[u8]) -> f64 {
    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let n = data.len() as f64;
    let mut h = 0.0;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn all_permutations_bijective() {
        for &id in PermuteId::all() {
            let fwd = forward(id);
            assert!(fwd.is_bijective(), "{id:?} is not bijective");
        }
    }

    #[test]
    fn all_permutations_round_trip() {
        for &id in PermuteId::all() {
            let fwd = forward(id);
            let inv = inverse(id);
            for i in 0..=255u8 {
                assert_eq!(inv.apply(fwd.apply(i)), i, "{id:?} failed at {i}");
            }
        }
    }

    #[test]
    fn gray_code_adjacent_differ_by_one_bit() {
        let gray = forward(PermuteId::GrayCode);
        for i in 0..255u8 {
            let a = gray.apply(i);
            let b = gray.apply(i + 1);
            let diff_bits = (a ^ b).count_ones();
            assert_eq!(
                diff_bits,
                1,
                "Gray({i}) and Gray({}) differ by {diff_bits} bits",
                i + 1
            );
        }
    }

    #[test]
    fn stratum_sort_groups_by_popcount() {
        let ss = forward(PermuteId::StratumSort);
        // Byte 0 (stratum 0) should map to position 0
        assert_eq!(ss.apply(0), 0);
        // Byte 255 (stratum 8) should map to position 255
        assert_eq!(ss.apply(255), 255);
    }

    #[test]
    fn neg_complement_is_successor() {
        let nc = forward(PermuteId::NegComplement);
        for i in 0..=255u8 {
            assert_eq!(nc.apply(i), i.wrapping_add(1));
        }
    }

    #[test]
    fn auto_select_returns_valid() {
        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let id = auto_select(&data);
        // Just verify it returns one of the valid IDs.
        assert!(PermuteId::all().contains(&id));
    }
}
