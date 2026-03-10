//! Stratum-Partitioned Entropy Coding (SPEC).
//!
//! Observable: `ByteDatum::stratum()` (popcount) partitions Z/256Z into 9
//! equivalence classes. Within each class, a byte is identified by its rank
//! (index among all bytes with the same popcount).
//!
//! This factoring exposes redundancy: when data clusters in certain strata,
//! the stratum stream has low entropy and the intra-stratum ranks use fewer
//! bits than a full 8-bit encoding.

use alloc::vec::Vec;
use hologram_core::lut::q0::stratum_q0;

/// Number of bytes with popcount k: C(8, k).
pub static STRATUM_SIZES: [u8; 9] = [1, 8, 28, 56, 70, 56, 28, 8, 1];

/// Cumulative offset into the sorted-by-stratum ordering.
/// STRATUM_OFFSETS[k] = sum of STRATUM_SIZES[0..k].
pub static STRATUM_OFFSETS: [u8; 9] = [0, 1, 9, 37, 93, 163, 219, 247, 255];

/// For each byte value (0..256), its (stratum, intra-stratum rank).
/// Built at compile time.
static BYTE_TO_STRATUM_RANK: [(u8, u8); 256] = {
    // First pass: for each stratum, count how many we've seen so far.
    let mut table = [(0u8, 0u8); 256];
    let mut counts = [0u8; 9];
    let mut i = 0u16;
    while i < 256 {
        let s = (i as u8).count_ones() as u8;
        table[i as usize] = (s, counts[s as usize]);
        counts[s as usize] += 1;
        i += 1;
    }
    table
};

/// Inverse table: (stratum, rank) → byte value.
/// Indexed as STRATUM_RANK_TO_BYTE[stratum][rank].
static STRATUM_RANK_TO_BYTE: [[u8; 70]; 9] = {
    let mut table = [[0u8; 70]; 9];
    let mut i = 0u16;
    while i < 256 {
        let (s, r) = BYTE_TO_STRATUM_RANK[i as usize];
        table[s as usize][r as usize] = i as u8;
        i += 1;
    }
    table
};

/// Map a byte to its (stratum, intra-stratum rank).
#[inline]
pub fn to_stratum_rank(byte: u8) -> (u8, u8) {
    BYTE_TO_STRATUM_RANK[byte as usize]
}

/// Reconstruct a byte from (stratum, rank).
#[inline]
pub fn from_stratum_rank(stratum: u8, rank: u8) -> u8 {
    STRATUM_RANK_TO_BYTE[stratum as usize][rank as usize]
}

/// Encode data into separate stratum and rank streams.
///
/// Returns `(strata, ranks)` where:
/// - `strata[i]` is the popcount of `data[i]` (range 0..=8)
/// - `ranks[i]` is the intra-stratum rank of `data[i]`
pub fn encode(data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut strata = Vec::with_capacity(data.len());
    let mut ranks = Vec::with_capacity(data.len());
    for &byte in data {
        let (s, r) = to_stratum_rank(byte);
        strata.push(s);
        ranks.push(r);
    }
    (strata, ranks)
}

/// Decode stratum + rank streams back to original bytes.
pub fn decode(strata: &[u8], ranks: &[u8]) -> Vec<u8> {
    debug_assert_eq!(strata.len(), ranks.len());
    let mut out = Vec::with_capacity(strata.len());
    for (&s, &r) in strata.iter().zip(ranks.iter()) {
        out.push(from_stratum_rank(s, r));
    }
    out
}

/// Compute a stratum histogram for a byte slice.
pub fn histogram(data: &[u8]) -> [u32; 9] {
    let mut h = [0u32; 9];
    for &byte in data {
        h[stratum_q0(byte) as usize] += 1;
    }
    h
}

/// Compute the Shannon entropy of the stratum distribution in bits.
pub fn stratum_entropy(data: &[u8]) -> f64 {
    let h = histogram(data);
    let n = data.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &count in &h {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_all_bytes() {
        for i in 0..=255u8 {
            let (s, r) = to_stratum_rank(i);
            assert_eq!(from_stratum_rank(s, r), i);
        }
    }

    #[test]
    fn stratum_sizes_sum_to_256() {
        let sum: u16 = STRATUM_SIZES.iter().map(|&s| s as u16).sum();
        assert_eq!(sum, 256);
    }

    #[test]
    fn stratum_sizes_match_binomial() {
        // C(8,k) for k=0..8
        let expected = [1, 8, 28, 56, 70, 56, 28, 8, 1];
        assert_eq!(STRATUM_SIZES, expected);
    }

    #[test]
    fn encode_decode_round_trip() {
        let data: Vec<u8> = (0..=255).collect();
        let (strata, ranks) = encode(&data);
        let recovered = decode(&strata, &ranks);
        assert_eq!(data, recovered);
    }

    #[test]
    fn stratum_complement_symmetry() {
        // stratum(bnot(x)) == 8 - stratum(x)
        for i in 0..=255u8 {
            let (s, _) = to_stratum_rank(i);
            let (s_bnot, _) = to_stratum_rank(!i);
            assert_eq!(s_bnot, 8 - s);
        }
    }

    #[test]
    fn histogram_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let h = histogram(&data);
        for (k, &count) in h.iter().enumerate() {
            assert_eq!(count, STRATUM_SIZES[k] as u32);
        }
    }
}
