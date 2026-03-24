//! Frequency counting and normalization for entropy coding.

use alloc::vec::Vec;

/// Maximum number of distinct symbols we support.
pub const MAX_SYMBOLS: usize = 256;

/// Power-of-2 denominator for normalized frequencies.
/// 2^14 = 16384 — large enough for good precision, small enough for fast rANS.
pub const FREQ_TOTAL_BITS: u32 = 14;
pub const FREQ_TOTAL: u32 = 1 << FREQ_TOTAL_BITS;

/// Raw frequency table: count of each symbol.
pub fn count_frequencies(data: &[u8], num_symbols: usize) -> Vec<u32> {
    let mut freq = alloc::vec![0u32; num_symbols];
    for &sym in data {
        freq[sym as usize] += 1;
    }
    freq
}

/// Normalize frequencies to sum to `FREQ_TOTAL`.
///
/// Ensures every symbol with count > 0 gets at least frequency 1.
/// Uses a proportional scaling with leftover distribution.
pub fn normalize_frequencies(raw: &[u32]) -> Vec<u32> {
    let total_raw: u64 = raw.iter().map(|&c| c as u64).sum();
    if total_raw == 0 {
        return alloc::vec![0u32; raw.len()];
    }

    let n = raw.len();
    let mut norm = alloc::vec![0u32; n];

    // Count non-zero symbols and assign minimum frequency of 1.
    let non_zero = raw.iter().filter(|&&c| c > 0).count() as u32;
    let remaining = FREQ_TOTAL.saturating_sub(non_zero);

    // Proportional allocation of the remaining budget.
    let mut allocated = 0u32;
    for i in 0..n {
        if raw[i] > 0 {
            let share = (raw[i] as u64 * remaining as u64 / total_raw) as u32;
            norm[i] = 1 + share;
            allocated += norm[i];
        }
    }

    // Distribute rounding remainder to the largest symbols.
    let target = FREQ_TOTAL;
    if allocated < target {
        let mut diff = target - allocated;
        // Find indices of non-zero symbols sorted by raw frequency (descending).
        let mut indices: Vec<usize> = (0..n).filter(|&i| raw[i] > 0).collect();
        indices.sort_unstable_by(|&a, &b| raw[b].cmp(&raw[a]));
        for &idx in indices.iter().cycle() {
            if diff == 0 {
                break;
            }
            norm[idx] += 1;
            diff -= 1;
        }
    } else if allocated > target {
        let mut diff = allocated - target;
        // Remove from the largest normalized symbols (but keep >= 1).
        let mut indices: Vec<usize> = (0..n).filter(|&i| norm[i] > 1).collect();
        indices.sort_unstable_by(|&a, &b| norm[b].cmp(&norm[a]));
        for &idx in indices.iter().cycle() {
            if diff == 0 {
                break;
            }
            if norm[idx] > 1 {
                norm[idx] -= 1;
                diff -= 1;
            }
        }
    }

    debug_assert_eq!(
        norm.iter().sum::<u32>(),
        FREQ_TOTAL,
        "normalized frequencies must sum to FREQ_TOTAL"
    );
    norm
}

/// Build cumulative frequency table from normalized frequencies.
/// `cum[i]` = sum of `freq[0..i]`.
pub fn cumulative_frequencies(freq: &[u32]) -> Vec<u32> {
    let mut cum = Vec::with_capacity(freq.len() + 1);
    cum.push(0);
    let mut acc = 0u32;
    for &f in freq {
        acc += f;
        cum.push(acc);
    }
    cum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_uniform() {
        let raw = alloc::vec![10u32; 256];
        let norm = normalize_frequencies(&raw);
        assert_eq!(norm.iter().sum::<u32>(), FREQ_TOTAL);
        // Uniform should give each symbol ~64
        for &f in &norm {
            assert!((60..=68).contains(&f));
        }
    }

    #[test]
    fn normalize_sparse() {
        let mut raw = alloc::vec![0u32; 256];
        raw[0] = 100;
        raw[1] = 1;
        let norm = normalize_frequencies(&raw);
        assert_eq!(norm.iter().sum::<u32>(), FREQ_TOTAL);
        assert!(norm[0] > norm[1]);
        assert!(norm[1] >= 1); // Non-zero symbols get at least 1
    }

    #[test]
    fn normalize_single_symbol() {
        let mut raw = alloc::vec![0u32; 256];
        raw[42] = 1000;
        let norm = normalize_frequencies(&raw);
        assert_eq!(norm.iter().sum::<u32>(), FREQ_TOTAL);
        assert_eq!(norm[42], FREQ_TOTAL);
    }

    #[test]
    fn normalize_empty() {
        let raw = alloc::vec![0u32; 256];
        let norm = normalize_frequencies(&raw);
        assert_eq!(norm.iter().sum::<u32>(), 0);
    }

    #[test]
    fn cumulative_correct() {
        let freq = alloc::vec![3, 5, 0, 2];
        let cum = cumulative_frequencies(&freq);
        assert_eq!(cum, alloc::vec![0, 3, 8, 8, 10]);
    }
}
