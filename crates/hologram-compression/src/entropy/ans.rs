//! rANS (range Asymmetric Numeral Systems) encoder/decoder.
//!
//! A modern entropy coder achieving near-Shannon-optimal compression.
//! Pure no_std, no external dependencies.
//!
//! 32-bit rANS with byte-level output. State lives in [RANS_L, RANS_L << 8).
//! Encodes in reverse order, decodes in forward order.

use alloc::vec::Vec;

use super::histogram::{FREQ_TOTAL, FREQ_TOTAL_BITS};

/// Lower bound of the rANS state range.
/// State is maintained in [RANS_L, RANS_L << 8) = [2^23, 2^31).
const RANS_L: u32 = 1 << 23;

/// Encode a sequence of symbols using rANS.
///
/// - `symbols`: the symbols to encode (each < `num_symbols`)
/// - `freq`: normalized frequency table (sums to `FREQ_TOTAL`)
/// - `cum`: cumulative frequency table (`cum[i]` = sum of `freq[0..i]`)
///
/// Returns the compressed byte stream.
pub fn encode(symbols: &[u8], freq: &[u32], cum: &[u32]) -> Vec<u8> {
    if symbols.is_empty() {
        return Vec::new();
    }

    let mut output: Vec<u8> = Vec::new();
    let mut state: u32 = RANS_L;

    // Encode in reverse order (rANS property).
    for &sym in symbols.iter().rev() {
        let s = sym as usize;
        let f = freq[s];
        if f == 0 {
            continue;
        }

        // Renormalize: flush bytes while state is too large.
        // max_state = ((RANS_L >> FREQ_TOTAL_BITS) << 8) * f
        // With RANS_L=2^23, FREQ_TOTAL_BITS=14: (2^9 << 8) * f = 2^17 * f
        // For f=16384(2^14): 2^17 * 2^14 = 2^31 — fits in u32.
        let max_state = ((RANS_L >> FREQ_TOTAL_BITS) << 8) * f;
        while state >= max_state {
            output.push((state & 0xFF) as u8);
            state >>= 8;
        }

        // Encode: state' = (state / f) * FREQ_TOTAL + cum[s] + (state % f)
        state = (state / f) * FREQ_TOTAL + cum[s] + (state % f);
    }

    // Flush final state as 4 bytes (little-endian).
    output.push((state & 0xFF) as u8);
    output.push(((state >> 8) & 0xFF) as u8);
    output.push(((state >> 16) & 0xFF) as u8);
    output.push(((state >> 24) & 0xFF) as u8);

    // Reverse so decoder reads forward.
    output.reverse();
    output
}

/// Decode symbols from a rANS byte stream.
///
/// - `data`: the compressed byte stream
/// - `freq`: normalized frequency table
/// - `cum`: cumulative frequency table
/// - `num_symbols`: number of distinct symbols
/// - `count`: how many symbols to decode
///
/// Returns the decoded symbols.
pub fn decode(data: &[u8], freq: &[u32], cum: &[u32], num_symbols: usize, count: usize) -> Vec<u8> {
    if count == 0 || data.len() < 4 {
        return Vec::new();
    }

    // Build reverse CDF lookup table for O(1) symbol identification.
    let mut sym_lookup = alloc::vec![0u8; FREQ_TOTAL as usize];
    for s in 0..num_symbols {
        let start = cum[s] as usize;
        let end = cum[s + 1] as usize;
        for entry in &mut sym_lookup[start..end] {
            *entry = s as u8;
        }
    }

    let mut pos = 0usize;

    // Initialize state from first 4 bytes (big-endian after reverse).
    let mut state: u32 = 0;
    for _ in 0..4 {
        state = (state << 8) | data[pos] as u32;
        pos += 1;
    }

    let mut output = Vec::with_capacity(count);

    for _ in 0..count {
        // Identify symbol from state.
        let slot = (state & (FREQ_TOTAL - 1)) as usize;
        let sym = sym_lookup[slot];
        let s = sym as usize;
        let f = freq[s];
        let c = cum[s];

        output.push(sym);

        // Advance state.
        state = f * (state >> FREQ_TOTAL_BITS) + (state & (FREQ_TOTAL - 1)) - c;

        // Renormalize: read bytes while state is below threshold.
        while state < RANS_L && pos < data.len() {
            state = (state << 8) | data[pos] as u32;
            pos += 1;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::histogram::{
        count_frequencies, cumulative_frequencies, normalize_frequencies,
    };
    use alloc::vec;
    use alloc::vec::Vec;

    fn round_trip(data: &[u8], num_symbols: usize) {
        let raw = count_frequencies(data, num_symbols);
        let freq = normalize_frequencies(&raw);
        let cum = cumulative_frequencies(&freq);

        let compressed = encode(data, &freq, &cum);
        let decoded = decode(&compressed, &freq, &cum, num_symbols, data.len());
        assert_eq!(data, decoded.as_slice(), "rANS round-trip failed");
    }

    #[test]
    fn round_trip_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        round_trip(&data, 256);
    }

    #[test]
    fn round_trip_single_symbol() {
        let data = vec![42u8; 100];
        round_trip(&data, 256);
    }

    #[test]
    fn round_trip_two_symbols() {
        let data: Vec<u8> = (0..200).map(|i| if i % 3 == 0 { 0 } else { 1 }).collect();
        round_trip(&data, 256);
    }

    #[test]
    fn round_trip_skewed() {
        let mut data = vec![0u8; 900];
        for i in 0..100 {
            data.push((i % 10) as u8);
        }
        round_trip(&data, 256);
    }

    #[test]
    fn compression_skewed() {
        let data = vec![0u8; 1000];
        let raw = count_frequencies(&data, 256);
        let freq = normalize_frequencies(&raw);
        let cum = cumulative_frequencies(&freq);
        let compressed = encode(&data, &freq, &cum);
        assert!(
            compressed.len() < 100,
            "expected good compression, got {} bytes",
            compressed.len()
        );
    }

    #[test]
    fn empty_data() {
        let data: Vec<u8> = Vec::new();
        let raw = count_frequencies(&data, 256);
        let freq = normalize_frequencies(&raw);
        let cum = cumulative_frequencies(&freq);
        let compressed = encode(&data, &freq, &cum);
        assert!(compressed.is_empty());
        let decoded = decode(&compressed, &freq, &cum, 256, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn round_trip_small_symbol_set() {
        let data: Vec<u8> = (0..500).map(|i| (i % 9) as u8).collect();
        round_trip(&data, 9);
    }
}
