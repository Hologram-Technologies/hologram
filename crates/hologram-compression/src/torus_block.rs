//! Orbit-Torus Blocked Coding.
//!
//! Observable: `torus_page_q0(x) = x / 8` and `torus_offset_q0(x) = x % 8`
//! split each byte into a page (0..31) and offset (0..7).
//!
//! When data clusters in a few torus pages (common for quantized weights near
//! a zero-point), the page stream compresses heavily via entropy coding while
//! offsets remain at 3 bits each.

use alloc::vec::Vec;
use hologram_core::lut::q0::{torus_offset_q0, torus_page_q0};

/// Split data into page and offset streams.
///
/// - `pages[i] = data[i] / 8` (range 0..32)
/// - `offsets[i] = data[i] % 8` (range 0..8)
pub fn encode(data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut pages = Vec::with_capacity(data.len());
    let mut offsets = Vec::with_capacity(data.len());
    for &byte in data {
        pages.push(torus_page_q0(byte));
        offsets.push(torus_offset_q0(byte));
    }
    (pages, offsets)
}

/// Reconstruct bytes from page and offset streams.
pub fn decode(pages: &[u8], offsets: &[u8]) -> Vec<u8> {
    debug_assert_eq!(pages.len(), offsets.len());
    let mut out = Vec::with_capacity(pages.len());
    for (&p, &o) in pages.iter().zip(offsets.iter()) {
        out.push(p * 8 + o);
    }
    out
}

/// Compute the page histogram (32 bins).
pub fn page_histogram(data: &[u8]) -> [u32; 32] {
    let mut h = [0u32; 32];
    for &byte in data {
        h[torus_page_q0(byte) as usize] += 1;
    }
    h
}

/// Shannon entropy of the page distribution in bits.
pub fn page_entropy(data: &[u8]) -> f64 {
    let h = page_histogram(data);
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
        let data: Vec<u8> = (0..=255).collect();
        let (pages, offsets) = encode(&data);
        let recovered = decode(&pages, &offsets);
        assert_eq!(data, recovered);
    }

    #[test]
    fn page_offset_decomposition() {
        for i in 0..=255u8 {
            let page = torus_page_q0(i);
            let offset = torus_offset_q0(i);
            assert_eq!(page * 8 + offset, i);
            assert!(page < 32);
            assert!(offset < 8);
        }
    }

    #[test]
    fn clustered_data_low_page_entropy() {
        // Data clustered in page 0 (values 0..7)
        let data: Vec<u8> = (0..100).map(|i| (i % 8) as u8).collect();
        let entropy = page_entropy(&data);
        assert!(entropy < 0.01); // Essentially one page
    }

    #[test]
    fn uniform_data_max_page_entropy() {
        let data: Vec<u8> = (0..=255).collect();
        let entropy = page_entropy(&data);
        // 32 pages, uniform → log2(32) = 5.0
        assert!((entropy - 5.0).abs() < 0.01);
    }
}
