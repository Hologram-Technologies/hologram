//! Pipeline orchestration: full compress/decompress with mode selection.
//!
//! ```text
//! Raw bytes → [Mode Select] → [Pre-Transform] → [Ring-Diff] → [SPEC/Torus] → [ANS] → compressed
//! ```

use alloc::vec::Vec;

use crate::codec::{CompressedBlock, CompressionMode};
use crate::entropy::ans;
use crate::entropy::histogram::{count_frequencies, cumulative_frequencies, normalize_frequencies};
use crate::header::{self, HEADER_SIZE};
use crate::permute::{self, PermuteId};
use crate::ring_diff::{self, PredictionOrder};
use crate::stratum;
use crate::torus_block;

/// Compress data using the specified mode.
///
/// Returns a `CompressedBlock` containing the header + compressed payload.
pub fn compress(data: &[u8], mode: CompressionMode) -> CompressedBlock {
    if data.is_empty() {
        return CompressedBlock {
            data: Vec::new(),
            original_len: 0,
            mode,
        };
    }

    match mode {
        CompressionMode::Generic => compress_generic(data),
        CompressionMode::Stratum => compress_stratum(data),
        CompressionMode::Float => compress_float(data),
        CompressionMode::Quantized => compress_quantized(data),
    }
}

/// Decompress a compressed block back to original bytes.
///
/// Returns `None` if the block is malformed.
pub fn decompress(compressed: &[u8]) -> Option<Vec<u8>> {
    if compressed.is_empty() {
        return Some(Vec::new());
    }

    let hdr = header::decode_header(compressed)?;
    let payload = &compressed[HEADER_SIZE..];

    match hdr.mode {
        CompressionMode::Generic => decompress_generic(payload, &hdr),
        CompressionMode::Stratum => decompress_stratum(payload, &hdr),
        CompressionMode::Float => decompress_float(payload, &hdr),
        CompressionMode::Quantized => decompress_quantized(payload, &hdr),
    }
}

/// Auto-select the best compression mode for the given data.
pub fn auto_select_mode(data: &[u8]) -> CompressionMode {
    if data.is_empty() {
        return CompressionMode::Generic;
    }

    // Check if data looks like f32 (length divisible by 4 and exponent bytes are low-entropy).
    if data.len() >= 16 && data.len().is_multiple_of(4) {
        let n = data.len() / 4;
        let mut exp_distinct = [false; 256];
        let mut exp_count = 0;
        for i in 0..n.min(256) {
            let msb = data[i * 4 + 3]; // f32 MSB in little-endian
            if !exp_distinct[msb as usize] {
                exp_distinct[msb as usize] = true;
                exp_count += 1;
            }
        }
        if exp_count <= 8 {
            return CompressionMode::Float;
        }
    }

    // Check stratum concentration.
    let hist = stratum::histogram(data);
    let total = data.len() as f64;
    let max_stratum_frac = hist
        .iter()
        .map(|&c| c as f64 / total)
        .fold(0.0f64, f64::max);
    if max_stratum_frac > 0.4 {
        return CompressionMode::Stratum;
    }

    // Check torus page concentration.
    let page_hist = torus_block::page_histogram(data);
    let max_page_frac = page_hist
        .iter()
        .map(|&c| c as f64 / total)
        .fold(0.0f64, f64::max);
    if max_page_frac > 0.3 {
        return CompressionMode::Quantized;
    }

    CompressionMode::Generic
}

// ── Mode 0: Generic (RDC + ANS) ─────────────────────────────────

fn compress_generic(data: &[u8]) -> CompressedBlock {
    let permute_id = permute::auto_select(data);
    let fwd = permute::forward(permute_id);

    // Apply pre-transform.
    let mut transformed = alloc::vec![0u8; data.len()];
    fwd.apply_to(data, &mut transformed);

    // Ring-differential coding.
    let residuals = ring_diff::encode(&transformed, PredictionOrder::Zero);

    // Entropy coding.
    let raw_freq = count_frequencies(&residuals, 256);
    let freq = normalize_frequencies(&raw_freq);
    let cum = cumulative_frequencies(&freq);
    let entropy_data = ans::encode(&residuals, &freq, &cum);

    // Pack: header + freq_table + entropy_data.
    let hdr = header::encode_header(
        CompressionMode::Generic,
        permute_id as u8,
        data.len() as u64,
    );
    let mut output = Vec::with_capacity(HEADER_SIZE + 256 * 2 + entropy_data.len());
    output.extend_from_slice(&hdr);
    // Store normalized frequency table (256 entries × 2 bytes each = 512 bytes).
    for &f in &freq {
        output.extend_from_slice(&(f as u16).to_le_bytes());
    }
    output.extend_from_slice(&entropy_data);

    CompressedBlock {
        data: output,
        original_len: data.len(),
        mode: CompressionMode::Generic,
    }
}

fn decompress_generic(payload: &[u8], hdr: &header::Header) -> Option<Vec<u8>> {
    let freq_table_size = 256 * 2;
    if payload.len() < freq_table_size {
        return None;
    }

    // Read frequency table.
    let mut freq = alloc::vec![0u32; 256];
    for i in 0..256 {
        freq[i] = u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]) as u32;
    }
    let cum = cumulative_frequencies(&freq);

    let entropy_data = &payload[freq_table_size..];
    let residuals = ans::decode(entropy_data, &freq, &cum, 256, hdr.original_len as usize);

    // Inverse ring-differential.
    let transformed = ring_diff::decode(&residuals, PredictionOrder::Zero);

    // Inverse pre-transform.
    let permute_id = PermuteId::from_byte(hdr.permute_id)?;
    let inv = permute::inverse(permute_id);
    let mut data = alloc::vec![0u8; transformed.len()];
    inv.apply_to(&transformed, &mut data);

    Some(data)
}

// ── Mode 1: Stratum (SPEC) ──────────────────────────────────────

fn compress_stratum(data: &[u8]) -> CompressedBlock {
    let (strata, ranks) = stratum::encode(data);

    // Entropy-code the stratum stream (9 symbols).
    let raw_freq_s = count_frequencies(&strata, 9);
    let freq_s = normalize_frequencies(&raw_freq_s);
    let cum_s = cumulative_frequencies(&freq_s);
    let enc_strata = ans::encode(&strata, &freq_s, &cum_s);

    // Entropy-code the rank stream (max symbol value = 69 for stratum 4).
    let raw_freq_r = count_frequencies(&ranks, 70);
    let freq_r = normalize_frequencies(&raw_freq_r);
    let cum_r = cumulative_frequencies(&freq_r);
    let enc_ranks = ans::encode(&ranks, &freq_r, &cum_r);

    // Pack: header + freq_s(9×2) + freq_r(70×2) + len(enc_strata) + enc_strata + enc_ranks.
    let hdr = header::encode_header(CompressionMode::Stratum, 0, data.len() as u64);
    let mut output = Vec::new();
    output.extend_from_slice(&hdr);

    // Stratum frequency table (9 entries × 2 bytes).
    for &f in &freq_s {
        output.extend_from_slice(&(f as u16).to_le_bytes());
    }
    // Rank frequency table (70 entries × 2 bytes).
    for &f in &freq_r {
        output.extend_from_slice(&(f as u16).to_le_bytes());
    }
    // Length of encoded strata stream.
    output.extend_from_slice(&(enc_strata.len() as u32).to_le_bytes());
    output.extend_from_slice(&enc_strata);
    output.extend_from_slice(&enc_ranks);

    CompressedBlock {
        data: output,
        original_len: data.len(),
        mode: CompressionMode::Stratum,
    }
}

fn decompress_stratum(payload: &[u8], hdr: &header::Header) -> Option<Vec<u8>> {
    let n = hdr.original_len as usize;
    let freq_s_size = 9 * 2;
    let freq_r_size = 70 * 2;
    let meta_size = freq_s_size + freq_r_size + 4; // +4 for strata length

    if payload.len() < meta_size {
        return None;
    }

    // Read stratum frequency table.
    let mut freq_s = alloc::vec![0u32; 9];
    for i in 0..9 {
        freq_s[i] = u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]) as u32;
    }
    let cum_s = cumulative_frequencies(&freq_s);

    // Read rank frequency table.
    let offset = freq_s_size;
    let mut freq_r = alloc::vec![0u32; 70];
    for i in 0..70 {
        freq_r[i] =
            u16::from_le_bytes([payload[offset + i * 2], payload[offset + i * 2 + 1]]) as u32;
    }
    let cum_r = cumulative_frequencies(&freq_r);

    // Read strata length.
    let len_offset = freq_s_size + freq_r_size;
    let strata_len = u32::from_le_bytes([
        payload[len_offset],
        payload[len_offset + 1],
        payload[len_offset + 2],
        payload[len_offset + 3],
    ]) as usize;

    let strata_start = meta_size;
    let strata_end = strata_start + strata_len;
    if payload.len() < strata_end {
        return None;
    }

    let strata = ans::decode(&payload[strata_start..strata_end], &freq_s, &cum_s, 9, n);
    let ranks = ans::decode(&payload[strata_end..], &freq_r, &cum_r, 70, n);

    Some(stratum::decode(&strata, &ranks))
}

// ── Mode 2: Float (byte-plane transpose + per-plane coding) ─────

fn compress_float(data: &[u8]) -> CompressedBlock {
    use crate::float_plane;

    // Transpose into 4 byte planes.
    let transposed = match float_plane::transpose_f32(data) {
        Some(t) => t,
        None => {
            // Fallback to generic if not valid f32 data.
            return compress_generic(data);
        }
    };

    let hdr = header::encode_header(CompressionMode::Float, 0, data.len() as u64);
    let mut output = Vec::new();
    output.extend_from_slice(&hdr);

    // Compress each plane independently with generic mode (RDC + ANS).
    // Store: [plane_0_compressed_len(4)] [plane_0_data] [plane_1...] ...
    for plane_idx in 0..4 {
        let plane = float_plane::plane_slice(&transposed, plane_idx, 4);
        let compressed_plane = compress_plane(plane);
        output.extend_from_slice(&(compressed_plane.len() as u32).to_le_bytes());
        output.extend_from_slice(&compressed_plane);
    }

    CompressedBlock {
        data: output,
        original_len: data.len(),
        mode: CompressionMode::Float,
    }
}

/// Compress a single byte plane using RDC + ANS (no header).
fn compress_plane(plane: &[u8]) -> Vec<u8> {
    let residuals = ring_diff::encode(plane, PredictionOrder::Zero);
    let raw_freq = count_frequencies(&residuals, 256);
    let freq = normalize_frequencies(&raw_freq);
    let cum = cumulative_frequencies(&freq);
    let entropy_data = ans::encode(&residuals, &freq, &cum);

    let mut out = Vec::with_capacity(256 * 2 + entropy_data.len());
    for &f in &freq {
        out.extend_from_slice(&(f as u16).to_le_bytes());
    }
    out.extend_from_slice(&entropy_data);
    out
}

/// Decompress a single byte plane.
fn decompress_plane(payload: &[u8], n: usize) -> Option<Vec<u8>> {
    let freq_size = 256 * 2;
    if payload.len() < freq_size {
        return None;
    }
    let mut freq = alloc::vec![0u32; 256];
    for i in 0..256 {
        freq[i] = u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]) as u32;
    }
    let cum = cumulative_frequencies(&freq);
    let residuals = ans::decode(&payload[freq_size..], &freq, &cum, 256, n);
    Some(ring_diff::decode(&residuals, PredictionOrder::Zero))
}

fn decompress_float(payload: &[u8], hdr: &header::Header) -> Option<Vec<u8>> {
    use crate::float_plane;

    let n = hdr.original_len as usize / 4;
    let mut offset = 0usize;
    let mut planes = Vec::with_capacity(hdr.original_len as usize);

    for _ in 0..4 {
        if offset + 4 > payload.len() {
            return None;
        }
        let plane_len = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + plane_len > payload.len() {
            return None;
        }
        let plane = decompress_plane(&payload[offset..offset + plane_len], n)?;
        planes.extend_from_slice(&plane);
        offset += plane_len;
    }

    float_plane::untranspose_f32(&planes)
}

// ── Mode 3: Quantized (torus-blocked coding) ────────────────────

fn compress_quantized(data: &[u8]) -> CompressedBlock {
    let (pages, offsets) = torus_block::encode(data);

    // Entropy-code the page stream (32 symbols).
    let raw_freq_p = count_frequencies(&pages, 32);
    let freq_p = normalize_frequencies(&raw_freq_p);
    let cum_p = cumulative_frequencies(&freq_p);
    let enc_pages = ans::encode(&pages, &freq_p, &cum_p);

    // Entropy-code the offset stream (8 symbols).
    let raw_freq_o = count_frequencies(&offsets, 8);
    let freq_o = normalize_frequencies(&raw_freq_o);
    let cum_o = cumulative_frequencies(&freq_o);
    let enc_offsets = ans::encode(&offsets, &freq_o, &cum_o);

    let hdr = header::encode_header(CompressionMode::Quantized, 0, data.len() as u64);
    let mut output = Vec::new();
    output.extend_from_slice(&hdr);

    // Page frequency table (32 × 2 bytes).
    for &f in &freq_p {
        output.extend_from_slice(&(f as u16).to_le_bytes());
    }
    // Offset frequency table (8 × 2 bytes).
    for &f in &freq_o {
        output.extend_from_slice(&(f as u16).to_le_bytes());
    }
    // Length of encoded pages.
    output.extend_from_slice(&(enc_pages.len() as u32).to_le_bytes());
    output.extend_from_slice(&enc_pages);
    output.extend_from_slice(&enc_offsets);

    CompressedBlock {
        data: output,
        original_len: data.len(),
        mode: CompressionMode::Quantized,
    }
}

fn decompress_quantized(payload: &[u8], hdr: &header::Header) -> Option<Vec<u8>> {
    let n = hdr.original_len as usize;
    let freq_p_size = 32 * 2;
    let freq_o_size = 8 * 2;
    let meta_size = freq_p_size + freq_o_size + 4;

    if payload.len() < meta_size {
        return None;
    }

    let mut freq_p = alloc::vec![0u32; 32];
    for i in 0..32 {
        freq_p[i] = u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]) as u32;
    }
    let cum_p = cumulative_frequencies(&freq_p);

    let off = freq_p_size;
    let mut freq_o = alloc::vec![0u32; 8];
    for i in 0..8 {
        freq_o[i] = u16::from_le_bytes([payload[off + i * 2], payload[off + i * 2 + 1]]) as u32;
    }
    let cum_o = cumulative_frequencies(&freq_o);

    let len_off = freq_p_size + freq_o_size;
    let pages_len = u32::from_le_bytes([
        payload[len_off],
        payload[len_off + 1],
        payload[len_off + 2],
        payload[len_off + 3],
    ]) as usize;

    let pages_start = meta_size;
    let pages_end = pages_start + pages_len;
    if payload.len() < pages_end {
        return None;
    }

    let pages = ans::decode(&payload[pages_start..pages_end], &freq_p, &cum_p, 32, n);
    let offsets = ans::decode(&payload[pages_end..], &freq_o, &cum_o, 8, n);

    Some(torus_block::decode(&pages, &offsets))
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use alloc::vec::Vec;

    fn round_trip_mode(data: &[u8], mode: CompressionMode) {
        let block = compress(data, mode);
        let recovered = decompress(&block.data).expect("decompression failed");
        assert_eq!(
            data,
            &recovered[..],
            "round-trip failed for {mode:?}, len={}",
            data.len()
        );
    }

    #[test]
    fn generic_round_trip() {
        let data: Vec<u8> = (0..=255).collect();
        round_trip_mode(&data, CompressionMode::Generic);
    }

    #[test]
    fn stratum_round_trip() {
        let data: Vec<u8> = (0..=255).collect();
        round_trip_mode(&data, CompressionMode::Stratum);
    }

    #[test]
    fn quantized_round_trip() {
        let data: Vec<u8> = (0..=255).collect();
        round_trip_mode(&data, CompressionMode::Quantized);
    }

    #[test]
    fn float_round_trip() {
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        round_trip_mode(&data, CompressionMode::Float);
    }

    #[test]
    fn generic_constant_data() {
        let data = vec![42u8; 1000];
        let block = compress(&data, CompressionMode::Generic);
        assert!(
            block.data.len() < data.len(),
            "constant data should compress: {} >= {}",
            block.data.len(),
            data.len()
        );
        let recovered = decompress(&block.data).unwrap();
        assert_eq!(data, recovered);
    }

    #[test]
    fn empty_data() {
        let data: Vec<u8> = Vec::new();
        for mode in [
            CompressionMode::Generic,
            CompressionMode::Stratum,
            CompressionMode::Quantized,
        ] {
            let block = compress(&data, mode);
            let recovered = decompress(&block.data).unwrap();
            assert!(recovered.is_empty());
        }
    }

    #[test]
    fn auto_select_reasonable() {
        // Uniform random-ish data → Generic
        let uniform: Vec<u8> = (0..=255).collect();
        let mode = auto_select_mode(&uniform);
        assert_eq!(mode, CompressionMode::Generic);
    }

    #[test]
    fn all_modes_round_trip_large() {
        let data: Vec<u8> = (0..2048).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        for mode in [
            CompressionMode::Generic,
            CompressionMode::Stratum,
            CompressionMode::Quantized,
        ] {
            round_trip_mode(&data, mode);
        }
    }
}
