//! Weight quantization via k-means clustering.
//!
//! Quantizes f32 weight matrices into Q4 (16 centroids) or Q8 (256 centroids)
//! representations. Each weight is replaced by an index into a centroid table.
//! At inference time, these indices drive Psumbook accumulation.

use super::psumbook::{Q4_LEVELS, Q8_LEVELS};

/// Number of k-means iterations for quantization.
const KMEANS_ITERS: usize = 10;

// --- Packing helpers ---

/// Pack two 4-bit indices into one byte (high nibble, low nibble).
#[inline]
#[must_use]
pub const fn pack_q4(hi: u8, lo: u8) -> u8 {
    (hi << 4) | (lo & 0x0F)
}

/// Unpack one byte into two 4-bit indices.
#[inline]
#[must_use]
pub const fn unpack_q4(packed: u8) -> (u8, u8) {
    (packed >> 4, packed & 0x0F)
}

/// Get the Q4 index for element at `(row, col)` in a matrix.
#[inline]
#[must_use]
pub fn get_q4_index(indices: &[u8], row: u32, col: u32, cols: u32) -> u8 {
    let flat = (row * cols + col) as usize;
    let byte_idx = flat / 2;
    if flat.is_multiple_of(2) {
        indices[byte_idx] >> 4
    } else {
        indices[byte_idx] & 0x0F
    }
}

// --- Quantized weight types ---

/// 4-bit quantized weight matrix (16 centroids).
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct QuantizedWeights4 {
    /// Packed indices: 2 per byte (high nibble first).
    pub indices: Vec<u8>,
    /// 16 cluster centroids learned via k-means.
    pub centroids: [f32; Q4_LEVELS],
    /// Number of rows (k dimension).
    pub rows: u32,
    /// Number of columns (n dimension).
    pub cols: u32,
}

/// 8-bit quantized weight matrix (256 centroids).
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct QuantizedWeights8 {
    /// One index byte per weight.
    pub indices: Vec<u8>,
    /// 256 cluster centroids learned via k-means.
    pub centroids: [f32; Q8_LEVELS],
    /// Number of rows (k dimension).
    pub rows: u32,
    /// Number of columns (n dimension).
    pub cols: u32,
}

/// Unified quantized weights (Q4 or Q8).
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum QuantizedWeights {
    /// 4-bit quantization (16 centroids).
    Q4(QuantizedWeights4),
    /// 8-bit quantization (256 centroids).
    Q8(Box<QuantizedWeights8>),
}

impl QuantizedWeights {
    /// Number of rows in the weight matrix.
    #[must_use]
    pub fn rows(&self) -> u32 {
        match self {
            Self::Q4(w) => w.rows,
            Self::Q8(w) => w.rows,
        }
    }

    /// Number of columns in the weight matrix.
    #[must_use]
    pub fn cols(&self) -> u32 {
        match self {
            Self::Q4(w) => w.cols,
            Self::Q8(w) => w.cols,
        }
    }
}

// --- K-means internals ---

/// Find the nearest centroid index for a value.
fn find_nearest(value: f32, centroids: &[f32]) -> u8 {
    let mut best = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &c) in centroids.iter().enumerate() {
        let d = (value - c) * (value - c);
        if d < best_dist {
            best_dist = d;
            best = i as u8;
        }
    }
    best
}

/// Assign each weight to its nearest centroid.
fn assign_all(weights: &[f32], centroids: &[f32]) -> Vec<u8> {
    weights
        .iter()
        .map(|&w| find_nearest(w, centroids))
        .collect()
}

/// Recompute centroids as mean of assigned weights.
fn update_centroids(weights: &[f32], assignments: &[u8], centroids: &mut [f32]) {
    let k = centroids.len();
    let mut sums = vec![0.0f32; k];
    let mut counts = vec![0u32; k];
    for (i, &w) in weights.iter().enumerate() {
        let idx = assignments[i] as usize;
        sums[idx] += w;
        counts[idx] += 1;
    }
    for i in 0..k {
        if counts[i] > 0 {
            centroids[i] = sums[i] / counts[i] as f32;
        }
    }
}

/// Initialize centroids uniformly between min and max.
fn init_centroids(weights: &[f32], k: usize) -> Vec<f32> {
    let (min, max) = min_max(weights);
    let range = max - min;
    (0..k)
        .map(|i| min + range * (i as f32) / (k - 1).max(1) as f32)
        .collect()
}

/// Find min and max of a slice.
fn min_max(data: &[f32]) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &v in data {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

// --- Public quantization API ---

/// Quantize weights to 4-bit (16 centroids) via k-means.
#[must_use]
pub fn quantize_4bit(weights: &[f32], rows: u32, cols: u32) -> QuantizedWeights4 {
    let mut centroids_vec = init_centroids(weights, Q4_LEVELS);
    let mut assignments = assign_all(weights, &centroids_vec);
    for _ in 0..KMEANS_ITERS {
        update_centroids(weights, &assignments, &mut centroids_vec);
        assignments = assign_all(weights, &centroids_vec);
    }
    let mut centroids = [0.0f32; Q4_LEVELS];
    centroids.copy_from_slice(&centroids_vec);
    let indices = pack_assignments_q4(&assignments);
    QuantizedWeights4 {
        indices,
        centroids,
        rows,
        cols,
    }
}

/// Pack u8 assignments into nibble-packed bytes.
fn pack_assignments_q4(assignments: &[u8]) -> Vec<u8> {
    assignments
        .chunks(2)
        .map(|chunk| {
            let hi = chunk[0];
            let lo = chunk.get(1).copied().unwrap_or(0);
            pack_q4(hi, lo)
        })
        .collect()
}

/// Quantize weights to 8-bit (256 centroids) via k-means.
#[must_use]
pub fn quantize_8bit(weights: &[f32], rows: u32, cols: u32) -> QuantizedWeights8 {
    let mut centroids_vec = init_centroids(weights, Q8_LEVELS);
    let mut assignments = assign_all(weights, &centroids_vec);
    for _ in 0..KMEANS_ITERS {
        update_centroids(weights, &assignments, &mut centroids_vec);
        assignments = assign_all(weights, &centroids_vec);
    }
    let mut centroids = [0.0f32; Q8_LEVELS];
    centroids.copy_from_slice(&centroids_vec);
    QuantizedWeights8 {
        indices: assignments,
        centroids,
        rows,
        cols,
    }
}

/// Quantize with auto-selection: tries Q4, falls back to Q8.
#[must_use]
pub fn quantize_auto(weights: &[f32], rows: u32, cols: u32) -> QuantizedWeights {
    let q4 = quantize_4bit(weights, rows, cols);
    let err = dequantize_error_q4(weights, &q4);
    if err < 0.05 {
        QuantizedWeights::Q4(q4)
    } else {
        QuantizedWeights::Q8(Box::new(quantize_8bit(weights, rows, cols)))
    }
}

/// Relative RMSE of Q4 quantization.
#[must_use]
pub fn dequantize_error_q4(original: &[f32], qw: &QuantizedWeights4) -> f32 {
    let mut sq_err = 0.0f32;
    let mut sq_orig = 0.0f32;
    for (i, &w) in original.iter().enumerate() {
        let idx = get_q4_flat_index(&qw.indices, i);
        let recon = qw.centroids[idx as usize];
        sq_err += (w - recon) * (w - recon);
        sq_orig += w * w;
    }
    if sq_orig > 0.0 {
        (sq_err / sq_orig).sqrt()
    } else {
        0.0
    }
}

/// Relative RMSE of Q8 quantization.
#[must_use]
pub fn dequantize_error_q8(original: &[f32], qw: &QuantizedWeights8) -> f32 {
    let mut sq_err = 0.0f32;
    let mut sq_orig = 0.0f32;
    for (i, &w) in original.iter().enumerate() {
        let recon = qw.centroids[qw.indices[i] as usize];
        sq_err += (w - recon) * (w - recon);
        sq_orig += w * w;
    }
    if sq_orig > 0.0 {
        (sq_err / sq_orig).sqrt()
    } else {
        0.0
    }
}

/// Get Q4 index by flat element position.
#[inline]
fn get_q4_flat_index(indices: &[u8], flat: usize) -> u8 {
    let byte_idx = flat / 2;
    if flat.is_multiple_of(2) {
        indices[byte_idx] >> 4
    } else {
        indices[byte_idx] & 0x0F
    }
}

/// Basic statistics for a weight slice.
#[derive(Debug, Clone, Copy)]
pub struct WeightStats {
    /// Arithmetic mean.
    pub mean: f32,
    /// Standard deviation.
    pub stddev: f32,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Fraction of values that are exactly zero.
    pub sparsity: f32,
}

/// Compute mean, stddev, min, max, and sparsity ratio for a weight slice.
#[must_use]
pub fn weight_stats(weights: &[f32]) -> WeightStats {
    if weights.is_empty() {
        return WeightStats {
            mean: 0.0,
            stddev: 0.0,
            min: 0.0,
            max: 0.0,
            sparsity: 0.0,
        };
    }
    let n = weights.len() as f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    let mut zeros = 0u32;
    for &w in weights {
        sum += w;
        sum_sq += w * w;
        if w < lo {
            lo = w;
        }
        if w > hi {
            hi = w;
        }
        if w == 0.0 {
            zeros += 1;
        }
    }
    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);
    let stddev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    WeightStats {
        mean,
        stddev,
        min: lo,
        max: hi,
        sparsity: zeros as f32 / n,
    }
}

/// Count zero-valued centroids in a `QuantizedWeights` and return the fraction.
///
/// A centroid is considered "zero" if its absolute value is below `f32::EPSILON`.
/// High sparsity ratios indicate the weight matrix may benefit from sparse
/// representations.
#[must_use]
pub fn sparsity_ratio(qw: &QuantizedWeights) -> f32 {
    match qw {
        QuantizedWeights::Q4(w) => {
            let zeros = w
                .centroids
                .iter()
                .filter(|&&c| c.abs() < f32::EPSILON)
                .count();
            zeros as f32 / w.centroids.len() as f32
        }
        QuantizedWeights::Q8(w) => {
            let zeros = w
                .centroids
                .iter()
                .filter(|&&c| c.abs() < f32::EPSILON)
                .count();
            zeros as f32 / w.centroids.len() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        for hi in 0..16u8 {
            for lo in 0..16u8 {
                let packed = pack_q4(hi, lo);
                assert_eq!(unpack_q4(packed), (hi, lo));
            }
        }
    }

    #[test]
    fn get_q4_index_row_col() {
        // 2x4 matrix: indices [0,1,2,3,4,5,6,7]
        let assignments: Vec<u8> = (0..8).collect();
        let packed = pack_assignments_q4(&assignments);
        // packed: [0x01, 0x23, 0x45, 0x67]
        for r in 0..2u32 {
            for c in 0..4u32 {
                let expected = (r * 4 + c) as u8;
                assert_eq!(get_q4_index(&packed, r, c, 4), expected);
            }
        }
    }

    #[test]
    fn quantize_4bit_basic() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let qw = quantize_4bit(&weights, 8, 8);
        assert_eq!(qw.rows, 8);
        assert_eq!(qw.cols, 8);
        assert_eq!(qw.indices.len(), 32); // 64/2
    }

    #[test]
    fn quantize_8bit_basic() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let qw = quantize_8bit(&weights, 8, 8);
        assert_eq!(qw.rows, 8);
        assert_eq!(qw.cols, 8);
        assert_eq!(qw.indices.len(), 64);
    }

    #[test]
    fn quantize_4bit_error_bounded() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let qw = quantize_4bit(&weights, 16, 16);
        let err = dequantize_error_q4(&weights, &qw);
        // 16 centroids on uniform data: error should be small
        assert!(err < 0.10, "Q4 error too high: {err}");
    }

    #[test]
    fn quantize_8bit_error_bounded() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let qw = quantize_8bit(&weights, 16, 16);
        let err = dequantize_error_q8(&weights, &qw);
        // 256 centroids on 256 values: should be near-zero
        assert!(err < 0.01, "Q8 error too high: {err}");
    }

    #[test]
    fn quantize_auto_selects_q4_for_uniform() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let qw = quantize_auto(&weights, 8, 8);
        // Uniform data with 64 values and 16 centroids: Q4 error < 5%
        assert!(matches!(qw, QuantizedWeights::Q4(_)));
    }

    #[test]
    fn quantize_constant_weights() {
        let weights = vec![42.0f32; 16];
        let qw = quantize_4bit(&weights, 4, 4);
        let err = dequantize_error_q4(&weights, &qw);
        assert!(err < 1e-6, "constant weights should quantize exactly");
    }

    #[test]
    fn quantized_weights_enum_accessors() {
        let weights: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let qw = QuantizedWeights::Q4(quantize_4bit(&weights, 3, 4));
        assert_eq!(qw.rows(), 3);
        assert_eq!(qw.cols(), 4);
    }

    #[test]
    fn rkyv_roundtrip_q4() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let qw = quantize_4bit(&weights, 4, 4);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<QuantizedWeights4>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.rows, 4);
        assert_eq!(archived.cols, 4);
    }

    #[test]
    fn rkyv_roundtrip_q8() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let qw = quantize_8bit(&weights, 4, 4);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<QuantizedWeights8>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.rows, 4);
        assert_eq!(archived.cols, 4);
    }

    #[test]
    fn rkyv_roundtrip_enum() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let qw = QuantizedWeights::Q4(quantize_4bit(&weights, 4, 4));
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap();
        let _archived =
            rkyv::access::<rkyv::Archived<QuantizedWeights>, rkyv::rancor::Error>(&bytes).unwrap();
    }

    #[test]
    fn find_nearest_basic() {
        let centroids = [0.0, 5.0, 10.0];
        assert_eq!(find_nearest(1.0, &centroids), 0);
        assert_eq!(find_nearest(4.0, &centroids), 1);
        assert_eq!(find_nearest(8.0, &centroids), 2);
    }

    #[test]
    fn odd_element_count_packing() {
        // 3 elements → 2 packed bytes (last byte has 0 in low nibble)
        let assignments = vec![5u8, 10, 15];
        let packed = pack_assignments_q4(&assignments);
        assert_eq!(packed.len(), 2);
        assert_eq!(unpack_q4(packed[0]), (5, 10));
        assert_eq!(unpack_q4(packed[1]), (15, 0));
    }

    #[test]
    fn dequantize_error_zero_weights() {
        let weights = vec![0.0f32; 16];
        let qw = quantize_4bit(&weights, 4, 4);
        let err = dequantize_error_q4(&weights, &qw);
        assert!(err == 0.0, "zero weights should have zero error");
    }
}
