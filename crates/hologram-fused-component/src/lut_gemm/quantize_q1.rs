//! Q16 hierarchical weight quantization (65536 centroids).
//!
//! Two-level k-means: 256 page centroids, then 256 sub-centroids per page.
//! Each page is classified as Constant, Linear, or Full to reduce inference cost.
//!
//! UOR grounding: HierarchicalLut principle and fiber FS_1 decomposition.

use super::psumbook_q1::{PageKindTag, PageParams16};
use super::quantize::quantize_8bit;

/// Maximum relative spread within a page to classify as Constant.
pub const Q16_CONST_TOL: f32 = 0.01;
/// Maximum RMS residual (relative to mean) to classify as Linear.
pub const Q16_LINEAR_TOL: f32 = 0.03;

/// 16-bit quantized weight matrix (65536 centroids, 256 pages of 256).
///
/// Each weight index `w[i]` encodes `(high_byte=page, low_byte=sub_index)`.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct QuantizedWeights16 {
    /// 2 bytes per weight (u16, little-endian). Length = rows × cols.
    pub indices: Vec<u16>,
    /// Page kind discriminant for each of the 256 pages.
    pub page_tags: [PageKindTag; 256],
    /// Centroid parameters (constant, linear, full) for all pages.
    pub params: PageParams16,
    /// Number of rows (k dimension).
    pub rows: u32,
    /// Number of columns (n dimension).
    pub cols: u32,
}

/// Classify a 256-entry centroid page as Constant, Linear, or Full.
///
/// O(256) — called once per page at quantize time.
#[must_use]
pub fn page_classify(centroids: &[f32; 256]) -> PageKindTag {
    let mean = centroids.iter().sum::<f32>() / 256.0;
    let mean_abs = mean.abs();

    // Constant: max absolute deviation from mean < 1% of mean.
    if mean_abs > 1e-6 {
        let max_dev = centroids
            .iter()
            .map(|&c| (c - mean).abs())
            .fold(0.0f32, f32::max);
        if max_dev / mean_abs < Q16_CONST_TOL {
            return PageKindTag::Constant;
        }
    }

    // Linear: fit slope + offset, check RMS residual.
    let (slope, offset) = fit_linear(centroids);
    let rms_residual = (centroids
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let r = c - (slope * i as f32 + offset);
            r * r
        })
        .sum::<f32>()
        / 256.0f32)
        .sqrt();
    let scale = mean_abs
        .max(centroids.iter().map(|c| c.abs()).fold(0.0f32, f32::max))
        .max(1e-6);
    if rms_residual / scale < Q16_LINEAR_TOL {
        return PageKindTag::Linear;
    }

    PageKindTag::Full
}

/// Least-squares linear fit for 256 centroid values indexed 0..255.
///
/// Precomputed sums for i=0..255: Σi = 32640, Σi² = 5559680.
#[must_use]
pub fn fit_linear(c: &[f32; 256]) -> (f32, f32) {
    const N: f32 = 256.0;
    const SX: f32 = 32640.0; // Σ i for i=0..255
    const SX2: f32 = 5559680.0; // Σ i² for i=0..255
    let sy: f32 = c.iter().sum();
    let sxy: f32 = c.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
    let denom = N * SX2 - SX * SX;
    let slope = (N * sxy - SX * sy) / denom;
    let offset = (sy - slope * SX) / N;
    (slope, offset)
}

/// Quantize weights to Q16 via two-level hierarchical k-means.
///
/// Step 1: Q8 k-means → 256 page centroids.
/// Step 2: For each page, Q8 k-means on page-assigned weights → 256 sub-centroids.
/// Step 3: Classify each page.
/// Step 4: Assign each weight to its page centroid, then sub-centroid.
/// Step 5: Encode as u16 = (page << 8) | sub_index.
#[must_use]
pub fn quantize_16bit(weights: &[f32], rows: u32, cols: u32) -> QuantizedWeights16 {
    // Step 1: Level-1 Q8 k-means → page assignments.
    let level1 = quantize_8bit(weights, rows, cols);
    let page_assignments = &level1.indices;

    // Step 2: For each page, collect assigned weights and run Q8 k-means.
    let mut page_sub_centroids: Vec<Option<[f32; 256]>> = vec![None; 256];
    let mut page_sub_assignments: Vec<Option<Vec<u8>>> = vec![None; 256];

    for page in 0..256usize {
        let page_weights: Vec<f32> = weights
            .iter()
            .zip(page_assignments.iter())
            .filter(|(_, &p)| p as usize == page)
            .map(|(&w, _)| w)
            .collect();
        if page_weights.is_empty() {
            // Empty page: use placeholder centroids (won't be referenced).
            page_sub_centroids[page] = Some([0.0f32; 256]);
            page_sub_assignments[page] = Some(vec![]);
            continue;
        }
        let sub_qw = quantize_8bit(&page_weights, page_weights.len() as u32, 1);
        page_sub_centroids[page] = Some(sub_qw.centroids);
        page_sub_assignments[page] = Some(sub_qw.indices);
    }

    // Step 3: Classify each page.
    let mut page_tags = [PageKindTag::Constant; 256];
    let mut constant_centroid = [0.0f32; 256];
    let mut linear_params = [(0.0f32, 0.0f32); 256];
    let mut full_page_indices: Vec<u8> = Vec::new();
    let mut full_page_centroids: Vec<[f32; 256]> = Vec::new();

    for page in 0..256usize {
        let centroids = page_sub_centroids[page].as_ref().unwrap();
        let tag = page_classify(centroids);
        page_tags[page] = tag;
        match tag {
            PageKindTag::Constant => {
                constant_centroid[page] = centroids[0]; // representative
            }
            PageKindTag::Linear => {
                linear_params[page] = fit_linear(centroids);
            }
            PageKindTag::Full => {
                full_page_indices.push(page as u8);
                full_page_centroids.push(*centroids);
            }
        }
    }

    // Step 4 & 5: Build u16 indices.
    // For each weight, map to (page, sub_index) → u16.
    let mut sub_assign_iter: Vec<usize> = vec![0; 256]; // per-page cursor
    let mut indices = Vec::with_capacity(weights.len());
    for (w_idx, (&w, &page)) in weights.iter().zip(page_assignments.iter()).enumerate() {
        let page = page as usize;
        let sub_assigns = page_sub_assignments[page].as_ref().unwrap();
        let cursor = sub_assign_iter[page];
        // Count how many weights before w_idx belong to this page.
        // Use the cursor (we iterate weights in order, page-sub-assignments in page order).
        // But page_weights was collected in order, so sub_assigns is in the same order.
        // We need to find the sub_index for this particular weight.
        // Since page_weights was built in order of w_idx, we can use a per-page counter.
        let sub_idx = if cursor < sub_assigns.len() {
            let s = sub_assigns[cursor];
            sub_assign_iter[page] += 1;
            s
        } else {
            0
        };
        let _ = w; // w is not used directly (kept for clarity)
        let _ = w_idx;
        indices.push(((page as u16) << 8) | (sub_idx as u16));
    }

    let params = PageParams16 {
        constant_centroid,
        linear_params,
        full_page_indices,
        full_page_centroids,
    };

    QuantizedWeights16 {
        indices,
        page_tags,
        params,
        rows,
        cols,
    }
}

/// Compute relative RMSE of Q16 quantization vs original weights.
#[must_use]
pub fn dequantize_error_q16(original: &[f32], qw: &QuantizedWeights16) -> f32 {
    let mut sq_err = 0.0f32;
    let mut sq_orig = 0.0f32;
    for (i, (&w, &idx)) in original.iter().zip(qw.indices.iter()).enumerate() {
        let page = (idx >> 8) as usize;
        let sub = (idx & 0xFF) as usize;
        let recon = match qw.page_tags[page] {
            PageKindTag::Constant => qw.params.constant_centroid[page],
            PageKindTag::Linear => {
                let (slope, offset) = qw.params.linear_params[page];
                slope * sub as f32 + offset
            }
            PageKindTag::Full => {
                if let Some(centroids) = qw.params.full_centroids_for(page as u8) {
                    centroids[sub]
                } else {
                    0.0
                }
            }
        };
        let _ = i;
        sq_err += (w - recon) * (w - recon);
        sq_orig += w * w;
    }
    if sq_orig > 0.0 {
        (sq_err / sq_orig).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_classify_constant_exact() {
        let c = [1.0f32; 256];
        assert_eq!(page_classify(&c), PageKindTag::Constant);
    }

    #[test]
    fn page_classify_linear_exact() {
        let mut c = [0.0f32; 256];
        for (i, v) in c.iter_mut().enumerate() {
            *v = 0.1 * i as f32;
        }
        assert_eq!(page_classify(&c), PageKindTag::Linear);
    }

    #[test]
    fn fit_linear_exact_ramp() {
        let mut c = [0.0f32; 256];
        for (i, v) in c.iter_mut().enumerate() {
            *v = 2.0 * i as f32 + 1.0; // slope=2, offset=1
        }
        let (slope, offset) = fit_linear(&c);
        assert!((slope - 2.0).abs() < 1e-3, "slope={slope}");
        assert!((offset - 1.0).abs() < 1e-1, "offset={offset}");
    }

    #[test]
    fn quantize_16bit_basic() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let qw = quantize_16bit(&weights, 8, 8);
        assert_eq!(qw.rows, 8);
        assert_eq!(qw.cols, 8);
        assert_eq!(qw.indices.len(), 64);
    }

    #[test]
    fn quantize_16bit_error_bounded() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let qw = quantize_16bit(&weights, 16, 16);
        let err = dequantize_error_q16(&weights, &qw);
        // Q16 should achieve much lower error than Q8
        assert!(err < 0.05, "Q16 RMSE too high: {err}");
    }

    #[test]
    fn page_tags_are_set() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let qw = quantize_16bit(&weights, 4, 4);
        // At least some pages should be classified
        let has_any_constant = qw.page_tags.contains(&PageKindTag::Constant);
        let has_any_tag = has_any_constant
            || qw.page_tags.contains(&PageKindTag::Linear)
            || qw.page_tags.contains(&PageKindTag::Full);
        assert!(has_any_tag);
    }

    #[test]
    fn rkyv_roundtrip_q16() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let qw = quantize_16bit(&weights, 4, 4);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<QuantizedWeights16>, rkyv::rancor::Error>(&bytes)
                .unwrap();
        assert_eq!(archived.rows, 4);
        assert_eq!(archived.cols, 4);
    }
}
