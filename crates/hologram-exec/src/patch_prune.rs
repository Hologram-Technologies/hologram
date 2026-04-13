//! PixelPrune: pre-ViT patch pruning via 2D predictive coding.
//!
//! Selects the most informative patches from an image by comparing
//! neighboring patches in raster-scan order. Redundant patches (solid
//! backgrounds, repeated patterns) are dropped, reducing the ViT token
//! count by 30–80% on document/GUI images with minimal accuracy loss.
//!
//! This kernel runs **before** the compiled ViT graph as a preprocessing
//! step. It produces `kept_indices` which the compiled graph consumes
//! via `Gather(axis=1)` nodes inserted by the `PatchPruneInjection` pass.
//!
//! # Algorithm (Pred-2D)
//!
//! For each patch at grid position `(r, c)`:
//! 1. Select a causal reference patch via median-edge predictor:
//!    - If `dist(diag, upper) < dist(diag, left)`: predict from left
//!    - Else: predict from upper
//! 2. If `max_pixel_diff(patch, predicted) > tau`: keep the patch.
//! 3. Anchor `(0, 0)` is always kept.
//!
//! When the number of kept patches exceeds `max_kept` (the compiled
//! budget), the kernel sorts by prediction error (descending) and
//! retains the top-K most informative patches.
//!
//! # References
//!
//! - PixelPrune: arXiv 2604.00886
//! - Plan 063: specs/plans/063-vit-patch-prune.md

/// Result of patch pruning: indices to select and an attention mask.
#[derive(Debug, Clone)]
pub struct PatchPruneResult {
    /// Flattened indices into the `[N_patches]` sequence dimension.
    /// Length == `max_kept`. Values in `[0, grid_h * grid_w)`.
    /// Padded positions use index 0 (the anchor, always valid).
    pub kept_indices: Vec<i64>,

    /// Attention mask. `1.0` for real patches, `0.0` for padding.
    /// Length == `max_kept`.
    pub attention_mask: Vec<f32>,

    /// Number of actually informative patches (before padding).
    pub n_kept: usize,
}

/// Parameters for the patch pruning kernel.
#[derive(Debug, Clone)]
pub struct PatchPruneParams {
    /// Number of color channels (typically 3 for RGB).
    pub channels: usize,
    /// Image height in pixels.
    pub img_h: usize,
    /// Image width in pixels.
    pub img_w: usize,
    /// Patch height in pixels (must evenly divide `img_h`).
    pub patch_h: usize,
    /// Patch width in pixels (must evenly divide `img_w`).
    pub patch_w: usize,
    /// Per-pixel difference threshold. `0.0` = lossless (exact match only).
    pub tau: f32,
    /// Budget — maximum patches to keep (from compiled graph).
    pub max_kept: usize,
}

/// Run Pred-2D patch pruning on raw pixel data.
///
/// `pixels` is raw image data as `&[f32]`, layout `[C, H, W]` (channel-first).
/// The batch dimension should be stripped before calling.
///
/// # Panics
///
/// Panics if `pixels.len() != channels * img_h * img_w` or if patch sizes
/// don't evenly divide image dimensions.
pub fn patch_prune(pixels: &[f32], params: &PatchPruneParams) -> PatchPruneResult {
    let PatchPruneParams {
        channels,
        img_h,
        img_w,
        patch_h,
        patch_w,
        tau,
        max_kept,
    } = *params;

    assert_eq!(
        pixels.len(),
        channels * img_h * img_w,
        "pixel buffer size mismatch: expected {}×{}×{} = {}, got {}",
        channels,
        img_h,
        img_w,
        channels * img_h * img_w,
        pixels.len(),
    );
    assert!(
        img_h.is_multiple_of(patch_h) && img_w.is_multiple_of(patch_w),
        "image dims ({img_h}×{img_w}) must be evenly divisible by patch size ({patch_h}×{patch_w})"
    );

    let grid_h = img_h / patch_h;
    let grid_w = img_w / patch_w;
    let patch_dim = channels * patch_h * patch_w;
    let total_patches = grid_h * grid_w;

    // Extract patches into a contiguous buffer: [grid_h, grid_w, patch_dim].
    // This converts from CHW pixel layout to a patch grid.
    let patches = extract_patches(pixels, channels, img_h, img_w, patch_h, patch_w);
    debug_assert_eq!(patches.len(), total_patches * patch_dim);

    // Helper to get a patch slice.
    let patch = |r: usize, c: usize| -> &[f32] {
        let offset = (r * grid_w + c) * patch_dim;
        &patches[offset..offset + patch_dim]
    };

    // Pred-2D scan: decide which patches to keep.
    // Each kept entry is (flat_index, prediction_error).
    let mut kept: Vec<(usize, f32)> = Vec::with_capacity(total_patches);

    for r in 0..grid_h {
        for c in 0..grid_w {
            let flat_idx = r * grid_w + c;

            // Anchor: always keep (0, 0).
            if r == 0 && c == 0 {
                kept.push((flat_idx, f32::MAX));
                continue;
            }

            // Select reference patch via median-edge predictor.
            let predicted = if r > 0 && c > 0 {
                let left = patch(r, c - 1);
                let upper = patch(r - 1, c);
                let diag = patch(r - 1, c - 1);
                let dist_diag_upper = max_pixel_diff(diag, upper);
                let dist_diag_left = max_pixel_diff(diag, left);
                if dist_diag_upper < dist_diag_left {
                    // Upper and diag agree → horizontal edge likely → predict from left.
                    left
                } else {
                    // Left and diag agree → vertical edge likely → predict from upper.
                    upper
                }
            } else if c > 0 {
                // First row: predict from left.
                patch(r, c - 1)
            } else {
                // First column: predict from upper.
                patch(r - 1, c)
            };

            let error = max_pixel_diff(patch(r, c), predicted);
            if error > tau {
                kept.push((flat_idx, error));
            }
        }
    }

    let n_actually_kept = kept.len();

    // Budget enforcement.
    if kept.len() > max_kept {
        // Sort by prediction error descending, keep top-K.
        // The anchor (error = f32::MAX) always survives.
        kept.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        kept.truncate(max_kept);
        // Re-sort by flat index to preserve spatial order (important for
        // position encoding and causal attention patterns).
        kept.sort_unstable_by_key(|&(idx, _)| idx);
    }

    // Build output arrays, padded to max_kept.
    let mut kept_indices = Vec::with_capacity(max_kept);
    let mut attention_mask = Vec::with_capacity(max_kept);

    for &(idx, _) in &kept {
        kept_indices.push(idx as i64);
        attention_mask.push(1.0f32);
    }

    // Pad remaining slots.
    let n_kept = kept.len();
    for _ in n_kept..max_kept {
        kept_indices.push(0); // anchor index (always valid)
        attention_mask.push(0.0);
    }

    PatchPruneResult {
        kept_indices,
        attention_mask,
        n_kept: n_actually_kept.min(max_kept),
    }
}

/// Maximum absolute per-element difference between two patch vectors.
#[inline]
fn max_pixel_diff(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |acc, (&x, &y)| acc.max((x - y).abs()))
}

/// Extract patches from CHW pixel layout into a contiguous
/// `[grid_h, grid_w, C * patch_h * patch_w]` buffer.
fn extract_patches(
    pixels: &[f32],
    channels: usize,
    img_h: usize,
    img_w: usize,
    patch_h: usize,
    patch_w: usize,
) -> Vec<f32> {
    let grid_h = img_h / patch_h;
    let grid_w = img_w / patch_w;
    let patch_dim = channels * patch_h * patch_w;

    let mut patches = vec![0.0f32; grid_h * grid_w * patch_dim];

    for gr in 0..grid_h {
        for gc in 0..grid_w {
            let patch_offset = (gr * grid_w + gc) * patch_dim;
            let mut idx = 0;
            for ch in 0..channels {
                let channel_base = ch * img_h * img_w;
                for py in 0..patch_h {
                    let row = gr * patch_h + py;
                    let src_base = channel_base + row * img_w + gc * patch_w;
                    patches[patch_offset + idx..patch_offset + idx + patch_w]
                        .copy_from_slice(&pixels[src_base..src_base + patch_w]);
                    idx += patch_w;
                }
            }
        }
    }

    patches
}

/// Convert `kept_indices` to bytes suitable for graph input (i64 LE).
pub fn indices_to_bytes(indices: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(indices.len() * 8);
    for &idx in indices {
        bytes.extend_from_slice(&idx.to_le_bytes());
    }
    bytes
}

/// Convert `attention_mask` to bytes suitable for graph input (f32 LE).
pub fn mask_to_bytes(mask: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(mask.len() * 4);
    for &v in mask {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(
        channels: usize,
        img_h: usize,
        img_w: usize,
        patch_h: usize,
        patch_w: usize,
        tau: f32,
        max_kept: usize,
    ) -> PatchPruneParams {
        PatchPruneParams {
            channels,
            img_h,
            img_w,
            patch_h,
            patch_w,
            tau,
            max_kept,
        }
    }

    /// Create a solid-color image (all pixels same value).
    /// Pred-2D with tau=0 should keep only the anchor.
    #[test]
    fn solid_image_keeps_only_anchor() {
        let p = params(3, 32, 32, 16, 16, 0.0, 4);
        let pixels = vec![0.5f32; p.channels * p.img_h * p.img_w];

        let result = patch_prune(&pixels, &p);

        assert_eq!(result.n_kept, 1);
        assert_eq!(result.kept_indices[0], 0);
        assert_eq!(result.attention_mask[0], 1.0);
        for i in 1..p.max_kept {
            assert_eq!(result.attention_mask[i], 0.0);
        }
    }

    /// Checkerboard image: every patch is unique.
    /// All patches should be kept.
    #[test]
    fn checkerboard_keeps_all() {
        let p = params(1, 32, 32, 16, 16, 0.0, 4);
        let mut pixels = vec![0.0f32; p.channels * p.img_h * p.img_w];
        let values = [0.0, 1.0, 0.5, 0.8];
        for gr in 0..2 {
            for gc in 0..2 {
                let val = values[gr * 2 + gc];
                for py in 0..p.patch_h {
                    for px in 0..p.patch_w {
                        let row = gr * p.patch_h + py;
                        let col = gc * p.patch_w + px;
                        pixels[row * p.img_w + col] = val;
                    }
                }
            }
        }

        let result = patch_prune(&pixels, &p);

        assert_eq!(result.n_kept, 4, "all 4 unique patches should be kept");
        for i in 0..4 {
            assert_eq!(result.attention_mask[i], 1.0);
        }
    }

    /// Budget overflow: more patches survive than the budget allows.
    /// Top-K by error should be selected.
    #[test]
    fn budget_overflow_keeps_top_k() {
        let p = params(1, 64, 64, 16, 16, 0.0, 4);
        let mut pixels = vec![0.0f32; p.channels * p.img_h * p.img_w];
        for gr in 0..4 {
            for gc in 0..4 {
                let val = (gr * 4 + gc) as f32 * 0.1;
                for py in 0..p.patch_h {
                    for px in 0..p.patch_w {
                        let row = gr * p.patch_h + py;
                        let col = gc * p.patch_w + px;
                        pixels[row * p.img_w + col] = val;
                    }
                }
            }
        }

        let result = patch_prune(&pixels, &p);

        assert_eq!(result.n_kept, 4);
        for i in 0..4 {
            assert_eq!(result.attention_mask[i], 1.0);
        }
        assert!(result.kept_indices.contains(&0));
    }

    /// Lossy pruning (tau > 0) drops near-matches.
    #[test]
    fn lossy_prune_drops_near_matches() {
        let p = params(1, 32, 32, 16, 16, 0.01, 4);
        let mut pixels = vec![0.5f32; p.channels * p.img_h * p.img_w];
        for row in 0..p.img_h {
            for col in (p.img_w / 2)..p.img_w {
                pixels[row * p.img_w + col] = 0.501;
            }
        }

        // tau=0.01: diff 0.001 < 0.01 → near-matches dropped.
        let result = patch_prune(&pixels, &p);
        assert_eq!(result.n_kept, 1, "lossy should drop near-matches");

        // tau=0.0005: diff 0.001 > 0.0005 → right patches kept.
        let p2 = params(1, 32, 32, 16, 16, 0.0005, 4);
        let result2 = patch_prune(&pixels, &p2);
        assert!(
            result2.n_kept > 1,
            "tight tau should keep patches with diff > tau"
        );
    }

    /// Indices are in ascending spatial order (for correct positional encoding).
    #[test]
    fn indices_in_spatial_order() {
        let p = params(1, 64, 64, 16, 16, 0.0, 16);
        let mut pixels = vec![0.0f32; p.channels * p.img_h * p.img_w];
        for i in 0..pixels.len() {
            pixels[i] = (i as f32) / (pixels.len() as f32);
        }

        let result = patch_prune(&pixels, &p);

        for i in 1..result.n_kept {
            assert!(
                result.kept_indices[i] > result.kept_indices[i - 1],
                "indices must be in ascending spatial order"
            );
        }
    }

    #[test]
    fn indices_to_bytes_roundtrip() {
        let indices = vec![0i64, 5, 10, 42];
        let bytes = indices_to_bytes(&indices);
        assert_eq!(bytes.len(), 32);
        for (i, &expected) in indices.iter().enumerate() {
            let actual = i64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().expect("8 bytes"));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn mask_to_bytes_roundtrip() {
        let mask = vec![1.0f32, 0.0, 1.0, 0.0];
        let bytes = mask_to_bytes(&mask);
        assert_eq!(bytes.len(), 16);
        for (i, &expected) in mask.iter().enumerate() {
            let actual = f32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().expect("4 bytes"));
            assert_eq!(actual, expected);
        }
    }
}
