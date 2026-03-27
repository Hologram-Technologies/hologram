//! Hierarchical partial sum accumulator for Q16 LUT-GEMM.
//!
//! Q16 uses 65536 centroids organized as 256 pages of 256 sub-centroids each.
//! Each page is classified as Constant (1 accumulator), Linear (2 accumulators),
//! or Full (256-entry sub-psumbook). Typical models have <10% Full pages.
//!
//! UOR grounding: HierarchicalLut + fiber decomposition FS_1 from hologram-core.
//! 16-bit index splits as `(high_byte, low_byte)` — lossless fiber decomposition.

use super::psumbook::Psumbook8;

/// Discriminant for per-page centroid distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(u8)]
pub enum PageKindTag {
    /// All centroids in this page are approximately equal: 1 accumulator needed.
    Constant = 0,
    /// Centroids form an approximate linear ramp: 2 accumulators (sum + weighted sum).
    Linear = 1,
    /// General distribution: 256-entry sub-psumbook required.
    Full = 2,
}

/// Centroid parameters for all 256 pages.
///
/// Stored once per `QuantizedWeights16` instance — not in the hot-path psumbook.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct PageParams16 {
    /// For Constant pages: the single representative centroid value.
    pub constant_centroid: [f32; 256],
    /// For Linear pages: (slope, offset) such that centroid[sub] ≈ slope*sub + offset.
    pub linear_params: [(f32, f32); 256],
    /// Sparse: page indices for Full pages (parallel with `full_page_centroids`).
    pub full_page_indices: Vec<u8>,
    /// Full centroid arrays for Full pages. `full_page_centroids[i]` corresponds
    /// to page `full_page_indices[i]`.
    pub full_page_centroids: Vec<[f32; 256]>,
}

impl PageParams16 {
    /// Look up full centroids for a given page index, if present.
    #[inline]
    pub fn full_centroids_for(&self, page: u8) -> Option<&[f32; 256]> {
        self.full_page_indices
            .iter()
            .position(|&p| p == page)
            .map(|idx| &self.full_page_centroids[idx])
    }
}

/// Hierarchical partial sum accumulator for Q16.
///
/// One instance per output element computation. Allocate once per matmul call
/// (outside the i×j output loop) and reset between columns.
pub struct HierarchicalPsumbook16 {
    /// Page kind tags (copied from QuantizedWeights16 at construction).
    tags: [PageKindTag; 256],
    /// Scalar sum accumulator per page (256 × f32 = 1 KB).
    page_sums: [f32; 256],
    /// Weighted-position accumulator per page (for Linear pages, 1 KB).
    page_wsums: [f32; 256],
    /// Full sub-psumbooks for Full pages. Allocated once; reset between columns.
    /// `(page_index, Psumbook8)` pairs.
    full_books: Vec<(u8, Psumbook8)>,
}

impl HierarchicalPsumbook16 {
    /// Construct from page kind tags. Heap-allocates Psumbook8 for each Full page.
    /// Call once per matmul invocation, outside the output-element loop.
    pub fn from_tags(tags: [PageKindTag; 256]) -> Self {
        let full_books = (0u8..=255)
            .filter(|&p| tags[p as usize] == PageKindTag::Full)
            .map(|p| (p, Psumbook8::zeroed()))
            .collect();
        Self {
            tags,
            page_sums: [0.0f32; 256],
            page_wsums: [0.0f32; 256],
            full_books,
        }
    }

    /// Accumulate one activation value with its 16-bit weight index.
    ///
    /// O(1) per call. Branch on page kind is branch-predicted after first pass.
    #[inline]
    pub fn accumulate(&mut self, idx: u16, value: f32) {
        let page = (idx >> 8) as usize;
        let sub = (idx & 0xFF) as usize;
        match self.tags[page] {
            PageKindTag::Constant => {
                self.page_sums[page] += value;
            }
            PageKindTag::Linear => {
                self.page_sums[page] += value;
                self.page_wsums[page] += value * (sub as f32);
            }
            PageKindTag::Full => {
                // Linear scan over full_books — typically very short (≤ F pages).
                for (pg, book) in self.full_books.iter_mut() {
                    if *pg as usize == page {
                        book.accumulate(sub as u8, value);
                        break;
                    }
                }
            }
        }
    }

    /// Compute the dot product using page parameters.
    ///
    /// O(C + 2L + 256F) where C, L, F = Constant/Linear/Full page counts.
    pub fn dot(&self, params: &PageParams16) -> f32 {
        let mut result = 0.0f32;
        for page in 0..256usize {
            match self.tags[page] {
                PageKindTag::Constant => {
                    result += params.constant_centroid[page] * self.page_sums[page];
                }
                PageKindTag::Linear => {
                    let (slope, offset) = params.linear_params[page];
                    result += slope * self.page_wsums[page] + offset * self.page_sums[page];
                }
                PageKindTag::Full => {}
            }
        }
        for (pg, book) in self.full_books.iter() {
            if let Some(centroids) = params.full_centroids_for(*pg) {
                result += book.dot(centroids);
            }
        }
        result
    }

    /// Reset all accumulators for reuse across output elements.
    ///
    /// Called between columns in the inner loop. No allocation.
    #[inline]
    pub fn reset(&mut self) {
        self.page_sums = [0.0f32; 256];
        self.page_wsums = [0.0f32; 256];
        for (_, book) in self.full_books.iter_mut() {
            book.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_constant_tags() -> [PageKindTag; 256] {
        [PageKindTag::Constant; 256]
    }

    fn all_linear_tags() -> [PageKindTag; 256] {
        [PageKindTag::Linear; 256]
    }

    fn make_constant_params(centroid: f32) -> PageParams16 {
        PageParams16 {
            constant_centroid: [centroid; 256],
            linear_params: [(0.0, 0.0); 256],
            full_page_indices: vec![],
            full_page_centroids: vec![],
        }
    }

    #[test]
    fn constant_page_accumulates_correctly() {
        let tags = all_constant_tags();
        let mut book = HierarchicalPsumbook16::from_tags(tags);
        // page 0, sub 5
        book.accumulate(0x0005u16, 3.0);
        // page 0, sub 7
        book.accumulate(0x0007u16, 2.0);
        // page 1, sub 0
        book.accumulate(0x0100u16, 4.0);

        let params = make_constant_params(1.0);
        // dot = (3+2) * 1.0 + 4 * 1.0 = 9.0
        let result = book.dot(&params);
        assert!((result - 9.0).abs() < 1e-5, "expected 9.0, got {result}");
    }

    #[test]
    fn linear_page_accumulates_correctly() {
        let tags = all_linear_tags();
        let mut book = HierarchicalPsumbook16::from_tags(tags);
        // page 0, sub 2 → page_sums[0] += 1.0, page_wsums[0] += 1.0 * 2 = 2.0
        book.accumulate(0x0002u16, 1.0);
        // page 0, sub 4 → page_sums[0] += 2.0, page_wsums[0] += 2.0 * 4 = 8.0
        book.accumulate(0x0004u16, 2.0);

        // slope=1.0, offset=0.0 → dot_page0 = 1.0*(2+8) + 0.0*(1+2) = 10.0
        let mut params = make_constant_params(0.0);
        params.linear_params[0] = (1.0, 0.0);
        let result = book.dot(&params);
        assert!((result - 10.0).abs() < 1e-5, "expected 10.0, got {result}");
    }

    #[test]
    fn full_page_accumulates_correctly() {
        let mut tags = [PageKindTag::Constant; 256];
        tags[3] = PageKindTag::Full;
        let mut book = HierarchicalPsumbook16::from_tags(tags);
        // page 3, sub 7 → full_books for page 3, bucket 7 += 5.0
        book.accumulate(0x0307u16, 5.0);
        book.accumulate(0x0307u16, 3.0); // same bucket → 8.0

        let mut centroids = [0.0f32; 256];
        centroids[7] = 2.0;
        let mut params = make_constant_params(0.0);
        params.full_page_indices = vec![3];
        params.full_page_centroids = vec![centroids];
        // dot = 8.0 * 2.0 = 16.0
        let result = book.dot(&params);
        assert!((result - 16.0).abs() < 1e-5, "expected 16.0, got {result}");
    }

    #[test]
    fn reset_clears_all_accumulators() {
        let mut tags = [PageKindTag::Constant; 256];
        tags[0] = PageKindTag::Full;
        let mut book = HierarchicalPsumbook16::from_tags(tags);
        book.accumulate(0x0000u16, 99.0);
        book.accumulate(0x0100u16, 99.0);
        book.reset();

        let mut centroids = [0.0f32; 256];
        centroids[0] = 1.0;
        let params = PageParams16 {
            constant_centroid: [1.0f32; 256],
            linear_params: [(1.0, 0.0); 256],
            full_page_indices: vec![0],
            full_page_centroids: vec![centroids],
        };
        let result = book.dot(&params);
        assert!(
            result.abs() < 1e-5,
            "after reset dot should be 0, got {result}"
        );
    }

    #[test]
    fn from_tags_full_books_len() {
        let mut tags = [PageKindTag::Constant; 256];
        tags[5] = PageKindTag::Full;
        tags[200] = PageKindTag::Full;
        let book = HierarchicalPsumbook16::from_tags(tags);
        assert_eq!(book.full_books.len(), 2);
    }
}
