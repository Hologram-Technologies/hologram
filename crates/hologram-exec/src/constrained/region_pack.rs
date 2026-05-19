//! Region-based weight packing for sequential IO.
//!
//! Weights grouped by execution region can be loaded with sequential reads
//! instead of random-access seeks. The compiler assigns `region_id`s based
//! on execution order; the runtime loads regions sequentially for optimal
//! prefetching and deterministic eviction.

use hologram_graph::constant::ConstantId;

/// A packed weight span within a region-aligned archive.
///
/// Maps a `ConstantId` to a byte range within a specific execution region.
/// Regions are compiler-assigned groups of weights accessed together during
/// a contiguous portion of tape execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedWeightSpan {
    /// Byte offset into the weight archive.
    pub offset: u64,
    /// Length in bytes.
    pub len: u32,
    /// Execution region identifier (compiler-assigned).
    /// Weights in the same region are accessed during the same execution phase.
    pub region_id: u32,
}

/// Mapping from constant IDs to their packed spans in the weight archive.
///
/// Built at tape compile time. At runtime, the constrained executor uses
/// this mapping to load weights region-by-region in sequential order,
/// enabling madvise prefetching and deterministic eviction.
#[derive(Debug, Clone, Default)]
pub struct RegionIndex {
    /// Spans keyed by `ConstantId::raw()` for O(1) lookup.
    spans: Vec<Option<PackedWeightSpan>>,
    /// Region count (0..n_regions).
    n_regions: u32,
}

impl RegionIndex {
    /// Create an empty region index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a span for a constant.
    pub fn insert(&mut self, cid: ConstantId, span: PackedWeightSpan) {
        let idx = cid.raw() as usize;
        if idx >= self.spans.len() {
            self.spans.resize(idx + 1, None);
        }
        self.spans[idx] = Some(span);
        if span.region_id + 1 > self.n_regions {
            self.n_regions = span.region_id + 1;
        }
    }

    /// Look up the span for a constant.
    #[must_use]
    pub fn get(&self, cid: ConstantId) -> Option<&PackedWeightSpan> {
        self.spans.get(cid.raw() as usize)?.as_ref()
    }

    /// Number of execution regions.
    #[must_use]
    pub fn n_regions(&self) -> u32 {
        self.n_regions
    }

    /// Iterate all constants in a given region, ordered by offset.
    pub fn constants_in_region(&self, region_id: u32) -> Vec<(ConstantId, PackedWeightSpan)> {
        let mut result: Vec<(ConstantId, PackedWeightSpan)> = self
            .spans
            .iter()
            .enumerate()
            .filter_map(|(i, span)| {
                span.filter(|s| s.region_id == region_id)
                    .map(|s| (ConstantId::new(i as u32), s))
            })
            .collect();
        result.sort_by_key(|(_, s)| s.offset);
        result
    }

    /// Byte range covering all constants in a region (for prefetch/madvise).
    #[must_use]
    pub fn region_byte_range(&self, region_id: u32) -> Option<(u64, u64)> {
        let mut min_start = u64::MAX;
        let mut max_end = 0u64;
        for span in self.spans.iter().flatten() {
            if span.region_id == region_id {
                min_start = min_start.min(span.offset);
                max_end = max_end.max(span.offset + span.len as u64);
            }
        }
        if min_start == u64::MAX {
            None
        } else {
            Some((min_start, max_end))
        }
    }

    /// Prefetch weight data for a region via `madvise(WILLNEED)`.
    ///
    /// No-op on non-Unix platforms or if the region has no constants.
    pub fn prefetch_region(&self, region_id: u32, weights: &[u8]) {
        if let Some((start, end)) = self.region_byte_range(region_id) {
            let start = start as usize;
            let end = (end as usize).min(weights.len());
            if start < end {
                #[cfg(unix)]
                {
                    let ptr = weights[start..].as_ptr();
                    let len = end - start;
                    // SAFETY: ptr is within the weights slice, len doesn't extend past it.
                    unsafe {
                        libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_WILLNEED);
                    }
                }
            }
        }
    }

    /// Release weight pages for a region via `madvise(DONTNEED)`.
    ///
    /// No-op on non-Unix platforms or if the region has no constants.
    pub fn release_region(&self, region_id: u32, weights: &[u8]) {
        if let Some((start, end)) = self.region_byte_range(region_id) {
            let start = start as usize;
            let end = (end as usize).min(weights.len());
            if start < end {
                #[cfg(unix)]
                {
                    let ptr = weights[start..].as_ptr();
                    let len = end - start;
                    // SAFETY: ptr is within the weights slice, len doesn't extend past it.
                    unsafe {
                        libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index() {
        let idx = RegionIndex::new();
        assert_eq!(idx.n_regions(), 0);
        assert!(idx.get(ConstantId::new(0)).is_none());
    }

    #[test]
    fn insert_and_get() {
        let mut idx = RegionIndex::new();
        let span = PackedWeightSpan {
            offset: 100,
            len: 500,
            region_id: 0,
        };
        idx.insert(ConstantId::new(3), span);
        assert_eq!(idx.get(ConstantId::new(3)), Some(&span));
        assert!(idx.get(ConstantId::new(0)).is_none());
        assert_eq!(idx.n_regions(), 1);
    }

    #[test]
    fn constants_in_region_sorted_by_offset() {
        let mut idx = RegionIndex::new();
        idx.insert(
            ConstantId::new(1),
            PackedWeightSpan {
                offset: 200,
                len: 100,
                region_id: 0,
            },
        );
        idx.insert(
            ConstantId::new(0),
            PackedWeightSpan {
                offset: 0,
                len: 200,
                region_id: 0,
            },
        );
        idx.insert(
            ConstantId::new(2),
            PackedWeightSpan {
                offset: 0,
                len: 50,
                region_id: 1,
            },
        );

        let r0 = idx.constants_in_region(0);
        assert_eq!(r0.len(), 2);
        assert_eq!(r0[0].0, ConstantId::new(0)); // offset 0 first
        assert_eq!(r0[1].0, ConstantId::new(1)); // offset 200 second

        let r1 = idx.constants_in_region(1);
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].0, ConstantId::new(2));
    }

    #[test]
    fn region_byte_range() {
        let mut idx = RegionIndex::new();
        idx.insert(
            ConstantId::new(0),
            PackedWeightSpan {
                offset: 100,
                len: 200,
                region_id: 0,
            },
        );
        idx.insert(
            ConstantId::new(1),
            PackedWeightSpan {
                offset: 500,
                len: 300,
                region_id: 0,
            },
        );
        assert_eq!(idx.region_byte_range(0), Some((100, 800)));
        assert_eq!(idx.region_byte_range(1), None);
    }

    #[test]
    fn multiple_regions() {
        let mut idx = RegionIndex::new();
        idx.insert(
            ConstantId::new(0),
            PackedWeightSpan {
                offset: 0,
                len: 100,
                region_id: 0,
            },
        );
        idx.insert(
            ConstantId::new(1),
            PackedWeightSpan {
                offset: 100,
                len: 100,
                region_id: 1,
            },
        );
        idx.insert(
            ConstantId::new(2),
            PackedWeightSpan {
                offset: 200,
                len: 100,
                region_id: 2,
            },
        );
        assert_eq!(idx.n_regions(), 3);
    }

    #[test]
    fn prefetch_and_release_noop_on_empty() {
        let idx = RegionIndex::new();
        let weights = vec![0u8; 1024];
        // Should not panic.
        idx.prefetch_region(0, &weights);
        idx.release_region(0, &weights);
    }
}
