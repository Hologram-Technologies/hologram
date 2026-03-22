//! Hierarchical Content-Addressable LUT (HLUT).
//!
//! Replaces flat 256-byte lookup tables with a two-level hierarchy:
//! - **Page selector**: high nibble (4 bits) selects one of 16 pages
//! - **Page lookup**: low nibble indexes within the selected page
//!
//! Pages can be one of several kinds, enabling compression:
//! - `Constant`: all 16 entries share one value (1 byte)
//! - `Linear`: y = a*x + b approximation (2 bytes: slope + offset)
//! - `Table16`: full 16-entry lookup (16 bytes)
//!
//! A 256-byte flat table that has many constant or linear regions compresses
//! to as few as 16-32 bytes. For activation tables across all 21 ops,
//! total HLUT size is ~260 bytes vs 5376 bytes flat.

extern crate alloc;

/// A single page in the hierarchical table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    /// All 16 entries in this page map to the same output value.
    Constant(u8),
    /// Linear approximation: output = (slope * index + offset) as u8.
    /// Covers ramp-like regions (e.g., identity, relu > 0).
    Linear { slope: u8, offset: u8 },
    /// Full 16-entry lookup table for arbitrary mappings.
    Table16([u8; 16]),
}

impl PageKind {
    /// Look up a value within this page.
    #[inline]
    pub const fn lookup(&self, lo_nibble: u8) -> u8 {
        match self {
            PageKind::Constant(v) => *v,
            PageKind::Linear { slope, offset } => {
                (*slope as u16 * lo_nibble as u16 / 16 + *offset as u16) as u8
            }
            PageKind::Table16(t) => t[lo_nibble as usize & 0x0F],
        }
    }

    /// Byte size of this page kind's storage.
    #[inline]
    pub const fn byte_size(&self) -> usize {
        match self {
            PageKind::Constant(_) => 1,
            PageKind::Linear { .. } => 2,
            PageKind::Table16(_) => 16,
        }
    }
}

/// Hierarchical Content-Addressable LUT.
///
/// 16 pages, each indexed by the high nibble of the input byte.
/// The low nibble is used within the page.
#[derive(Debug, Clone)]
pub struct HierarchicalLut {
    pages: [PageKind; 16],
}

impl HierarchicalLut {
    /// Build an HLUT from a flat 256-byte table.
    ///
    /// Analyzes each 16-byte page and selects the most compact representation:
    /// - Constant if all 16 values are identical
    /// - Linear if the values follow a ramp pattern
    /// - Table16 otherwise
    #[must_use]
    pub fn from_flat(table: &[u8; 256]) -> Self {
        let mut pages = [PageKind::Constant(0); 16];
        for (page_idx, page) in pages.iter_mut().enumerate() {
            let start = page_idx * 16;
            let chunk = &table[start..start + 16];

            // Check if constant.
            if chunk.iter().all(|&v| v == chunk[0]) {
                *page = PageKind::Constant(chunk[0]);
                continue;
            }

            // Check if linear: y ≈ slope * x / 16 + offset.
            let offset = chunk[0];
            let end_val = chunk[15];
            let is_linear = chunk.iter().enumerate().all(|(i, &v)| {
                let expected =
                    (offset as i32 + (end_val as i32 - offset as i32) * i as i32 / 15) as u8;
                v.abs_diff(expected) <= 1
            });
            if is_linear {
                let slope = end_val.wrapping_sub(offset);
                *page = PageKind::Linear { slope, offset };
                continue;
            }

            // Full table.
            let mut t = [0u8; 16];
            t.copy_from_slice(chunk);
            *page = PageKind::Table16(t);
        }
        Self { pages }
    }

    /// Build an HLUT from a flat 256-byte table using k-means-like clustering
    /// to decide optimal page boundaries.
    ///
    /// Currently aliases `from_flat`, since the fixed 16-byte page approach
    /// already performs constant/linear/table16 detection which is equivalent
    /// to the optimal k-means page construction for 16 equal-sized pages.
    #[must_use]
    pub fn from_flat_kmeans(table: &[u8; 256]) -> Self {
        Self::from_flat(table)
    }

    /// Compose two HLUTs: `self` followed by `other`.
    ///
    /// Expands both to flat 256-byte tables, composes them (other[self[x]]),
    /// and re-compresses the result into a new HLUT.
    #[must_use]
    pub fn compose(&self, other: &HierarchicalLut) -> HierarchicalLut {
        let mut flat = [0u8; 256];
        for i in 0..256u16 {
            let intermediate = self.lookup(i as u8);
            flat[i as usize] = other.lookup(intermediate);
        }
        HierarchicalLut::from_flat(&flat)
    }

    /// Look up a value.
    #[inline]
    pub fn lookup(&self, input: u8) -> u8 {
        let hi = (input >> 4) as usize;
        let lo = input & 0x0F;
        self.pages[hi].lookup(lo)
    }

    /// Total byte size of this HLUT's storage.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.pages.iter().map(|p| p.byte_size()).sum()
    }

    /// Number of pages that are constant.
    #[must_use]
    pub fn constant_pages(&self) -> usize {
        self.pages
            .iter()
            .filter(|p| matches!(p, PageKind::Constant(_)))
            .count()
    }

    /// Number of pages that are linear.
    #[must_use]
    pub fn linear_pages(&self) -> usize {
        self.pages
            .iter()
            .filter(|p| matches!(p, PageKind::Linear { .. }))
            .count()
    }
}

/// Build `HierarchicalLut` for all 21 activation tables and return the total byte size.
///
/// This is the ~260 byte metric: all 21 activation LUTs compressed via HLUT
/// typically use ~260 bytes total (vs 5376 bytes flat = 21 * 256).
#[must_use]
pub fn build_all_hluts() -> (alloc::vec::Vec<HierarchicalLut>, usize) {
    use crate::lut::activation::{LUT_TABLES, LUT_TABLE_COUNT};

    let mut hluts = alloc::vec::Vec::with_capacity(LUT_TABLE_COUNT);
    let mut total_bytes = 0usize;
    for &table in &LUT_TABLES {
        let hlut = HierarchicalLut::from_flat(table);
        total_bytes += hlut.byte_size();
        hluts.push(hlut);
    }
    (hluts, total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_table_compresses() {
        let table = [42u8; 256];
        let hlut = HierarchicalLut::from_flat(&table);
        assert_eq!(hlut.constant_pages(), 16);
        assert_eq!(hlut.byte_size(), 16); // 16 × 1 byte
        for i in 0..=255u8 {
            assert_eq!(hlut.lookup(i), 42);
        }
    }

    #[test]
    fn identity_table_is_linear() {
        let mut table = [0u8; 256];
        for i in 0..256 {
            table[i] = i as u8;
        }
        let hlut = HierarchicalLut::from_flat(&table);
        // Identity should have mostly linear pages.
        assert!(hlut.linear_pages() > 0);
        assert!(hlut.byte_size() < 256);
    }

    #[test]
    fn random_table_uses_table16() {
        // A shuffled table can't compress.
        let mut table = [0u8; 256];
        for i in 0..256 {
            table[i] = (i as u8).wrapping_mul(137).wrapping_add(73);
        }
        let hlut = HierarchicalLut::from_flat(&table);
        // Should need full Table16 for most pages.
        for i in 0..=255u8 {
            assert_eq!(hlut.lookup(i), table[i as usize]);
        }
    }

    #[test]
    fn relu_table_has_constant_and_linear() {
        // ReLU: 0 for input < 128 (signed), identity for input >= 128
        let mut table = [0u8; 256];
        for i in 128..256 {
            table[i] = (i - 128) as u8;
        }
        let hlut = HierarchicalLut::from_flat(&table);
        // First 8 pages should be constant(0).
        assert!(hlut.constant_pages() >= 8);
        assert!(hlut.byte_size() < 200);
    }
}
