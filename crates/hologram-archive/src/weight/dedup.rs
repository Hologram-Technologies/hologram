//! Content-addressable weight storage for pipeline archives.
//!
//! Each unique weight blob is stored once; sub-archives reference by hash.
//! Follows the `hologram-compression` pattern: a self-contained algebraic
//! primitive consumed by the pipeline assembly path.
//!
//! # Usage
//!
//! ```ignore
//! let mut store = WeightStore::new();
//! let ref_a = store.insert(weights_prefill);  // stores blob
//! let ref_b = store.insert(weights_decode);   // same content → same ref
//! assert_eq!(ref_a, ref_b);
//!
//! let (blob, index) = store.build();
//! // blob: deduplicated weight bytes
//! // index: maps WeightRef → (offset, len) in blob
//! ```

use alloc::vec::Vec;

extern crate alloc;

use crate::checksum;
use crate::section::{EmbeddableSection, SECTION_CUSTOM_BASE};

/// Section kind for weight deduplication index.
pub const SECTION_WEIGHT_DEDUP: u32 = SECTION_CUSTOM_BASE + 0x20;

/// Reference to a deduplicated weight block.
///
/// Two `WeightRef`s are equal iff they point to byte-identical weight blobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightRef {
    /// CRC32 of the weight blob (fast identity check).
    pub checksum: u32,
    /// Byte size of the weight blob.
    pub size: u64,
    /// Internal index into `WeightStore::blocks`.
    index: usize,
}

/// Content-addressable weight storage.
///
/// Inserts weight blobs and deduplicates by content identity (CRC32 + exact
/// byte comparison on collision). Call [`build`](Self::build) to produce
/// the final deduplicated blob and index.
pub struct WeightStore {
    /// Unique weight blocks: (checksum, data).
    blocks: Vec<(u32, Vec<u8>)>,
}

impl Default for WeightStore {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightStore {
    /// Create an empty weight store.
    #[must_use]
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Insert a weight blob. Returns a reference.
    ///
    /// If a byte-identical blob already exists, returns the existing ref
    /// without storing a duplicate.
    pub fn insert(&mut self, data: Vec<u8>) -> WeightRef {
        let cksum = checksum::crc32(&data);

        // Check for existing identical block.
        for (i, (existing_cksum, existing_data)) in self.blocks.iter().enumerate() {
            if *existing_cksum == cksum
                && existing_data.len() == data.len()
                && *existing_data == data
            {
                return WeightRef {
                    checksum: cksum,
                    size: data.len() as u64,
                    index: i,
                };
            }
        }

        // New unique block.
        let index = self.blocks.len();
        let size = data.len() as u64;
        self.blocks.push((cksum, data));

        WeightRef {
            checksum: cksum,
            size,
            index,
        }
    }

    /// Number of unique weight blocks stored.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total deduplicated byte size across all unique blocks.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.blocks.iter().map(|(_, d)| d.len() as u64).sum()
    }

    /// Build the final deduplicated blob and index.
    ///
    /// Returns:
    /// - `blob`: concatenated unique weight bytes (page-aligned between blocks)
    /// - `index`: serializable index mapping block index → (offset, len, checksum)
    pub fn build(self) -> (Vec<u8>, WeightDedupIndex) {
        let mut blob = Vec::new();
        let mut entries = Vec::new();

        for (cksum, data) in &self.blocks {
            let offset = blob.len() as u64;
            let size = data.len() as u64;
            entries.push(WeightDedupEntry {
                offset,
                size,
                checksum: *cksum,
            });
            blob.extend_from_slice(data);
            // Page-align between blocks (matches PipelineWriter pattern).
            let aligned = crate::format::align_to_page(blob.len() as u64) as usize;
            blob.resize(aligned, 0);
        }

        (blob, WeightDedupIndex { entries })
    }

    /// Resolve a `WeightRef` to the raw weight bytes.
    ///
    /// Returns `None` if the ref is invalid (from a different store).
    #[must_use]
    pub fn get(&self, weight_ref: &WeightRef) -> Option<&[u8]> {
        self.blocks.get(weight_ref.index).map(|(_, d)| d.as_slice())
    }
}

/// Serializable index for the deduplicated weight blob.
///
/// Stored as `SECTION_WEIGHT_DEDUP` in the pipeline wrapper archive.
/// At load time, the runtime resolves weight refs to offsets in the
/// shared blob — zero overhead during execution.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WeightDedupIndex {
    /// One entry per unique weight block.
    pub entries: Vec<WeightDedupEntry>,
}

/// A single entry in the dedup index: where a unique block lives in the blob.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WeightDedupEntry {
    /// Byte offset in the deduplicated blob.
    pub offset: u64,
    /// Byte size.
    pub size: u64,
    /// CRC32 for integrity verification at load time.
    pub checksum: u32,
}

impl EmbeddableSection for WeightDedupIndex {
    fn section_kind(&self) -> u32 {
        SECTION_WEIGHT_DEDUP
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("WeightDedupIndex serialization should not fail")
            .to_vec()
    }
}

impl WeightDedupIndex {
    /// Deserialize from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }

    /// Zero-copy access from raw bytes (e.g. memory-mapped archive).
    pub fn access(bytes: &[u8]) -> Result<&ArchivedWeightDedupIndex, rkyv::rancor::Error> {
        rkyv::access::<ArchivedWeightDedupIndex, rkyv::rancor::Error>(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_store() {
        let store = WeightStore::new();
        assert_eq!(store.unique_count(), 0);
        assert_eq!(store.total_bytes(), 0);
        let (blob, index) = store.build();
        assert!(blob.is_empty());
        assert!(index.entries.is_empty());
    }

    #[test]
    fn single_insert() {
        let mut store = WeightStore::new();
        let data = vec![1u8, 2, 3, 4];
        let r = store.insert(data);
        assert_eq!(r.size, 4);
        assert_eq!(store.unique_count(), 1);
    }

    #[test]
    fn dedup_identical_blobs() {
        let mut store = WeightStore::new();
        let data = vec![0u8; 1024];
        let r1 = store.insert(data.clone());
        let r2 = store.insert(data);
        assert_eq!(r1, r2);
        assert_eq!(store.unique_count(), 1);
    }

    #[test]
    fn distinct_blobs_stored_separately() {
        let mut store = WeightStore::new();
        let r1 = store.insert(vec![1u8; 512]);
        let r2 = store.insert(vec![2u8; 512]);
        assert_ne!(r1, r2);
        assert_eq!(store.unique_count(), 2);
    }

    #[test]
    fn build_produces_valid_blob() {
        let mut store = WeightStore::new();
        store.insert(vec![0xAA; 100]);
        store.insert(vec![0xBB; 200]);
        let (blob, index) = store.build();
        assert_eq!(index.entries.len(), 2);

        // First block at offset 0.
        assert_eq!(index.entries[0].offset, 0);
        assert_eq!(index.entries[0].size, 100);

        // Second block after page-aligned first.
        assert!(index.entries[1].offset >= 100);
        assert_eq!(index.entries[1].size, 200);

        // Verify content.
        let e0 = &index.entries[0];
        assert!(
            blob[e0.offset as usize..e0.offset as usize + e0.size as usize]
                .iter()
                .all(|&b| b == 0xAA)
        );
        let e1 = &index.entries[1];
        assert!(
            blob[e1.offset as usize..e1.offset as usize + e1.size as usize]
                .iter()
                .all(|&b| b == 0xBB)
        );
    }

    #[test]
    fn dedup_saves_space() {
        let mut store = WeightStore::new();
        let weights = vec![0xCC; 4096];
        store.insert(weights.clone());
        store.insert(weights.clone());
        store.insert(weights);
        // 3 inserts of identical data → 1 unique block.
        assert_eq!(store.unique_count(), 1);
        assert_eq!(store.total_bytes(), 4096);
    }

    #[test]
    fn get_returns_correct_data() {
        let mut store = WeightStore::new();
        let data = vec![42u8; 256];
        let r = store.insert(data.clone());
        let retrieved = store.get(&r).expect("should find ref");
        assert_eq!(retrieved, &data);
    }

    #[test]
    fn index_rkyv_roundtrip() {
        let index = WeightDedupIndex {
            entries: vec![
                WeightDedupEntry {
                    offset: 0,
                    size: 1024,
                    checksum: 0xDEAD,
                },
                WeightDedupEntry {
                    offset: 4096,
                    size: 2048,
                    checksum: 0xBEEF,
                },
            ],
        };
        let bytes = index.to_bytes();
        let deserialized =
            WeightDedupIndex::from_bytes(&bytes).expect("deserialization should succeed");
        assert_eq!(deserialized, index);
    }

    #[test]
    fn index_zero_copy_access() {
        let index = WeightDedupIndex {
            entries: vec![WeightDedupEntry {
                offset: 0,
                size: 512,
                checksum: 0x1234,
            }],
        };
        let bytes = index.to_bytes();
        let archived = WeightDedupIndex::access(&bytes).expect("zero-copy access should succeed");
        assert_eq!(archived.entries.len(), 1);
        assert_eq!(archived.entries[0].checksum, 0x1234);
    }
}
