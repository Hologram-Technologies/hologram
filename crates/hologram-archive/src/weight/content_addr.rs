//! Content-addressed weight index for UOR-based weight resolution.
//!
//! Maps BLAKE3 content digests to byte offsets within the weight blob,
//! enabling address-based resolution that unifies mmap, in-memory, and
//! streaming paths behind a single lookup mechanism.

use crate::section::EmbeddableSection;

/// Section kind identifier for the content address index.
pub const SECTION_CONTENT_ADDRESS_INDEX: u32 = 6;

/// A single entry in the content address index.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ContentAddressEntry {
    /// BLAKE3 content hash (the content address).
    pub digest: [u8; 32],
    /// Byte offset within the weight blob.
    pub offset: u64,
    /// Byte size of the encoded tensor data.
    pub size: u64,
}

/// Index mapping content addresses (BLAKE3 digests) to weight blob locations.
///
/// Entries are sorted by digest for O(log n) binary search resolution.
/// Built in-memory during archive writing and serialized as the final section.
#[derive(
    Debug, Clone, PartialEq, Eq, Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ContentAddressIndex {
    /// Entries sorted by digest for binary search.
    pub entries: Vec<ContentAddressEntry>,
}

impl ContentAddressIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create an index with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Insert an entry. The caller must call `sort()` after all insertions.
    pub fn insert(&mut self, digest: [u8; 32], offset: u64, size: u64) {
        self.entries.push(ContentAddressEntry {
            digest,
            offset,
            size,
        });
    }

    /// Sort entries by digest for binary search. Call after all insertions.
    pub fn sort(&mut self) {
        self.entries.sort_by_key(|a| a.digest);
    }

    /// Resolve a content address to its blob location.
    /// Returns `None` if the digest is not found.
    #[must_use]
    pub fn resolve(&self, digest: &[u8; 32]) -> Option<&ContentAddressEntry> {
        self.entries
            .binary_search_by(|e| e.digest.cmp(digest))
            .ok()
            .map(|i| &self.entries[i])
    }

    /// Number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl EmbeddableSection for ContentAddressIndex {
    fn section_kind(&self) -> u32 {
        SECTION_CONTENT_ADDRESS_INDEX
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("ContentAddressIndex serialization should not fail")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index() {
        let idx = ContentAddressIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.resolve(&[0u8; 32]).is_none());
    }

    #[test]
    fn insert_and_resolve() {
        let mut idx = ContentAddressIndex::new();
        let digest_a = [0xAAu8; 32];
        let digest_b = [0xBBu8; 32];

        idx.insert(digest_a, 0, 1024);
        idx.insert(digest_b, 1024, 2048);
        idx.sort();

        let entry = idx.resolve(&digest_a).expect("should find digest_a");
        assert_eq!(entry.offset, 0);
        assert_eq!(entry.size, 1024);

        let entry = idx.resolve(&digest_b).expect("should find digest_b");
        assert_eq!(entry.offset, 1024);
        assert_eq!(entry.size, 2048);

        assert!(idx.resolve(&[0xCCu8; 32]).is_none());
    }

    #[test]
    fn sorted_order() {
        let mut idx = ContentAddressIndex::new();
        // Insert in reverse order
        idx.insert([0xFF; 32], 200, 100);
        idx.insert([0x00; 32], 0, 100);
        idx.insert([0x80; 32], 100, 100);
        idx.sort();

        assert_eq!(idx.entries[0].digest, [0x00; 32]);
        assert_eq!(idx.entries[1].digest, [0x80; 32]);
        assert_eq!(idx.entries[2].digest, [0xFF; 32]);
    }

    #[test]
    fn rkyv_round_trip() {
        let mut idx = ContentAddressIndex::new();
        idx.insert([0xAA; 32], 0, 1024);
        idx.insert([0xBB; 32], 4096, 2048);
        idx.sort();

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&idx).expect("serialize");
        let deserialized = rkyv::from_bytes::<ContentAddressIndex, rkyv::rancor::Error>(&bytes)
            .expect("deserialize");
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized.entries[0].digest, [0xAA; 32]);
        assert_eq!(deserialized.entries[1].offset, 4096);
    }

    #[test]
    fn section_kind() {
        let idx = ContentAddressIndex::new();
        assert_eq!(idx.section_kind(), SECTION_CONTENT_ADDRESS_INDEX);
    }

    #[test]
    fn embeddable_section_round_trip() {
        let mut idx = ContentAddressIndex::new();
        idx.insert([0x42; 32], 0, 512);
        idx.sort();

        let bytes = idx.to_bytes();
        let deserialized = rkyv::from_bytes::<ContentAddressIndex, rkyv::rancor::Error>(&bytes)
            .expect("deserialize section bytes");
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized.entries[0].size, 512);
    }
}
