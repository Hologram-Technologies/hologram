//! Content-addressable weight deduplication for pipeline archives.
//!
//! Components that share a `weight_group` (e.g., LLM prefill/decode sharing
//! the same transformer weights) store the weight blob only once. The
//! `WeightStore` builder collects weight blobs keyed by group, deduplicates
//! identical content via BLAKE3 cryptographic hashing, and produces a
//! `WeightDedupIndex` section for the pipeline archive.
//!
//! At load time, [`LoadedPipeline`](crate::loader::pipeline::LoadedPipeline)
//! resolves the index: components without inline weights are given a slice of
//! the shared weight blob. Runtime weight access remains zero-indirection —
//! dedup is fully resolved at archive-load time.

use alloc::string::String;
use alloc::vec::Vec;
use std::collections::HashMap;

extern crate alloc;

use crate::section::EmbeddableSection;

/// BLAKE3 digest used as content address for weight blobs.
type Blake3Hash = [u8; 32];

/// A reference to a deduplicated weight blob in the shared store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightRef {
    /// Offset into the shared weight blob.
    pub offset: u64,
    /// Size of this weight block.
    pub size: u64,
}

/// Content-addressable weight storage for pipeline archives.
///
/// Collects weight blobs by group name, deduplicating identical content
/// via BLAKE3 hash. Components register with both a component name (used
/// in the output index) and a group name (the dedup key). The output
/// `WeightDedupIndex` is keyed by **component name** so the pipeline
/// loader can resolve weights without knowing about group semantics.
pub struct WeightStore {
    /// group_name → index into `blobs`
    group_to_blob: HashMap<String, usize>,
    /// BLAKE3 hash → index into `blobs` (for cross-group content dedup)
    hash_to_blob: HashMap<Blake3Hash, usize>,
    /// component_name → index into `blobs` (for per-component output)
    component_to_blob: HashMap<String, usize>,
    /// Unique weight blobs in insertion order.
    blobs: Vec<Vec<u8>>,
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
        Self {
            group_to_blob: HashMap::new(),
            hash_to_blob: HashMap::new(),
            component_to_blob: HashMap::new(),
            blobs: Vec::new(),
        }
    }

    /// Register a weight blob for a component.
    ///
    /// `component` is the pipeline entry name (e.g., `"lm.prefill"`).
    /// `group` is the dedup key — components sharing a group with
    /// identical content store the blob once.
    ///
    /// Returns `None` if `data` is empty (no weights to store).
    pub fn insert(&mut self, component: &str, group: &str, data: &[u8]) -> Option<WeightRef> {
        if data.is_empty() {
            return None;
        }

        // If this group was already registered, reuse its blob.
        if let Some(&blob_idx) = self.group_to_blob.get(group) {
            self.component_to_blob
                .insert(component.to_string(), blob_idx);
            return Some(self.ref_for_blob(blob_idx));
        }

        let hash: Blake3Hash = *blake3::hash(data).as_bytes();

        // Content-match against existing blobs via BLAKE3.
        if let Some(&blob_idx) = self.hash_to_blob.get(&hash) {
            self.group_to_blob.insert(group.to_string(), blob_idx);
            self.component_to_blob
                .insert(component.to_string(), blob_idx);
            return Some(self.ref_for_blob(blob_idx));
        }

        // New unique blob.
        let blob_idx = self.blobs.len();
        self.blobs.push(data.to_vec());
        self.hash_to_blob.insert(hash, blob_idx);
        self.group_to_blob.insert(group.to_string(), blob_idx);
        self.component_to_blob
            .insert(component.to_string(), blob_idx);
        Some(self.ref_for_blob(blob_idx))
    }

    /// Check if a group has already been registered.
    #[must_use]
    pub fn contains_group(&self, group: &str) -> bool {
        self.group_to_blob.contains_key(group)
    }

    /// Number of unique weight blobs stored.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.blobs.len()
    }

    /// Total deduplicated bytes (sum of unique blobs).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.blobs.iter().map(|b| b.len() as u64).sum()
    }

    /// Build the final shared weight blob and deduplication index.
    ///
    /// Concatenates all unique blobs into a single byte vector and
    /// produces a [`WeightDedupIndex`] keyed by component name. The
    /// pipeline loader uses this to resolve weights at archive-load time.
    #[must_use]
    pub fn build(self) -> (Vec<u8>, WeightDedupIndex) {
        // Compute offset for each blob in the concatenated output.
        let mut blob_offsets: Vec<u64> = Vec::with_capacity(self.blobs.len());
        let mut cursor: u64 = 0;
        for blob in &self.blobs {
            blob_offsets.push(cursor);
            cursor += blob.len() as u64;
        }

        // Build entries: one per registered component.
        let mut entries: Vec<WeightDedupEntry> = self
            .component_to_blob
            .iter()
            .map(|(component, &blob_idx)| {
                let offset = blob_offsets[blob_idx];
                let size = self.blobs[blob_idx].len() as u64;
                WeightDedupEntry {
                    component: component.clone(),
                    offset,
                    size,
                }
            })
            .collect();
        // Sort for deterministic output.
        entries.sort_by(|a, b| a.component.cmp(&b.component));

        // Concatenate blobs.
        let total_len = self.blobs.iter().map(|b| b.len()).sum();
        let mut blob = Vec::with_capacity(total_len);
        for b in self.blobs {
            blob.extend_from_slice(&b);
        }

        (blob, WeightDedupIndex { entries })
    }

    /// Compute the `WeightRef` for a blob by index.
    fn ref_for_blob(&self, blob_idx: usize) -> WeightRef {
        let offset: u64 = self.blobs[..blob_idx].iter().map(|b| b.len() as u64).sum();
        WeightRef {
            offset,
            size: self.blobs[blob_idx].len() as u64,
        }
    }
}

/// Index mapping components to offsets in a shared weight blob.
///
/// Serialized via rkyv and embedded as `SECTION_WEIGHT_DEDUP` in the
/// pipeline wrapper archive. Keyed by component name (matching
/// `PipelineEntry.name`) so the loader can resolve without external
/// metadata.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WeightDedupIndex {
    /// Per-component entries mapping to (offset, size) in shared blob.
    pub entries: Vec<WeightDedupEntry>,
}

impl WeightDedupIndex {
    /// Look up a component by name.
    #[must_use]
    pub fn find_component(&self, component: &str) -> Option<&WeightDedupEntry> {
        self.entries.iter().find(|e| e.component == component)
    }

    /// Whether this index is empty (no entries).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Deserialize from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl EmbeddableSection for WeightDedupIndex {
    fn section_kind(&self) -> u32 {
        crate::section::SECTION_WEIGHT_DEDUP
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("WeightDedupIndex serialization should not fail")
            .to_vec()
    }
}

/// A single entry in the weight deduplication index.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WeightDedupEntry {
    /// Component name (matches `PipelineEntry.name`).
    pub component: String,
    /// Byte offset in the shared weight blob.
    pub offset: u64,
    /// Byte size of this component's weights.
    pub size: u64,
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
        assert!(index.is_empty());
    }

    #[test]
    fn insert_empty_data_returns_none() {
        let mut store = WeightStore::new();
        assert!(store.insert("comp_a", "group_a", &[]).is_none());
        assert_eq!(store.unique_count(), 0);
    }

    #[test]
    fn single_component() {
        let mut store = WeightStore::new();
        let weights = vec![1u8, 2, 3, 4, 5];
        let r = store
            .insert("model.forward", "lm", &weights)
            .expect("non-empty insert");
        assert_eq!(r.offset, 0);
        assert_eq!(r.size, 5);
        assert_eq!(store.unique_count(), 1);

        let (blob, index) = store.build();
        assert_eq!(blob, weights);
        assert_eq!(index.entries.len(), 1);
        assert_eq!(index.entries[0].component, "model.forward");
        assert_eq!(index.entries[0].offset, 0);
        assert_eq!(index.entries[0].size, 5);
    }

    #[test]
    fn same_group_deduplicates() {
        let mut store = WeightStore::new();
        let weights = vec![42u8; 256];
        let r1 = store
            .insert("lm.prefill", "lm", &weights)
            .expect("insert prefill");
        let r2 = store
            .insert("lm.decode", "lm", &weights)
            .expect("insert decode");
        assert_eq!(r1, r2);
        assert_eq!(store.unique_count(), 1);
        assert_eq!(store.total_bytes(), 256);

        let (blob, index) = store.build();
        assert_eq!(blob.len(), 256);
        assert_eq!(index.entries.len(), 2);
        let e1 = index.find_component("lm.prefill").expect("find prefill");
        let e2 = index.find_component("lm.decode").expect("find decode");
        assert_eq!(e1.offset, e2.offset);
        assert_eq!(e1.size, e2.size);
    }

    #[test]
    fn different_groups_same_content_deduplicates() {
        let mut store = WeightStore::new();
        let weights = vec![42u8; 256];
        let r1 = store
            .insert("encoder", "group_a", &weights)
            .expect("insert encoder");
        let r2 = store
            .insert("decoder", "group_b", &weights)
            .expect("insert decoder");
        assert_eq!(r1, r2);
        assert_eq!(store.unique_count(), 1);
    }

    #[test]
    fn different_content_not_deduplicated() {
        let mut store = WeightStore::new();
        let w1 = vec![1u8; 100];
        let w2 = vec![2u8; 200];
        let r1 = store
            .insert("encoder", "vision", &w1)
            .expect("insert encoder");
        let r2 = store
            .insert("decoder", "text", &w2)
            .expect("insert decoder");
        assert_ne!(r1, r2);
        assert_eq!(store.unique_count(), 2);
        assert_eq!(store.total_bytes(), 300);

        let (blob, index) = store.build();
        assert_eq!(blob.len(), 300);
        let e1 = index.find_component("decoder").expect("find decoder");
        let e2 = index.find_component("encoder").expect("find encoder");
        assert_eq!(e2.offset, 0);
        assert_eq!(e2.size, 100);
        assert_eq!(e1.offset, 100);
        assert_eq!(e1.size, 200);
    }

    #[test]
    fn contains_group() {
        let mut store = WeightStore::new();
        assert!(!store.contains_group("lm"));
        store.insert("lm.prefill", "lm", &[1, 2, 3]);
        assert!(store.contains_group("lm"));
        assert!(!store.contains_group("other"));
    }

    #[test]
    fn four_components_two_groups() {
        let mut store = WeightStore::new();
        let shared_a = vec![0xAAu8; 512];
        let shared_b = vec![0xBBu8; 128];

        store.insert("ae.encoder", "ae", &shared_a);
        store.insert("ae.decoder", "ae", &shared_a);
        store.insert("backbone", "lm", &shared_b);
        store.insert("head", "lm", &shared_b);

        assert_eq!(store.unique_count(), 2);
        assert_eq!(store.total_bytes(), 512 + 128);

        let (blob, index) = store.build();
        assert_eq!(blob.len(), 640);
        assert_eq!(index.entries.len(), 4);

        let ae_enc = index.find_component("ae.encoder").expect("ae.encoder");
        let ae_dec = index.find_component("ae.decoder").expect("ae.decoder");
        assert_eq!(ae_enc.offset, ae_dec.offset);
        assert_eq!(ae_enc.size, 512);

        let bb = index.find_component("backbone").expect("backbone");
        let hd = index.find_component("head").expect("head");
        assert_eq!(bb.offset, hd.offset);
        assert_eq!(bb.size, 128);
    }

    #[test]
    fn rkyv_round_trip() {
        let index = WeightDedupIndex {
            entries: vec![
                WeightDedupEntry {
                    component: "lm.prefill".into(),
                    offset: 0,
                    size: 1024,
                },
                WeightDedupEntry {
                    component: "lm.decode".into(),
                    offset: 0,
                    size: 1024,
                },
            ],
        };

        let bytes = index.to_bytes();
        let deserialized = WeightDedupIndex::from_bytes(&bytes).expect("rkyv deserialization");
        assert_eq!(deserialized, index);
    }

    #[test]
    fn embeddable_section_kind() {
        let index = WeightDedupIndex { entries: vec![] };
        assert_eq!(index.section_kind(), crate::section::SECTION_WEIGHT_DEDUP);
    }
}
