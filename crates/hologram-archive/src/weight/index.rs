//! Per-tensor weight offset index with layer group annotations.
//!
//! Maps each weight tensor in the archive's weight blob to its byte range
//! and logical layer group (e.g., `"layers.0"`, `"embed"`, `"lm_head"`).
//!
//! At load time, consumers can aggregate entries by group to issue
//! `madvise(MADV_WILLNEED)` for upcoming layers, load partial weight
//! ranges for layer-granular offloading, or inspect the archive without
//! loading the full weight blob.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::section::EmbeddableSection;

/// A single tensor's location within the weight blob.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WeightIndexEntry {
    /// Tensor name from the source model (e.g.,
    /// `"model.layers.0.self_attn.q_proj.weight"`).
    pub tensor_name: String,
    /// Normalized layer group (e.g., `"layers.0"`, `"embed"`, `"norm"`,
    /// `"lm_head"`, `"other"`).
    pub group: String,
    /// Byte offset within the weight blob.
    pub offset: u64,
    /// Byte size of this tensor.
    pub size: u64,
}

/// Index mapping individual tensors to byte ranges in the weight blob,
/// annotated with layer group membership.
///
/// Entries are sorted by offset (matching weight blob layout order).
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WeightIndex {
    /// Per-tensor entries, sorted by `offset`.
    pub entries: Vec<WeightIndexEntry>,
}

impl WeightIndex {
    /// Unique group names in sorted order.
    #[must_use]
    pub fn groups(&self) -> BTreeSet<&str> {
        self.entries.iter().map(|e| e.group.as_str()).collect()
    }

    /// All entries belonging to a given group.
    #[must_use]
    pub fn entries_for_group<'a>(&'a self, group: &str) -> Vec<&'a WeightIndexEntry> {
        self.entries.iter().filter(|e| e.group == group).collect()
    }

    /// Bounding byte range `(min_offset, end)` for a group, where `end` is
    /// one past the last byte. Returns `None` if the group has no entries.
    #[must_use]
    pub fn group_byte_range(&self, group: &str) -> Option<(u64, u64)> {
        let entries = self.entries_for_group(group);
        if entries.is_empty() {
            return None;
        }
        let min_offset = entries.iter().map(|e| e.offset).min().unwrap_or(0);
        let max_end = entries.iter().map(|e| e.offset + e.size).max().unwrap_or(0);
        Some((min_offset, max_end))
    }

    /// Deserialize from raw section bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl EmbeddableSection for WeightIndex {
    fn section_kind(&self) -> u32 {
        crate::section::SECTION_WEIGHT_INDEX
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("WeightIndex serialization should not fail")
            .to_vec()
    }
}

/// Derive a normalized layer group name from a tensor name.
///
/// Handles both ONNX-style (`model.layers.{N}.*`) and GGUF-style
/// (`blk.{N}.*`) naming conventions. Non-layer tensors are classified
/// as `"embed"`, `"norm"`, `"lm_head"`, or `"other"`.
#[must_use]
pub fn derive_layer_group(tensor_name: &str) -> String {
    // Try to extract a layer index from known prefix patterns.
    for prefix in &[
        "model.layers.",
        "encoder.layers.",
        "decoder.layers.",
        "transformer.h.",
        "blk.",
    ] {
        if let Some(rest) = tensor_name.strip_prefix(prefix) {
            if let Some(n) = extract_leading_number(rest) {
                return alloc::format!("layers.{n}");
            }
        }
    }

    // Embedding tensors.
    if tensor_name.contains("embed") || tensor_name.contains("token_embedding") {
        return String::from("embed");
    }

    // LM head (output projection).
    if tensor_name.starts_with("lm_head") || tensor_name == "output.weight" {
        return String::from("lm_head");
    }

    // Top-level normalization (not inside a layer block).
    if tensor_name.contains("norm") {
        return String::from("norm");
    }

    String::from("other")
}

/// Extract a leading decimal number from a string (e.g., `"12.foo"` → `Some(12)`).
fn extract_leading_number(s: &str) -> Option<u32> {
    let digits: &str = s.split(|c: char| !c.is_ascii_digit()).next()?;
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── derive_layer_group ──────────────────────────────────────────

    #[test]
    fn onnx_layer_names() {
        assert_eq!(
            derive_layer_group("model.layers.0.self_attn.q_proj.weight"),
            "layers.0"
        );
        assert_eq!(
            derive_layer_group("model.layers.31.mlp.gate_proj.weight"),
            "layers.31"
        );
    }

    #[test]
    fn gguf_layer_names() {
        assert_eq!(derive_layer_group("blk.0.attn_q.weight"), "layers.0");
        assert_eq!(derive_layer_group("blk.15.ffn_down.weight"), "layers.15");
    }

    #[test]
    fn encoder_decoder_layers() {
        assert_eq!(
            derive_layer_group("encoder.layers.2.self_attn.weight"),
            "layers.2"
        );
        assert_eq!(
            derive_layer_group("decoder.layers.5.cross_attn.weight"),
            "layers.5"
        );
    }

    #[test]
    fn gpt2_style() {
        assert_eq!(
            derive_layer_group("transformer.h.11.attn.c_attn.weight"),
            "layers.11"
        );
    }

    #[test]
    fn embed_tokens() {
        assert_eq!(derive_layer_group("model.embed_tokens.weight"), "embed");
        assert_eq!(derive_layer_group("token_embedding.weight"), "embed");
    }

    #[test]
    fn lm_head() {
        assert_eq!(derive_layer_group("lm_head.weight"), "lm_head");
        assert_eq!(derive_layer_group("output.weight"), "lm_head");
    }

    #[test]
    fn top_level_norm() {
        assert_eq!(derive_layer_group("model.norm.weight"), "norm");
    }

    #[test]
    fn unknown_fallback() {
        assert_eq!(derive_layer_group("some_random_tensor"), "other");
    }

    // ── WeightIndex helpers ─────────────────────────────────────────

    fn sample_index() -> WeightIndex {
        WeightIndex {
            entries: vec![
                WeightIndexEntry {
                    tensor_name: "model.embed_tokens.weight".into(),
                    group: "embed".into(),
                    offset: 0,
                    size: 1000,
                },
                WeightIndexEntry {
                    tensor_name: "model.layers.0.self_attn.q_proj.weight".into(),
                    group: "layers.0".into(),
                    offset: 1000,
                    size: 500,
                },
                WeightIndexEntry {
                    tensor_name: "model.layers.0.self_attn.v_proj.weight".into(),
                    group: "layers.0".into(),
                    offset: 1500,
                    size: 500,
                },
                WeightIndexEntry {
                    tensor_name: "model.layers.1.self_attn.q_proj.weight".into(),
                    group: "layers.1".into(),
                    offset: 2000,
                    size: 500,
                },
            ],
        }
    }

    #[test]
    fn groups_sorted() {
        let idx = sample_index();
        let groups: Vec<&str> = idx.groups().into_iter().collect();
        assert_eq!(groups, vec!["embed", "layers.0", "layers.1"]);
    }

    #[test]
    fn entries_for_group_filters() {
        let idx = sample_index();
        let l0 = idx.entries_for_group("layers.0");
        assert_eq!(l0.len(), 2);
        assert!(l0.iter().all(|e| e.group == "layers.0"));
    }

    #[test]
    fn group_byte_range_covers_all() {
        let idx = sample_index();
        assert_eq!(idx.group_byte_range("layers.0"), Some((1000, 2000)));
        assert_eq!(idx.group_byte_range("embed"), Some((0, 1000)));
        assert_eq!(idx.group_byte_range("nonexistent"), None);
    }

    #[test]
    fn section_kind_is_weight_index() {
        let idx = WeightIndex { entries: vec![] };
        assert_eq!(idx.section_kind(), crate::section::SECTION_WEIGHT_INDEX);
    }

    // ── rkyv round-trip ─────────────────────────────────────────────

    #[test]
    fn rkyv_round_trip() {
        let idx = sample_index();
        let bytes = idx.to_bytes();
        let restored = WeightIndex::from_bytes(&bytes).expect("deserialization");
        assert_eq!(restored, idx);
    }
}
