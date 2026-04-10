//! Extensible section system for archive metadata.

pub mod compile_unit_meta;
pub mod conformance_shape;
pub mod model_meta;
pub mod table;
pub mod tokenizer;

/// Well-known section kind: weight index.
pub const SECTION_WEIGHT_INDEX: u32 = 1;
/// Well-known section kind: layer header.
pub const SECTION_LAYER_HEADER: u32 = 2;
/// Well-known section kind: pipeline header.
pub const SECTION_PIPELINE: u32 = 3;
/// Well-known section kind: weight deduplication index.
pub const SECTION_WEIGHT_DEDUP: u32 = 4;
/// Well-known section kind: CompileUnit metadata.
pub const SECTION_COMPILE_UNIT_META: u32 = 5;
/// Well-known section kind: conformance shape declaration (v0.2.0).
///
/// Identifies which `Shape` the archive's compiled tape conforms to.
/// Loaders use this to refuse mismatched archives at load time, before
/// any execution proceeds. Per the v0.2.0 conformance-first contract,
/// every archive emitted by `hologram-compiler` carries this section.
pub const SECTION_CONFORMANCE_SHAPE: u32 = 6;
/// Base kind for custom sections.
pub const SECTION_CUSTOM_BASE: u32 = 0x1000;

/// Trait for types that can be embedded as archive sections.
pub trait EmbeddableSection {
    /// Section kind identifier.
    fn section_kind(&self) -> u32;
    /// Serialize this section to bytes.
    fn to_bytes(&self) -> Vec<u8>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_kinds_distinct() {
        let kinds = [
            SECTION_WEIGHT_INDEX,
            SECTION_LAYER_HEADER,
            SECTION_PIPELINE,
            SECTION_WEIGHT_DEDUP,
            SECTION_COMPILE_UNIT_META,
            SECTION_CONFORMANCE_SHAPE,
            SECTION_CUSTOM_BASE,
        ];
        for (i, a) in kinds.iter().enumerate() {
            for (j, b) in kinds.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn custom_base_above_built_in() {
        const { assert!(SECTION_CUSTOM_BASE > SECTION_PIPELINE) };
        const { assert!(SECTION_CUSTOM_BASE > SECTION_CONFORMANCE_SHAPE) };
    }
}
