//! Model metadata section — declares model kind, architecture, and capabilities.
//!
//! Embedded by the compiler so that `hologram run` can adapt its I/O
//! behavior (e.g. autoregressive generation for LLMs, typed output
//! formatting for vision models, etc.).

use super::{EmbeddableSection, SECTION_CUSTOM_BASE};

/// Section kind for model metadata.
pub const SECTION_MODEL_META: u32 = SECTION_CUSTOM_BASE + 0x02;

/// What kind of model this archive contains.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ModelKind {
    /// Autoregressive text generation (LLaMA, GPT, Mistral).
    TextLlm,
    /// Text embeddings (BERT, sentence-transformers).
    TextEncoder,
    /// Image classification, detection, segmentation (ViT, ResNet, YOLO).
    Vision,
    /// Speech recognition, audio classification (Whisper).
    Audio,
    /// Image generation (diffusion, VAE).
    ImageGen,
    /// Audio/speech synthesis (TTS).
    AudioGen,
    /// Video generation.
    VideoGen,
    /// Combined modalities (LLaVA, CLIP).
    MultiModal,
    /// Unknown or custom model type.
    Generic,
}

/// Model metadata embedded in a `.holo` archive.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ModelMetaSection {
    /// What kind of model this is.
    pub kind: ModelKind,
    /// Architecture identifier (e.g. "llama", "whisper", "vit").
    pub arch: String,
    /// Human-readable description.
    pub description: String,
    /// Maximum sequence length / context window (0 if not applicable).
    pub max_seq_len: u32,
    /// Whether `--prompt` autoregressive generation is supported.
    pub supports_prompt: bool,
    /// Number of transformer layers (0 if not applicable / no KV cache).
    pub n_layers: u32,
    /// Number of KV attention heads (0 if not applicable).
    pub n_kv_heads: u32,
    /// Dimension per attention head (0 if not applicable).
    pub head_dim: u32,
}

impl ModelMetaSection {
    /// Zero-copy access from raw section bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedModelMetaSection, rkyv::rancor::Error> {
        rkyv::access::<ArchivedModelMetaSection, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw section bytes into an owned value.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl EmbeddableSection for ModelMetaSection {
    fn section_kind(&self) -> u32 {
        SECTION_MODEL_META
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("ModelMetaSection serialization should not fail")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rkyv_roundtrip() {
        let section = ModelMetaSection {
            kind: ModelKind::TextLlm,
            arch: "llama".into(),
            description: "TinyLlama 1.1B".into(),
            max_seq_len: 2048,
            supports_prompt: true,
            n_layers: 22,
            n_kv_heads: 4,
            head_dim: 64,
        };
        let bytes = section.to_bytes();
        let deserialized = ModelMetaSection::deserialize_from(&bytes).unwrap();
        assert_eq!(deserialized.kind, ModelKind::TextLlm);
        assert_eq!(deserialized.arch, "llama");
        assert_eq!(deserialized.max_seq_len, 2048);
        assert!(deserialized.supports_prompt);
    }

    #[test]
    fn zero_copy_access() {
        let section = ModelMetaSection {
            kind: ModelKind::Vision,
            arch: "vit".into(),
            description: "ViT-B/16".into(),
            max_seq_len: 0,
            supports_prompt: false,
            n_layers: 0,
            n_kv_heads: 0,
            head_dim: 0,
        };
        let bytes = section.to_bytes();
        let archived = ModelMetaSection::from_bytes(&bytes).unwrap();
        assert_eq!(archived.arch.as_str(), "vit");
        assert!(!archived.supports_prompt);
    }

    #[test]
    fn section_kind_correct() {
        let section = ModelMetaSection {
            kind: ModelKind::Generic,
            arch: String::new(),
            description: String::new(),
            max_seq_len: 0,
            supports_prompt: false,
            n_layers: 0,
            n_kv_heads: 0,
            head_dim: 0,
        };
        assert_eq!(section.section_kind(), SECTION_MODEL_META);
        assert_eq!(SECTION_MODEL_META, 0x1002);
    }

    #[test]
    fn all_model_kinds() {
        let kinds = [
            ModelKind::TextLlm,
            ModelKind::TextEncoder,
            ModelKind::Vision,
            ModelKind::Audio,
            ModelKind::ImageGen,
            ModelKind::AudioGen,
            ModelKind::VideoGen,
            ModelKind::MultiModal,
            ModelKind::Generic,
        ];
        for kind in kinds {
            let section = ModelMetaSection {
                kind: kind.clone(),
                arch: String::new(),
                description: String::new(),
                max_seq_len: 0,
                supports_prompt: false,
                n_layers: 0,
                n_kv_heads: 0,
                head_dim: 0,
            };
            let bytes = section.to_bytes();
            let de = ModelMetaSection::deserialize_from(&bytes).unwrap();
            assert_eq!(de.kind, kind);
        }
    }
}
