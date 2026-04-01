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
    /// KV cache K bit-width (0=F32, 1=Q8, 2=Q4). Default: 0.
    pub kv_k_bits: u8,
    /// KV cache V bit-width (0=F32, 1=Q8, 2=Q4). Default: 0.
    pub kv_v_bits: u8,
    /// Number of boundary layers kept at f32 for KV. Default: 2.
    pub kv_boundary_layers: u8,
    /// Whether Walsh-Hadamard rotation is applied to V before quantization.
    pub kv_wht: bool,
}

impl ModelMetaSection {
    /// Zero-copy access from raw section bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedModelMetaSection, rkyv::rancor::Error> {
        rkyv::access::<ArchivedModelMetaSection, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw section bytes into an owned value.
    ///
    /// Backward compatible: if deserialization fails (e.g., archive compiled
    /// before KV config fields were added), tries the legacy format and fills
    /// new fields with defaults.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        // Try current format first.
        if let Ok(v) = rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes) {
            return Ok(v);
        }
        // Fallback: try legacy format (without kv_* fields).
        let legacy = rkyv::from_bytes::<ModelMetaSectionLegacy, rkyv::rancor::Error>(bytes)?;
        Ok(Self {
            kind: legacy.kind,
            arch: legacy.arch,
            description: legacy.description,
            max_seq_len: legacy.max_seq_len,
            supports_prompt: legacy.supports_prompt,
            n_layers: legacy.n_layers,
            n_kv_heads: legacy.n_kv_heads,
            head_dim: legacy.head_dim,
            kv_k_bits: 0,
            kv_v_bits: 0,
            kv_boundary_layers: 2,
            kv_wht: false,
        })
    }
}

/// Legacy format for backward-compatible deserialization of archives compiled
/// before the KV cache config fields were added.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ModelMetaSectionLegacy {
    pub kind: ModelKind,
    pub arch: String,
    pub description: String,
    pub max_seq_len: u32,
    pub supports_prompt: bool,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
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
            kv_k_bits: 0,
            kv_v_bits: 0,
            kv_boundary_layers: 2,
            kv_wht: false,
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
            kv_k_bits: 0,
            kv_v_bits: 0,
            kv_boundary_layers: 2,
            kv_wht: false,
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
            kv_k_bits: 0,
            kv_v_bits: 0,
            kv_boundary_layers: 2,
            kv_wht: false,
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
                kv_k_bits: 0,
                kv_v_bits: 0,
                kv_boundary_layers: 2,
                kv_wht: false,
            };
            let bytes = section.to_bytes();
            let de = ModelMetaSection::deserialize_from(&bytes).unwrap();
            assert_eq!(de.kind, kind);
        }
    }

    #[test]
    fn kv_config_roundtrip() {
        let section = ModelMetaSection {
            kind: ModelKind::TextLlm,
            arch: "llama".into(),
            description: "test".into(),
            max_seq_len: 2048,
            supports_prompt: true,
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 64,
            kv_k_bits: 1, // Q8
            kv_v_bits: 2, // Q4
            kv_boundary_layers: 2,
            kv_wht: true,
        };
        let bytes = section.to_bytes();
        let de = ModelMetaSection::deserialize_from(&bytes).unwrap();
        assert_eq!(de.kv_k_bits, 1);
        assert_eq!(de.kv_v_bits, 2);
        assert_eq!(de.kv_boundary_layers, 2);
        assert!(de.kv_wht);
    }
}
