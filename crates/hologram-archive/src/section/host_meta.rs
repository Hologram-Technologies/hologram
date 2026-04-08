//! Host-facing metadata section — prompt/chat templates, sampling defaults,
//! port names, and model card.
//!
//! Sibling to `ModelMetaSection`. Where `ModelMetaSection` holds
//! execution-shape metadata consumed by runtime kernels (KV dims, layer
//! counts, arch name), `HostMetaSection` holds annotation-level metadata
//! consumed by *applications* calling into the archive. Separating the
//! two keeps kernel metadata frozen at compile time while allowing
//! host-facing fields to evolve independently.
//!
//! All fields are optional. A compile that supplies none of them simply
//! omits the section; a reader that does not know about this section
//! kind skips it.

use super::{EmbeddableSection, SECTION_CUSTOM_BASE};

/// Section kind for host-facing metadata.
pub const SECTION_HOST_META: u32 = SECTION_CUSTOM_BASE + 0x03;

/// Current on-disk format version. Bump on any field addition (which must
/// be append-only, per the enum-append-only discipline).
pub const HOST_META_VERSION: u8 = 1;

/// Sampling defaults shipped alongside a model archive.
///
/// All fields are optional; hosts should treat `None` as "fall back to
/// whatever default the host itself provides". These are intentionally
/// minimal — no beam search, no mirostat, no grammar constraints.
#[derive(Debug, Clone, Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop: Vec<String>,
}

/// Descriptive model card metadata. Not used for execution — shown by
/// `holo info` and surfaced to hosts that want to display provenance.
#[derive(Debug, Clone, Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ModelCard {
    pub author: Option<String>,
    pub license: Option<String>,
    pub source_url: Option<String>,
    pub tags: Vec<String>,
}

/// A single port name mapping: logical name → graph port identifier.
///
/// Stored as a `Vec<PortBinding>` instead of a `HashMap` because rkyv
/// zero-copy access over maps is clumsier than over vecs, and port
/// tables are tiny (typically < 10 entries).
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct PortBinding {
    pub logical_name: String,
    pub graph_port: String,
}

/// Host-facing metadata embedded in a `.holo` archive.
///
/// All fields are optional / empty-by-default so that a reader can
/// treat a missing section and an empty-but-present section identically.
#[derive(Debug, Clone, Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct HostMetaSection {
    /// Format version. Start at `HOST_META_VERSION = 1`.
    pub version: u8,
    /// Simple prompt template (single-turn), e.g. `"<|user|>{prompt}<|assistant|>"`.
    pub prompt_template: Option<String>,
    /// Jinja-style chat template (multi-turn). Often auto-populated from
    /// GGUF v3 `tokenizer.chat_template` at import time.
    pub chat_template: Option<String>,
    /// Sampling defaults. `None` means "host decides".
    pub sampling: Option<SamplingDefaults>,
    /// Logical → graph port bindings. Empty vec means "use positional order".
    pub ports: Vec<PortBinding>,
    /// Provenance / model card.
    pub model_card: Option<ModelCard>,
}

impl HostMetaSection {
    /// Construct an empty section at the current version.
    pub fn new() -> Self {
        Self {
            version: HOST_META_VERSION,
            ..Self::default()
        }
    }

    /// Returns true iff every annotation field is unset. Callers can use
    /// this to decide whether to bother embedding the section at all.
    pub fn is_empty(&self) -> bool {
        self.prompt_template.is_none()
            && self.chat_template.is_none()
            && self.sampling.is_none()
            && self.ports.is_empty()
            && self.model_card.is_none()
    }

    /// Zero-copy access from raw section bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedHostMetaSection, rkyv::rancor::Error> {
        rkyv::access::<ArchivedHostMetaSection, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw section bytes into an owned value.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl EmbeddableSection for HostMetaSection {
    fn section_kind(&self) -> u32 {
        SECTION_HOST_META
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("HostMetaSection serialization should not fail")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fully_populated() -> HostMetaSection {
        HostMetaSection {
            version: HOST_META_VERSION,
            prompt_template: Some("<|user|>{prompt}<|assistant|>".into()),
            chat_template: Some("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
            sampling: Some(SamplingDefaults {
                temperature: Some(0.7),
                top_k: Some(40),
                top_p: Some(0.95),
                repetition_penalty: Some(1.3),
                stop: vec!["</s>".into(), "<|end|>".into()],
            }),
            ports: vec![
                PortBinding {
                    logical_name: "logits".into(),
                    graph_port: "output_0".into(),
                },
                PortBinding {
                    logical_name: "hidden".into(),
                    graph_port: "output_1".into(),
                },
            ],
            model_card: Some(ModelCard {
                author: Some("TinyLlama Team".into()),
                license: Some("Apache-2.0".into()),
                source_url: Some(
                    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
                ),
                tags: vec!["llm".into(), "chat".into()],
            }),
        }
    }

    #[test]
    fn new_is_empty() {
        let s = HostMetaSection::new();
        assert!(s.is_empty());
        assert_eq!(s.version, HOST_META_VERSION);
    }

    #[test]
    fn rkyv_roundtrip_full() {
        let section = fully_populated();
        let bytes = section.to_bytes();
        let de = HostMetaSection::deserialize_from(&bytes).expect("deserialize");
        assert_eq!(de.version, HOST_META_VERSION);
        assert_eq!(
            de.prompt_template.as_deref(),
            Some("<|user|>{prompt}<|assistant|>")
        );
        let sampling = de.sampling.expect("sampling populated");
        assert_eq!(sampling.temperature, Some(0.7));
        assert_eq!(sampling.top_k, Some(40));
        assert_eq!(sampling.stop.len(), 2);
        assert_eq!(de.ports.len(), 2);
        assert_eq!(de.ports[0].logical_name, "logits");
        let card = de.model_card.expect("card populated");
        assert_eq!(card.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(card.tags, vec!["llm".to_string(), "chat".to_string()]);
    }

    #[test]
    fn rkyv_roundtrip_empty() {
        let section = HostMetaSection::new();
        let bytes = section.to_bytes();
        let de = HostMetaSection::deserialize_from(&bytes).expect("deserialize empty");
        assert!(de.is_empty());
    }

    #[test]
    fn zero_copy_access() {
        let section = fully_populated();
        let bytes = section.to_bytes();
        let archived = HostMetaSection::from_bytes(&bytes).expect("access");
        assert_eq!(archived.version, HOST_META_VERSION);
        assert_eq!(
            archived.prompt_template.as_ref().map(|s| s.as_str()),
            Some("<|user|>{prompt}<|assistant|>"),
        );
        assert_eq!(archived.ports.len(), 2);
    }

    #[test]
    fn section_kind_correct() {
        let section = HostMetaSection::new();
        assert_eq!(section.section_kind(), SECTION_HOST_META);
        assert_eq!(SECTION_HOST_META, 0x1003);
    }

    #[test]
    fn is_empty_detects_any_field() {
        let mut s = HostMetaSection::new();
        assert!(s.is_empty());

        s.prompt_template = Some("x".into());
        assert!(!s.is_empty());
        s.prompt_template = None;

        s.chat_template = Some("y".into());
        assert!(!s.is_empty());
        s.chat_template = None;

        s.sampling = Some(SamplingDefaults::default());
        assert!(!s.is_empty());
        s.sampling = None;

        s.ports.push(PortBinding {
            logical_name: "a".into(),
            graph_port: "b".into(),
        });
        assert!(!s.is_empty());
        s.ports.clear();

        s.model_card = Some(ModelCard::default());
        assert!(!s.is_empty());
    }
}
