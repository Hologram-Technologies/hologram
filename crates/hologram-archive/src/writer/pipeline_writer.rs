//! Pipeline writer for multi-model archives.

use crate::checksum;
use crate::error::{ArchiveError, ArchiveResult};
use crate::format::align_to_page;
use crate::section::{EmbeddableSection, SECTION_PIPELINE};

/// Entry for a model within a pipeline archive.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct PipelineEntry {
    /// Model name.
    pub name: String,
    /// Byte offset of this model's sub-archive.
    pub offset: u64,
    /// Byte size of this model's sub-archive.
    pub size: u64,
    /// CRC32 of the sub-archive bytes.
    pub checksum: u32,
}

/// Header for multi-model pipeline archives.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct PipelineHeader {
    /// Models in the pipeline.
    pub models: Vec<PipelineEntry>,
}

impl EmbeddableSection for PipelineHeader {
    fn section_kind(&self) -> u32 {
        SECTION_PIPELINE
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("pipeline header serialization")
            .to_vec()
    }
}

/// Builder for multi-model pipeline archives.
///
/// Each model is a complete .holo archive. The pipeline wraps them
/// with a `PipelineHeader` section for indexed access.
pub struct PipelineWriter {
    models: Vec<(String, Vec<u8>)>,
}

impl Default for PipelineWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineWriter {
    /// Create a new empty pipeline writer.
    #[must_use]
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Add a named model (as a complete .holo archive).
    #[must_use]
    pub fn add_model(mut self, name: impl Into<String>, archive: Vec<u8>) -> Self {
        self.models.push((name.into(), archive));
        self
    }

    /// Build the pipeline archive.
    ///
    /// Creates a wrapper .holo archive containing a `PipelineHeader`
    /// section and all model sub-archives in the weights section.
    pub fn build(self) -> ArchiveResult<Vec<u8>> {
        if self.models.is_empty() {
            return Err(ArchiveError::GraphError(
                "pipeline must have at least one model".into(),
            ));
        }

        // Concatenate model archives, track offsets
        let mut combined = Vec::new();
        let mut entries = Vec::new();
        for (name, data) in &self.models {
            let offset = combined.len() as u64;
            let size = data.len() as u64;
            let cksum = checksum::crc32(data);
            entries.push(PipelineEntry {
                name: name.clone(),
                offset,
                size,
                checksum: cksum,
            });
            combined.extend_from_slice(data);
            // Page-align between models
            let aligned = align_to_page(combined.len() as u64) as usize;
            combined.resize(aligned, 0);
        }

        let pipeline_header = PipelineHeader { models: entries };

        // Build wrapper archive with pipeline header as section
        use crate::writer::holo_writer::HoloWriter;
        HoloWriter::new()
            .set_weights(combined)
            .add_section(&pipeline_header)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::holo_writer::HoloWriter;
    use hologram_graph::graph::GraphOp;
    use hologram_graph::Graph;

    fn make_simple_archive() -> Vec<u8> {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        HoloWriter::new().set_graph(&g).build().unwrap()
    }

    #[test]
    fn pipeline_empty_errors() {
        let result = PipelineWriter::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_single_model() {
        let archive = make_simple_archive();
        let pipeline = PipelineWriter::new()
            .add_model("model_a", archive)
            .build()
            .unwrap();
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn pipeline_two_models() {
        let a1 = make_simple_archive();
        let a2 = make_simple_archive();
        let pipeline = PipelineWriter::new()
            .add_model("encoder", a1)
            .add_model("decoder", a2)
            .build()
            .unwrap();
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn pipeline_header_embeddable() {
        let ph = PipelineHeader {
            models: vec![PipelineEntry {
                name: "test".into(),
                offset: 0,
                size: 100,
                checksum: 0xABCD,
            }],
        };
        assert_eq!(ph.section_kind(), SECTION_PIPELINE);
        assert!(!ph.to_bytes().is_empty());
    }
}
