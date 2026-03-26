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
    /// BLAKE3 hash of the sub-archive bytes.
    pub checksum: [u8; 32],
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
/// with a `PipelineHeader` section for indexed access. Additional
/// metadata sections (e.g., component roles, connections) can be
/// embedded in the wrapper via [`add_section`](Self::add_section).
pub struct PipelineWriter {
    models: Vec<(String, Vec<u8>)>,
    extra_sections: Vec<(u32, Vec<u8>)>,
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
        Self {
            models: Vec::new(),
            extra_sections: Vec::new(),
        }
    }

    /// Add a named model (as a complete .holo archive).
    #[must_use]
    pub fn add_model(mut self, name: impl Into<String>, archive: Vec<u8>) -> Self {
        self.models.push((name.into(), archive));
        self
    }

    /// Add a raw section to the pipeline wrapper archive.
    ///
    /// These sections are embedded in the outer `.holo` wrapper alongside
    /// the `PipelineHeader`. Use this for pipeline-level metadata like
    /// component roles, weight deduplication indices, etc.
    #[must_use]
    pub fn add_section(mut self, kind: u32, bytes: Vec<u8>) -> Self {
        self.extra_sections.push((kind, bytes));
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
            let cksum = checksum::checksum(data);
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

        // Build wrapper archive with pipeline header + extra sections
        use crate::writer::holo_writer::HoloWriter;
        let mut writer = HoloWriter::new()
            .set_weights(combined)
            .add_section(&pipeline_header);

        for (kind, bytes) in self.extra_sections {
            writer = writer.add_raw_section(kind, bytes);
        }

        writer.build()
    }

    /// Build a pipeline archive with shared (deduplicated) weights.
    ///
    /// Sub-archives contain only graph + sections (no embedded weights).
    /// All weights are stored once in the shared blob, referenced via
    /// `WeightDedupIndex`. This halves the archive size and enables
    /// zero-copy mmap loading.
    ///
    /// Layout in the wrapper's weight region:
    /// ```text
    /// [sub-archive 0 (graph only)] [pad] [sub-archive 1 (graph only)] [pad] [shared weights blob]
    /// ```
    pub fn build_with_shared_weights(
        self,
        shared_weights: Vec<u8>,
        dedup_index: &crate::weight::dedup::WeightDedupIndex,
    ) -> ArchiveResult<Vec<u8>> {
        if self.models.is_empty() {
            return Err(ArchiveError::GraphError(
                "pipeline must have at least one model".into(),
            ));
        }

        // Concatenate graph-only sub-archives, then append shared weights.
        let mut combined = Vec::new();
        let mut entries = Vec::new();
        for (name, data) in &self.models {
            let offset = combined.len() as u64;
            let size = data.len() as u64;
            let cksum = checksum::checksum(data);
            entries.push(PipelineEntry {
                name: name.clone(),
                offset,
                size,
                checksum: cksum,
            });
            combined.extend_from_slice(data);
            let aligned = align_to_page(combined.len() as u64) as usize;
            combined.resize(aligned, 0);
        }

        // Append the shared weight blob (page-aligned).
        let aligned = align_to_page(combined.len() as u64) as usize;
        combined.resize(aligned, 0);
        combined.extend_from_slice(&shared_weights);

        let pipeline_header = PipelineHeader { models: entries };

        use crate::section::SECTION_WEIGHT_DEDUP;
        use crate::writer::holo_writer::HoloWriter;

        let dedup_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(dedup_index)
            .map_err(|e| ArchiveError::GraphError(format!("dedup index serialization: {e}")))?
            .to_vec();

        let mut writer = HoloWriter::new()
            .set_weights(combined)
            .add_section(&pipeline_header)
            .add_raw_section(SECTION_WEIGHT_DEDUP, dedup_bytes);

        for (kind, bytes) in self.extra_sections {
            writer = writer.add_raw_section(kind, bytes);
        }

        writer.build()
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
                checksum: [0xAB; 32],
            }],
        };
        assert_eq!(ph.section_kind(), SECTION_PIPELINE);
        assert!(!ph.to_bytes().is_empty());
    }
}
