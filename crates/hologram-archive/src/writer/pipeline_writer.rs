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

/// A model in the pipeline — either fully in-memory or with streamed weights.
enum PipelineModel {
    /// Complete sub-archive in memory (small models or non-streaming).
    InMemory(Vec<u8>),
    /// Sub-archive with graph/sections in memory and weights on disk.
    /// The sub-archive bytes contain a valid .holo with an empty weight
    /// region; the real weights are in the `WeightSource`.
    Streaming {
        /// Sub-archive bytes (graph + sections, placeholder weights).
        sub_archive: Vec<u8>,
        /// Real model weights to stream into the sub-archive's weight region.
        weight_source: crate::writer::holo_writer::WeightSource,
    },
}

/// Builder for multi-model pipeline archives.
///
/// Each model is a complete .holo archive. The pipeline wraps them
/// with a `PipelineHeader` section for indexed access. Additional
/// metadata sections (e.g., component roles, connections) can be
/// embedded in the wrapper via [`add_section`](Self::add_section).
pub struct PipelineWriter {
    models: Vec<(String, PipelineModel)>,
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
        self.models
            .push((name.into(), PipelineModel::InMemory(archive)));
        self
    }

    /// Add a named model with streaming weights.
    ///
    /// The `sub_archive` contains the compiled graph and sections with
    /// an empty weight region. The real weights are read from
    /// `weight_source` at build time, never held in memory.
    #[must_use]
    pub fn add_model_streaming(
        mut self,
        name: impl Into<String>,
        sub_archive: Vec<u8>,
        weight_source: crate::writer::holo_writer::WeightSource,
    ) -> Self {
        self.models.push((
            name.into(),
            PipelineModel::Streaming {
                sub_archive,
                weight_source,
            },
        ));
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

    /// Build the pipeline archive in memory.
    ///
    /// Creates a wrapper .holo archive containing a `PipelineHeader`
    /// section and all model sub-archives in the weights section.
    ///
    /// All models must be `InMemory`. For streaming models, use
    /// [`build_to_file`](Self::build_to_file).
    pub fn build(self) -> ArchiveResult<Vec<u8>> {
        if self.models.is_empty() {
            return Err(ArchiveError::GraphError(
                "pipeline must have at least one model".into(),
            ));
        }

        // Concatenate model archives, track offsets
        let mut combined = Vec::new();
        let mut entries = Vec::new();
        for (name, model) in &self.models {
            let data = match model {
                PipelineModel::InMemory(d) => d,
                PipelineModel::Streaming { .. } => {
                    return Err(ArchiveError::GraphError(
                        "streaming models require build_to_file, not build".into(),
                    ));
                }
            };
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

    /// Build the pipeline archive to a file, streaming weights from disk.
    ///
    /// Assembles the pipeline weight region (concatenated sub-archives)
    /// into `scratch_path`, streaming large model weights directly from
    /// their `WeightSource` without loading them into memory.
    /// The caller is responsible for providing (and later cleaning up)
    /// `scratch_path` — typically a temp file.
    pub fn build_to_file(
        self,
        output_path: &std::path::Path,
        scratch_path: &std::path::Path,
    ) -> ArchiveResult<()> {
        use crate::writer::holo_writer::{HoloWriter, WeightSource};
        use std::io::Write;

        if self.models.is_empty() {
            return Err(ArchiveError::GraphError(
                "pipeline must have at least one model".into(),
            ));
        }

        // Build the concatenated sub-archive region in the scratch file.
        let mut file = std::fs::File::create(scratch_path).map_err(ArchiveError::Io)?;
        let mut entries = Vec::new();
        let mut pos: u64 = 0;

        for (name, model) in &self.models {
            let offset = pos;

            match model {
                PipelineModel::InMemory(data) => {
                    let size = data.len() as u64;
                    entries.push(PipelineEntry {
                        name: name.clone(),
                        offset,
                        size,
                        checksum: [0u8; 32],
                    });
                    file.write_all(data).map_err(ArchiveError::Io)?;
                    pos += size;
                }
                PipelineModel::Streaming {
                    sub_archive,
                    weight_source,
                } => {
                    // The sub-archive was built with empty weights.
                    // Patch the header with the real weight size, then
                    // stream the real weights after the prefix.
                    let sub_header = crate::format::header::HoloHeader::from_bytes(sub_archive)
                        .ok_or_else(|| {
                            ArchiveError::ValidationFailed("sub-archive header parse failed".into())
                        })?;

                    let prefix_len = sub_header.weights_offset as usize;
                    let real_weight_len = weight_source.len();

                    // Patch header: correct weights_size, zero checksum.
                    let mut new_header = sub_header;
                    new_header.weights_size = real_weight_len;
                    new_header.weights_checksum = [0u8; 32];
                    let header_bytes = new_header.as_bytes();

                    // Write patched header.
                    file.write_all(header_bytes).map_err(ArchiveError::Io)?;
                    // Write the rest of the prefix (graph, sections, section table).
                    let header_len = header_bytes.len();
                    if prefix_len > header_len {
                        file.write_all(&sub_archive[header_len..prefix_len])
                            .map_err(ArchiveError::Io)?;
                    }

                    // Stream real weights from source.
                    match weight_source {
                        WeightSource::Bytes(v) => {
                            file.write_all(v).map_err(ArchiveError::Io)?;
                        }
                        WeightSource::File { path, len } => {
                            let mut src = std::fs::File::open(path).map_err(ArchiveError::Io)?;
                            let copied =
                                std::io::copy(&mut src, &mut file).map_err(ArchiveError::Io)?;
                            if copied != *len {
                                return Err(ArchiveError::ValidationFailed(format!(
                                    "weight file size mismatch: expected {len}, copied {copied}"
                                )));
                            }
                        }
                    }

                    let total_size = prefix_len as u64 + real_weight_len;
                    entries.push(PipelineEntry {
                        name: name.clone(),
                        offset,
                        size: total_size,
                        checksum: [0u8; 32],
                    });
                    pos += total_size;
                }
            }

            // Page-align between models.
            let aligned = align_to_page(pos);
            if aligned > pos {
                let padding = vec![0u8; (aligned - pos) as usize];
                file.write_all(&padding).map_err(ArchiveError::Io)?;
                pos = aligned;
            }
        }

        file.flush().map_err(ArchiveError::Io)?;
        drop(file);

        // Build the outer wrapper archive using the scratch file as weights.
        let pipeline_header = PipelineHeader { models: entries };
        let mut writer = HoloWriter::new()
            .set_weight_source(WeightSource::File {
                path: scratch_path.to_path_buf(),
                len: pos,
            })
            .add_section(&pipeline_header);

        for (kind, bytes) in self.extra_sections {
            writer = writer.add_raw_section(kind, bytes);
        }

        writer.build_to_file(output_path)
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
        for (name, model) in &self.models {
            let data = match model {
                PipelineModel::InMemory(d) => d,
                PipelineModel::Streaming { .. } => {
                    return Err(ArchiveError::GraphError(
                        "streaming models require build_to_file".into(),
                    ));
                }
            };
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
