//! Pipeline loader: access individual models from a pipeline archive.

use crate::error::{ArchiveError, ArchiveResult};
use crate::loader::bytes::load_from_bytes;
use crate::loader::plan::LoadedPlan;
use crate::section::{SECTION_PIPELINE, SECTION_WEIGHT_DEDUP};
use crate::weight::dedup::WeightDedupIndex;
use crate::writer::pipeline_writer::PipelineHeader;

/// Loaded pipeline with access to individual models.
pub struct LoadedPipeline {
    header: PipelineHeader,
    models: Vec<(String, LoadedPlan)>,
}

impl LoadedPipeline {
    /// Load a pipeline archive from bytes.
    ///
    /// First loads the wrapper archive, extracts the pipeline header
    /// section, then loads each sub-archive from the weights region.
    pub fn from_bytes(data: &[u8]) -> ArchiveResult<Self> {
        let wrapper = load_from_bytes(data)?;

        let pipeline_entry = wrapper
            .sections()
            .find(SECTION_PIPELINE)
            .ok_or(ArchiveError::SectionNotFound(SECTION_PIPELINE))?;

        let section_start = pipeline_entry.offset as usize;
        let section_end = section_start + pipeline_entry.size as usize;
        if section_end > data.len() {
            return Err(ArchiveError::OutOfBounds {
                offset: pipeline_entry.offset,
                size: pipeline_entry.size,
            });
        }

        let section_bytes = &data[section_start..section_end];
        let ph: PipelineHeader =
            rkyv::from_bytes::<PipelineHeader, rkyv::rancor::Error>(section_bytes)
                .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))?;

        // Parse optional weight dedup index from wrapper sections.
        let dedup_index = Self::parse_dedup_index(data, &wrapper);

        // Each model is a sub-archive within the wrapper's weights
        let weights = wrapper.weights();
        let mut models = Vec::new();
        for entry in &ph.models {
            let start = entry.offset as usize;
            let end = start + entry.size as usize;
            if end > weights.len() {
                return Err(ArchiveError::OutOfBounds {
                    offset: entry.offset,
                    size: entry.size,
                });
            }
            let mut model_plan = load_from_bytes(&weights[start..end])?;

            // Resolve deduplicated weights: if the sub-archive has empty
            // weights and the dedup index has an entry for this component,
            // graft the shared weights onto the loaded plan.
            if model_plan.weights().is_empty() {
                if let Some(ref idx) = dedup_index {
                    if let Some(dedup_entry) = idx.find_component(&entry.name) {
                        let w_start = dedup_entry.offset as usize;
                        let w_end = w_start + dedup_entry.size as usize;
                        if w_end <= weights.len() {
                            model_plan.set_weights(weights[w_start..w_end].to_vec());
                        }
                    }
                }
            }

            models.push((entry.name.clone(), model_plan));
        }

        Ok(Self { header: ph, models })
    }

    /// Number of models in the pipeline.
    #[must_use]
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get a model by index.
    #[must_use]
    pub fn model(&self, index: usize) -> Option<&LoadedPlan> {
        self.models.get(index).map(|(_, p)| p)
    }

    /// Get a model by name.
    #[must_use]
    pub fn model_by_name(&self, name: &str) -> Option<&LoadedPlan> {
        self.models.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }

    /// Pipeline header.
    #[must_use]
    pub fn header(&self) -> &PipelineHeader {
        &self.header
    }

    /// Consume the pipeline and return the first model's [`LoadedPlan`].
    ///
    /// Useful for single-component pipelines where callers want to treat
    /// the archive as a flat model without pipeline awareness.
    #[must_use]
    pub fn into_first_model(mut self) -> Option<LoadedPlan> {
        if self.models.is_empty() {
            None
        } else {
            Some(self.models.remove(0).1)
        }
    }

    /// Load a pipeline archive with zero-copy weight access.
    ///
    /// Sub-archive graphs are deserialized (unavoidable), but weights are
    /// borrowed directly from the input slice (mmap). No weight data is
    /// copied at any point.
    ///
    /// # Safety
    /// The caller must ensure `data` outlives the returned `LoadedPipeline`.
    pub unsafe fn from_bytes_zero_copy(data: &[u8]) -> ArchiveResult<Self> {
        use crate::loader::bytes::{load_from_bytes_unchecked, load_from_bytes_zero_copy};

        let wrapper = load_from_bytes_zero_copy(data)?;

        let pipeline_entry = wrapper
            .sections()
            .find(SECTION_PIPELINE)
            .ok_or(ArchiveError::SectionNotFound(SECTION_PIPELINE))?;

        let section_start = pipeline_entry.offset as usize;
        let section_end = section_start + pipeline_entry.size as usize;
        if section_end > data.len() {
            return Err(ArchiveError::OutOfBounds {
                offset: pipeline_entry.offset,
                size: pipeline_entry.size,
            });
        }

        let section_bytes = &data[section_start..section_end];
        let ph: PipelineHeader =
            rkyv::from_bytes::<PipelineHeader, rkyv::rancor::Error>(section_bytes)
                .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))?;

        let dedup_index = Self::parse_dedup_index(data, &wrapper);
        let weights = wrapper.weights();

        let mut models = Vec::new();
        for entry in &ph.models {
            let start = entry.offset as usize;
            let end = start + entry.size as usize;
            if end > weights.len() {
                return Err(ArchiveError::OutOfBounds {
                    offset: entry.offset,
                    size: entry.size,
                });
            }

            // Use unchecked (skips BLAKE3) for sub-archive graph deserialization.
            // Sub-archive weights may be empty (shared via dedup index).
            let mut model_plan = load_from_bytes_unchecked(&weights[start..end])?;

            // Zero-copy weight resolution: borrow from the wrapper's mmap.
            if model_plan.weights().is_empty() {
                if let Some(ref idx) = dedup_index {
                    if let Some(dedup_entry) = idx.find_component(&entry.name) {
                        let w_start = dedup_entry.offset as usize;
                        let w_end = w_start + dedup_entry.size as usize;
                        if w_end <= weights.len() {
                            // Borrow shared weights directly from mmap — no copy.
                            model_plan.set_weights_borrowed(&weights[w_start..w_end]);
                        }
                    }
                }
            }

            models.push((entry.name.clone(), model_plan));
        }

        Ok(Self { header: ph, models })
    }

    /// Parse the `WeightDedupIndex` from the wrapper archive, if present.
    fn parse_dedup_index(data: &[u8], wrapper: &LoadedPlan) -> Option<WeightDedupIndex> {
        let entry = wrapper.sections().find(SECTION_WEIGHT_DEDUP)?;
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end > data.len() {
            return None;
        }
        rkyv::from_bytes::<WeightDedupIndex, rkyv::rancor::Error>(&data[start..end]).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::holo_writer::HoloWriter;
    use crate::writer::pipeline_writer::PipelineWriter;
    use hologram_graph::graph::GraphOp;
    use hologram_graph::Graph;

    fn make_archive() -> Vec<u8> {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        HoloWriter::new().set_graph(&g).build().unwrap()
    }

    #[test]
    fn pipeline_round_trip() {
        let pipeline = PipelineWriter::new()
            .add_model("encoder", make_archive())
            .add_model("decoder", make_archive())
            .build()
            .unwrap();

        let loaded = LoadedPipeline::from_bytes(&pipeline).unwrap();
        assert_eq!(loaded.model_count(), 2);
    }

    #[test]
    fn model_by_name() {
        let pipeline = PipelineWriter::new()
            .add_model("alpha", make_archive())
            .add_model("beta", make_archive())
            .build()
            .unwrap();

        let loaded = LoadedPipeline::from_bytes(&pipeline).unwrap();
        assert!(loaded.model_by_name("alpha").is_some());
        assert!(loaded.model_by_name("beta").is_some());
        assert!(loaded.model_by_name("gamma").is_none());
    }

    #[test]
    fn model_by_index() {
        let pipeline = PipelineWriter::new()
            .add_model("only", make_archive())
            .build()
            .unwrap();

        let loaded = LoadedPipeline::from_bytes(&pipeline).unwrap();
        assert!(loaded.model(0).is_some());
        assert!(loaded.model(1).is_none());
    }

    /// Build a graph-only sub-archive (no weights).
    fn make_graph_only_archive() -> Vec<u8> {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Output);
        HoloWriter::new().set_graph(&g).build().unwrap()
    }

    #[test]
    fn shared_weights_round_trip() {
        use crate::weight::dedup::WeightStore;

        // Build shared weights via WeightStore.
        let mut store = WeightStore::new();
        store.insert("lm.prefill", "lm", &[1u8, 2, 3, 4]);
        store.insert("lm.decode", "lm", &[5u8, 6, 7, 8]);
        let (shared_blob, dedup_index) = store.build();

        // Build pipeline with shared weights.
        let pipeline_bytes = PipelineWriter::new()
            .add_model("lm.prefill", make_graph_only_archive())
            .add_model("lm.decode", make_graph_only_archive())
            .build_with_shared_weights(shared_blob.clone(), &dedup_index)
            .unwrap();

        // Load and verify both models get weights from the shared blob.
        let loaded = LoadedPipeline::from_bytes(&pipeline_bytes).unwrap();
        assert_eq!(loaded.model_count(), 2);

        // Sub-archives had empty weights, so they should be grafted from dedup.
        let prefill = loaded.model_by_name("lm.prefill").unwrap();
        let decode = loaded.model_by_name("lm.decode").unwrap();

        // Both share the same weight group "lm", so they get the same blob.
        assert!(!prefill.weights().is_empty());
        assert!(!decode.weights().is_empty());
    }

    #[test]
    fn shared_weights_zero_copy() {
        use crate::weight::dedup::WeightStore;

        let mut store = WeightStore::new();
        store.insert("lm.prefill", "lm", &[10u8; 64]);
        store.insert("lm.decode", "lm", &[20u8; 64]);
        let (shared_blob, dedup_index) = store.build();

        let pipeline_bytes = PipelineWriter::new()
            .add_model("lm.prefill", make_graph_only_archive())
            .add_model("lm.decode", make_graph_only_archive())
            .build_with_shared_weights(shared_blob, &dedup_index)
            .unwrap();

        // Zero-copy load.
        // SAFETY: pipeline_bytes outlives loaded.
        let loaded = unsafe { LoadedPipeline::from_bytes_zero_copy(&pipeline_bytes) }.unwrap();
        assert_eq!(loaded.model_count(), 2);

        let prefill = loaded.model_by_name("lm.prefill").unwrap();
        let decode = loaded.model_by_name("lm.decode").unwrap();
        assert!(!prefill.weights().is_empty());
        assert!(!decode.weights().is_empty());
    }
}
