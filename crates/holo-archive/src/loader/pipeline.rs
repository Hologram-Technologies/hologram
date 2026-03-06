//! Pipeline loader: access individual models from a pipeline archive.

use rkyv::Deserialize;

use crate::error::{ArchiveError, ArchiveResult};
use crate::loader::bytes::load_from_bytes;
use crate::loader::plan::LoadedPlan;
use crate::section::SECTION_PIPELINE;
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
        let archived =
            rkyv::check_archived_root::<PipelineHeader>(section_bytes)
                .map_err(|e| {
                    ArchiveError::ValidationFailed(format!("{e}"))
                })?;
        let ph: PipelineHeader = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|e| {
                ArchiveError::ValidationFailed(format!("{e:?}"))
            })?;

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
            let model_plan = load_from_bytes(&weights[start..end])?;
            models.push((entry.name.clone(), model_plan));
        }

        Ok(Self {
            header: ph,
            models,
        })
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
        self.models
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
    }

    /// Pipeline header.
    #[must_use]
    pub fn header(&self) -> &PipelineHeader {
        &self.header
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::holo_writer::HoloWriter;
    use crate::writer::pipeline_writer::PipelineWriter;
    use holo_graph::graph::GraphOp;
    use holo_graph::Graph;

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
}
