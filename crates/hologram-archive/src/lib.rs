//! `.holo` archive format with execution entrypoints.
//!
//! Provides a single clean archive format for serializing compiled graphs,
//! weights, and execution metadata. Uses rkyv for zero-copy serialization
//! of variable-length data, bytemuck for fixed-layout header, and supports
//! memory-mapped loading.

pub mod checksum;
pub mod entrypoint;
pub mod error;
pub mod format;
pub mod layer;
pub mod loader;
pub mod section;
pub mod weight;
pub mod writer;

// Re-exports for convenience.
pub use entrypoint::schedule::LayerHeader;
pub use entrypoint::{LayerDescriptor, LayerEntrypoint, LayerId};
pub use error::{ArchiveError, ArchiveResult};
pub use format::header::HoloHeader;
pub use loader::bytes::{
    decompress_archive, is_compressed, load_auto, load_from_bytes, load_from_bytes_unchecked,
    load_from_bytes_zero_copy,
};
pub use loader::plan::LoadedPlan;
pub use weight::dedup::{WeightDedupIndex, WeightRef, WeightStore};
pub use writer::holo_writer::HoloWriter;

#[cfg(feature = "std")]
pub use loader::HoloLoader;

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ir::builder::GraphBuilder;
    use hologram_ir::graph::GraphOp;
    use hologram_ir::Graph;

    #[test]
    fn empty_archive_round_trip() {
        let archive = HoloWriter::new().build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.node_count(), 0);
        assert!(plan.weights().is_empty());
        assert!(plan.header().is_valid_magic());
        assert!(plan.header().is_supported_version());
    }

    #[test]
    fn graph_weights_sections_round_trip() {
        use crate::entrypoint::schedule::LayerHeader;
        use hologram_core::op::LutOp;

        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();

        let weights = vec![42u8; 256];
        let archive = HoloWriter::new()
            .set_graph(&g)
            .set_weights(weights.clone())
            .add_section(&LayerHeader::new())
            .build()
            .unwrap();

        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.node_count(), 3);
        assert_eq!(plan.weights(), &weights);
        assert!(plan
            .sections()
            .find(section::SECTION_LAYER_HEADER)
            .is_some());
    }

    #[test]
    fn pipeline_round_trip() {
        use loader::pipeline::LoadedPipeline;
        use writer::pipeline_writer::PipelineWriter;

        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let model = HoloWriter::new().set_graph(&g).build().unwrap();

        let pipeline = PipelineWriter::new()
            .add_model("encoder", model.clone())
            .add_model("decoder", model)
            .build()
            .unwrap();

        let loaded = LoadedPipeline::from_bytes(&pipeline).unwrap();
        assert_eq!(loaded.model_count(), 2);
        assert!(loaded.model_by_name("encoder").is_some());
        assert!(loaded.model_by_name("decoder").is_some());
        assert_eq!(loaded.model(0).unwrap().node_count(), 1);
    }

    #[test]
    fn header_accessible_from_plan() {
        let archive = HoloWriter::new()
            .set_weights(vec![1, 2, 3])
            .build()
            .unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        let header = plan.header();
        assert_eq!(header.magic, format::HOLO_MAGIC);
        assert_eq!(header.version, format::FORMAT_VERSION);
        assert!(header.weights_size > 0);
        assert_eq!(plan.weights(), &[1, 2, 3]);
    }

    #[test]
    fn mmap_loader_integration() {
        use std::io::Write;

        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Output);
        let archive = HoloWriter::new()
            .set_graph(&g)
            .set_weights(vec![99u8; 32])
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_holo_integration.holo");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }

        let loader = HoloLoader::open(&path).unwrap();
        let plan = loader.load().unwrap();
        assert_eq!(plan.node_count(), 2);
        assert_eq!(plan.weights().len(), 32);

        std::fs::remove_file(&path).ok();
    }
}
