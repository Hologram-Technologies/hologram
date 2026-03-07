//! Load a .holo archive from a byte slice (WASM/embedded compatible).

use crate::checksum;
use crate::entrypoint::schedule::LayerHeader;
use crate::error::{ArchiveError, ArchiveResult};
use crate::format::graph::SerializedGraph;
use crate::format::header::{HoloHeader, HEADER_SIZE};
use crate::loader::plan::LoadedPlan;
use crate::section::table::SectionTable;
use crate::section::SECTION_LAYER_HEADER;

/// Validate the archive header bytes.
///
/// Reads the fixed-layout header via bytemuck, then checks magic and version.
pub fn validate_header(data: &[u8]) -> ArchiveResult<HoloHeader> {
    let header = HoloHeader::from_bytes(data).ok_or_else(|| {
        ArchiveError::ValidationFailed(format!(
            "header too short: need {HEADER_SIZE} bytes, got {}",
            data.len()
        ))
    })?;

    if !header.is_valid_magic() {
        return Err(ArchiveError::InvalidMagic);
    }
    if !header.is_supported_version() {
        return Err(ArchiveError::UnsupportedVersion(header.version));
    }
    Ok(header)
}

/// Load a .holo archive from a byte slice.
///
/// Validates magic, version, and checksums, then deserializes the
/// graph and extracts weight data.
pub fn load_from_bytes(data: &[u8]) -> ArchiveResult<LoadedPlan> {
    // Find header: it's the rkyv-serialized HoloHeader at offset 0.
    // We need to figure out where the header ends. The header is
    // rkyv-serialized, so we try to parse it from the start of data.
    // The graph_offset tells us where the header region ends.
    let header = find_and_validate_header(data)?;

    let graph = deserialize_graph(data, &header)?;
    let weights = extract_weights(data, &header)?;
    let section_table = deserialize_section_table(data, &header)?;
    let layer_header = extract_layer_header(data, &section_table)?;

    verify_checksums(data, &header)?;

    Ok(LoadedPlan::new(
        header,
        graph,
        weights,
        section_table,
        layer_header,
    ))
}

fn find_and_validate_header(data: &[u8]) -> ArchiveResult<HoloHeader> {
    // The header is a fixed-layout struct at offset 0 (HEADER_SIZE bytes).
    validate_header(data)
}

fn deserialize_graph(data: &[u8], header: &HoloHeader) -> ArchiveResult<SerializedGraph> {
    let start = header.graph_offset as usize;
    let end = start + header.graph_size as usize;
    if end > data.len() {
        return Err(ArchiveError::OutOfBounds {
            offset: header.graph_offset,
            size: header.graph_size,
        });
    }
    if header.graph_size == 0 {
        return Ok(SerializedGraph {
            nodes: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            output_node_ids: Vec::new(),
            constants: hologram_graph::constant::ConstantStore::new(),
        });
    }
    let graph_bytes = &data[start..end];
    rkyv::from_bytes::<SerializedGraph, rkyv::rancor::Error>(graph_bytes)
        .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))
}

fn extract_weights(data: &[u8], header: &HoloHeader) -> ArchiveResult<Vec<u8>> {
    if header.weights_size == 0 {
        return Ok(Vec::new());
    }
    let start = header.weights_offset as usize;
    let end = start + header.weights_size as usize;
    if end > data.len() {
        return Err(ArchiveError::OutOfBounds {
            offset: header.weights_offset,
            size: header.weights_size,
        });
    }
    Ok(data[start..end].to_vec())
}

fn deserialize_section_table(data: &[u8], header: &HoloHeader) -> ArchiveResult<SectionTable> {
    if header.section_count == 0 {
        return Ok(SectionTable::new());
    }
    let start = header.section_table_offset as usize;
    let end = start + header.section_table_size as usize;
    if end > data.len() {
        return Err(ArchiveError::OutOfBounds {
            offset: header.section_table_offset,
            size: header.section_table_size,
        });
    }
    let table_bytes = &data[start..end];
    rkyv::from_bytes::<SectionTable, rkyv::rancor::Error>(table_bytes)
        .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))
}

/// Extract the `LayerHeader` section from the archive, if present.
fn extract_layer_header(data: &[u8], table: &SectionTable) -> ArchiveResult<Option<LayerHeader>> {
    let entry = match table.find(SECTION_LAYER_HEADER) {
        Some(e) => e,
        None => return Ok(None),
    };
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > data.len() {
        return Err(ArchiveError::OutOfBounds {
            offset: entry.offset,
            size: entry.size,
        });
    }
    let lh = rkyv::from_bytes::<LayerHeader, rkyv::rancor::Error>(&data[start..end])
        .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))?;
    Ok(Some(lh))
}

fn verify_checksums(data: &[u8], header: &HoloHeader) -> ArchiveResult<()> {
    // Verify graph checksum
    if header.graph_size > 0 {
        let start = header.graph_offset as usize;
        let end = start + header.graph_size as usize;
        let actual = checksum::crc32(&data[start..end]);
        if actual != header.graph_checksum {
            return Err(ArchiveError::ChecksumMismatch {
                expected: header.graph_checksum,
                actual,
            });
        }
    }
    // Verify weights checksum
    if header.weights_size > 0 {
        let start = header.weights_offset as usize;
        let end = start + header.weights_size as usize;
        let actual = checksum::crc32(&data[start..end]);
        if actual != header.weights_checksum {
            return Err(ArchiveError::ChecksumMismatch {
                expected: header.weights_checksum,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::holo_writer::HoloWriter;
    use hologram_graph::graph::GraphOp;
    use hologram_graph::Graph;

    #[test]
    fn round_trip_empty() {
        let archive = HoloWriter::new().build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.node_count(), 0);
        assert!(plan.weights().is_empty());
    }

    #[test]
    fn round_trip_with_graph() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Output);
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.node_count(), 2);
    }

    #[test]
    fn round_trip_with_weights() {
        let weights = vec![10u8, 20, 30, 40];
        let archive = HoloWriter::new()
            .set_weights(weights.clone())
            .build()
            .unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.weights(), &weights);
    }

    #[test]
    fn round_trip_with_section() {
        use crate::entrypoint::schedule::LayerHeader;
        use crate::section::SECTION_LAYER_HEADER;

        let header = LayerHeader::new();
        let archive = HoloWriter::new().add_section(&header).build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert!(plan.sections().find(SECTION_LAYER_HEADER).is_some());
    }

    #[test]
    fn invalid_magic() {
        let bad = vec![0u8; 4096];
        let result = load_from_bytes(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn checksum_verified() {
        let weights = vec![1u8, 2, 3];
        let archive = HoloWriter::new().set_weights(weights).build().unwrap();
        // Loading should succeed (checksums match)
        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.weights(), &[1, 2, 3]);
    }

    #[test]
    fn full_round_trip() {
        use crate::entrypoint::schedule::LayerHeader;
        use hologram_core::op::LutOp;
        use hologram_graph::builder::GraphBuilder;

        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();

        let archive = HoloWriter::new()
            .set_graph(&g)
            .set_weights(vec![42u8; 64])
            .add_section(&LayerHeader::new())
            .build()
            .unwrap();

        let plan = load_from_bytes(&archive).unwrap();
        assert_eq!(plan.node_count(), 3);
        assert_eq!(plan.weights().len(), 64);
        assert!(plan.header().is_valid_magic());
        assert!(plan.header().is_supported_version());
    }

    #[test]
    fn layer_header_extracted() {
        use crate::entrypoint::LayerEntrypoint;
        use hologram_graph::builder::GraphBuilder;

        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Output, &[0])
            .output("y", 1)
            .build();

        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        let lh = plan.layer_header().expect("layer header present");
        assert_eq!(lh.layer_count(), 1);
        let layer = &lh.layers[0];
        assert_eq!(layer.name, "main");
        assert_eq!(layer.entrypoint, LayerEntrypoint::Graph);
    }

    #[test]
    fn no_layer_header_without_graph() {
        let archive = HoloWriter::new().build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert!(plan.layer_header().is_none());
    }
}
