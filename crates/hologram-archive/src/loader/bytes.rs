//! Load a .holo archive from a byte slice (WASM/embedded compatible).

use crate::checksum;
use crate::entrypoint::schedule::LayerHeader;
use crate::error::{ArchiveError, ArchiveResult};
use crate::format::graph::SerializedGraph;
use crate::format::header::{HoloHeader, HEADER_SIZE};
use crate::loader::plan::LoadedPlan;
use crate::section::table::SectionTable;
use crate::section::SECTION_LAYER_HEADER;

/// Check if an archive has any compressed sections (graph or weights).
///
/// Only reads the header (first 128 bytes) — does not parse the full archive.
/// Use this to decide whether to decompress to a cache file before mmap loading.
pub fn is_compressed(data: &[u8]) -> bool {
    validate_header(data)
        .map(|h| h.is_graph_compressed() || h.is_weights_compressed())
        .unwrap_or(false)
}

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

/// Decompress an archive into an uncompressed version.
///
/// Reads the compressed archive, decompresses graph and weight sections,
/// and rebuilds as an uncompressed archive. The result can be written to
/// a cache file for instant mmap loading on subsequent runs.
///
/// Returns `None` if the archive is already uncompressed.
pub fn decompress_archive(data: &[u8]) -> ArchiveResult<Option<Vec<u8>>> {
    let header = validate_header(data)?;
    if !header.is_graph_compressed() && !header.is_weights_compressed() {
        return Ok(None); // Already uncompressed
    }

    // Load via the checked path (handles decompression internally).
    let plan = load_from_bytes(data)?;

    // Rebuild uncompressed: graph via rkyv serialization, weights as-is.
    let graph = plan.graph();
    let graph_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(graph)
        .map_err(|e| ArchiveError::ValidationFailed(format!("re-serializing graph: {e}")))?
        .to_vec();

    // Rebuild via HoloWriter with compression disabled (the default).
    let mut writer = crate::writer::holo_writer::HoloWriter::new()
        .set_graph_bytes_uncompressed(graph_bytes)
        .set_weights(plan.weights().to_vec());

    // Preserve sections from the original archive.
    let sections = plan.sections();
    for entry in &sections.entries {
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end <= data.len() {
            writer = writer.add_raw_section(entry.kind, data[start..end].to_vec());
        }
    }

    Ok(Some(writer.build()?))
}

/// Load a .holo archive from a byte slice.
///
/// Validates magic, version, and checksums, then deserializes the
/// graph and extracts weight data.
pub fn load_from_bytes(data: &[u8]) -> ArchiveResult<LoadedPlan> {
    load_from_bytes_inner(data, true)
}

/// Load a `.holo` archive, auto-detecting pipeline format.
///
/// If the archive is a pipeline wrapper (empty graph with a pipeline section),
/// returns the first model's [`LoadedPlan`]. Otherwise identical to [`load_from_bytes`].
pub fn load_auto(data: &[u8]) -> ArchiveResult<LoadedPlan> {
    let plan = load_from_bytes_inner(data, true)?;
    if plan.graph().nodes.is_empty()
        && plan
            .sections()
            .find(crate::section::SECTION_PIPELINE)
            .is_some()
    {
        let pipeline = crate::loader::pipeline::LoadedPipeline::from_bytes(data)?;
        if let Some(model) = pipeline.into_first_model() {
            return Ok(model);
        }
    }
    Ok(plan)
}

/// Load a .holo archive without checksum verification.
///
/// Skips the BLAKE3 checksum on the weights region, which avoids reading
/// the entire multi-GB weight blob on load. Use for mmap-backed archives
/// where the OS guarantees data integrity via the filesystem.
pub fn load_from_bytes_unchecked(data: &[u8]) -> ArchiveResult<LoadedPlan> {
    load_from_bytes_inner(data, false)
}

/// Load a .holo archive with zero-copy weight access.
///
/// Weights are borrowed directly from the input slice (no allocation or copy).
/// Skips checksum verification. Ideal for mmap-backed archives.
///
/// # Safety
/// The caller must ensure `data` outlives the returned `LoadedPlan`.
pub unsafe fn load_from_bytes_zero_copy(data: &[u8]) -> ArchiveResult<LoadedPlan> {
    let header = find_and_validate_header(data)?;
    let section_table = deserialize_section_table(data, &header)?;
    let layer_header = extract_layer_header(data, &section_table)?;

    // Extract graph bytes without deserializing — lazy deser on first access.
    let graph_bytes = extract_graph_bytes(data, &header)?;

    // If weights are compressed, we must decompress (allocate).
    // Otherwise, zero-copy: borrow directly from the mmap'd data.
    if header.is_weights_compressed() {
        let weights = extract_weights(data, &header)?;
        if graph_bytes.is_empty() {
            Ok(LoadedPlan::new(
                header,
                SerializedGraph::empty(),
                weights,
                section_table,
                layer_header,
            ))
        } else {
            Ok(LoadedPlan::new_with_archived_graph(
                header,
                graph_bytes,
                weights,
                section_table,
                layer_header,
            ))
        }
    } else {
        let weights = if header.weights_size > 0 {
            let start = header.weights_offset as usize;
            let end = start + header.weights_size as usize;
            if end > data.len() {
                return Err(ArchiveError::OutOfBounds {
                    offset: header.weights_offset,
                    size: header.weights_size,
                });
            }
            &data[start..end]
        } else {
            &[]
        };
        if graph_bytes.is_empty() {
            Ok(LoadedPlan::new_borrowed(
                header,
                SerializedGraph::empty(),
                weights,
                section_table,
                layer_header,
            ))
        } else {
            Ok(LoadedPlan::new_with_archived_graph_borrowed(
                header,
                graph_bytes,
                weights,
                section_table,
                layer_header,
            ))
        }
    }
}

fn load_from_bytes_inner(data: &[u8], verify: bool) -> ArchiveResult<LoadedPlan> {
    let header = find_and_validate_header(data)?;
    let graph = deserialize_graph(data, &header)?;
    let weights = extract_weights(data, &header)?;
    let section_table = deserialize_section_table(data, &header)?;
    let layer_header = extract_layer_header(data, &section_table)?;

    if verify {
        verify_checksums(data, &header)?;
    }

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
    let aligned = extract_graph_bytes(data, header)?;
    if aligned.is_empty() {
        return Ok(SerializedGraph::empty());
    }
    rkyv::from_bytes::<SerializedGraph, rkyv::rancor::Error>(&aligned)
        .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))
}

/// Extract graph bytes as aligned bytes WITHOUT deserializing.
///
/// For uncompressed archives, this returns the raw rkyv bytes in an AlignedVec.
/// These bytes can be stored as `GraphAccess::Archived` for lazy deserialization.
fn extract_graph_bytes(
    data: &[u8],
    header: &HoloHeader,
) -> ArchiveResult<rkyv::util::AlignedVec<16>> {
    let start = header.graph_offset as usize;
    let end = start + header.graph_size as usize;
    if end > data.len() {
        return Err(ArchiveError::OutOfBounds {
            offset: header.graph_offset,
            size: header.graph_size,
        });
    }
    if header.graph_size == 0 {
        return Ok(rkyv::util::AlignedVec::<16>::new());
    }
    let graph_bytes = &data[start..end];

    // Decompress if compressed, then align for rkyv.
    let aligned = if header.is_graph_compressed() {
        let decompressed = hologram_compression::decompress(graph_bytes).ok_or_else(|| {
            ArchiveError::ValidationFailed("failed to decompress graph section".into())
        })?;
        let mut av = rkyv::util::AlignedVec::<16>::with_capacity(decompressed.len());
        av.extend_from_slice(&decompressed);
        av
    } else {
        let mut av = rkyv::util::AlignedVec::<16>::with_capacity(graph_bytes.len());
        av.extend_from_slice(graph_bytes);
        av
    };
    Ok(aligned)
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
    let weight_bytes = &data[start..end];

    // Decompress if the weights section is compressed.
    if header.is_weights_compressed() {
        return hologram_compression::decompress(weight_bytes).ok_or_else(|| {
            ArchiveError::ValidationFailed("failed to decompress weights section".into())
        });
    }

    Ok(weight_bytes.to_vec())
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

    SectionTable::from_raw_bytes(table_bytes).map_err(|e| ArchiveError::ValidationFailed(e.into()))
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
    let section_bytes = &data[start..end];
    let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(section_bytes.len());
    aligned.extend_from_slice(section_bytes);
    let lh = rkyv::from_bytes::<LayerHeader, rkyv::rancor::Error>(&aligned)
        .map_err(|e| ArchiveError::ValidationFailed(format!("{e}")))?;
    Ok(Some(lh))
}

fn verify_checksums(data: &[u8], header: &HoloHeader) -> ArchiveResult<()> {
    // Verify graph checksum
    if header.graph_size > 0 {
        let start = header.graph_offset as usize;
        let end = start + header.graph_size as usize;
        let actual = checksum::checksum(&data[start..end]);
        if actual != header.graph_checksum {
            return Err(ArchiveError::ChecksumMismatch {
                expected: header.graph_checksum,
                actual,
            });
        }
    }
    // Verify weights checksum (skip if all-zeros — streaming archives
    // can't compute the checksum without loading all weights into memory).
    if header.weights_size > 0 && header.weights_checksum != [0u8; 32] {
        let start = header.weights_offset as usize;
        let end = start + header.weights_size as usize;
        let actual = checksum::checksum(&data[start..end]);
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
            // ADR-053: Relu (idx 1) requires shape coverage for v3.
            .set_node_shape(1, vec![4])
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

    #[test]
    fn unchecked_skips_checksum() {
        let weights = vec![42u8; 256];
        let archive = HoloWriter::new()
            .set_weights(weights.clone())
            .build()
            .unwrap();
        // Unchecked load should succeed and return correct weights.
        let plan = load_from_bytes_unchecked(&archive).unwrap();
        assert_eq!(plan.weights(), &weights);
    }

    #[test]
    fn zero_copy_matches_checked() {
        let weights = vec![7u8; 128];
        let archive = HoloWriter::new()
            .set_weights(weights.clone())
            .build()
            .unwrap();

        let plan_checked = load_from_bytes(&archive).unwrap();
        // SAFETY: archive outlives plan (both in this scope).
        let plan_zero = unsafe { load_from_bytes_zero_copy(&archive) }.unwrap();

        // Both should produce identical weight content.
        assert_eq!(plan_checked.weights().len(), plan_zero.weights().len());
        assert_eq!(plan_checked.weights(), plan_zero.weights());
    }

    #[test]
    fn zero_copy_with_graph() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Output);
        let weights = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let archive = HoloWriter::new()
            .set_graph(&g)
            .set_weights(weights.clone())
            .build()
            .unwrap();

        let plan_checked = load_from_bytes(&archive).unwrap();
        // SAFETY: archive outlives plan.
        let plan_zero = unsafe { load_from_bytes_zero_copy(&archive) }.unwrap();

        assert_eq!(plan_checked.node_count(), plan_zero.node_count());
        assert_eq!(plan_checked.weights(), plan_zero.weights());
    }

    #[test]
    fn zero_copy_empty_archive() {
        let archive = HoloWriter::new().build().unwrap();
        // SAFETY: archive outlives plan.
        let plan = unsafe { load_from_bytes_zero_copy(&archive) }.unwrap();
        assert_eq!(plan.node_count(), 0);
        assert!(plan.weights().is_empty());
    }
}
