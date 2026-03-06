//! Builder for constructing .holo archives in memory.

use crate::checksum;
use crate::error::{ArchiveError, ArchiveResult};
use crate::format::graph::SerializedGraph;
use crate::format::header::HoloHeader;
use crate::format::{align_to_page, FORMAT_VERSION, HOLO_MAGIC, PAGE_SIZE};
use crate::section::table::{SectionEntry, SectionTable};
use crate::section::EmbeddableSection;

/// Builder for constructing a .holo archive in memory.
///
/// Uses builder pattern: set graph, weights, and sections, then call `build()`.
pub struct HoloWriter {
    graph_bytes: Option<Vec<u8>>,
    weight_bytes: Option<Vec<u8>>,
    sections: Vec<(u32, Vec<u8>)>,
}

impl Default for HoloWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl HoloWriter {
    /// Create a new empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph_bytes: None,
            weight_bytes: None,
            sections: Vec::new(),
        }
    }

    /// Set the graph from a live `Graph` (serialized via rkyv).
    #[must_use]
    pub fn set_graph(mut self, graph: &holo_graph::Graph) -> Self {
        let sg = SerializedGraph::from_graph(graph);
        let bytes = rkyv::to_bytes::<_, 4096>(&sg)
            .expect("graph serialization")
            .to_vec();
        self.graph_bytes = Some(bytes);
        self
    }

    /// Set the graph from pre-serialized bytes.
    #[must_use]
    pub fn set_graph_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.graph_bytes = Some(bytes);
        self
    }

    /// Set raw weight data.
    #[must_use]
    pub fn set_weights(mut self, weights: Vec<u8>) -> Self {
        self.weight_bytes = Some(weights);
        self
    }

    /// Add a section from an `EmbeddableSection` implementor.
    #[must_use]
    pub fn add_section(mut self, section: &dyn EmbeddableSection) -> Self {
        self.sections
            .push((section.section_kind(), section.to_bytes()));
        self
    }

    /// Build the complete archive as a byte vector.
    pub fn build(self) -> ArchiveResult<Vec<u8>> {
        let graph_data = self.graph_bytes.unwrap_or_default();
        let weight_data = self.weight_bytes.unwrap_or_default();

        let layout = compute_layout(
            graph_data.len() as u64,
            weight_data.len() as u64,
            &self.sections,
        );

        let header = build_header(&layout, &graph_data, &weight_data);
        assemble_archive(header, &layout, &graph_data, &weight_data, &self.sections)
    }
}

/// Computed byte offsets for each archive section.
struct ArchiveLayout {
    graph_offset: u64,
    section_offsets: Vec<u64>,
    section_table_offset: u64,
    section_table_size: u64,
    weights_offset: u64,
    total_size: u64,
}

fn compute_layout(
    graph_size: u64,
    weights_size: u64,
    sections: &[(u32, Vec<u8>)],
) -> ArchiveLayout {
    // Header serialized size (estimate; we'll use a fixed page)
    let header_size = PAGE_SIZE;
    let graph_offset = header_size;

    // After graph: page-align for sections
    let after_graph = align_to_page(graph_offset + graph_size);

    // Section data: concatenated, each page-aligned
    let mut cursor = after_graph;
    let mut section_offsets = Vec::new();
    for (_, data) in sections {
        section_offsets.push(cursor);
        cursor += data.len() as u64;
        cursor = align_to_page(cursor);
    }
    // Section table after section data
    let section_table_offset = cursor;
    let table = build_section_table(sections, &section_offsets);
    let table_bytes = rkyv::to_bytes::<_, 1024>(&table)
        .expect("section table serialization");
    let section_table_size = table_bytes.len() as u64;
    cursor += section_table_size;
    cursor = align_to_page(cursor);

    // Weights after sections
    let weights_offset = cursor;
    let total_size = weights_offset + weights_size;

    ArchiveLayout {
        graph_offset,
        section_offsets,
        section_table_offset,
        section_table_size,
        weights_offset,
        total_size,
    }
}

fn build_section_table(
    sections: &[(u32, Vec<u8>)],
    offsets: &[u64],
) -> SectionTable {
    let mut table = SectionTable::new();
    for ((kind, data), &offset) in sections.iter().zip(offsets.iter()) {
        table.push(SectionEntry {
            kind: *kind,
            offset,
            size: data.len() as u64,
            checksum: checksum::crc32(data),
        });
    }
    table
}

fn build_header(
    layout: &ArchiveLayout,
    graph_data: &[u8],
    weight_data: &[u8],
) -> HoloHeader {
    HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: layout.graph_offset,
        graph_size: graph_data.len() as u64,
        weights_offset: layout.weights_offset,
        weights_size: weight_data.len() as u64,
        section_table_offset: layout.section_table_offset,
        section_table_size: layout.section_table_size,
        total_size: layout.total_size,
        graph_checksum: checksum::crc32(graph_data),
        weights_checksum: checksum::crc32(weight_data),
        section_count: layout.section_offsets.len() as u32,
        flags: 0,
    }
}

fn assemble_archive(
    header: HoloHeader,
    layout: &ArchiveLayout,
    graph_data: &[u8],
    weight_data: &[u8],
    sections: &[(u32, Vec<u8>)],
) -> ArchiveResult<Vec<u8>> {
    let mut buf = vec![0u8; layout.total_size as usize];

    // Write header (fixed-layout via bytemuck)
    let header_bytes = header.as_bytes();
    buf[..header_bytes.len()].copy_from_slice(header_bytes);

    // Write graph
    let go = layout.graph_offset as usize;
    buf[go..go + graph_data.len()].copy_from_slice(graph_data);

    // Write section data
    for ((_, data), &offset) in sections.iter().zip(layout.section_offsets.iter()) {
        let o = offset as usize;
        buf[o..o + data.len()].copy_from_slice(data);
    }

    // Write section table
    let table = build_section_table(sections, &layout.section_offsets);
    let table_bytes = rkyv::to_bytes::<_, 1024>(&table)
        .map_err(|e| ArchiveError::GraphError(format!("{e}")))?;
    let sto = layout.section_table_offset as usize;
    buf[sto..sto + table_bytes.len()].copy_from_slice(&table_bytes);

    // Write weights
    let wo = layout.weights_offset as usize;
    buf[wo..wo + weight_data.len()].copy_from_slice(weight_data);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use holo_graph::graph::GraphOp;
    use holo_graph::Graph;

    #[test]
    fn build_empty() {
        let archive = HoloWriter::new().build().unwrap();
        assert!(!archive.is_empty());
        // Should start with rkyv-serialized header containing HOLO magic
        // Validate by checking that the serialized header can be read
    }

    #[test]
    fn build_with_graph() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        assert!(archive.len() >= PAGE_SIZE as usize);
    }

    #[test]
    fn build_with_weights() {
        let weights = vec![1u8, 2, 3, 4];
        let archive = HoloWriter::new()
            .set_weights(weights.clone())
            .build()
            .unwrap();
        // Weights should be embedded in the archive
        assert!(archive.len() >= weights.len());
    }

    #[test]
    fn build_with_section() {
        use crate::entrypoint::schedule::LayerHeader;
        let header = LayerHeader::new();
        let archive = HoloWriter::new()
            .add_section(&header)
            .build()
            .unwrap();
        assert!(!archive.is_empty());
    }

    #[test]
    fn build_full() {
        use crate::entrypoint::schedule::LayerHeader;

        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Output);

        let archive = HoloWriter::new()
            .set_graph(&g)
            .set_weights(vec![0u8; 128])
            .add_section(&LayerHeader::new())
            .build()
            .unwrap();

        // Verify: archive is large enough for header + graph + weights
        assert!(archive.len() >= PAGE_SIZE as usize + 128);
    }

    #[test]
    fn graph_offset_page_aligned() {
        let layout = compute_layout(100, 50, &[]);
        assert_eq!(layout.graph_offset % PAGE_SIZE, 0);
        assert_eq!(layout.weights_offset % PAGE_SIZE, 0);
    }

    #[test]
    fn header_has_correct_checksums() {
        let graph_data = vec![1, 2, 3];
        let weight_data = vec![4, 5, 6];
        let layout = compute_layout(3, 3, &[]);
        let header = build_header(&layout, &graph_data, &weight_data);
        assert_eq!(header.graph_checksum, checksum::crc32(&graph_data));
        assert_eq!(header.weights_checksum, checksum::crc32(&weight_data));
    }

    #[test]
    fn default_impl() {
        let w = HoloWriter::default();
        let archive = w.build().unwrap();
        assert!(!archive.is_empty());
    }
}
