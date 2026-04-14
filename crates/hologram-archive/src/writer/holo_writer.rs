//! Builder for constructing .holo archives in memory.

use crate::checksum;
use crate::entrypoint::schedule::LayerHeader;
use crate::entrypoint::{LayerDescriptor, LayerEntrypoint, LayerId, TensorPort};
use crate::error::ArchiveResult;
use crate::format::graph::SerializedGraph;
use crate::format::header::HoloHeader;
use crate::format::{align_to_page, FORMAT_VERSION, HOLO_MAGIC, PAGE_SIZE};
use crate::section::table::{SectionEntry, SectionTable};
use crate::section::{EmbeddableSection, SECTION_LAYER_HEADER};
use crate::weight::WeightDType;

/// Builder for constructing a .holo archive in memory.
///
/// Uses builder pattern: set graph, weights, and sections, then call `build()`.
/// Automatically generates a `LayerHeader` section with a default "main"
/// entrypoint when a graph is set and no `LayerHeader` is explicitly added.
pub struct HoloWriter {
    graph_bytes: Option<Vec<u8>>,
    graph_input_names: Vec<String>,
    graph_output_names: Vec<String>,
    weight_bytes: Option<Vec<u8>>,
    /// File-backed weight source for large models. When set, weights are
    /// streamed from this file during `build_to_file()` — never held in memory.
    weight_file: Option<WeightSource>,
    sections: Vec<(u32, Vec<u8>)>,
    /// When true, `graph_bytes` are already compressed and should not be
    /// compressed again during `build()`.
    graph_pre_compressed: bool,
    /// When false, skip weight compression. Default: false (no compression).
    compress_weights: bool,
    /// When false, skip graph compression. Default: false (no compression).
    /// Enables zero-copy rkyv::access on the graph without decompression.
    compress_graph: bool,
    /// When true, the `FLAG_TENSOR_PAGE_ALIGNED` flag is set in the header.
    /// The caller is responsible for page-aligning tensors within the weight
    /// blob before calling `set_weights()` (see `page_align_weight_blob`).
    tensor_page_aligned: bool,
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
            graph_input_names: Vec::new(),
            graph_output_names: Vec::new(),
            weight_bytes: None,
            weight_file: None,
            sections: Vec::new(),
            graph_pre_compressed: false,
            compress_weights: false,
            compress_graph: false,
            tensor_page_aligned: false,
        }
    }

    /// Set the graph from a live `Graph` (serialized via rkyv).
    #[must_use]
    pub fn set_graph(mut self, graph: &hologram_graph::Graph) -> Self {
        let sg = SerializedGraph::from_graph(graph);
        self.graph_input_names = sg.input_names.clone();
        self.graph_output_names = sg.output_names.clone();
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&sg)
            .expect("graph serialization")
            .to_vec();
        self.graph_bytes = Some(bytes);
        self
    }

    /// Set the graph from pre-serialized (and already compressed) bytes.
    ///
    /// These bytes are assumed to already be compressed and will NOT be
    /// compressed again during `build()`. Use this when extracting graph
    /// bytes from an existing archive for rebuild.
    #[must_use]
    pub fn set_graph_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.graph_bytes = Some(bytes);
        self.graph_pre_compressed = true;
        self
    }

    /// Set the graph from pre-serialized uncompressed rkyv bytes.
    ///
    /// The bytes will be compressed during `build()` only if `compress_graph`
    /// is enabled. Use this when rebuilding an archive from decompressed data.
    #[must_use]
    pub fn set_graph_bytes_uncompressed(mut self, bytes: Vec<u8>) -> Self {
        self.graph_bytes = Some(bytes);
        self.graph_pre_compressed = false;
        self
    }

    /// Set raw weight data (in-memory).
    #[must_use]
    pub fn set_weights(mut self, weights: Vec<u8>) -> Self {
        self.weight_bytes = Some(weights);
        self
    }

    /// Set weight data from a file on disk (streaming).
    ///
    /// The weights are read from the file at build time and streamed
    /// directly to the output archive — never held in memory. Use this
    /// for large models (SDXL UNet = 10 GB) to keep RSS bounded.
    #[must_use]
    pub fn set_weight_source(mut self, source: WeightSource) -> Self {
        match source {
            WeightSource::Bytes(v) => self.weight_bytes = Some(v),
            WeightSource::File { .. } => self.weight_file = Some(source),
        }
        self
    }

    /// Mark that tensors within the weight blob are page-aligned.
    ///
    /// When enabled, sets `FLAG_TENSOR_PAGE_ALIGNED` in the archive header.
    /// The caller must ensure tensors are actually page-aligned within the
    /// weight blob (see [`page_align_weight_blob`]).
    #[must_use]
    pub fn tensor_page_aligned(mut self, enabled: bool) -> Self {
        self.tensor_page_aligned = enabled;
        self
    }

    /// Enable weight compression (off by default).
    ///
    /// Compresses the weight region for smaller archives. Disables zero-copy
    /// mmap loading — the entire weight region must be decompressed on load.
    /// Use for distribution/storage; omit for fast local inference.
    #[must_use]
    pub fn compress_weights(mut self) -> Self {
        self.compress_weights = true;
        self
    }

    /// Enable graph compression (off by default).
    ///
    /// Compresses the graph section for smaller archives. Disables zero-copy
    /// rkyv::access — the graph must be decompressed and deserialized on load.
    #[must_use]
    pub fn compress_graph(mut self) -> Self {
        self.compress_graph = true;
        self
    }

    /// Add a section from an `EmbeddableSection` implementor.
    #[must_use]
    pub fn add_section(mut self, section: &dyn EmbeddableSection) -> Self {
        self.sections
            .push((section.section_kind(), section.to_bytes()));
        self
    }

    /// Add a section from raw bytes (kind + pre-serialized data).
    #[must_use]
    pub fn add_raw_section(mut self, kind: u32, bytes: Vec<u8>) -> Self {
        self.sections.push((kind, bytes));
        self
    }

    /// Build the complete archive as a byte vector.
    ///
    /// If a graph was set via `set_graph` and no `LayerHeader` section was
    /// explicitly added, a default one is generated with a single "main"
    /// layer using `LayerEntrypoint::Graph`.
    pub fn build(mut self) -> ArchiveResult<Vec<u8>> {
        self.ensure_layer_header();
        let graph_data = self.graph_bytes.unwrap_or_default();
        let weight_data = self.weight_bytes.unwrap_or_default();

        // Compress graph and weight sections (in parallel when feature enabled).
        let (graph_data, weight_data, flags) = {
            use crate::format::header::{FLAG_GRAPH_COMPRESSED, FLAG_WEIGHTS_COMPRESSED};
            use hologram_compression::codec::CompressionMode;

            let compress_graph_flag = self.compress_graph;
            let pre_compressed = self.graph_pre_compressed;
            let compress_weights_flag = self.compress_weights;

            let compress_graph = move || -> (Vec<u8>, u32) {
                if graph_data.is_empty() {
                    (graph_data, 0)
                } else if pre_compressed {
                    (graph_data, FLAG_GRAPH_COMPRESSED)
                } else if compress_graph_flag {
                    let block =
                        hologram_compression::compress(&graph_data, CompressionMode::Generic);
                    (block.data, FLAG_GRAPH_COMPRESSED)
                } else {
                    (graph_data, 0)
                }
            };

            let compress_weights = move || -> (Vec<u8>, u32) {
                if !weight_data.is_empty() && compress_weights_flag {
                    let mode = hologram_compression::pipeline::auto_select_mode(&weight_data);
                    let block = hologram_compression::compress(&weight_data, mode);
                    (block.data, FLAG_WEIGHTS_COMPRESSED)
                } else {
                    (weight_data, 0)
                }
            };

            #[cfg(feature = "parallel")]
            let ((graph_data, gf), (weight_data, wf)) =
                rayon::join(compress_graph, compress_weights);
            #[cfg(not(feature = "parallel"))]
            let ((graph_data, gf), (weight_data, wf)) = (compress_graph(), compress_weights());

            (graph_data, weight_data, gf | wf)
        };

        let (layout, table_bytes) = compute_layout(
            graph_data.len() as u64,
            weight_data.len() as u64,
            &self.sections,
        );

        let mut header = build_header(&layout, &graph_data, &weight_data);
        header.flags = flags
            | if self.tensor_page_aligned {
                crate::format::header::FLAG_TENSOR_PAGE_ALIGNED
            } else {
                0
            };
        assemble_archive(
            header,
            &layout,
            &graph_data,
            &weight_data,
            &self.sections,
            &table_bytes,
        )
    }

    /// Build the archive to a file, streaming weights from disk.
    ///
    /// Unlike `build()` which holds the entire archive in memory, this writes
    /// each section to the output file via buffered I/O. Peak memory is bounded
    /// by the graph + sections (~tens of MB), not the weight data.
    ///
    /// Use `set_weight_source(WeightSource::File { .. })` to provide file-backed
    /// weights. Falls back to in-memory weights from `set_weights()` if no
    /// file source was set.
    pub fn build_to_file(mut self, output_path: &std::path::Path) -> ArchiveResult<()> {
        self.ensure_layer_header();
        let graph_data = self.graph_bytes.unwrap_or_default();

        // Determine weight source: prefer file-backed, fall back to in-memory.
        let weight_source = self
            .weight_file
            .unwrap_or_else(|| WeightSource::Bytes(self.weight_bytes.unwrap_or_default()));

        // Compute layout using weight byte length (no data loaded).
        let weight_len = weight_source.len();
        let (layout, table_bytes) =
            compute_layout(graph_data.len() as u64, weight_len, &self.sections);

        let header = {
            // For file-backed weights, we can't compute a checksum without
            // loading the data. Use zero checksum — the loader validates
            // via section checksums instead.
            let empty_weights: Vec<u8> = Vec::new();
            let weight_ref = match &weight_source {
                WeightSource::Bytes(v) => v.as_slice(),
                WeightSource::File { .. } => &empty_weights,
            };
            let mut h = build_header(&layout, &graph_data, weight_ref);
            if self.tensor_page_aligned {
                h.flags |= crate::format::header::FLAG_TENSOR_PAGE_ALIGNED;
            }
            h
        };

        assemble_archive_to_file(
            header,
            &layout,
            &graph_data,
            &weight_source,
            &self.sections,
            &table_bytes,
            output_path,
        )
    }

    /// Add a default `LayerHeader` if a graph was set and none was provided.
    fn ensure_layer_header(&mut self) {
        if self.graph_bytes.is_none() {
            return;
        }
        let has_layer_header = self
            .sections
            .iter()
            .any(|(k, _)| *k == SECTION_LAYER_HEADER);
        if has_layer_header {
            return;
        }
        let lh = build_default_layer_header(&self.graph_input_names, &self.graph_output_names);
        self.sections.push((lh.section_kind(), lh.to_bytes()));
    }
}

/// Build a default `LayerHeader` with a single "main" graph entrypoint.
fn build_default_layer_header(inputs: &[String], outputs: &[String]) -> LayerHeader {
    let descriptor = LayerDescriptor {
        id: LayerId(0),
        name: "main".into(),
        entrypoint: LayerEntrypoint::Graph,
        inputs: inputs.iter().map(|n| default_port(n)).collect(),
        outputs: outputs.iter().map(|n| default_port(n)).collect(),
        group: 0,
        plan_offset: 0,
        plan_size: 0,
    };
    LayerHeader {
        layers: vec![descriptor],
        schedule: vec![vec![LayerId(0)]],
    }
}

/// Build a default tensor port with U8 dtype and scalar shape.
fn default_port(name: &str) -> TensorPort {
    TensorPort {
        name: name.to_string(),
        shape: vec![1],
        dtype: WeightDType::U8,
    }
}

/// Computed byte offsets for each archive section.
struct ArchiveLayout {
    graph_offset: u64,
    section_offsets: Vec<u64>,
    section_table_offset: u64,
    section_table_size: u64,
    weights_offset: u64,
    weights_size: u64,
    total_size: u64,
}

/// Compute the archive layout and build the section table bytes once.
///
/// Returns `(layout, table_bytes)`. The table is serialized via
/// `SectionTable::to_raw_bytes()` (bytemuck fixed-layout), avoiding
/// rkyv overhead. CRC32 per section is computed exactly once here;
/// `assemble_archive` reuses the returned `table_bytes` directly.
fn compute_layout(
    graph_size: u64,
    weights_size: u64,
    sections: &[(u32, Vec<u8>)],
) -> (ArchiveLayout, Vec<u8>) {
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
    // Section table after section data — build once via bytemuck.
    let section_table_offset = cursor;
    let table = build_section_table(sections, &section_offsets);
    let table_bytes = table.to_raw_bytes();
    let section_table_size = table_bytes.len() as u64;
    cursor += section_table_size;
    cursor = align_to_page(cursor);

    // Weights after sections
    let weights_offset = cursor;
    let total_size = weights_offset + weights_size;

    (
        ArchiveLayout {
            graph_offset,
            section_offsets,
            section_table_offset,
            section_table_size,
            weights_offset,
            weights_size,
            total_size,
        },
        table_bytes,
    )
}

fn build_section_table(sections: &[(u32, Vec<u8>)], offsets: &[u64]) -> SectionTable {
    let mut table = SectionTable::new();
    for ((kind, data), &offset) in sections.iter().zip(offsets.iter()) {
        table.push(SectionEntry {
            kind: *kind,
            offset,
            size: data.len() as u64,
            checksum: checksum::checksum(data),
        });
    }
    table
}

fn build_header(layout: &ArchiveLayout, graph_data: &[u8], weight_data: &[u8]) -> HoloHeader {
    HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: layout.graph_offset,
        graph_size: graph_data.len() as u64,
        weights_offset: layout.weights_offset,
        weights_size: layout.weights_size,
        section_table_offset: layout.section_table_offset,
        section_table_size: layout.section_table_size,
        total_size: layout.total_size,
        certificate_offset: 0,
        certificate_size: 0,
        graph_checksum: checksum::checksum(graph_data),
        weights_checksum: checksum::checksum(weight_data),
        unit_address: [0u8; 32],
        section_count: layout.section_offsets.len() as u32,
        flags: 0,
    }
}

/// Source for weight data — either in-memory bytes or a file on disk.
pub enum WeightSource {
    /// In-memory byte vector (small models).
    Bytes(Vec<u8>),
    /// File on disk (large models). The weights are read from the file
    /// at build time and streamed to the output, never held in memory.
    File { path: std::path::PathBuf, len: u64 },
}

impl WeightSource {
    /// Byte length of the weight data.
    pub fn len(&self) -> u64 {
        match self {
            Self::Bytes(v) => v.len() as u64,
            Self::File { len, .. } => *len,
        }
    }

    /// Whether the source is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build a .holo archive to a file, streaming weights from the source.
///
/// Unlike `assemble_archive` (which allocates the entire archive in memory),
/// this writes each section to the output file via buffered I/O. Peak memory
/// is bounded by the graph + sections (~tens of MB), not the weight data.
fn assemble_archive_to_file(
    header: HoloHeader,
    layout: &ArchiveLayout,
    graph_data: &[u8],
    weight_source: &WeightSource,
    sections: &[(u32, Vec<u8>)],
    table_bytes: &[u8],
    output_path: &std::path::Path,
) -> ArchiveResult<()> {
    use std::io::{Seek, SeekFrom, Write};

    let mut w = std::fs::File::create(output_path).map_err(crate::error::ArchiveError::Io)?;
    // Pre-allocate the file to the final size for efficient sequential writes.
    w.set_len(layout.total_size)
        .map_err(crate::error::ArchiveError::Io)?;

    // Header (fixed-layout via bytemuck).
    w.write_all(header.as_bytes())
        .map_err(crate::error::ArchiveError::Io)?;

    // Graph — write sequentially after header (graph_offset is always right after header).
    w.seek(SeekFrom::Start(layout.graph_offset))
        .map_err(crate::error::ArchiveError::Io)?;
    w.write_all(graph_data)
        .map_err(crate::error::ArchiveError::Io)?;

    // Sections.
    for ((_, data), &offset) in sections.iter().zip(layout.section_offsets.iter()) {
        w.seek(SeekFrom::Start(offset))
            .map_err(crate::error::ArchiveError::Io)?;
        w.write_all(data).map_err(crate::error::ArchiveError::Io)?;
    }

    // Section table.
    w.seek(SeekFrom::Start(layout.section_table_offset))
        .map_err(crate::error::ArchiveError::Io)?;
    w.write_all(table_bytes)
        .map_err(crate::error::ArchiveError::Io)?;

    // Weights — streamed from source with manual buffering.
    w.seek(SeekFrom::Start(layout.weights_offset))
        .map_err(crate::error::ArchiveError::Io)?;
    match weight_source {
        WeightSource::Bytes(data) => {
            w.write_all(data).map_err(crate::error::ArchiveError::Io)?;
        }
        WeightSource::File { path, len } => {
            let mut src = std::fs::File::open(path).map_err(crate::error::ArchiveError::Io)?;
            let mut remaining = *len as usize;
            let mut buf = vec![0u8; 1024 * 1024]; // 1 MB streaming buffer
            while remaining > 0 {
                let to_read = remaining.min(buf.len());
                std::io::Read::read_exact(&mut src, &mut buf[..to_read])
                    .map_err(crate::error::ArchiveError::Io)?;
                w.write_all(&buf[..to_read])
                    .map_err(crate::error::ArchiveError::Io)?;
                remaining -= to_read;
            }
        }
    }

    w.sync_all().map_err(crate::error::ArchiveError::Io)?;
    Ok(())
}

fn assemble_archive(
    header: HoloHeader,
    layout: &ArchiveLayout,
    graph_data: &[u8],
    weight_data: &[u8],
    sections: &[(u32, Vec<u8>)],
    table_bytes: &[u8],
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

    // Write section table — reuse pre-computed bytes from compute_layout.
    let sto = layout.section_table_offset as usize;
    buf[sto..sto + table_bytes.len()].copy_from_slice(table_bytes);

    // Write weights
    let wo = layout.weights_offset as usize;
    buf[wo..wo + weight_data.len()].copy_from_slice(weight_data);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_graph::graph::GraphOp;
    use hologram_graph::Graph;

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
        let archive = HoloWriter::new().add_section(&header).build().unwrap();
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
        let (layout, _) = compute_layout(100, 50, &[]);
        assert_eq!(layout.graph_offset % PAGE_SIZE, 0);
        assert_eq!(layout.weights_offset % PAGE_SIZE, 0);
    }

    #[test]
    fn header_has_correct_checksums() {
        let graph_data = vec![1, 2, 3];
        let weight_data = vec![4, 5, 6];
        let (layout, _) = compute_layout(3, 3, &[]);
        let header = build_header(&layout, &graph_data, &weight_data);
        assert_eq!(header.graph_checksum, checksum::checksum(&graph_data));
        assert_eq!(header.weights_checksum, checksum::checksum(&weight_data));
    }

    #[test]
    fn default_impl() {
        let w = HoloWriter::default();
        let archive = w.build().unwrap();
        assert!(!archive.is_empty());
    }

    #[test]
    fn auto_generates_layer_header() {
        use crate::loader::bytes::load_from_bytes;
        use hologram_graph::builder::GraphBuilder;

        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Output, &[0])
            .output("y", 1)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        assert!(plan.sections().find(SECTION_LAYER_HEADER).is_some());
    }

    #[test]
    fn explicit_layer_header_not_duplicated() {
        use crate::entrypoint::schedule::LayerHeader;
        use crate::load_from_bytes;

        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let custom = LayerHeader::new();
        let archive = HoloWriter::new()
            .set_graph(&g)
            .add_section(&custom)
            .build()
            .unwrap();
        let plan = load_from_bytes(&archive).unwrap();
        let count = plan
            .sections()
            .entries
            .iter()
            .filter(|e| e.kind == SECTION_LAYER_HEADER)
            .count();
        assert_eq!(count, 1);
    }
}
