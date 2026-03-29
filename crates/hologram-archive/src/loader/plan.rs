//! Loaded and validated archive plan.

use crate::entrypoint::schedule::LayerHeader;
use crate::format::graph::SerializedGraph;
use crate::format::header::HoloHeader;
use crate::section::table::SectionTable;
use std::borrow::Cow;
use std::sync::OnceLock;

/// How the graph is stored — either fully deserialized (owned) or as raw
/// archived bytes that are deserialized lazily on first access.
///
/// The `Archived` variant enables zero-copy graph loading: uncompressed graph
/// bytes from mmap are kept as-is, avoiding the 1.5s `rkyv::from_bytes`
/// deserialization cost until the graph is actually needed.
enum GraphAccess {
    /// Fully deserialized graph (compressed archives, or after lazy deser).
    Owned(SerializedGraph),
    /// Raw archived bytes (uncompressed, 16-byte aligned). The graph is
    /// deserialized lazily on first `graph()` call via `OnceLock`.
    Archived {
        bytes: rkyv::util::AlignedVec<16>,
        cache: OnceLock<SerializedGraph>,
    },
}

/// A loaded and validated archive.
///
/// Provides access to the deserialized graph, raw weight bytes,
/// the section table, and the layer header (if present).
///
/// Weight bytes are borrowed when loaded from mmap (zero-copy) or
/// owned when loaded from a network buffer or compressed archive.
///
/// Graph bytes may be stored as raw archived bytes (uncompressed archives)
/// and deserialized lazily on first access, avoiding the 1.5s deserialization
/// cost for large graphs like TinyLlama's 199MB graph.
pub struct LoadedPlan {
    header: HoloHeader,
    graph: GraphAccess,
    weights: Cow<'static, [u8]>,
    section_table: SectionTable,
    layer_header: Option<LayerHeader>,
}

impl LoadedPlan {
    /// Create a new LoadedPlan with an owned (fully deserialized) graph and owned weight bytes.
    pub(crate) fn new(
        header: HoloHeader,
        graph: SerializedGraph,
        weights: Vec<u8>,
        section_table: SectionTable,
        layer_header: Option<LayerHeader>,
    ) -> Self {
        Self {
            header,
            graph: GraphAccess::Owned(graph),
            weights: Cow::Owned(weights),
            section_table,
            layer_header,
        }
    }

    /// Create a new LoadedPlan with borrowed weight bytes (zero-copy from mmap).
    ///
    /// # Safety
    /// The caller must ensure the weight bytes outlive this LoadedPlan.
    /// This is guaranteed when the bytes come from an mmap stored in the same
    /// struct (e.g., HoloRunner holds both the mmap and the plan).
    pub(crate) unsafe fn new_borrowed(
        header: HoloHeader,
        graph: SerializedGraph,
        weights: &[u8],
        section_table: SectionTable,
        layer_header: Option<LayerHeader>,
    ) -> Self {
        let weights_static: &'static [u8] =
            std::slice::from_raw_parts(weights.as_ptr(), weights.len());
        Self {
            header,
            graph: GraphAccess::Owned(graph),
            weights: Cow::Borrowed(weights_static),
            section_table,
            layer_header,
        }
    }

    /// Create a LoadedPlan with raw archived graph bytes (zero-copy graph).
    ///
    /// The graph is NOT deserialized until `graph()` is called. This avoids
    /// the 1.5s `rkyv::from_bytes` cost for large graphs.
    pub(crate) fn new_with_archived_graph(
        header: HoloHeader,
        graph_bytes: rkyv::util::AlignedVec<16>,
        weights: Vec<u8>,
        section_table: SectionTable,
        layer_header: Option<LayerHeader>,
    ) -> Self {
        Self {
            header,
            graph: GraphAccess::Archived {
                bytes: graph_bytes,
                cache: OnceLock::new(),
            },
            weights: Cow::Owned(weights),
            section_table,
            layer_header,
        }
    }

    /// Create a LoadedPlan with raw archived graph bytes and borrowed weights.
    ///
    /// # Safety
    /// The caller must ensure the weight bytes outlive this LoadedPlan.
    pub(crate) unsafe fn new_with_archived_graph_borrowed(
        header: HoloHeader,
        graph_bytes: rkyv::util::AlignedVec<16>,
        weights: &[u8],
        section_table: SectionTable,
        layer_header: Option<LayerHeader>,
    ) -> Self {
        let weights_static: &'static [u8] =
            std::slice::from_raw_parts(weights.as_ptr(), weights.len());
        Self {
            header,
            graph: GraphAccess::Archived {
                bytes: graph_bytes,
                cache: OnceLock::new(),
            },
            weights: Cow::Borrowed(weights_static),
            section_table,
            layer_header,
        }
    }

    /// The archive header.
    #[must_use]
    pub fn header(&self) -> &HoloHeader {
        &self.header
    }

    /// The deserialized graph (lazy — deserialized on first call for archived graphs).
    #[must_use]
    pub fn graph(&self) -> &SerializedGraph {
        match &self.graph {
            GraphAccess::Owned(sg) => sg,
            GraphAccess::Archived { bytes, cache } => cache.get_or_init(|| {
                rkyv::from_bytes::<SerializedGraph, rkyv::rancor::Error>(bytes)
                    .expect("archived graph bytes should be valid rkyv")
            }),
        }
    }

    /// Raw weight bytes.
    #[must_use]
    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    /// The section table.
    #[must_use]
    pub fn sections(&self) -> &SectionTable {
        &self.section_table
    }

    /// The layer header with execution entrypoints, if present.
    #[must_use]
    pub fn layer_header(&self) -> Option<&LayerHeader> {
        self.layer_header.as_ref()
    }

    /// Extract the weight index from raw archive bytes.
    ///
    /// The weight index maps each tensor to its byte range in the weight blob
    /// and its layer group. Requires the raw archive bytes since section data
    /// is referenced by offset. Returns `None` if absent or unparseable.
    #[must_use]
    pub fn weight_index_from_bytes(
        &self,
        archive_bytes: &[u8],
    ) -> Option<crate::weight::index::WeightIndex> {
        let entry = self
            .section_table
            .find(crate::section::SECTION_WEIGHT_INDEX)?;
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end > archive_bytes.len() {
            return None;
        }
        crate::weight::index::WeightIndex::from_bytes(&archive_bytes[start..end]).ok()
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph().node_count()
    }

    /// Replace weights with owned data (copies into heap).
    pub(crate) fn set_weights(&mut self, weights: Vec<u8>) {
        self.weights = Cow::Owned(weights);
    }

    /// Replace weights with a borrowed slice (zero-copy from mmap).
    ///
    /// # Safety
    /// The caller must ensure `weights` outlives this LoadedPlan.
    pub unsafe fn set_weights_borrowed(&mut self, weights: &[u8]) {
        let w: &'static [u8] = std::slice::from_raw_parts(weights.as_ptr(), weights.len());
        self.weights = Cow::Borrowed(w);
    }
}
