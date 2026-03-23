//! Loaded and validated archive plan.

use crate::entrypoint::schedule::LayerHeader;
use crate::format::graph::SerializedGraph;
use crate::format::header::HoloHeader;
use crate::section::table::SectionTable;
use std::borrow::Cow;

/// A loaded and validated archive.
///
/// Provides access to the deserialized graph, raw weight bytes,
/// the section table, and the layer header (if present).
///
/// Weight bytes are borrowed when loaded from mmap (zero-copy) or
/// owned when loaded from a network buffer or compressed archive.
pub struct LoadedPlan {
    header: HoloHeader,
    graph: SerializedGraph,
    weights: Cow<'static, [u8]>,
    section_table: SectionTable,
    layer_header: Option<LayerHeader>,
}

impl LoadedPlan {
    /// Create a new LoadedPlan with owned weight bytes.
    pub(crate) fn new(
        header: HoloHeader,
        graph: SerializedGraph,
        weights: Vec<u8>,
        section_table: SectionTable,
        layer_header: Option<LayerHeader>,
    ) -> Self {
        Self {
            header,
            graph,
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
        // Extend lifetime to 'static — the caller guarantees the backing
        // storage (mmap or Vec) outlives this LoadedPlan.
        let weights_static: &'static [u8] =
            std::slice::from_raw_parts(weights.as_ptr(), weights.len());
        Self {
            header,
            graph,
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

    /// The deserialized graph.
    #[must_use]
    pub fn graph(&self) -> &SerializedGraph {
        &self.graph
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

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Replace weights (used by pipeline loader for weight dedup resolution).
    pub(crate) fn set_weights(&mut self, weights: Vec<u8>) {
        self.weights = Cow::Owned(weights);
    }
}
