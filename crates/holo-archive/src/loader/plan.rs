//! Loaded and validated archive plan.

use crate::format::graph::SerializedGraph;
use crate::format::header::HoloHeader;
use crate::section::table::SectionTable;

/// A loaded and validated archive.
///
/// Provides access to the deserialized graph, raw weight bytes,
/// and the section table for locating custom sections.
pub struct LoadedPlan {
    header: HoloHeader,
    graph: SerializedGraph,
    weights: Vec<u8>,
    section_table: SectionTable,
}

impl LoadedPlan {
    /// Create a new LoadedPlan (crate-internal).
    pub(crate) fn new(
        header: HoloHeader,
        graph: SerializedGraph,
        weights: Vec<u8>,
        section_table: SectionTable,
    ) -> Self {
        Self {
            header,
            graph,
            weights,
            section_table,
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

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
}
