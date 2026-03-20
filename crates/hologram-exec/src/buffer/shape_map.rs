//! Tracks logical tensor shapes alongside `BufferArena` byte buffers.
//!
//! The executor stores raw `Vec<u8>` per node. `ShapeMap` records the
//! N-dimensional shape so that ops like Reshape and Transpose can
//! interpret the flat byte buffer correctly.
//!
//! Uses flat `Vec` indexing by `NodeId::index()` for O(1) lookup.

use hologram_graph::graph::node::NodeId;

/// Maps `NodeId` → logical tensor shape (`Vec<usize>`).
///
/// Flat Vec-indexed by `NodeId::index()` for O(1) access without hashing.
#[derive(Debug, Default)]
pub struct ShapeMap {
    shapes: Vec<Option<Vec<usize>>>,
}

impl ShapeMap {
    /// Create an empty shape map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite the shape for a node.
    #[track_caller]
    pub fn insert(&mut self, id: NodeId, shape: Vec<usize>) {
        let idx = id.index() as usize;
        if idx >= self.shapes.len() {
            let new_len = (idx + 1).max(self.shapes.len() * 2);
            self.shapes.resize_with(new_len, || None);
        }
        self.shapes[idx] = Some(shape);
    }

    /// Get the shape for a node, if known.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&[usize]> {
        let idx = id.index() as usize;
        if idx < self.shapes.len() {
            self.shapes[idx].as_deref()
        } else {
            None
        }
    }

    /// Infer a 1-D shape from byte length, assuming f32 (4 bytes per element).
    #[must_use]
    pub fn infer_1d(byte_len: usize) -> Vec<usize> {
        vec![byte_len / 4]
    }

    /// Infer a 1-D shape from byte length using the given element size.
    #[must_use]
    pub fn infer_1d_with_elem_size(byte_len: usize, elem_size: usize) -> Vec<usize> {
        let es = elem_size.max(1);
        vec![byte_len / es]
    }

    /// Clone all shapes into an owned `HashMap`.
    ///
    /// Intended for conformance testing — snapshots the shape state.
    #[cfg(feature = "profile")]
    #[must_use]
    pub fn snapshot(&self) -> std::collections::HashMap<NodeId, Vec<usize>> {
        let mut map = std::collections::HashMap::new();
        for (idx, slot) in self.shapes.iter().enumerate() {
            if let Some(shape) = slot {
                map.insert(NodeId::new(idx as u32, 0), shape.clone());
            }
        }
        map
    }
}
