//! Tracks logical tensor shapes alongside `BufferArena` byte buffers.
//!
//! The executor stores raw `Vec<u8>` per node. `ShapeMap` records the
//! N-dimensional shape so that ops like Reshape and Transpose can
//! interpret the flat byte buffer correctly.

use hologram_graph::graph::node::NodeId;
use std::collections::HashMap;

/// Maps `NodeId` → logical tensor shape (`Vec<usize>`).
#[derive(Debug, Default)]
pub struct ShapeMap {
    shapes: HashMap<NodeId, Vec<usize>>,
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
        self.shapes.insert(id, shape);
    }

    /// Get the shape for a node, if known.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&[usize]> {
        self.shapes.get(&id).map(|v| v.as_slice())
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
}
