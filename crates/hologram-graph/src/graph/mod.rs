//! Core graph types: GraphOp, SubgraphId, and the arena-based Graph.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use std::collections::HashMap;

pub mod edge;
pub mod node;
mod op;
pub mod validate;

use crate::constant::{ConstantData, ConstantId, ConstantStore};
use hologram_core::op::FloatDType;
use node::{InputSlot, InputSource, Node, NodeId};
pub use op::{CustomOpId, GraphOp, SubgraphId};

// --- Arena slot ---

/// Arena slot: occupied with a node or free for reuse.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum Slot {
    Occupied(Node),
    Free { next_free: Option<u32> },
}

// --- Graph ---

/// Arena-based compute graph.
///
/// Single type for construction, optimization, and serialization.
/// Replaces v1's separate `OperationGraph` + `CompileGraph`.
#[derive(Debug, Clone)]
pub struct Graph {
    slots: Vec<Slot>,
    generations: Vec<u32>,
    free_head: Option<u32>,
    node_count: usize,
    graph_inputs: Vec<String>,
    graph_outputs: Vec<(String, NodeId)>,
    constants: ConstantStore,
    constant_shapes: HashMap<ConstantId, Vec<usize>>,
    /// Compiled N-D output shapes per node.
    ///
    /// Populated during lowering from the AI-level IR which has complete shape
    /// information. Dimensions that are symbolic at compile time use 0 as a
    /// sentinel. The executor uses these shapes as ground truth, resolving 0s
    /// from actual buffer sizes at runtime.
    node_shapes: HashMap<NodeId, Vec<usize>>,
    /// Compiled output dtype per node.
    ///
    /// Populated during lowering from the AI-level IR. Defaults to F32 when
    /// absent. The executor uses this to dispatch type-aware operations
    /// (e.g., i64 shape subgraphs vs f32 tensor data).
    node_dtypes: HashMap<NodeId, FloatDType>,
    subgraphs: Vec<crate::subgraph::SubgraphDef>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            free_head: None,
            node_count: 0,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            constants: ConstantStore::new(),
            constant_shapes: HashMap::new(),
            node_shapes: HashMap::new(),
            node_dtypes: HashMap::new(),
            subgraphs: Vec::new(),
        }
    }

    /// Create an empty graph with preallocated capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            generations: Vec::with_capacity(cap),
            free_head: None,
            node_count: 0,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            constants: ConstantStore::new(),
            constant_shapes: HashMap::new(),
            node_shapes: HashMap::new(),
            node_dtypes: HashMap::new(),
            subgraphs: Vec::new(),
        }
    }

    // --- Node management ---

    /// Add a node with the given operation, returning its ID.
    pub fn add_node(&mut self, op: GraphOp) -> NodeId {
        let (index, gen) = self.allocate_slot();
        let id = NodeId::new(index, gen);
        self.slots[index as usize] = Slot::Occupied(Node::new(id, op));
        self.node_count += 1;
        id
    }

    /// Remove a node, returning it if it existed.
    pub fn remove_node(&mut self, id: NodeId) -> Option<Node> {
        if !self.is_valid_id(id) {
            return None;
        }
        let idx = id.index() as usize;
        let old = core::mem::replace(
            &mut self.slots[idx],
            Slot::Free {
                next_free: self.free_head,
            },
        );
        self.free_head = Some(id.index());
        self.generations[idx] += 1;
        self.node_count -= 1;
        match old {
            Slot::Occupied(node) => Some(node),
            Slot::Free { .. } => None,
        }
    }

    /// Get an immutable reference to a node.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        if !self.is_valid_id(id) {
            return None;
        }
        match &self.slots[id.index() as usize] {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        }
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        if !self.is_valid_id(id) {
            return None;
        }
        match &mut self.slots[id.index() as usize] {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        }
    }

    /// Whether the graph contains a live node with this ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.get(id).is_some()
    }

    /// Replace a node's operation in place.
    pub fn replace_op(&mut self, id: NodeId, op: GraphOp) -> bool {
        if let Some(node) = self.get_mut(id) {
            node.op = op;
            true
        } else {
            false
        }
    }

    // --- Edges ---

    /// Add an edge: `source` output feeds `target` as a new input.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> bool {
        if !self.contains(source) || !self.contains(target) {
            return false;
        }
        if let Some(node) = self.get_mut(target) {
            node.inputs.push(InputSlot::from_node(source));
            true
        } else {
            false
        }
    }

    /// Predecessor NodeIds of a node.
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        self.get(id)
            .map(|n| n.dependencies().collect())
            .unwrap_or_default()
    }

    /// Successor NodeIds of a node (nodes that use this node as input).
    ///
    /// This performs a full graph scan — O(V+E). For multiple lookups,
    /// use [`build_successor_index`] to build the index once, then
    /// call [`successors_from_index`] for O(degree) per lookup.
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        for node in self.nodes() {
            if node.dependencies().any(|dep| dep == id) {
                result.push(node.id);
            }
        }
        result
    }

    /// Build a reverse-edge index: for each node, the list of its successors.
    ///
    /// Built in O(V+E) via a single pass over all edges. Returns a flat
    /// Vec indexed by `NodeId::index()`. Use [`successors_from_index`]
    /// for O(degree) successor lookups after building this once.
    #[must_use]
    pub fn build_successor_index(&self) -> Vec<Vec<NodeId>> {
        let len = self.slots.len();
        let mut index: Vec<Vec<NodeId>> = Vec::with_capacity(len);
        index.resize_with(len, Vec::new);
        for node in self.nodes() {
            // Deduplicate: if a node lists the same dep twice, only record
            // the successor once. Matches successors() which uses .any().
            let mut seen_deps = Vec::new();
            for dep in node.dependencies() {
                let dep_idx = dep.index() as usize;
                if dep_idx < len && !seen_deps.contains(&dep) {
                    seen_deps.push(dep);
                    index[dep_idx].push(node.id);
                }
            }
        }
        index
    }

    /// Look up successors from a pre-built reverse-edge index. O(degree).
    ///
    /// The index must have been built by [`build_successor_index`] on the
    /// same graph state. Returns an empty slice for unknown or out-of-range IDs.
    #[must_use]
    pub fn successors_from_index(id: NodeId, index: &[Vec<NodeId>]) -> &[NodeId] {
        let idx = id.index() as usize;
        if idx < index.len() {
            &index[idx]
        } else {
            &[]
        }
    }

    /// All edges as (source, target) pairs.
    pub fn edges(&self) -> Vec<(NodeId, NodeId)> {
        let mut result = Vec::new();
        for node in self.nodes() {
            for dep in node.dependencies() {
                result.push((dep, node.id));
            }
        }
        result
    }

    // --- Iteration ---

    /// Iterator over all live nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.slots.iter().filter_map(|s| match s {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        })
    }

    /// Mutable iterator over all live nodes.
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.slots.iter_mut().filter_map(|s| match s {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        })
    }

    /// Collect all live NodeIds.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes().map(|n| n.id).collect()
    }

    /// Number of live nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    // --- I/O ---

    /// Register a named graph input, returning its index.
    pub fn add_input(&mut self, name: impl Into<String>) -> u32 {
        let idx = self.graph_inputs.len() as u32;
        self.graph_inputs.push(name.into());
        idx
    }

    /// Register a named graph output connected to a node.
    pub fn add_output(&mut self, name: impl Into<String>, node: NodeId) {
        self.graph_outputs.push((name.into(), node));
    }

    /// Named graph inputs.
    #[must_use]
    pub fn inputs(&self) -> &[String] {
        &self.graph_inputs
    }

    /// Named graph outputs with their source nodes.
    #[must_use]
    pub fn outputs(&self) -> &[(String, NodeId)] {
        &self.graph_outputs
    }

    /// Source nodes (nodes with no predecessors).
    pub fn sources(&self) -> Vec<NodeId> {
        self.nodes()
            .filter(|n| n.dependencies().next().is_none())
            .map(|n| n.id)
            .collect()
    }

    /// Sink nodes (nodes with no successors).
    ///
    /// Uses a pre-built reverse-edge index for O(V+E) total instead of O(V^2).
    pub fn sinks(&self) -> Vec<NodeId> {
        let succ_index = self.build_successor_index();
        self.nodes()
            .filter(|n| Self::successors_from_index(n.id, &succ_index).is_empty())
            .map(|n| n.id)
            .collect()
    }

    // --- Constants ---

    /// Add a constant and return its ID.
    pub fn add_constant(&mut self, data: ConstantData) -> ConstantId {
        self.constants.insert(data)
    }

    /// Look up a constant by ID.
    #[must_use]
    pub fn get_constant(&self, id: ConstantId) -> Option<&ConstantData> {
        self.constants.get(id)
    }

    /// Replace a constant's data. Returns true if found and replaced.
    pub fn replace_constant(&mut self, id: ConstantId, data: ConstantData) -> bool {
        self.constants.replace(id, data)
    }

    /// Reference to the constant store.
    #[must_use]
    pub fn constant_store(&self) -> &ConstantStore {
        &self.constants
    }

    /// Set the N-D shape for a constant (e.g. weight matrix shape).
    pub fn set_constant_shape(&mut self, id: ConstantId, shape: Vec<usize>) {
        self.constant_shapes.insert(id, shape);
    }

    /// Get the N-D shape for a constant, if recorded.
    #[must_use]
    pub fn constant_shape(&self, id: ConstantId) -> Option<&[usize]> {
        self.constant_shapes.get(&id).map(|v| v.as_slice())
    }

    /// All recorded constant shapes.
    #[must_use]
    pub fn constant_shapes(&self) -> &HashMap<ConstantId, Vec<usize>> {
        &self.constant_shapes
    }

    // --- Node shapes ---

    /// Set the compiled N-D output shape for a node.
    ///
    /// Use 0 for dimensions that are symbolic at compile time (batch, seq_len).
    /// The executor resolves 0s from actual buffer sizes at runtime.
    pub fn set_node_shape(&mut self, id: NodeId, shape: Vec<usize>) {
        self.node_shapes.insert(id, shape);
    }

    /// Get the compiled N-D output shape for a node, if recorded.
    #[must_use]
    pub fn node_shape(&self, id: NodeId) -> Option<&[usize]> {
        self.node_shapes.get(&id).map(|v| v.as_slice())
    }

    /// All recorded node shapes.
    #[must_use]
    pub fn node_shapes(&self) -> &HashMap<NodeId, Vec<usize>> {
        &self.node_shapes
    }

    // --- Node dtypes ---

    /// Set the compiled output dtype for a node.
    pub fn set_node_dtype(&mut self, id: NodeId, dtype: FloatDType) {
        self.node_dtypes.insert(id, dtype);
    }

    /// Get the compiled output dtype for a node, if recorded.
    #[must_use]
    pub fn node_dtype(&self, id: NodeId) -> Option<FloatDType> {
        self.node_dtypes.get(&id).copied()
    }

    /// All recorded node dtypes.
    #[must_use]
    pub fn node_dtypes(&self) -> &HashMap<NodeId, FloatDType> {
        &self.node_dtypes
    }

    // --- Subgraphs ---

    /// Register a subgraph template, returning its ID.
    pub fn register_subgraph(&mut self, def: crate::subgraph::SubgraphDef) -> SubgraphId {
        let id = SubgraphId(self.subgraphs.len() as u32);
        self.subgraphs.push(def);
        id
    }

    /// Look up a subgraph by ID.
    #[must_use]
    pub fn get_subgraph(&self, id: SubgraphId) -> Option<&crate::subgraph::SubgraphDef> {
        self.subgraphs.get(id.0 as usize)
    }

    // --- Rewire ---

    /// Rewire all successors of `old` to point to `new` instead.
    pub fn rewire_successors(&mut self, old: NodeId, new: NodeId) {
        for slot in &mut self.slots {
            if let Slot::Occupied(node) = slot {
                for input in &mut node.inputs {
                    if input.source == InputSource::Node(old) {
                        input.source = InputSource::Node(new);
                    }
                }
            }
        }
    }

    /// Rewire successors of `old` to point to `new`, using a pre-built index.
    ///
    /// Only visits actual successors from the index instead of scanning all slots.
    /// O(degree) instead of O(V×E).
    pub fn rewire_successors_indexed(
        &mut self,
        old: NodeId,
        new: NodeId,
        succ_index: &[Vec<NodeId>],
    ) {
        let successors: Vec<NodeId> = Self::successors_from_index(old, succ_index).to_vec();
        for succ_id in successors {
            if let Some(node) = self.get_mut(succ_id) {
                for input in &mut node.inputs {
                    if input.source == InputSource::Node(old) {
                        input.source = InputSource::Node(new);
                    }
                }
            }
        }
    }

    // --- Private ---

    /// Allocate or reuse a slot, returning (index, generation).
    fn allocate_slot(&mut self) -> (u32, u32) {
        if let Some(free_idx) = self.free_head {
            let idx = free_idx as usize;
            if let Slot::Free { next_free } = &self.slots[idx] {
                self.free_head = *next_free;
            }
            (free_idx, self.generations[idx])
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot::Free { next_free: None });
            self.generations.push(0);
            (idx, 0)
        }
    }

    /// Check if a NodeId is valid (correct generation).
    fn is_valid_id(&self, id: NodeId) -> bool {
        let idx = id.index() as usize;
        idx < self.slots.len() && self.generations[idx] == id.generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::LutOp;

    #[test]
    fn empty_graph() {
        let g = Graph::new();
        assert!(g.is_empty());
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn add_and_get() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Input);
        assert!(g.contains(id));
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.get(id).unwrap().op, GraphOp::Input);
    }

    #[test]
    fn remove_and_stale() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Input);
        g.remove_node(id);
        assert!(!g.contains(id)); // stale
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn slot_reuse() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        g.remove_node(a);
        let b = g.add_node(GraphOp::Output);
        // Same slot index, different generation
        assert_eq!(b.index(), a.index());
        assert_ne!(b.generation(), a.generation());
        assert!(!g.contains(a));
        assert!(g.contains(b));
    }

    #[test]
    fn edges() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        assert!(g.add_edge(a, b));
        assert_eq!(g.predecessors(b), alloc::vec![a]);
        assert_eq!(g.successors(a), alloc::vec![b]);
    }

    #[test]
    fn named_io() {
        let mut g = Graph::new();
        let idx = g.add_input("x");
        assert_eq!(idx, 0);
        let node = g.add_node(GraphOp::Output);
        g.add_output("y", node);
        assert_eq!(g.inputs().len(), 1);
        assert_eq!(g.outputs().len(), 1);
    }

    #[test]
    fn constants() {
        let mut g = Graph::new();
        let cid = g.add_constant(ConstantData::Bytes(alloc::vec![42]));
        assert_eq!(
            g.get_constant(cid),
            Some(&ConstantData::Bytes(alloc::vec![42]))
        );
    }

    #[test]
    fn sources_and_sinks() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        g.add_edge(a, b);
        assert!(g.sources().contains(&a));
        assert!(g.sinks().contains(&b));
    }

    #[test]
    fn rewire_successors() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Input);
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        g.add_edge(a, c);
        g.rewire_successors(a, b);
        assert_eq!(g.predecessors(c), alloc::vec![b]);
    }

    #[test]
    fn replace_op() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Lut(LutOp::Relu));
        assert!(g.replace_op(id, GraphOp::Lut(LutOp::Sigmoid)));
        assert_eq!(g.get(id).unwrap().op, GraphOp::Lut(LutOp::Sigmoid));
    }

    #[test]
    fn successor_index() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);

        let index = g.build_successor_index();
        assert_eq!(Graph::successors_from_index(a, &index), &[b]);
        assert_eq!(Graph::successors_from_index(b, &index), &[c]);
        assert!(Graph::successors_from_index(c, &index).is_empty());
    }
}
