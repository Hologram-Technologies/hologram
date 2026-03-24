//! Compact, rkyv-serializable snapshot of a Graph.
//!
//! `Graph` uses an arena free-list internally, which is a runtime artifact
//! unsuitable for serialization. `SerializedGraph` extracts only live nodes
//! into a dense representation.

use std::collections::HashMap;

use hologram_core::op::FloatDType;
use hologram_graph::constant::{ConstantId, ConstantStore};
use hologram_graph::graph::node::{InputSource, Node, NodeId};
use hologram_graph::graph::GraphOp;
use hologram_graph::Graph;

/// Compact, rkyv-serializable snapshot of a Graph.
///
/// Extracts only live nodes (no free-list gaps) and includes graph I/O
/// metadata and constants. This is the on-disk graph representation.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedGraph {
    /// Dense array of live nodes.
    pub nodes: Vec<Node>,
    /// Named graph inputs.
    pub input_names: Vec<String>,
    /// Named graph outputs.
    pub output_names: Vec<String>,
    /// Output source node IDs (parallel to output_names).
    pub output_node_ids: Vec<NodeId>,
    /// Constant store.
    pub constants: ConstantStore,
    /// N-D shapes for constant nodes (weight matrices).
    ///
    /// Flat list of `(ConstantId, shape)` pairs. Converted to `HashMap` via
    /// `constant_shapes_map()` for efficient lookup. Uses a flat vec rather
    /// than `HashMap` because rkyv's archived key types need `Hash + Eq`.
    pub constant_shapes: Vec<(ConstantId, Vec<usize>)>,
    /// Compiled N-D output shapes per node.
    ///
    /// Flat list of `(NodeId, shape)` pairs. Populated during lowering from
    /// the AI-level IR. Dimensions that are symbolic at compile time use 0
    /// as a sentinel. The executor resolves 0s from actual buffer sizes.
    pub node_shapes: Vec<(NodeId, Vec<usize>)>,
    /// Compiled output dtype per node.
    ///
    /// Flat list of `(NodeId, FloatDType)` pairs. Populated during lowering.
    /// Defaults to F32 when absent. Used by the executor to dispatch
    /// type-aware operations (e.g., i64 shape subgraphs vs f32 tensor data).
    pub node_dtypes: Vec<(NodeId, FloatDType)>,
}

impl SerializedGraph {
    /// Create an empty graph (no nodes, no constants).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            output_node_ids: Vec::new(),
            constants: ConstantStore::new(),
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        }
    }

    /// Create from a live Graph by extracting live nodes.
    ///
    /// If no outputs are explicitly registered via `graph.add_output()`,
    /// auto-detects `GraphOp::Output` nodes and registers them with empty
    /// names. This ensures graphs built without explicit output registration
    /// (common for ONNX imports) still produce output when executed.
    #[must_use]
    pub fn from_graph(graph: &Graph) -> Self {
        let nodes: Vec<Node> = graph.nodes().cloned().collect();
        let input_names: Vec<String> = graph.inputs().to_vec();
        let (mut output_names, mut output_node_ids): (Vec<_>, Vec<_>) =
            graph.outputs().iter().cloned().unzip();

        // Fallback: if no outputs are explicitly registered, auto-detect
        // GraphOp::Output nodes in the graph.
        if output_node_ids.is_empty() {
            for node in &nodes {
                if matches!(node.op, GraphOp::Output) {
                    output_names.push(String::new());
                    output_node_ids.push(node.id);
                }
            }
        }
        let constants = graph.constant_store().clone();
        let constant_shapes: Vec<(ConstantId, Vec<usize>)> = graph
            .constant_shapes()
            .iter()
            .map(|(&k, v)| (k, v.clone()))
            .collect();
        let node_shapes: Vec<(NodeId, Vec<usize>)> = graph
            .node_shapes()
            .iter()
            .map(|(&k, v)| (k, v.clone()))
            .collect();
        let node_dtypes: Vec<(NodeId, FloatDType)> =
            graph.node_dtypes().iter().map(|(&k, &v)| (k, v)).collect();
        Self {
            nodes,
            input_names,
            output_names,
            output_node_ids,
            constants,
            constant_shapes,
            node_shapes,
            node_dtypes,
        }
    }

    /// Number of nodes in the serialized graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Build a `HashMap<NodeId, Vec<usize>>` from the flat node_shapes list.
    #[must_use]
    pub fn node_shapes_map(&self) -> HashMap<NodeId, Vec<usize>> {
        self.node_shapes.iter().cloned().collect()
    }

    /// Build a `HashMap<ConstantId, Vec<usize>>` from the flat constant_shapes list.
    #[must_use]
    pub fn constant_shapes_map(&self) -> HashMap<ConstantId, Vec<usize>> {
        self.constant_shapes.iter().cloned().collect()
    }

    /// Build a `HashMap<NodeId, FloatDType>` from the flat node_dtypes list.
    #[must_use]
    pub fn node_dtypes_map(&self) -> HashMap<NodeId, FloatDType> {
        self.node_dtypes.iter().copied().collect()
    }

    /// Reconstruct a live Graph from this serialized snapshot.
    #[must_use]
    pub fn to_graph(&self) -> Graph {
        let mut graph = Graph::new();
        let id_map = insert_nodes(&mut graph, &self.nodes);
        wire_edges(&mut graph, &self.nodes, &id_map);
        restore_io(&mut graph, self, &id_map);
        // Restore constant shapes.
        for (cid, shape) in &self.constant_shapes {
            graph.set_constant_shape(*cid, shape.clone());
        }
        // Restore node shapes (remapped to new IDs).
        for (old_id, shape) in &self.node_shapes {
            if let Some(&new_id) = id_map.get(old_id) {
                graph.set_node_shape(new_id, shape.clone());
            }
        }
        // Restore node dtypes (remapped to new IDs).
        for &(old_id, dtype) in &self.node_dtypes {
            if let Some(&new_id) = id_map.get(&old_id) {
                graph.set_node_dtype(new_id, dtype);
            }
        }
        graph
    }
}

/// Insert all nodes, building old→new ID mapping.
fn insert_nodes(graph: &mut Graph, nodes: &[Node]) -> HashMap<NodeId, NodeId> {
    nodes
        .iter()
        .map(|n| (n.id, graph.add_node(n.op.clone())))
        .collect()
}

/// Wire edges using remapped IDs.
fn wire_edges(graph: &mut Graph, nodes: &[Node], id_map: &HashMap<NodeId, NodeId>) {
    for node in nodes {
        let new_target = id_map[&node.id];
        for input in &node.inputs {
            if let InputSource::Node(old_src) = input.source {
                if let Some(&new_src) = id_map.get(&old_src) {
                    graph.add_edge(new_src, new_target);
                }
            }
        }
    }
}

/// Restore graph-level I/O metadata.
fn restore_io(graph: &mut Graph, sg: &SerializedGraph, id_map: &HashMap<NodeId, NodeId>) {
    for name in &sg.input_names {
        graph.add_input(name.clone());
    }
    for (name, old_id) in sg.output_names.iter().zip(&sg.output_node_ids) {
        if let Some(&new_id) = id_map.get(old_id) {
            graph.add_output(name.clone(), new_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::LutOp;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::constant::ConstantData;
    use hologram_graph::graph::GraphOp;

    #[test]
    fn from_empty_graph() {
        let g = Graph::new();
        let sg = SerializedGraph::from_graph(&g);
        assert_eq!(sg.node_count(), 0);
        assert!(sg.input_names.is_empty());
        assert!(sg.output_names.is_empty());
    }

    #[test]
    fn from_single_node() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let sg = SerializedGraph::from_graph(&g);
        assert_eq!(sg.node_count(), 1);
    }

    #[test]
    fn from_diamond_graph() {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .output("y", 1)
            .build();
        let sg = SerializedGraph::from_graph(&g);
        assert_eq!(sg.node_count(), 3);
        assert_eq!(sg.input_names, vec!["x"]);
        assert_eq!(sg.output_names, vec!["y"]);
        assert_eq!(sg.output_node_ids.len(), 1);
    }

    #[test]
    fn preserves_constants() {
        let g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![42]))
            .build();
        let sg = SerializedGraph::from_graph(&g);
        assert_eq!(sg.node_count(), 1);
        assert!(!sg.constants.is_empty());
    }

    #[test]
    fn rkyv_round_trip() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .build();
        let sg = SerializedGraph::from_graph(&g);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&sg).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<SerializedGraph>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.nodes.len(), 2);
    }
}
