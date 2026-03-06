//! Compact, rkyv-serializable snapshot of a Graph.
//!
//! `Graph` uses an arena free-list internally, which is a runtime artifact
//! unsuitable for serialization. `SerializedGraph` extracts only live nodes
//! into a dense representation.

use holo_graph::constant::ConstantStore;
use holo_graph::graph::node::{Node, NodeId};
use holo_graph::Graph;

/// Compact, rkyv-serializable snapshot of a Graph.
///
/// Extracts only live nodes (no free-list gaps) and includes graph I/O
/// metadata and constants. This is the on-disk graph representation.
#[derive(
    Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
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
}

impl SerializedGraph {
    /// Create from a live Graph by extracting live nodes.
    #[must_use]
    pub fn from_graph(graph: &Graph) -> Self {
        let nodes: Vec<Node> = graph.nodes().cloned().collect();
        let input_names: Vec<String> = graph.inputs().to_vec();
        let (output_names, output_node_ids): (Vec<_>, Vec<_>) =
            graph.outputs().iter().cloned().unzip();
        let constants = graph.constant_store().clone();
        Self {
            nodes,
            input_names,
            output_names,
            output_node_ids,
            constants,
        }
    }

    /// Number of nodes in the serialized graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use holo_graph::builder::GraphBuilder;
    use holo_graph::constant::ConstantData;
    use holo_graph::graph::GraphOp;
    use holo_core::op::LutOp;

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
        let bytes = rkyv::to_bytes::<_, 1024>(&sg).unwrap();
        let archived =
            rkyv::check_archived_root::<SerializedGraph>(&bytes).unwrap();
        assert_eq!(archived.nodes.len(), 2);
    }
}
