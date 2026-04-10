//! Edge connection helpers.

use super::node::{InputSlot, NodeId};
use super::Graph;

/// Connect source output to target at a specific input index.
///
/// If the input index doesn't exist yet, slots are appended up to it.
pub fn connect(graph: &mut Graph, source: NodeId, target: NodeId, slot: usize) {
    if let Some(node) = graph.get_mut(target) {
        while node.inputs.len() <= slot {
            node.inputs.push(InputSlot::default());
        }
        node.inputs[slot] = InputSlot::from_node(source);
    }
}

/// Connect a graph-level input to a target node's input slot.
pub fn connect_graph_input(graph: &mut Graph, input_idx: u32, target: NodeId, slot: usize) {
    if let Some(node) = graph.get_mut(target) {
        while node.inputs.len() <= slot {
            node.inputs.push(InputSlot::default());
        }
        node.inputs[slot] = InputSlot::from_graph_input(input_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use hologram_core::op::LutOp;

    #[test]
    fn connect_two_nodes() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        connect(&mut g, a, b, 0);
        assert_eq!(g.predecessors(b), vec![a]);
    }

    #[test]
    fn connect_graph_input_to_node() {
        let mut g = Graph::new();
        g.add_input("x");
        let b = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        connect_graph_input(&mut g, 0, b, 0);
        let node = g.get(b).unwrap();
        assert!(!node.inputs[0].is_empty());
    }

    #[test]
    fn connect_grows_inputs() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Output);
        connect(&mut g, a, b, 3); // slot 3, skipping 0-2
        assert_eq!(g.get(b).unwrap().inputs.len(), 4);
    }
}
