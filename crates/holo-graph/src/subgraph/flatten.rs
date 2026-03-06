//! Subgraph flattening: template instantiation with ID remapping.
//!
//! Three-phase algorithm:
//! 1. Copy all template nodes (without inputs) into parent, building id_map.
//! 2. Rewire inputs: Node refs via id_map, GraphInput via bindings.
//! 3. Map output NodeIds via id_map.

use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::node::{InputSlot, InputSource, NodeId};
use crate::graph::{Graph, SubgraphId};

/// Result of flattening a subgraph into a parent graph.
#[derive(Debug)]
pub struct FlattenResult {
    /// Maps template NodeIds → parent NodeIds.
    pub id_map: HashMap<NodeId, NodeId>,
    /// Output node IDs in the parent (in port order).
    pub output_ids: Vec<NodeId>,
}

/// Flatten a registered subgraph into the parent graph.
///
/// `input_bindings` maps subgraph input index → parent NodeId.
pub fn flatten_subgraph(
    parent: &mut Graph,
    subgraph_id: SubgraphId,
    input_bindings: &[(u32, NodeId)],
) -> Result<FlattenResult, GraphError> {
    let template = parent
        .get_subgraph(subgraph_id)
        .ok_or(GraphError::InvalidSubgraph(subgraph_id.raw()))?
        .graph
        .clone();

    let binding_map: HashMap<u32, NodeId> = input_bindings.iter().copied().collect();
    let mut id_map = HashMap::new();

    // Phase 1: add all template nodes to parent (no inputs yet).
    for node in template.nodes() {
        let new_id = parent.add_node(node.op.clone());
        id_map.insert(node.id, new_id);
    }

    // Phase 2: rewire inputs using id_map and binding_map.
    for node in template.nodes() {
        let new_id = id_map[&node.id];
        let remapped: Vec<InputSlot> = node
            .inputs
            .iter()
            .map(|slot| remap_input(slot, &id_map, &binding_map))
            .collect();
        if let Some(new_node) = parent.get_mut(new_id) {
            new_node.inputs = remapped;
        }
    }

    // Phase 3: map output NodeIds.
    let output_ids = template
        .outputs()
        .iter()
        .filter_map(|(_, old_id)| id_map.get(old_id).copied())
        .collect();

    Ok(FlattenResult { id_map, output_ids })
}

/// Remap a single InputSlot during flattening.
fn remap_input(
    slot: &InputSlot,
    id_map: &HashMap<NodeId, NodeId>,
    binding_map: &HashMap<u32, NodeId>,
) -> InputSlot {
    match slot.source {
        InputSource::Node(old_id) => {
            if let Some(&new_id) = id_map.get(&old_id) {
                InputSlot::from_node_port(new_id, slot.output_port)
            } else {
                slot.clone()
            }
        }
        InputSource::GraphInput { index } => {
            if let Some(&parent_id) = binding_map.get(&index) {
                InputSlot::from_node_port(parent_id, slot.output_port)
            } else {
                slot.clone()
            }
        }
        InputSource::None => slot.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::graph::GraphOp;
    use crate::subgraph::SubgraphDef;
    use holo_core::op::LutOp;

    /// Build a simple subgraph: GraphInput(0) → Relu → Output
    fn make_relu_subgraph() -> SubgraphDef {
        let inner = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .output("y", 1)
            .build();
        SubgraphDef::new("relu_block", inner)
    }

    #[test]
    fn flatten_simple() {
        let mut g = Graph::new();
        let input_node = g.add_node(GraphOp::Input);
        let sub_id = g.register_subgraph(make_relu_subgraph());
        let result = flatten_subgraph(&mut g, sub_id, &[(0, input_node)]).unwrap();
        assert_eq!(result.output_ids.len(), 1);
        assert_eq!(result.id_map.len(), 2);
        // Parent now has 3 nodes: original Input + 2 from subgraph
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn flatten_preserves_connectivity() {
        let mut g = Graph::new();
        let input_node = g.add_node(GraphOp::Input);
        let sub_id = g.register_subgraph(make_relu_subgraph());
        let result = flatten_subgraph(&mut g, sub_id, &[(0, input_node)]).unwrap();
        let out_id = result.output_ids[0];
        // The output node should depend on the flattened input node
        let preds = g.predecessors(out_id);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn flatten_invalid_subgraph() {
        let mut g = Graph::new();
        let bad_id = SubgraphId::new(99);
        let result = flatten_subgraph(&mut g, bad_id, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn flatten_chain_two() {
        let mut g = Graph::new();
        let input_node = g.add_node(GraphOp::Input);
        let sub_id = g.register_subgraph(make_relu_subgraph());
        // First instantiation
        let r1 = flatten_subgraph(&mut g, sub_id, &[(0, input_node)]).unwrap();
        // Second instantiation chained to first output
        let r2 = flatten_subgraph(&mut g, sub_id, &[(0, r1.output_ids[0])]).unwrap();
        // 1 original + 2 from first + 2 from second = 5
        assert_eq!(g.node_count(), 5);
        assert_eq!(r2.output_ids.len(), 1);
    }
}
