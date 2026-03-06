//! Common subexpression elimination (CSE).
//!
//! Hash-based dedup: nodes with identical (op, sorted predecessors) share results.

use std::collections::HashMap;

use crate::graph::node::NodeId;
use crate::graph::Graph;

/// Signature for CSE: (op hash-eq key, sorted predecessor ids).
#[derive(Hash, PartialEq, Eq)]
struct NodeSignature {
    op: crate::graph::GraphOp,
    preds: Vec<NodeId>,
}

/// Eliminate common subexpressions in the graph.
///
/// For each node in the given topological order, if a node with the same
/// (op, sorted_preds) already exists, rewire all successors to the canonical
/// node and remove the duplicate. Returns the number of nodes eliminated.
pub fn eliminate_common_subexpressions(graph: &mut Graph, order: &[NodeId]) -> usize {
    let mut canonical: HashMap<NodeSignature, NodeId> = HashMap::new();
    let mut eliminated = 0;

    for &id in order {
        let node = match graph.get(id) {
            Some(n) => n,
            None => continue,
        };

        if !node.op.is_pure() {
            continue;
        }

        let op = node.op.clone();
        let mut preds: Vec<NodeId> = node.dependencies().collect();
        preds.sort_by_key(|n| (n.index(), n.generation()));

        let sig = NodeSignature { op, preds };

        if let Some(&canon_id) = canonical.get(&sig) {
            if canon_id != id {
                graph.rewire_successors(id, canon_id);
                graph.remove_node(id);
                eliminated += 1;
            }
        } else {
            canonical.insert(sig, id);
        }
    }

    eliminated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::graph::GraphOp;
    use crate::schedule::toposort;
    use holo_core::op::{LutOp, PrimOp};

    #[test]
    fn dedup_identical_ops() {
        // Input → Relu (a), Input → Relu (b) → Add(a, b)
        // Both Relus have same (op, pred), so one should be eliminated.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 2 (dup of 1)
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = eliminate_common_subexpressions(&mut g, &order);
        assert_eq!(count, 1);
        // 3 live nodes remain: Input, Relu, Add
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn no_dedup_different_ops() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 2
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = eliminate_common_subexpressions(&mut g, &order);
        assert_eq!(count, 0);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn no_dedup_impure() {
        // Input and Output are not pure, so they should not be deduped.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = eliminate_common_subexpressions(&mut g, &order);
        assert_eq!(count, 0);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn dedup_multiple() {
        // 3 identical Relu nodes from same input → 2 eliminated
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 2
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = eliminate_common_subexpressions(&mut g, &order);
        assert_eq!(count, 2);
        assert_eq!(g.node_count(), 2);
    }
}
