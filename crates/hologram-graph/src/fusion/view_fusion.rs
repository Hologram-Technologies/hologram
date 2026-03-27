//! View fusion: collapse unary chains into a single `FusedView`.
//!
//! Walks backward from each fusable unary node, composing `ElementWiseView`
//! tables via `then()`. Chains of N ops become a single 256-byte LUT.
//! When two ops compose to the identity (e.g. neg∘neg), the node is replaced
//! with `GraphOp::Passthrough` — a zero-cost identity at execution time.

use smallvec::SmallVec;

use crate::graph::node::NodeId;
use crate::graph::{Graph, GraphOp};
use hologram_core::op::PrimOp;

/// Try to fuse a unary node backward into its predecessor chain.
///
/// If the node and its sole predecessor are both fusable unary ops,
/// compose their views, replace this node with the composed FusedView,
/// and remove the predecessor (if it has no other successors).
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_unary_backward(graph: &mut Graph, id: NodeId, succ_index: &[Vec<NodeId>]) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    let this_view = match node.op.to_view() {
        Some(v) => v,
        None => return false,
    };

    // Need exactly one predecessor — SmallVec<[NodeId; 1]> avoids heap alloc
    let preds: SmallVec<[NodeId; 1]> = node.dependencies().collect();
    if preds.len() != 1 {
        return false;
    }
    let pred_id = preds[0];

    let pred = match graph.get(pred_id) {
        Some(n) => n,
        None => return false,
    };

    let pred_view = match pred.op.to_view() {
        Some(v) => v,
        None => return false,
    };

    // Only fuse if predecessor has exactly one successor (this node).
    let pred_succs = Graph::successors_from_index(pred_id, succ_index);
    if pred_succs.len() != 1 {
        return false;
    }

    // UOR fast path: identical self-inverse (involution) ops cancel to identity.
    // Neg∘Neg = id, Bnot∘Bnot = id — algebraically determined at compile time.
    // Skip the O(256) table composition and memcmp entirely.
    let both_same_involution = {
        let pred_op = graph.get(pred_id).map(|n| &n.op);
        let this_op = graph.get(id).map(|n| &n.op);
        matches!(
            (pred_op, this_op),
            (
                Some(GraphOp::Prim(PrimOp::Neg)),
                Some(GraphOp::Prim(PrimOp::Neg))
            ) | (
                Some(GraphOp::Prim(PrimOp::Bnot)),
                Some(GraphOp::Prim(PrimOp::Bnot))
            )
        )
    };

    // Compose: pred_view.then(this_view) = apply pred first, then this.
    // If the composition is the identity (e.g. neg∘neg, bnot∘bnot), emit
    // Passthrough — a zero-copy identity that costs nothing at runtime.
    // PartialEq on ElementWiseView compiles to a 256-byte memcmp (SIMD).
    if both_same_involution {
        graph.replace_op(id, GraphOp::Passthrough);
    } else {
        let composed = pred_view.then(&this_view);
        if composed == hologram_core::view::ElementWiseView::identity() {
            graph.replace_op(id, GraphOp::Passthrough);
        } else {
            graph.replace_op(id, GraphOp::FusedView(composed));
        }
    }

    // Rewire: this node now takes pred's inputs
    let pred_inputs = graph.get(pred_id).unwrap().inputs.clone();
    if let Some(node) = graph.get_mut(id) {
        node.inputs = pred_inputs;
    }

    graph.remove_node(pred_id);
    true
}

/// Fuse all unary chains in topological order. Returns nodes fused count.
pub fn fuse_unary_chains(graph: &mut Graph, order: &[NodeId]) -> usize {
    let mut fused = 0;
    let succ_index = graph.build_successor_index();
    for &id in order {
        if graph.get(id).is_none() {
            continue;
        }
        while try_fuse_unary_backward(graph, id, &succ_index) {
            fused += 1;
        }
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::schedule::toposort;
    use hologram_core::op::LutOp;

    #[test]
    fn fuse_two_unary() {
        // Input → Sigmoid → Relu → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1]) // 2
            .node_with_inputs(GraphOp::Output, &[2]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert_eq!(count, 1);
        // Sigmoid removed, Relu replaced with FusedView
        assert_eq!(g.node_count(), 3); // Input, FusedView, Output
        let ids = g.node_ids();
        let fused_node = ids
            .iter()
            .find(|&&id| matches!(g.get(id).unwrap().op, GraphOp::FusedView(_)))
            .expect("should have a FusedView node");
        // Verify: FusedView(x) = Relu(Sigmoid(x))
        if let GraphOp::FusedView(v) = &g.get(*fused_node).unwrap().op {
            let sig = LutOp::Sigmoid;
            let relu = LutOp::Relu;
            for x in 0..=255u8 {
                let expected = relu.apply(sig.apply(x));
                assert_eq!(v.apply(x), expected, "mismatch at x={x}");
            }
        }
    }

    #[test]
    fn fuse_three_unary() {
        // Input → Sigmoid → Tanh → Relu → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[1]) // 2
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert_eq!(count, 2);
        assert_eq!(g.node_count(), 3); // Input, FusedView, Output
    }

    #[test]
    fn no_fuse_fan_out() {
        // Input → Sigmoid → [Relu, Tanh]
        // Sigmoid has 2 successors, so it shouldn't be fused into either.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1]) // 2
            .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[1]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert_eq!(count, 0);
        assert_eq!(g.node_count(), 4);
    }

    #[test]
    fn no_fuse_binary_pred() {
        // Two inputs → Add → Relu
        // Add is binary, not fusable unary
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .node_with_inputs(GraphOp::Prim(hologram_core::op::PrimOp::Add), &[0, 1]) // 2
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[2]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert_eq!(count, 0);
    }

    #[test]
    fn fuse_already_fused() {
        // FusedView → Relu should compose
        use hologram_core::view::ElementWiseView;
        let view = ElementWiseView::new(|x| x.wrapping_add(1));
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::FusedView(view), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1]) // 2
            .node_with_inputs(GraphOp::Output, &[2]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert_eq!(count, 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn fuse_double_neg_becomes_passthrough() {
        use hologram_core::op::PrimOp;
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[0]) // 1
            .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[1]) // 2
            .node_with_inputs(GraphOp::Output, &[2]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert!(count >= 1);
        let has_passthrough = g
            .node_ids()
            .iter()
            .any(|&id| matches!(g.get(id).map(|n| &n.op), Some(GraphOp::Passthrough)));
        assert!(has_passthrough, "neg∘neg should fuse to Passthrough");
    }

    #[test]
    fn fuse_double_bnot_becomes_passthrough() {
        use hologram_core::op::PrimOp;
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Prim(PrimOp::Bnot), &[0]) // 1
            .node_with_inputs(GraphOp::Prim(PrimOp::Bnot), &[1]) // 2
            .node_with_inputs(GraphOp::Output, &[2]) // 3
            .build();
        let order = toposort::toposort(&g).unwrap();
        let count = fuse_unary_chains(&mut g, &order);
        assert!(count >= 1);
        let has_passthrough = g
            .node_ids()
            .iter()
            .any(|&id| matches!(g.get(id).map(|n| &n.op), Some(GraphOp::Passthrough)));
        assert!(has_passthrough, "bnot∘bnot should fuse to Passthrough");
    }
}
