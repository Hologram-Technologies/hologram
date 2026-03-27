//! Q1 view fusion: collapse Q1 unary chains into a single `FusedView16`.
//!
//! Mirrors `view_fusion.rs` for the Q1 (16-bit) domain. Walks backward from
//! each Q1-fusable unary node, composing `ElementWiseView16` tables via `then()`.
//! Chains of N Q1 ops become a single 128KB LUT, materialized once at compile time.
//! When two Q1 involutions compose to the identity (Neg∘Neg, Bnot∘Bnot), the node
//! is replaced with `GraphOp::Passthrough`.

use smallvec::SmallVec;

use crate::graph::node::NodeId;
use crate::graph::{Graph, GraphOp};
use hologram_core::op::{PrimOp, RingLevel};

/// Try to fuse a Q1 unary node backward into its predecessor chain.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_q1_unary_backward(
    graph: &mut Graph,
    id: NodeId,
    succ_index: &[Vec<NodeId>],
) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    let this_view = match node.op.to_view16() {
        Some(v) => v,
        None => return false,
    };

    let preds: SmallVec<[NodeId; 1]> = node.dependencies().collect();
    if preds.len() != 1 {
        return false;
    }
    let pred_id = preds[0];

    let pred = match graph.get(pred_id) {
        Some(n) => n,
        None => return false,
    };

    let pred_view = match pred.op.to_view16() {
        Some(v) => v,
        None => return false,
    };

    let pred_succs = Graph::successors_from_index(pred_id, succ_index);
    if pred_succs.len() != 1 {
        return false;
    }

    // UOR fast path: identical Q1 involutions cancel to identity.
    let both_same_involution = {
        let pred_op = graph.get(pred_id).map(|n| &n.op);
        let this_op = graph.get(id).map(|n| &n.op);
        matches!(
            (pred_op, this_op),
            (
                Some(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1)),
                Some(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1))
            ) | (
                Some(GraphOp::RingPrimUnary(PrimOp::Bnot, RingLevel::Q1)),
                Some(GraphOp::RingPrimUnary(PrimOp::Bnot, RingLevel::Q1))
            )
        )
    };

    if both_same_involution {
        graph.replace_op(id, GraphOp::Passthrough);
    } else {
        let composed = pred_view.then(&this_view);
        if composed == hologram_core::q1::view::ElementWiseView16::identity() {
            graph.replace_op(id, GraphOp::Passthrough);
        } else {
            graph.replace_op(id, GraphOp::FusedView16(Box::new(composed)));
        }
    }

    let pred_inputs = graph.get(pred_id).unwrap().inputs.clone();
    if let Some(node) = graph.get_mut(id) {
        node.inputs = pred_inputs;
    }

    graph.remove_node(pred_id);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::schedule::toposort;

    #[test]
    fn fuse_q1_double_neg_passthrough() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1), &[0])
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_q1_unary_backward(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert!(fused >= 1, "should fuse Q1 Neg∘Neg");
        let has_passthrough = g
            .node_ids()
            .iter()
            .any(|&id| matches!(g.get(id).map(|n| &n.op), Some(GraphOp::Passthrough)));
        assert!(has_passthrough, "Q1 Neg∘Neg should become Passthrough");
    }

    #[test]
    fn fuse_q1_double_bnot_passthrough() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Bnot, RingLevel::Q1), &[0])
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Bnot, RingLevel::Q1), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_q1_unary_backward(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert!(fused >= 1);
        let has_passthrough = g
            .node_ids()
            .iter()
            .any(|&id| matches!(g.get(id).map(|n| &n.op), Some(GraphOp::Passthrough)));
        assert!(has_passthrough, "Q1 Bnot∘Bnot should become Passthrough");
    }

    #[test]
    fn fuse_q1_neg_bnot_composed() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1), &[0])
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Bnot, RingLevel::Q1), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_q1_unary_backward(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1, "should fuse Q1 Neg+Bnot");
        let has_fused_view16 = g
            .node_ids()
            .iter()
            .any(|&id| matches!(g.get(id).map(|n| &n.op), Some(GraphOp::FusedView16(_))));
        assert!(has_fused_view16, "Neg∘Bnot should produce FusedView16");
    }

    #[test]
    fn no_cross_level_fusion() {
        use hologram_core::op::LutOp;
        // Q0 Lut followed by Q1 RingPrimUnary — must NOT fuse
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_q1_unary_backward(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        // Q0 Lut has no to_view16(), so Q1 fusion should not absorb it
        assert_eq!(fused, 0, "cross-level fusion must not occur");
    }
}
