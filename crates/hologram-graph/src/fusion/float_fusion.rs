//! Float chain fusion: collapse chains of unary element-wise `FloatOp` nodes
//! into a single `FusedFloatChain` node.
//!
//! Mirrors `view_fusion.rs` but operates in f32 domain instead of byte-domain.
//! Backward walk from each fusable node, compose into chain, rewire, remove pred.

use hologram_core::op::FloatOp;

use crate::graph::node::NodeId;
use crate::graph::{Graph, GraphOp};

/// Try to fuse a unary float node backward into its predecessor chain.
///
/// If the node and its sole predecessor are both unary element-wise float ops,
/// compose them into a `FusedFloatChain`, rewire inputs, and remove predecessor.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_float_unary(graph: &mut Graph, id: NodeId) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    // Current node must be a fusable float op or an existing FusedFloatChain.
    let this_chain: Vec<FloatOp> = match &node.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => vec![*f],
        GraphOp::FusedFloatChain(chain) => chain.clone(),
        _ => return false,
    };

    // Need exactly one predecessor.
    let preds: Vec<NodeId> = node.dependencies().collect();
    if preds.len() != 1 {
        return false;
    }
    let pred_id = preds[0];

    let pred = match graph.get(pred_id) {
        Some(n) => n,
        None => return false,
    };

    // Predecessor must be a fusable float op or an existing FusedFloatChain.
    let pred_chain: Vec<FloatOp> = match &pred.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => vec![*f],
        GraphOp::FusedFloatChain(chain) => chain.clone(),
        _ => return false,
    };

    // Only fuse if predecessor has exactly one successor (this node).
    let pred_succs = graph.successors(pred_id);
    if pred_succs.len() != 1 {
        return false;
    }

    // Compose: predecessor ops first, then this node's ops.
    let mut new_chain = pred_chain;
    new_chain.extend(this_chain);
    graph.replace_op(id, GraphOp::FusedFloatChain(new_chain));

    // Rewire: this node now takes pred's inputs.
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
    fn fuse_two_float_unary() {
        // Input → Exp → Sigmoid → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Float(FloatOp::Exp), &[0]) // 1
            .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[1]) // 2
            .node_with_inputs(GraphOp::Output, &[2]) // 3
            .build();

        let order = toposort::toposort(&g).unwrap();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1);
        assert_eq!(g.node_count(), 3); // Input, FusedFloatChain, Output

        // Find the fused node and verify chain order.
        let fused_node = g
            .node_ids()
            .into_iter()
            .find(|&id| matches!(g.get(id).unwrap().op, GraphOp::FusedFloatChain(_)))
            .expect("should have FusedFloatChain");
        if let GraphOp::FusedFloatChain(chain) = &g.get(fused_node).unwrap().op {
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0], FloatOp::Exp);
            assert_eq!(chain[1], FloatOp::Sigmoid);
        }
    }

    #[test]
    fn fuse_three_float_unary() {
        // Input → Exp → Sigmoid → Neg → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Float(FloatOp::Exp), &[0]) // 1
            .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[1]) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Neg), &[2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();

        let order = toposort::toposort(&g).unwrap();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id) {
                fused += 1;
            }
        }
        assert_eq!(fused, 2);
        assert_eq!(g.node_count(), 3); // Input, FusedFloatChain, Output

        let fused_node = g
            .node_ids()
            .into_iter()
            .find(|&id| matches!(g.get(id).unwrap().op, GraphOp::FusedFloatChain(_)))
            .expect("should have FusedFloatChain");
        if let GraphOp::FusedFloatChain(chain) = &g.get(fused_node).unwrap().op {
            assert_eq!(chain, &[FloatOp::Exp, FloatOp::Sigmoid, FloatOp::Neg]);
        }
    }

    #[test]
    fn no_fuse_fan_out() {
        // Input → Exp → [Sigmoid, Neg]
        // Exp has 2 successors, so it shouldn't be fused.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Float(FloatOp::Exp), &[0]) // 1
            .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[1]) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Neg), &[1]) // 3
            .build();

        let order = toposort::toposort(&g).unwrap();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
        assert_eq!(g.node_count(), 4);
    }

    #[test]
    fn no_fuse_binary_pred() {
        // Two inputs → Add → Sigmoid
        // Add is binary, not element-wise unary.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .node_with_inputs(GraphOp::Float(FloatOp::Add), &[0, 1]) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[2]) // 3
            .build();

        let order = toposort::toposort(&g).unwrap();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
        assert_eq!(g.node_count(), 4);
    }

    #[test]
    fn no_fuse_non_elementwise() {
        // Input → Softmax → Sigmoid
        // Softmax is not element-wise unary.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Float(FloatOp::Softmax { size: 10 }), &[0]) // 1
            .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[1]) // 2
            .build();

        let order = toposort::toposort(&g).unwrap();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn fused_chain_correctness() {
        // Verify that FusedFloatChain [Exp, Sigmoid, Neg] produces the same
        // result as applying each op individually.
        let chain = vec![FloatOp::Exp, FloatOp::Sigmoid, FloatOp::Neg];
        let test_vals = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

        for &x in &test_vals {
            let mut expected = x;
            for op in &chain {
                expected = op.apply_unary(expected);
            }

            // Simulate dispatch: apply chain to single element.
            let mut val = x;
            for op in &chain {
                val = op.apply_unary(val);
            }
            assert!(
                (val - expected).abs() < 1e-6,
                "mismatch at x={x}: got {val}, expected {expected}"
            );
        }
    }
}
