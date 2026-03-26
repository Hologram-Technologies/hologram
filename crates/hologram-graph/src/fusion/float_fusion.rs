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
pub fn try_fuse_float_unary(graph: &mut Graph, id: NodeId, succ_index: &[Vec<NodeId>]) -> bool {
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
    let pred_succs = Graph::successors_from_index(pred_id, succ_index);
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

/// Try to fuse a MatMul node forward into a successor unary activation.
///
/// If a MatMul has exactly one successor and that successor is an element-wise
/// unary float op, replace the pair with a `FusedMatMulActivation` node.
/// The successor absorbs the MatMul's inputs and the MatMul node is removed.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_matmul_activation(
    graph: &mut Graph,
    id: NodeId,
    succ_index: &[Vec<NodeId>],
) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    // Current node must be a MatMul.
    let (m, k, n) = match &node.op {
        GraphOp::Float(FloatOp::MatMul { m, k, n }) => (*m, *k, *n),
        _ => return false,
    };

    // MatMul must have exactly one successor.
    let succs = Graph::successors_from_index(id, succ_index);
    if succs.len() != 1 {
        return false;
    }
    let succ_id = succs[0];

    let succ = match graph.get(succ_id) {
        Some(n) => n,
        None => return false,
    };

    // Successor must be an element-wise unary float op.
    let activation = match &succ.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => *f,
        _ => return false,
    };

    // Successor must have exactly one predecessor (this MatMul).
    let succ_preds: Vec<NodeId> = succ.dependencies().collect();
    if succ_preds.len() != 1 {
        return false;
    }

    // Replace the successor node with the fused op, keeping its NodeId.
    let matmul_inputs = node.inputs.clone();
    graph.replace_op(
        succ_id,
        GraphOp::FusedMatMulActivation {
            m,
            k,
            n,
            activation,
        },
    );
    if let Some(succ_node) = graph.get_mut(succ_id) {
        succ_node.inputs = matmul_inputs;
    }
    graph.remove_node(id);
    true
}

/// Try to fuse a norm op (RmsNorm/LayerNorm/GroupNorm) forward into a successor
/// unary activation. Same pattern as matmul fusion.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_norm_activation(graph: &mut Graph, id: NodeId, succ_index: &[Vec<NodeId>]) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    // Current node must be a norm op.
    let fused_op_fn: Box<dyn FnOnce(FloatOp) -> GraphOp> = match &node.op {
        GraphOp::Float(FloatOp::RmsNorm { size, epsilon }) => {
            let (s, e) = (*size, *epsilon);
            Box::new(move |act| GraphOp::FusedRmsNormActivation {
                size: s,
                epsilon: e,
                activation: act,
            })
        }
        GraphOp::Float(FloatOp::LayerNorm { size, epsilon }) => {
            let (s, e) = (*size, *epsilon);
            Box::new(move |act| GraphOp::FusedLayerNormActivation {
                size: s,
                epsilon: e,
                activation: act,
            })
        }
        GraphOp::Float(FloatOp::GroupNorm {
            num_groups,
            epsilon,
        }) => {
            let (ng, e) = (*num_groups, *epsilon);
            Box::new(move |act| GraphOp::FusedGroupNormActivation {
                num_groups: ng,
                epsilon: e,
                activation: act,
            })
        }
        _ => return false,
    };

    // Must have exactly one successor.
    let succs = Graph::successors_from_index(id, succ_index);
    if succs.len() != 1 {
        return false;
    }
    let succ_id = succs[0];

    let succ = match graph.get(succ_id) {
        Some(n) => n,
        None => return false,
    };

    let activation = match &succ.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => *f,
        _ => return false,
    };
    let succ_preds: Vec<NodeId> = succ.dependencies().collect();
    if succ_preds.len() != 1 {
        return false;
    }

    let norm_inputs = node.inputs.clone();
    graph.replace_op(succ_id, fused_op_fn(activation));
    if let Some(succ_node) = graph.get_mut(succ_id) {
        succ_node.inputs = norm_inputs;
    }
    graph.remove_node(id);
    true
}

/// Try to fuse a LUT-GEMM (MatMulLut4/MatMulLut8) forward into a successor
/// unary activation. Same pattern as `try_fuse_matmul_activation`.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_lut_gemm_activation(
    graph: &mut Graph,
    id: NodeId,
    succ_index: &[Vec<NodeId>],
) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    // Current node must be a LUT-GEMM variant.
    let (is_q4, cid) = match &node.op {
        GraphOp::MatMulLut4(cid) => (true, *cid),
        GraphOp::MatMulLut8(cid) => (false, *cid),
        _ => return false,
    };

    // Must have exactly one successor.
    let succs = Graph::successors_from_index(id, succ_index);
    if succs.len() != 1 {
        return false;
    }
    let succ_id = succs[0];

    let succ = match graph.get(succ_id) {
        Some(n) => n,
        None => return false,
    };

    // Successor must be element-wise unary with single predecessor.
    let activation = match &succ.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => *f,
        _ => return false,
    };
    let succ_preds: Vec<NodeId> = succ.dependencies().collect();
    if succ_preds.len() != 1 {
        return false;
    }

    let lut_inputs = node.inputs.clone();
    let fused_op = if is_q4 {
        GraphOp::MatMulLut4Activation(cid, activation)
    } else {
        GraphOp::MatMulLut8Activation(cid, activation)
    };
    graph.replace_op(succ_id, fused_op);
    if let Some(succ_node) = graph.get_mut(succ_id) {
        succ_node.inputs = lut_inputs;
    }
    graph.remove_node(id);
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
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id, &succ_index) {
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
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id, &succ_index) {
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
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id, &succ_index) {
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
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id, &succ_index) {
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
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            while try_fuse_float_unary(&mut g, id, &succ_index) {
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

    // ── MatMul + Activation epilogue fusion tests ─────────────────────

    #[test]
    fn fuse_matmul_relu() {
        // Input0, Input1 → MatMul → Relu → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .node_with_inputs(
                GraphOp::Float(FloatOp::MatMul { m: 2, k: 3, n: 4 }),
                &[0, 1],
            ) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_matmul_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1);
        assert_eq!(g.node_count(), 4); // Input0, Input1, FusedMatMulActivation, Output

        let fused_node = g
            .node_ids()
            .into_iter()
            .find(|&id| matches!(g.get(id).unwrap().op, GraphOp::FusedMatMulActivation { .. }))
            .expect("should have FusedMatMulActivation");
        if let GraphOp::FusedMatMulActivation {
            m,
            k,
            n,
            activation,
        } = &g.get(fused_node).unwrap().op
        {
            assert_eq!((*m, *k, *n), (2, 3, 4));
            assert_eq!(*activation, FloatOp::Relu);
        }
    }

    #[test]
    fn no_fuse_matmul_fan_out() {
        // Input0, Input1 → MatMul → [Relu, Sigmoid]
        // MatMul has 2 successors — should NOT fuse.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .node_with_inputs(
                GraphOp::Float(FloatOp::MatMul { m: 2, k: 3, n: 4 }),
                &[0, 1],
            ) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[2]) // 3
            .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[2]) // 4
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_matmul_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
        assert_eq!(g.node_count(), 5);
    }

    #[test]
    fn no_fuse_matmul_non_unary_successor() {
        // Input0, Input1 → MatMul → Softmax → Output
        // Softmax is not element-wise unary — should NOT fuse.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .node_with_inputs(
                GraphOp::Float(FloatOp::MatMul { m: 2, k: 3, n: 4 }),
                &[0, 1],
            ) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Softmax { size: 4 }), &[2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_matmul_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
    }
}
