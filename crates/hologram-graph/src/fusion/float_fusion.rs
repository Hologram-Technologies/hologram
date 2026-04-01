//! Float chain fusion: collapse chains of unary element-wise `FloatOp` nodes
//! into a single `FusedFloatChain` node.
//!
//! Mirrors `view_fusion.rs` but operates in f32 domain instead of byte-domain.
//! Backward walk from each fusable node, compose into chain, rewire, remove pred.

use hologram_core::op::FloatOp;

use crate::graph::node::{InputSource, NodeId};
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

/// Try to fuse a MatMul → Add(constant bias) → Activation into a single
/// `FusedMatMulBiasActivation` node. This 3-node pattern appears in every
/// Linear+Activation layer and eliminates two intermediate buffers.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_matmul_bias_activation(
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

    // MatMul must have exactly one successor (the Add).
    let succs = Graph::successors_from_index(id, succ_index);
    if succs.len() != 1 {
        return false;
    }
    let add_id = succs[0];

    let add_node = match graph.get(add_id) {
        Some(n) => n,
        None => return false,
    };

    // Successor must be Float(Add).
    if !matches!(&add_node.op, GraphOp::Float(FloatOp::Add)) {
        return false;
    }

    // Add must have exactly 2 inputs. One is the MatMul, the other must be a Constant (bias).
    if add_node.inputs.len() != 2 {
        return false;
    }
    // Find the bias constant node ID (the Add input that isn't the MatMul).
    let bias_node_id = {
        let mut found = None;
        for slot in &add_node.inputs {
            if let InputSource::Node(pred_id) = &slot.source {
                if *pred_id == id {
                    continue; // This is the MatMul input, skip.
                }
                // Check if this predecessor is a Constant node.
                if let Some(pred_node) = graph.get(*pred_id) {
                    if matches!(&pred_node.op, GraphOp::Constant(_)) {
                        found = Some(*pred_id);
                    }
                }
            }
        }
        match found {
            Some(nid) => nid,
            None => return false, // No constant bias found.
        }
    };

    // Add must have exactly one successor (the Activation).
    let add_succs = Graph::successors_from_index(add_id, succ_index);
    if add_succs.len() != 1 {
        return false;
    }
    let act_id = add_succs[0];

    let act_node = match graph.get(act_id) {
        Some(n) => n,
        None => return false,
    };

    // Activation must be element-wise unary.
    let activation = match &act_node.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => *f,
        _ => return false,
    };

    // Activation must have exactly one predecessor (the Add).
    let act_preds: Vec<NodeId> = act_node.dependencies().collect();
    if act_preds.len() != 1 {
        return false;
    }

    // Build fused inputs: [matmul_input0, matmul_input1, bias_constant_node].
    // Bias stays in the graph as a constant node — zero-copy from arena.
    let mut fused_inputs = node.inputs.clone();
    fused_inputs.push(crate::graph::node::InputSlot {
        source: InputSource::Node(bias_node_id),
        output_port: 0,
    });

    graph.replace_op(
        act_id,
        GraphOp::FusedMatMulBiasActivation {
            m,
            k,
            n,
            activation,
        },
    );
    if let Some(act_node) = graph.get_mut(act_id) {
        act_node.inputs = fused_inputs;
    }

    // Remove the MatMul and Add nodes (bias constant stays).
    graph.remove_node(id);
    graph.remove_node(add_id);
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
        GraphOp::Float(FloatOp::AddRmsNorm { size, epsilon }) => {
            let (s, e) = (*size, *epsilon);
            Box::new(move |act| GraphOp::FusedAddRmsNormActivation {
                size: s,
                epsilon: e,
                activation: act,
            })
        }
        GraphOp::Float(FloatOp::InstanceNorm { size, epsilon }) => {
            let (s, e) = (*size, *epsilon);
            Box::new(move |act| GraphOp::FusedInstanceNormActivation {
                size: s,
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

/// Try to eliminate a Transpose whose single successor is another Transpose
/// that is the exact inverse permutation. The composition is identity →
/// replace with Passthrough.
///
/// Returns `true` if elimination occurred.
pub fn try_eliminate_inverse_transpose(
    graph: &mut Graph,
    id: NodeId,
    succ_index: &[Vec<NodeId>],
) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    let (perm1, ndim1) = match &node.op {
        GraphOp::Float(FloatOp::Transpose { perm, ndim }) => (*perm, *ndim as usize),
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

    let (perm2, ndim2) = match &succ.op {
        GraphOp::Float(FloatOp::Transpose { perm, ndim }) => (*perm, *ndim as usize),
        _ => return false,
    };

    if ndim1 != ndim2 {
        return false;
    }

    // Check if perm2 is the inverse of perm1: perm2[perm1[i]] == i for all i.
    let is_inverse = (0..ndim1).all(|i| {
        let p1 = perm1[i] as usize;
        p1 < ndim2 && perm2[p1] as usize == i
    });

    if !is_inverse {
        return false;
    }

    // Compose to identity → replace successor with Passthrough, remove current.
    let transpose_inputs = node.inputs.clone();
    graph.replace_op(succ_id, GraphOp::Passthrough);
    if let Some(succ_node) = graph.get_mut(succ_id) {
        succ_node.inputs = transpose_inputs;
    }
    graph.remove_node(id);
    true
}

/// Try to fuse a Silu → Mul pattern into FusedSwiGLU.
///
/// Detects: Silu(gate) with single successor Mul, where Mul's other input
/// is the "up" path. Replaces with Float(FusedSwiGLU) taking [gate, up].
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_swiglu(graph: &mut Graph, id: NodeId, succ_index: &[Vec<NodeId>]) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    // Current node must be Silu.
    if !matches!(&node.op, GraphOp::Float(FloatOp::Silu)) {
        return false;
    }

    // Silu must have exactly one successor.
    let succs = Graph::successors_from_index(id, succ_index);
    if succs.len() != 1 {
        return false;
    }
    let mul_id = succs[0];

    let mul_node = match graph.get(mul_id) {
        Some(n) => n,
        None => return false,
    };

    // Successor must be Mul.
    if !matches!(&mul_node.op, GraphOp::Float(FloatOp::Mul)) {
        return false;
    }

    // Mul must have exactly 2 inputs.
    if mul_node.inputs.len() != 2 {
        return false;
    }

    // Find the "up" input (the Mul input that isn't the Silu output).
    let up_slot = {
        let mut found = None;
        for (i, slot) in mul_node.inputs.iter().enumerate() {
            if let InputSource::Node(pred_id) = &slot.source {
                if *pred_id != id {
                    found = Some((i, slot.clone()));
                }
            }
        }
        match found {
            Some((_, slot)) => slot,
            None => return false,
        }
    };

    // Silu's input is the "gate" path.
    let gate_slot = match node.inputs.first() {
        Some(slot) => slot.clone(),
        None => return false,
    };

    // Replace Mul with FusedSwiGLU([gate, up]).
    graph.replace_op(mul_id, GraphOp::Float(FloatOp::FusedSwiGLU));
    if let Some(mul_node) = graph.get_mut(mul_id) {
        let mut inputs = tinyvec::TinyVec::new();
        inputs.push(gate_slot);
        inputs.push(up_slot);
        mul_node.inputs = inputs;
    }
    graph.remove_node(id);
    true
}

/// Try to fuse a Conv2d + Add(constant bias) + Activation into a single
/// `FusedConv2dBiasActivation` node. Same 3-node pattern as matmul bias fusion.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_conv2d_bias_activation(
    graph: &mut Graph,
    id: NodeId,
    succ_index: &[Vec<NodeId>],
) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    let conv_params = match &node.op {
        GraphOp::Float(FloatOp::Conv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            ..
        }) => (
            *kernel_h,
            *kernel_w,
            *stride_h,
            *stride_w,
            *pad_h,
            *pad_w,
            *dilation_h,
            *dilation_w,
            *group,
        ),
        _ => return false,
    };

    // Conv2d must have exactly one successor (the Add).
    let succs = Graph::successors_from_index(id, succ_index);
    if succs.len() != 1 {
        return false;
    }
    let add_id = succs[0];

    let add_node = match graph.get(add_id) {
        Some(n) => n,
        None => return false,
    };

    if !matches!(&add_node.op, GraphOp::Float(FloatOp::Add)) {
        return false;
    }

    if add_node.inputs.len() != 2 {
        return false;
    }

    // Find the bias constant.
    let bias_node_id = {
        let mut found = None;
        for slot in &add_node.inputs {
            if let InputSource::Node(pred_id) = &slot.source {
                if *pred_id == id {
                    continue;
                }
                if let Some(pred_node) = graph.get(*pred_id) {
                    if matches!(&pred_node.op, GraphOp::Constant(_)) {
                        found = Some(*pred_id);
                    }
                }
            }
        }
        match found {
            Some(nid) => nid,
            None => return false,
        }
    };

    // Add must have exactly one successor (the Activation).
    let add_succs = Graph::successors_from_index(add_id, succ_index);
    if add_succs.len() != 1 {
        return false;
    }
    let act_id = add_succs[0];

    let act_node = match graph.get(act_id) {
        Some(n) => n,
        None => return false,
    };

    let activation = match &act_node.op {
        GraphOp::Float(f) if f.is_elementwise_unary() => *f,
        _ => return false,
    };

    let act_preds: Vec<NodeId> = act_node.dependencies().collect();
    if act_preds.len() != 1 {
        return false;
    }

    // Build fused inputs: [conv_data, conv_weight, bias_constant].
    // Conv2d has 3 inputs [data, weight, original_bias]; replace original_bias with the Add's bias.
    let mut fused_inputs = node.inputs.clone();
    // If Conv2d already had a bias input, replace it. Otherwise append.
    if fused_inputs.len() >= 3 {
        fused_inputs[2] = crate::graph::node::InputSlot {
            source: InputSource::Node(bias_node_id),
            output_port: 0,
        };
    } else {
        fused_inputs.push(crate::graph::node::InputSlot {
            source: InputSource::Node(bias_node_id),
            output_port: 0,
        });
    }

    let (kh, kw, sh, sw, ph, pw, dh, dw, g) = conv_params;
    graph.replace_op(
        act_id,
        GraphOp::FusedConv2dBiasActivation {
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
            dilation_h: dh,
            dilation_w: dw,
            group: g,
            activation,
        },
    );
    if let Some(act_node) = graph.get_mut(act_id) {
        act_node.inputs = fused_inputs;
    }

    graph.remove_node(id);
    graph.remove_node(add_id);
    true
}

/// Try to fuse a Conv2d forward into a successor unary activation.
/// Same 2-node pattern as `try_fuse_matmul_activation`.
///
/// Returns `true` if fusion occurred.
pub fn try_fuse_conv2d_activation(
    graph: &mut Graph,
    id: NodeId,
    succ_index: &[Vec<NodeId>],
) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    let conv_params = match &node.op {
        GraphOp::Float(FloatOp::Conv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            ..
        }) => (
            *kernel_h,
            *kernel_w,
            *stride_h,
            *stride_w,
            *pad_h,
            *pad_w,
            *dilation_h,
            *dilation_w,
            *group,
        ),
        _ => return false,
    };

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

    let conv_inputs = node.inputs.clone();
    let (kh, kw, sh, sw, ph, pw, dh, dw, g) = conv_params;
    graph.replace_op(
        succ_id,
        GraphOp::FusedConv2dActivation {
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
            dilation_h: dh,
            dilation_w: dw,
            group: g,
            activation,
        },
    );
    if let Some(succ_node) = graph.get_mut(succ_id) {
        succ_node.inputs = conv_inputs;
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

    // ── Conv2d fusion tests ──────────────────────────────────────────

    fn make_conv2d_op() -> GraphOp {
        GraphOp::Float(FloatOp::Conv2d {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
            input_h: 32,
            input_w: 32,
        })
    }

    #[test]
    fn fuse_conv2d_silu() {
        // Input, Weight, Bias → Conv2d → SiLU → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0: data
            .node(GraphOp::Input) // 1: weight
            .node(GraphOp::Input) // 2: bias
            .node_with_inputs(make_conv2d_op(), &[0, 1, 2]) // 3: Conv2d
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[3]) // 4: SiLU
            .node_with_inputs(GraphOp::Output, &[4]) // 5: Output
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_conv2d_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1);
        // Input, Weight, Bias, FusedConv2dActivation, Output = 5 nodes
        assert_eq!(g.node_count(), 5);

        let fused_node = g
            .node_ids()
            .into_iter()
            .find(|&id| matches!(g.get(id).unwrap().op, GraphOp::FusedConv2dActivation { .. }))
            .expect("should have FusedConv2dActivation");
        if let GraphOp::FusedConv2dActivation {
            activation,
            kernel_h,
            kernel_w,
            ..
        } = &g.get(fused_node).unwrap().op
        {
            assert_eq!(*activation, FloatOp::Silu);
            assert_eq!((*kernel_h, *kernel_w), (3, 3));
        }
    }

    #[test]
    fn no_fuse_conv2d_fan_out() {
        // Conv2d has 2 successors → should NOT fuse.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node_with_inputs(make_conv2d_op(), &[0, 1, 2])
            .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[3])
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[3])
            .node_with_inputs(GraphOp::Output, &[4])
            .node_with_inputs(GraphOp::Output, &[5])
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_conv2d_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
    }

    #[test]
    fn fuse_conv2d_via_full_pass() {
        // Verify Conv2d fusion works through the full `fuse()` pass.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node_with_inputs(make_conv2d_op(), &[0, 1, 2])
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[3])
            .node_with_inputs(GraphOp::Output, &[4])
            .build();

        let stats = crate::fusion::fuse(&mut g).unwrap();
        assert!(
            stats.matmul_activations_fused >= 1,
            "Conv2d+SiLU should fuse via full pass"
        );

        let has_fused = g
            .node_ids()
            .into_iter()
            .any(|id| matches!(g.get(id).unwrap().op, GraphOp::FusedConv2dActivation { .. }));
        assert!(
            has_fused,
            "should have FusedConv2dActivation after full fuse()"
        );
    }

    // ── AddRmsNorm + InstanceNorm fusion tests ───────────────────────

    #[test]
    fn fuse_add_rms_norm_silu() {
        // Input, Residual, Weight → AddRmsNorm → SiLU → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0: x
            .node(GraphOp::Input) // 1: residual
            .node(GraphOp::Input) // 2: weight
            .node_with_inputs(
                GraphOp::Float(FloatOp::AddRmsNorm {
                    size: 64,
                    epsilon: f32::to_bits(1e-5),
                }),
                &[0, 1, 2],
            ) // 3
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[3]) // 4
            .node_with_inputs(GraphOp::Output, &[4]) // 5
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_norm_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1);

        let has_fused = g.node_ids().into_iter().any(|id| {
            matches!(
                g.get(id).unwrap().op,
                GraphOp::FusedAddRmsNormActivation { .. }
            )
        });
        assert!(has_fused, "should have FusedAddRmsNormActivation");
    }

    #[test]
    fn fuse_instance_norm_relu() {
        // Input, Scale, Bias → InstanceNorm → Relu → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node_with_inputs(
                GraphOp::Float(FloatOp::InstanceNorm {
                    size: 256,
                    epsilon: f32::to_bits(1e-5),
                }),
                &[0, 1, 2],
            )
            .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[3])
            .node_with_inputs(GraphOp::Output, &[4])
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_norm_activation(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1);

        let has_fused = g.node_ids().into_iter().any(|id| {
            matches!(
                g.get(id).unwrap().op,
                GraphOp::FusedInstanceNormActivation { .. }
            )
        });
        assert!(has_fused, "should have FusedInstanceNormActivation");
    }

    #[test]
    fn fuse_add_rms_norm_via_full_pass() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node_with_inputs(
                GraphOp::Float(FloatOp::AddRmsNorm {
                    size: 64,
                    epsilon: f32::to_bits(1e-5),
                }),
                &[0, 1, 2],
            )
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[3])
            .node_with_inputs(GraphOp::Output, &[4])
            .build();

        let stats = crate::fusion::fuse(&mut g).unwrap();
        assert!(stats.matmul_activations_fused >= 1);

        let has_fused = g.node_ids().into_iter().any(|id| {
            matches!(
                g.get(id).unwrap().op,
                GraphOp::FusedAddRmsNormActivation { .. }
            )
        });
        assert!(
            has_fused,
            "should have FusedAddRmsNormActivation after full fuse()"
        );
    }

    // ── Transpose elimination tests ──────────────────────────────────

    #[test]
    fn eliminate_inverse_transpose_pair() {
        // Input → Transpose([1,0,2]) → Transpose([1,0,2]) → Output
        // [1,0,2] is its own inverse → should collapse to Passthrough.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(
                GraphOp::Float(FloatOp::Transpose {
                    perm: [1, 0, 2, 0, 0, 0, 0, 0],
                    ndim: 3,
                }),
                &[0],
            )
            .node_with_inputs(
                GraphOp::Float(FloatOp::Transpose {
                    perm: [1, 0, 2, 0, 0, 0, 0, 0],
                    ndim: 3,
                }),
                &[1],
            )
            .node_with_inputs(GraphOp::Output, &[2])
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut eliminated = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_eliminate_inverse_transpose(&mut g, id, &succ_index) {
                eliminated += 1;
            }
        }
        assert_eq!(eliminated, 1);
        // Should have: Input, Passthrough, Output (3 nodes).
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn eliminate_transpose_via_full_pass() {
        // Verify transpose elimination works through the full fuse() pass.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(
                GraphOp::Float(FloatOp::Transpose {
                    perm: [1, 0, 2, 0, 0, 0, 0, 0],
                    ndim: 3,
                }),
                &[0],
            )
            .node_with_inputs(
                GraphOp::Float(FloatOp::Transpose {
                    perm: [1, 0, 2, 0, 0, 0, 0, 0],
                    ndim: 3,
                }),
                &[1],
            )
            .node_with_inputs(GraphOp::Output, &[2])
            .build();

        let stats = crate::fusion::fuse(&mut g).unwrap();
        assert!(
            stats.matmul_activations_fused >= 1,
            "inverse transpose pair should be eliminated"
        );

        let has_passthrough = g
            .node_ids()
            .into_iter()
            .any(|id| matches!(g.get(id).unwrap().op, GraphOp::Passthrough));
        assert!(
            has_passthrough,
            "should have Passthrough after eliminating inverse transposes"
        );
    }

    #[test]
    fn no_eliminate_non_inverse_transpose() {
        // [0,2,1] followed by [2,0,1] — NOT inverses → should NOT eliminate.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(
                GraphOp::Float(FloatOp::Transpose {
                    perm: [0, 2, 1, 0, 0, 0, 0, 0],
                    ndim: 3,
                }),
                &[0],
            )
            .node_with_inputs(
                GraphOp::Float(FloatOp::Transpose {
                    perm: [2, 0, 1, 0, 0, 0, 0, 0],
                    ndim: 3,
                }),
                &[1],
            )
            .node_with_inputs(GraphOp::Output, &[2])
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut eliminated = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_eliminate_inverse_transpose(&mut g, id, &succ_index) {
                eliminated += 1;
            }
        }
        assert_eq!(
            eliminated, 0,
            "non-inverse transposes should NOT be eliminated"
        );
    }

    // ── SwiGLU fusion tests ──────────────────────────────────────────

    #[test]
    fn fuse_silu_mul_to_swiglu() {
        // gate_input → Silu → Mul(silu_out, up_input) → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0: gate
            .node(GraphOp::Input) // 1: up
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[0]) // 2: Silu(gate)
            .node_with_inputs(GraphOp::Float(FloatOp::Mul), &[2, 1]) // 3: Mul(silu, up)
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_swiglu(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 1);
        // Input(gate), Input(up), FusedSwiGLU, Output = 4 nodes
        assert_eq!(g.node_count(), 4);

        let has_swiglu = g
            .node_ids()
            .into_iter()
            .any(|id| matches!(g.get(id).unwrap().op, GraphOp::Float(FloatOp::FusedSwiGLU)));
        assert!(has_swiglu, "should have FusedSwiGLU");
    }

    #[test]
    fn no_fuse_silu_with_fan_out() {
        // Silu has 2 successors → should NOT fuse.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[0])
            .node_with_inputs(GraphOp::Float(FloatOp::Mul), &[2, 1])
            .node_with_inputs(GraphOp::Float(FloatOp::Add), &[2, 1])
            .node_with_inputs(GraphOp::Output, &[3])
            .node_with_inputs(GraphOp::Output, &[4])
            .build();

        let order = toposort::toposort(&g).unwrap();
        let succ_index = g.build_successor_index();
        let mut fused = 0;
        for &id in &order {
            if g.get(id).is_none() {
                continue;
            }
            if try_fuse_swiglu(&mut g, id, &succ_index) {
                fused += 1;
            }
        }
        assert_eq!(fused, 0);
    }

    #[test]
    fn fuse_swiglu_via_full_pass() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[0])
            .node_with_inputs(GraphOp::Float(FloatOp::Mul), &[2, 1])
            .node_with_inputs(GraphOp::Output, &[3])
            .build();

        let stats = crate::fusion::fuse(&mut g).unwrap();
        assert!(
            stats.matmul_activations_fused >= 1,
            "Silu+Mul should fuse to SwiGLU"
        );

        let has_swiglu = g
            .node_ids()
            .into_iter()
            .any(|id| matches!(g.get(id).unwrap().op, GraphOp::Float(FloatOp::FusedSwiGLU)));
        assert!(has_swiglu, "should have FusedSwiGLU after full fuse()");
    }
}
