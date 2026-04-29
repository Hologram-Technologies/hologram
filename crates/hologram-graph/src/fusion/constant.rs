//! Constant folding: evaluate ops on constant inputs at compile time.

use crate::constant::ConstantData;
use crate::graph::node::NodeId;
use crate::graph::{Graph, GraphOp};

/// Try to fold a node if all its predecessors are constants.
///
/// Returns `true` if the node was folded into a Constant.
pub fn try_fold_constant(graph: &mut Graph, id: NodeId) -> bool {
    let node = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };

    if !node.op.is_pure() {
        return false;
    }

    // Collect constant predecessor values
    let pred_ids: Vec<NodeId> = node.dependencies().collect();
    if pred_ids.is_empty() {
        return false;
    }

    let mut values: Vec<u8> = Vec::new();
    for &pid in &pred_ids {
        match graph.get(pid).map(|n| &n.op) {
            Some(GraphOp::Constant(cid)) => {
                if let Some(ConstantData::Bytes(bytes)) = graph.get_constant(*cid) {
                    if let Some(&val) = bytes.first() {
                        values.push(val);
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // Evaluate the operation on constant inputs
    let op = graph.get(id).unwrap().op.clone();
    let result = match &op {
        GraphOp::Lut(lut_op) if values.len() == 1 => lut_op.apply(values[0]),
        GraphOp::Prim(prim_op) if prim_op.arity() == 1 && values.len() == 1 => {
            prim_op.apply_unary(values[0])
        }
        GraphOp::Prim(prim_op) if prim_op.arity() == 2 && values.len() == 2 => {
            prim_op.apply_binary(values[0], values[1])
        }
        GraphOp::FusedView(view) if values.len() == 1 => view.apply(values[0]),
        // RingPrimUnary/Binary at Q0 are semantically identical to Prim — foldable.
        // Q1/Q2 constant data is a single byte; multi-byte ring folding is not supported.
        GraphOp::RingPrimUnary(prim_op, hologram_core::op::RingLevel::Q0) if values.len() == 1 => {
            prim_op.apply_unary(values[0])
        }
        GraphOp::RingPrimBinary(prim_op, hologram_core::op::RingLevel::Q0) if values.len() == 2 => {
            prim_op.apply_binary(values[0], values[1])
        }
        _ => return false,
    };

    let cid = graph.add_constant(ConstantData::Bytes(vec![result]));
    // ADR-053: v3 archives require constant_shapes coverage. The folded
    // constant is a single byte; record its shape so the writer's
    // validation pass accepts it.
    graph.set_constant_shape(cid, vec![1]);
    graph.replace_op(id, GraphOp::Constant(cid));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use hologram_core::op::{LutOp, PrimOp};

    #[test]
    fn fold_unary_lut() {
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![42]))
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .build();
        let ids = g.node_ids();
        let relu_id = ids[1];
        assert!(try_fold_constant(&mut g, relu_id));
        let node = g.get(relu_id).unwrap();
        assert!(matches!(node.op, GraphOp::Constant(_)));
    }

    #[test]
    fn fold_unary_prim() {
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![1]))
            .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[0])
            .build();
        let ids = g.node_ids();
        let neg_id = ids[1];
        assert!(try_fold_constant(&mut g, neg_id));
        // Check folded value: neg(1) = 255
        if let GraphOp::Constant(cid) = g.get(neg_id).unwrap().op {
            let data = g.get_constant(cid).unwrap();
            assert_eq!(data, &ConstantData::Bytes(vec![255]));
        } else {
            panic!("expected Constant");
        }
    }

    #[test]
    fn fold_binary() {
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![10]))
            .constant(ConstantData::Bytes(vec![20]))
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1])
            .build();
        let ids = g.node_ids();
        let add_id = ids[2];
        assert!(try_fold_constant(&mut g, add_id));
        if let GraphOp::Constant(cid) = g.get(add_id).unwrap().op {
            let data = g.get_constant(cid).unwrap();
            assert_eq!(data, &ConstantData::Bytes(vec![30]));
        } else {
            panic!("expected Constant");
        }
    }

    #[test]
    fn fold_ring_prim_unary_q0() {
        use hologram_core::op::RingLevel;
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![42]))
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q0), &[0])
            .build();
        let ids = g.node_ids();
        let neg_id = ids[1];
        assert!(try_fold_constant(&mut g, neg_id));
        let node = g.get(neg_id).unwrap();
        assert!(matches!(node.op, GraphOp::Constant(_)));
        // Value should be neg(42) mod 256 = 214.
        if let GraphOp::Constant(cid) = node.op {
            let data = g.get_constant(cid).unwrap();
            assert_eq!(data, &ConstantData::Bytes(vec![42u8.wrapping_neg()]));
        }
    }

    #[test]
    fn fold_ring_prim_binary_q0() {
        use hologram_core::op::RingLevel;
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![10]))
            .constant(ConstantData::Bytes(vec![20]))
            .node_with_inputs(GraphOp::RingPrimBinary(PrimOp::Add, RingLevel::Q0), &[0, 1])
            .build();
        let ids = g.node_ids();
        let add_id = ids[2];
        assert!(try_fold_constant(&mut g, add_id));
        if let GraphOp::Constant(cid) = g.get(add_id).unwrap().op {
            let data = g.get_constant(cid).unwrap();
            assert_eq!(data, &ConstantData::Bytes(vec![30])); // add_q0(10, 20) == 30
        } else {
            panic!("expected Constant");
        }
    }

    #[test]
    fn ring_prim_q1_not_folded() {
        // Q1 ring prim: constant data is a single byte; multi-byte folding not supported.
        use hologram_core::op::RingLevel;
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![42]))
            .node_with_inputs(GraphOp::RingPrimUnary(PrimOp::Neg, RingLevel::Q1), &[0])
            .build();
        let ids = g.node_ids();
        assert!(!try_fold_constant(&mut g, ids[1])); // should NOT fold
    }

    #[test]
    fn no_fold_non_constant() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .build();
        let ids = g.node_ids();
        assert!(!try_fold_constant(&mut g, ids[1]));
    }

    #[test]
    fn no_fold_input() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Input);
        assert!(!try_fold_constant(&mut g, id));
    }
}
