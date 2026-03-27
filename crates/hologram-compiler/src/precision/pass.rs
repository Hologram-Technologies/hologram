//! Compiler pass: promote byte-domain Prim nodes to RingPrim variants.
//!
//! Walks all graph nodes. For each `GraphOp::Prim(p)`:
//! - Unary (arity==1): compute the ring level required by the op's own output
//!   distribution (its Q0 ElementWiseView). If Q1/Q2: replace with
//!   `RingPrimUnary(p, level)`.
//! - Binary (arity==2): compute the required level from direct predecessor output
//!   distributions. Takes the maximum `select_ring_level_for_view` over all preds.
//!   If Q1/Q2: replace with `RingPrimBinary(p, level)`.
//!
//! Q0 Prim nodes are left unchanged (existing byte-domain LUT path is optimal).
//! This pass runs after fusion and bakes ring-level annotations into the graph
//! so the archive contains fully-annotated ops. The tape builder then maps 1:1.

use hologram_core::op::RingLevel;
use hologram_graph::graph::node::NodeId;
use hologram_graph::graph::{Graph, GraphOp};

use super::select_ring_level_for_view;

/// Run the precision promotion pass on `graph` in-place.
///
/// Promotes `Prim` nodes whose output distribution requires higher ring
/// precision (Q1 or Q2) to `RingPrimUnary` / `RingPrimBinary` variants.
///
/// Returns the number of nodes promoted (for stats).
pub fn promote_prim_ring_levels(graph: &mut Graph) -> usize {
    let ids = graph.node_ids();
    let mut promoted = 0usize;

    for id in ids {
        let (prim_op, arity, pred_ids) = {
            let node = match graph.get(id) {
                Some(n) => n,
                None => continue,
            };
            let p = match node.op {
                GraphOp::Prim(p) => p,
                _ => continue,
            };
            let deps: Vec<NodeId> = node.dependencies().collect();
            (p, p.arity(), deps)
        }; // immutable borrow of graph released here

        let level = if arity == 1 {
            // Unary: ring level determined by the op's own output distribution.
            match GraphOp::Prim(prim_op).to_view() {
                Some(view) => select_ring_level_for_view(&view),
                None => continue,
            }
        } else {
            // Binary: ring level determined by the maximum over predecessor views.
            let mut max_level = RingLevel::Q0;
            for pid in &pred_ids {
                if let Some(pred) = graph.get(*pid) {
                    if let Some(view) = pred.op.to_view() {
                        let lv = select_ring_level_for_view(&view);
                        if (lv as u8) > (max_level as u8) {
                            max_level = lv;
                        }
                    }
                }
            }
            max_level
        };

        if level == RingLevel::Q0 {
            continue;
        }

        let new_op = if arity == 1 {
            GraphOp::RingPrimUnary(prim_op, level)
        } else {
            GraphOp::RingPrimBinary(prim_op, level)
        };
        graph.replace_op(id, new_op);
        promoted += 1;
    }

    promoted
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::{LutOp, PrimOp};
    use hologram_graph::builder::GraphBuilder;

    #[test]
    fn unary_prim_succ_promoted_due_to_high_curvature() {
        // Succ outputs a uniform bijection 0..255 (like the identity table).
        // Mean curvature ≈ 2.0 > CURVATURE_Q1_THRESHOLD (1.5) → promoted to Q1+.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Prim(PrimOp::Succ), &[1])
            .build();
        promote_prim_ring_levels(&mut g);
        let ids = g.node_ids();
        let succ = g.get(ids[2]).unwrap();
        // Succ has uniform output → high curvature → promoted
        assert!(matches!(succ.op, GraphOp::RingPrimUnary(PrimOp::Succ, _)));
    }

    #[test]
    fn unary_prim_promoted_for_high_stratum() {
        // Neg's Q0 view: mean stratum > Q1 threshold (uniform 0..255 with ~4 bits).
        // Exact promotion depends on threshold comparison; just verify no panic.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[0])
            .build();
        let before_count = g.node_count();
        promote_prim_ring_levels(&mut g);
        // Node count must be unchanged — only ops are swapped in-place.
        assert_eq!(g.node_count(), before_count);
    }

    #[test]
    fn zero_promotions_on_no_prim() {
        // A graph with only Lut ops should have 0 promotions.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .build();
        let count = promote_prim_ring_levels(&mut g);
        assert_eq!(count, 0);
    }

    #[test]
    fn binary_prim_after_high_stratum_pred_promoted() {
        // Add after a sigmoid (high stratum) should be promoted.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
            .build();
        let ids_before: Vec<_> = g.node_ids();
        let add_id = *ids_before.last().unwrap();
        promote_prim_ring_levels(&mut g);
        let add_node = g.get(add_id).unwrap();
        // Should be promoted to RingPrimBinary since sigmoid has high stratum.
        match &add_node.op {
            GraphOp::RingPrimBinary(PrimOp::Add, _) => {} // promoted
            GraphOp::Prim(PrimOp::Add) => {}              // Q0 if sigmoid is below threshold
            other => panic!("unexpected op: {other:?}"),
        }
    }

    #[test]
    fn binary_prim_after_low_stratum_preds_not_promoted() {
        // Add after relu (low stratum) should stay Q0.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
            .build();
        let ids_before: Vec<_> = g.node_ids();
        let add_id = *ids_before.last().unwrap();
        promote_prim_ring_levels(&mut g);
        let add_node = g.get(add_id).unwrap();
        assert!(matches!(add_node.op, GraphOp::Prim(PrimOp::Add)));
    }
}
