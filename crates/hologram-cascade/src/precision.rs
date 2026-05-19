//! Observable-guided quantum-level selection.
//!
//! Uses the Q0 stratum (Hamming weight) and curvature (carry-depth) observables
//! to select the minimum ring precision tier (Q0/Q1/Q2) required for each
//! byte-domain node.

use hologram_core::lut::q0::{CURVATURE_Q0, STRATUM_Q0};
use hologram_core::op::RingLevel;
use hologram_core::view::ElementWiseView;
use hologram_graph::graph::node::NodeId;
use hologram_graph::graph::{Graph, GraphOp};
use uor_foundation::QuantumLevel;

/// Mean stratum (Hamming weight) of a 256-entry output distribution.
#[inline]
#[must_use]
pub fn mean_stratum_q0(table: &[u8; 256]) -> f32 {
    table
        .iter()
        .map(|&x| STRATUM_Q0[x as usize] as f32)
        .sum::<f32>()
        / 256.0
}

/// Mean curvature (carry-depth) of a 256-entry output distribution.
#[inline]
#[must_use]
pub fn mean_curvature_q0(table: &[u8; 256]) -> f32 {
    table
        .iter()
        .map(|&x| CURVATURE_Q0[x as usize] as f32)
        .sum::<f32>()
        / 256.0
}

/// Stratum threshold above which Q1 ring precision is required.
pub const STRATUM_Q1_THRESHOLD: f32 = 4.0;
/// Stratum threshold above which Q2 ring precision is required.
pub const STRATUM_Q2_THRESHOLD: f32 = 6.0;
/// Curvature threshold above which Q1 ring precision is required.
pub const CURVATURE_Q1_THRESHOLD: f32 = 1.5;

/// Select the minimum sufficient `QuantumLevel` for a byte-domain node with the given LUT table.
#[must_use]
pub fn select_ring_level(table: &[u8; 256]) -> QuantumLevel {
    let s = mean_stratum_q0(table);
    let c = mean_curvature_q0(table);
    if s > STRATUM_Q2_THRESHOLD {
        QuantumLevel::Q2
    } else if s > STRATUM_Q1_THRESHOLD || c > CURVATURE_Q1_THRESHOLD {
        QuantumLevel::Q1
    } else {
        QuantumLevel::Q0
    }
}

/// Select the minimum sufficient `QuantumLevel` for a graph node given its `ElementWiseView`.
#[must_use]
pub fn select_ring_level_for_view(view: &ElementWiseView) -> QuantumLevel {
    select_ring_level(view.table())
}

/// Promote `Prim` nodes to `RingPrimUnary`/`RingPrimBinary` based on output distribution.
/// Returns the number of nodes promoted.
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
        };

        let level = if arity == 1 {
            match GraphOp::Prim(prim_op).to_view() {
                Some(view) => select_ring_level_for_view(&view),
                None => continue,
            }
        } else {
            let mut max_level = QuantumLevel::Q0;
            for pid in &pred_ids {
                if let Some(pred) = graph.get(*pid) {
                    if let Some(view) = pred.op.to_view() {
                        let lv = select_ring_level_for_view(&view);
                        if lv.index() > max_level.index() {
                            max_level = lv;
                        }
                    }
                }
            }
            max_level
        };

        if level == QuantumLevel::Q0 {
            continue;
        }

        let new_op = if arity == 1 {
            GraphOp::RingPrimUnary(prim_op, RingLevel::from(level))
        } else {
            GraphOp::RingPrimBinary(prim_op, RingLevel::from(level))
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

    fn all_same_table(val: u8) -> [u8; 256] {
        [val; 256]
    }

    #[test]
    fn constant_zero_table_is_q0() {
        assert_eq!(select_ring_level(&all_same_table(0)), QuantumLevel::Q0);
    }

    #[test]
    fn all_ff_table_is_q2() {
        assert_eq!(select_ring_level(&all_same_table(0xFF)), QuantumLevel::Q2);
    }

    #[test]
    fn unary_prim_succ_promoted() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Prim(PrimOp::Succ), &[1])
            .build();
        promote_prim_ring_levels(&mut g);
        let ids = g.node_ids();
        let succ = g.get(ids[2]).unwrap();
        assert!(matches!(succ.op, GraphOp::RingPrimUnary(PrimOp::Succ, _)));
    }

    #[test]
    fn zero_promotions_on_no_prim() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .build();
        assert_eq!(promote_prim_ring_levels(&mut g), 0);
    }

    #[test]
    fn binary_prim_after_low_stratum_not_promoted() {
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
            .build();
        let ids: Vec<_> = g.node_ids();
        let add_id = *ids.last().unwrap();
        promote_prim_ring_levels(&mut g);
        let add = g.get(add_id).unwrap();
        assert!(matches!(add.op, GraphOp::Prim(PrimOp::Add)));
    }
}
