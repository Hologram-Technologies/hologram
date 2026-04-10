//! Observable-guided ring-level inference.
//!
//! Uses the Q0 stratum (Hamming weight) and curvature (carry-depth) observables
//! to infer the minimum ring precision tier (Q0/Q1/Q2) required for each
//! byte-domain node.
//!
//! # Architectural role
//!
//! This module is a **structure finder**, not an optimizer. The minimum
//! sufficient ring level for a node is a property the source already
//! determines via its output distribution — this analysis reads it off rather
//! than imposing it. Per Prism section 4 of the SCS framework, the compiler
//! is a structure-finder, and this is one of the structures it finds.
//!
//! # Performance
//!
//! O(N · 256) per pass: one mean-stratum scan over each node's 256-entry
//! LUT table. **Perf: NEUTRAL** — pure compile-time work.

use hologram_core::lut::q0::{CURVATURE_Q0, STRATUM_Q0};
use hologram_core::op::{RingLevel, WittLevel};
use hologram_core::view::ElementWiseView;

use crate::graph::node::NodeId;
use crate::graph::{Graph, GraphOp};

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

/// Select the minimum sufficient `WittLevel` for a byte-domain node with the given LUT table.
#[must_use]
pub fn select_ring_level(table: &[u8; 256]) -> WittLevel {
    let s = mean_stratum_q0(table);
    let c = mean_curvature_q0(table);
    if s > STRATUM_Q2_THRESHOLD {
        WittLevel::W24
    } else if s > STRATUM_Q1_THRESHOLD || c > CURVATURE_Q1_THRESHOLD {
        WittLevel::W16
    } else {
        WittLevel::W8
    }
}

/// Select the minimum sufficient `WittLevel` for a graph node given its `ElementWiseView`.
#[must_use]
pub fn select_ring_level_for_view(view: &ElementWiseView) -> WittLevel {
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
            let mut max_level = WittLevel::W8;
            for pid in &pred_ids {
                if let Some(pred) = graph.get(*pid) {
                    if let Some(view) = pred.op.to_view() {
                        let lv = select_ring_level_for_view(&view);
                        if lv.witt_length() > max_level.witt_length() {
                            max_level = lv;
                        }
                    }
                }
            }
            max_level
        };

        if level == WittLevel::W8 {
            continue;
        }

        // `select_ring_level` only ever returns one of W8/W16/W24, so the
        // `from_witt_level` conversion is total here. The `expect` documents
        // the invariant rather than hiding it behind a silent fallback.
        let ring_level = RingLevel::from_witt_level(level)
            .expect("precision pass selects only spec-named WittLevels (W8/W16/W24)");
        let new_op = if arity == 1 {
            GraphOp::RingPrimUnary(prim_op, ring_level)
        } else {
            GraphOp::RingPrimBinary(prim_op, ring_level)
        };
        graph.replace_op(id, new_op);
        promoted += 1;
    }

    promoted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use hologram_core::op::{LutOp, PrimOp};

    fn all_same_table(val: u8) -> [u8; 256] {
        [val; 256]
    }

    #[test]
    fn constant_zero_table_is_q0() {
        assert_eq!(select_ring_level(&all_same_table(0)), WittLevel::W8);
    }

    #[test]
    fn all_ff_table_is_q2() {
        assert_eq!(select_ring_level(&all_same_table(0xFF)), WittLevel::W24);
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
