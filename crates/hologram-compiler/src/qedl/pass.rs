//! QEDL boundary insertion pass.
//!
//! Walks the graph in topological order. For each edge that crosses from
//! byte-domain to float-domain (or vice versa), emits a boundary annotation
//! with the optimal encoding for that crossing.
//!
//! Replaces the `Vec::new()` stub in `compiler/mod.rs`.

use hologram_core::op::FloatOp;
use hologram_core::view::ElementWiseView;
use hologram_graph::graph::node::NodeId;
use hologram_graph::graph::GraphOp;
use hologram_graph::Graph;

use super::{compute_profile, select_encoding, EncodingId};
use crate::compiler::QedlBoundary;

/// Domain classification for a `GraphOp`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Domain {
    Byte,
    Float,
    Unknown,
}

/// Classify the output domain of a `GraphOp`.
fn domain(op: &GraphOp) -> Domain {
    match op {
        GraphOp::Input
        | GraphOp::Lut(_)
        | GraphOp::FusedView(_)
        | GraphOp::FusedView16(_)
        | GraphOp::Prim(_)
        | GraphOp::RingPrimUnary(_, _)
        | GraphOp::RingPrimBinary(_, _) => Domain::Byte,

        GraphOp::Float(_)
        | GraphOp::FusedFloatChain(_)
        | GraphOp::FusedMatMulActivation { .. }
        | GraphOp::FusedMatMulBiasActivation { .. }
        | GraphOp::FusedRmsNormActivation { .. }
        | GraphOp::FusedLayerNormActivation { .. }
        | GraphOp::FusedGroupNormActivation { .. }
        | GraphOp::MatMulLut4(_)
        | GraphOp::MatMulLut8(_)
        | GraphOp::MatMulLut16(_)
        | GraphOp::MatMulLut4Activation(_, _)
        | GraphOp::MatMulLut8Activation(_, _)
        | GraphOp::BatchMatMulLut4(_)
        | GraphOp::BatchMatMulLut8(_)
        | GraphOp::BatchMatMulLut16(_) => Domain::Float,

        // Output, Constant, Passthrough, CallSubgraph, Custom: domain depends on context.
        GraphOp::Output
        | GraphOp::Constant(_)
        | GraphOp::Passthrough
        | GraphOp::CallSubgraph(_)
        | GraphOp::Custom { .. } => Domain::Unknown,
    }
}

/// Extract the `FloatOp` from a node if it is a float-domain op.
fn as_float_op(op: &GraphOp) -> Option<FloatOp> {
    match op {
        GraphOp::Float(f) => Some(*f),
        GraphOp::FusedFloatChain(chain) => chain.first().copied(),
        _ => None,
    }
}

/// Walk the graph in topological order, emitting a QEDL boundary annotation
/// for each edge that crosses between byte-domain and float-domain.
///
/// Annotations are placed on the *consuming* node (the node whose input crosses
/// the boundary). The encoding is derived from the predecessor's output LUT profile.
///
/// Replaces `Vec::new()` in `emit_stage`.
#[must_use]
pub fn insert_qedl_boundaries(
    graph: &Graph,
    order: &[NodeId],
) -> Vec<(NodeId, QedlBoundary, EncodingId)> {
    let mut result = Vec::new();

    for &id in order {
        let node = match graph.get(id) {
            Some(n) => n,
            None => continue,
        };
        let this_domain = domain(&node.op);
        if this_domain == Domain::Unknown {
            continue;
        }
        // Downstream FloatOp for encoding selection (falls back to Add for non-float consumers).
        let downstream_float_op = as_float_op(&node.op).unwrap_or(FloatOp::Add);

        for input_slot in node.inputs.iter() {
            let pred_id = match input_slot.source {
                hologram_graph::graph::node::InputSource::Node(pid) => pid,
                _ => continue,
            };
            let pred = match graph.get(pred_id) {
                Some(n) => n,
                None => continue,
            };
            let pred_domain = domain(&pred.op);

            match (pred_domain, this_domain) {
                (Domain::Byte, Domain::Float) => {
                    // Dequantize boundary: byte → float.
                    let view = pred.op.to_view().unwrap_or_else(ElementWiseView::identity);
                    let profile = compute_profile(&view);
                    let enc = select_encoding(&profile, &downstream_float_op);
                    result.push((id, QedlBoundary::Dequantize, enc));
                }
                (Domain::Float, Domain::Byte) => {
                    // Quantize boundary: float → byte. Unsigned encoding for re-quantize.
                    result.push((id, QedlBoundary::Quantize, EncodingId::Unsigned));
                }
                _ => {}
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::LutOp;
    use hologram_graph::graph::{Graph, GraphOp};

    /// Collect all NodeIds from a schedule in topological order.
    fn schedule_order(schedule: &hologram_graph::schedule::ExecutionSchedule) -> Vec<NodeId> {
        schedule
            .levels
            .iter()
            .flat_map(|level| level.node_ids.iter().copied())
            .collect()
    }

    /// Build a graph: Input → Relu(Lut/byte) → MatMulLut8(float) → Output
    fn build_lut_then_float_graph() -> Graph {
        let mut g = Graph::new();
        let input = g.add_node(GraphOp::Input);
        let relu = g.add_node(GraphOp::Lut(LutOp::Relu));
        let cid = hologram_graph::constant::ConstantId::new(1);
        let matmul = g.add_node(GraphOp::MatMulLut8(cid));
        let output = g.add_node(GraphOp::Output);
        g.add_edge(input, relu);
        g.add_edge(relu, matmul);
        g.add_edge(matmul, output);
        g
    }

    #[test]
    fn qedl_pass_non_empty_for_mixed_graph() {
        let g = build_lut_then_float_graph();
        let schedule = hologram_graph::schedule::ExecutionSchedule::build(&g).unwrap();
        let order = schedule_order(&schedule);
        let boundaries = insert_qedl_boundaries(&g, &order);
        // The Relu→MatMulLut8 edge is a byte→float crossing → at least one Dequantize.
        assert!(
            !boundaries.is_empty(),
            "QEDL pass should annotate domain crossings"
        );
        assert!(
            boundaries
                .iter()
                .any(|(_, b, _)| *b == QedlBoundary::Dequantize),
            "should find at least one Dequantize boundary"
        );
    }

    #[test]
    fn qedl_pass_empty_for_pure_byte_graph() {
        // Input → Relu → Relu → Output: no float ops → no QEDL boundaries.
        let mut g = Graph::new();
        let input = g.add_node(GraphOp::Input);
        let r1 = g.add_node(GraphOp::Lut(LutOp::Relu));
        let r2 = g.add_node(GraphOp::Lut(LutOp::Relu));
        let output = g.add_node(GraphOp::Output);
        g.add_edge(input, r1);
        g.add_edge(r1, r2);
        g.add_edge(r2, output);
        let schedule = hologram_graph::schedule::ExecutionSchedule::build(&g).unwrap();
        let order = schedule_order(&schedule);
        let boundaries = insert_qedl_boundaries(&g, &order);
        assert!(
            boundaries.is_empty(),
            "pure byte graph should have no QEDL boundaries"
        );
    }
}
