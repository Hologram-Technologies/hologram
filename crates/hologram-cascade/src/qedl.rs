//! QEDL (Quantize/Encode/Dequantize/Lift) boundary analysis.
//!
//! Detects domain crossings between byte-domain (Z/256Z) ops and float-domain ops.
//! For each crossing, selects the minimum-error encoding based on the upstream
//! op's curvature profile.

use hologram_core::op::FloatOp;
use hologram_core::view::ElementWiseView;
use hologram_graph::graph::node::NodeId;
use hologram_graph::graph::GraphOp;
use hologram_graph::Graph;

/// QEDL boundary type: direction of domain crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QedlBoundary {
    /// Dequantize: byte-domain -> float-domain.
    Dequantize,
    /// Quantize: float-domain -> byte-domain.
    Quantize,
}

/// Encoding identifier for a QEDL dequantize boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodingId {
    Raw = 0,
    Unsigned = 1,
    Signed = 2,
    Angle = 3,
}

/// Algebraic profile of a byte-domain op's output distribution.
#[derive(Clone, Copy, Debug)]
pub struct CurvatureProfile {
    pub mean_trailing_ones: f32,
    pub output_entropy: f32,
    pub zero_crossing: bool,
    pub is_bijective: bool,
}

/// Compute `CurvatureProfile` from an `ElementWiseView` output LUT.
#[must_use]
pub fn compute_profile(view: &ElementWiseView) -> CurvatureProfile {
    let table = view.table();
    let mean_trailing_ones = table.iter().map(|&x| x.trailing_ones() as f32).sum::<f32>() / 256.0;

    let mut counts = [0u32; 256];
    for &x in table.iter() {
        counts[x as usize] += 1;
    }
    let output_entropy = -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / 256.0;
            p * p.log2()
        })
        .sum::<f32>()
        / 8.0;

    let zero_crossing = table.iter().any(|&x| x >= 128) && table.iter().any(|&x| x < 128);

    CurvatureProfile {
        mean_trailing_ones,
        output_entropy,
        zero_crossing,
        is_bijective: view.is_bijective(),
    }
}

/// Select the minimum-error encoding for a QEDL dequantize boundary.
#[must_use]
pub fn select_encoding(profile: &CurvatureProfile, downstream: &FloatOp) -> EncodingId {
    let downstream_is_additive = matches!(downstream, FloatOp::Add | FloatOp::Sub);
    if profile.zero_crossing && downstream_is_additive {
        return EncodingId::Signed;
    }
    if profile.is_bijective && !profile.zero_crossing {
        return EncodingId::Raw;
    }
    if profile.mean_trailing_ones < 1.5 {
        return EncodingId::Raw;
    }
    if profile.output_entropy < 0.3 {
        return EncodingId::Unsigned;
    }
    EncodingId::Signed
}

/// Domain classification for a `GraphOp`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Domain {
    Byte,
    Float,
    Unknown,
}

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

        GraphOp::Output
        | GraphOp::Constant(_)
        | GraphOp::Passthrough
        | GraphOp::CallSubgraph(_)
        | GraphOp::Custom { .. } => Domain::Unknown,
    }
}

fn as_float_op(op: &GraphOp) -> Option<FloatOp> {
    match op {
        GraphOp::Float(f) => Some(*f),
        GraphOp::FusedFloatChain(chain) => chain.first().copied(),
        _ => None,
    }
}

/// Walk the graph in topological order, emitting QEDL boundary annotations.
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
                    let view = pred.op.to_view().unwrap_or_else(ElementWiseView::identity);
                    let profile = compute_profile(&view);
                    let enc = select_encoding(&profile, &downstream_float_op);
                    result.push((id, QedlBoundary::Dequantize, enc));
                }
                (Domain::Float, Domain::Byte) => {
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
    use hologram_graph::graph::Graph;

    fn schedule_order(schedule: &hologram_graph::schedule::ExecutionSchedule) -> Vec<NodeId> {
        schedule
            .levels
            .iter()
            .flat_map(|level| level.node_ids.iter().copied())
            .collect()
    }

    #[test]
    fn qedl_non_empty_for_mixed_graph() {
        let mut g = Graph::new();
        let input = g.add_node(GraphOp::Input);
        let relu = g.add_node(GraphOp::Lut(LutOp::Relu));
        let cid = hologram_graph::constant::ConstantId::new(1);
        let matmul = g.add_node(GraphOp::MatMulLut8(cid));
        let output = g.add_node(GraphOp::Output);
        g.add_edge(input, relu);
        g.add_edge(relu, matmul);
        g.add_edge(matmul, output);

        let schedule = hologram_graph::schedule::ExecutionSchedule::build(&g).unwrap();
        let order = schedule_order(&schedule);
        let boundaries = insert_qedl_boundaries(&g, &order);
        assert!(!boundaries.is_empty());
        assert!(boundaries.iter().any(|(_, b, _)| *b == QedlBoundary::Dequantize));
    }

    #[test]
    fn qedl_empty_for_pure_byte_graph() {
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
        assert!(boundaries.is_empty());
    }

    #[test]
    fn profile_identity_view() {
        let id = ElementWiseView::identity();
        let p = compute_profile(&id);
        assert!(p.is_bijective);
        assert!((p.output_entropy - 1.0).abs() < 1e-3);
        assert!(p.zero_crossing);
    }

    #[test]
    fn encoding_bijective_no_crossing_is_raw() {
        let profile = CurvatureProfile {
            mean_trailing_ones: 0.5,
            output_entropy: 1.0,
            zero_crossing: false,
            is_bijective: true,
        };
        assert_eq!(select_encoding(&profile, &FloatOp::Mul), EncodingId::Raw);
    }
}
