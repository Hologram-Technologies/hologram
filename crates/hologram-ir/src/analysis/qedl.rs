//! QEDL (Quantize/Encode/Dequantize/Lift) boundary detection.
//!
//! Detects domain crossings between byte-domain (Z/256Z) ops and float-domain ops.
//! For each crossing, selects the minimum-error encoding based on the upstream
//! op's curvature profile.
//!
//! # Architectural role
//!
//! This module is a **structure finder**, not a constructor. Domain-crossing
//! boundaries are intrinsic properties of the source graph — they exist
//! because the graph mixes byte-domain and float-domain primitives, not
//! because the compiler decides to insert them. The selected encoding is the
//! minimum-error encoding for the *existing* downstream op, read off from
//! the curvature profile.
//!
//! # Performance
//!
//! O(N · 256) per pass: one curvature profile per byte→float crossing.
//! **Perf: NEUTRAL** — pure compile-time work.

use hologram_core::op::FloatOp;
use hologram_core::view::ElementWiseView;

use crate::graph::node::{InputSource, NodeId};
use crate::graph::GraphOp;
use crate::Graph;

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
    /// Raw byte interpretation (no transform).
    Raw = 0,
    /// Unsigned scaled-integer encoding.
    Unsigned = 1,
    /// Signed scaled-integer encoding (two's complement).
    Signed = 2,
    /// Angle/phase encoding for periodic ranges.
    Angle = 3,
}

/// Algebraic profile of a byte-domain op's output distribution.
#[derive(Clone, Copy, Debug)]
pub struct CurvatureProfile {
    /// Mean number of trailing ones in the output values.
    pub mean_trailing_ones: f32,
    /// Shannon entropy of the output distribution (normalized).
    pub output_entropy: f32,
    /// Whether the output range crosses the signed zero boundary.
    pub zero_crossing: bool,
    /// Whether the underlying view is a bijection.
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
        | GraphOp::RingPrimBinary(_, _)
        | GraphOp::RingActivation(_, _)
        | GraphOp::RingAccumulate(_)
        | GraphOp::RingReduce { .. } => Domain::Byte,

        GraphOp::Float(_)
        | GraphOp::FusedFloatChain(_)
        | GraphOp::FusedMatMulActivation { .. }
        | GraphOp::FusedMatMulBiasActivation { .. }
        | GraphOp::FusedConv2dActivation { .. }
        | GraphOp::FusedConv2dBiasActivation { .. }
        | GraphOp::FusedRmsNormActivation { .. }
        | GraphOp::FusedAddRmsNormActivation { .. }
        | GraphOp::FusedInstanceNormActivation { .. }
        | GraphOp::FusedLayerNormActivation { .. }
        | GraphOp::FusedGroupNormActivation { .. }
        | GraphOp::MatMulLut4(_)
        | GraphOp::MatMulLut8(_)
        | GraphOp::MatMulLut16(_)
        | GraphOp::MatMulLut4Activation(_, _)
        | GraphOp::MatMulLut8Activation(_, _)
        | GraphOp::BatchMatMulLut4(_)
        | GraphOp::BatchMatMulLut8(_)
        | GraphOp::BatchMatMulLut16(_)
        | GraphOp::MatMulLut2(_)
        | GraphOp::MatMulLut2Activation(_, _)
        | GraphOp::Conv2dLut4 { .. } => Domain::Float,

        GraphOp::Output
        | GraphOp::Constant(_)
        | GraphOp::Passthrough
        | GraphOp::CallSubgraph(_) => Domain::Unknown,
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
                InputSource::Node(pid) => pid,
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
    use crate::constant::ConstantId;
    use crate::graph::Graph;
    use crate::schedule::ExecutionSchedule;
    use hologram_core::op::LutOp;

    fn schedule_order(schedule: &ExecutionSchedule) -> Vec<NodeId> {
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
        let cid = ConstantId::new(1);
        let matmul = g.add_node(GraphOp::MatMulLut8(cid));
        let output = g.add_node(GraphOp::Output);
        g.add_edge(input, relu);
        g.add_edge(relu, matmul);
        g.add_edge(matmul, output);

        let schedule = ExecutionSchedule::build(&g).unwrap();
        let order = schedule_order(&schedule);
        let boundaries = insert_qedl_boundaries(&g, &order);
        assert!(!boundaries.is_empty());
        assert!(boundaries
            .iter()
            .any(|(_, b, _)| *b == QedlBoundary::Dequantize));
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

        let schedule = ExecutionSchedule::build(&g).unwrap();
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
