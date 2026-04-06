//! `--detail graph` output.

use hologram_archive::LoadedPlan;
use hologram_graph::constant::ConstantStore;
use hologram_graph::graph::node::{InputSlot, InputSource, Node};
use hologram_graph::graph::GraphOp;

/// Print every node with its operation and input edges.
pub fn print(plan: &LoadedPlan) {
    let sg = plan.graph();
    println!("Graph ({} nodes):", sg.node_count());
    for (idx, node) in sg.nodes.iter().enumerate() {
        println!("  {}", format_node(idx, node, &sg.constants));
    }
}

/// Format a single node line.
fn format_node(idx: usize, node: &Node, constants: &ConstantStore) -> String {
    let op = format_op(&node.op, constants);
    let edges = format_inputs(&node.inputs);
    if edges.is_empty() {
        format!("[{idx}] {op}")
    } else {
        format!("[{idx}] {op} <- {edges}")
    }
}

/// Format the operation name with metadata.
fn format_op(op: &GraphOp, constants: &ConstantStore) -> String {
    match op {
        GraphOp::Input => "Input".into(),
        GraphOp::Output => "Output".into(),
        GraphOp::Prim(p) => format!("Prim({})", p.name()),
        GraphOp::Lut(l) => format!("Lut({})", l.name()),
        GraphOp::FusedView(_) => "FusedView (256-byte table)".into(),
        GraphOp::FusedView16(_) => "FusedView16 (128KB Q1 table)".into(),
        GraphOp::Constant(id) => format_constant(id, constants),
        GraphOp::CallSubgraph(s) => format!("CallSubgraph({})", s.raw()),
        GraphOp::MatMulLut { bits, cid } => format!("MatMulLut{bits}(id={})", cid.raw()),
        GraphOp::BatchMatMulLut { bits, cid } => format!("BatchMatMulLut{bits}(id={})", cid.raw()),
        GraphOp::RingPrimUnary(p, level) => format!("RingPrimUnary({}, {:?})", p.name(), level),
        GraphOp::RingPrimBinary(p, level) => format!("RingPrimBinary({}, {:?})", p.name(), level),
        GraphOp::Float(f) => f.name().to_string(),
        GraphOp::FusedFloatChain(chain) => {
            let names: Vec<&str> = chain.iter().map(|f| f.name()).collect();
            format!("FusedFloatChain({})", names.join(" → "))
        }
        GraphOp::FusedMatMulActivation {
            m,
            k,
            n,
            activation,
        } => format!("MatMul[{m},{k},{n}]+{}", activation.name()),
        GraphOp::FusedMatMulBiasActivation {
            m,
            k,
            n,
            activation,
        } => {
            format!("MatMul[{m},{k},{n}]+Bias+{}", activation.name())
        }
        GraphOp::MatMulLutActivation {
            bits,
            cid,
            activation,
        } => {
            format!("MatMulLut{bits}(id={})+{}", cid.raw(), activation.name())
        }
        GraphOp::FusedRmsNormActivation { activation, .. } => {
            format!("RmsNorm+{}", activation.name())
        }
        GraphOp::FusedLayerNormActivation { activation, .. } => {
            format!("LayerNorm+{}", activation.name())
        }
        GraphOp::FusedGroupNormActivation { activation, .. } => {
            format!("GroupNorm+{}", activation.name())
        }
        GraphOp::Custom { id, arity } => {
            format!("Custom(id={}, arity={})", id.raw(), arity)
        }
        GraphOp::Passthrough => "Passthrough".into(),
    }
}

/// Format a constant reference with its byte size.
fn format_constant(id: &hologram_graph::constant::ConstantId, constants: &ConstantStore) -> String {
    let size = constants.get(*id).map_or(0, |c| c.byte_size());
    format!("Constant(id={}) ({} bytes)", id.raw(), size)
}

/// Format input edges as a bracketed list.
fn format_inputs(inputs: &[InputSlot]) -> String {
    let parts: Vec<String> = inputs
        .iter()
        .filter_map(|slot| match slot.source {
            InputSource::Node(id) => Some(format!("{}", id.index())),
            InputSource::GraphInput { index } => Some(format!("input[{index}]")),
            InputSource::None => None,
        })
        .collect();
    if parts.is_empty() {
        String::new()
    } else {
        format!("[{}]", parts.join(", "))
    }
}
