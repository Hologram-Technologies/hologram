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
        GraphOp::MatMulLut2(cid) => format!("MatMulLut2(id={})", cid.raw()),
        GraphOp::MatMulLut4(cid) => format!("MatMulLut4(id={})", cid.raw()),
        GraphOp::MatMulLut8(cid) => format!("MatMulLut8(id={})", cid.raw()),
        GraphOp::MatMulLut16(cid) => format!("MatMulLut16(id={})", cid.raw()),
        GraphOp::BatchMatMulLut4(cid) => format!("BatchMatMulLut4(id={})", cid.raw()),
        GraphOp::BatchMatMulLut8(cid) => format!("BatchMatMulLut8(id={})", cid.raw()),
        GraphOp::BatchMatMulLut16(cid) => format!("BatchMatMulLut16(id={})", cid.raw()),
        GraphOp::RingPrimUnary(p, level) => format!("RingPrimUnary({}, {:?})", p.name(), level),
        GraphOp::RingPrimBinary(p, level) => format!("RingPrimBinary({}, {:?})", p.name(), level),
        GraphOp::RingActivation(act, level) => format!("RingActivation({:?}, {:?})", act, level),
        GraphOp::RingAccumulate(level) => format!("RingAccumulate({:?})", level),
        GraphOp::RingReduce { op, axis, level } => {
            format!("RingReduce({}, axis={}, {:?})", op.name(), axis, level)
        }
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
        GraphOp::MatMulLut4Activation(cid, activation) => {
            format!("MatMulLut4(id={})+{}", cid.raw(), activation.name())
        }
        GraphOp::MatMulLut8Activation(cid, activation) => {
            format!("MatMulLut8(id={})+{}", cid.raw(), activation.name())
        }
        GraphOp::MatMulLut2Activation(id, act) => {
            format!("MatMulLut2(id={})+{}", id.raw(), act.name())
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
        GraphOp::FusedAddRmsNormActivation { activation, .. } => {
            format!("AddRmsNorm+{}", activation.name())
        }
        GraphOp::FusedInstanceNormActivation { activation, .. } => {
            format!("InstanceNorm+{}", activation.name())
        }
        GraphOp::FusedConv2dActivation { activation, .. } => {
            format!("Conv2d+{}", activation.name())
        }
        GraphOp::FusedConv2dBiasActivation { activation, .. } => {
            format!("Conv2d+Bias+{}", activation.name())
        }
        GraphOp::Custom { id, arity } => {
            format!("Custom(id={}, arity={})", id.raw(), arity)
        }
        GraphOp::Passthrough => "Passthrough".into(),
        GraphOp::Conv2dLut4 {
            cid,
            kernel_h,
            kernel_w,
            group,
            ..
        } => {
            format!(
                "Conv2dLut4(id={}, {}x{}, g={})",
                cid.raw(),
                kernel_h,
                kernel_w,
                group
            )
        }
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
