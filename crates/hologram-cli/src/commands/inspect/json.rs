//! `--detail json` output.

use hologram_archive::section::{
    SECTION_CUSTOM_BASE, SECTION_LAYER_HEADER, SECTION_PIPELINE, SECTION_WEIGHT_INDEX,
};
use hologram_archive::weight::TensorMetadata;
use hologram_archive::LoadedPlan;
use hologram_graph::constant::ConstantStore;
use hologram_graph::graph::node::{InputSlot, InputSource, Node};
use hologram_graph::graph::GraphOp;
use hologram_graph::ExecutionSchedule;
use serde_json::{json, Value};

use super::InspectArgs;

/// Print the full archive as a JSON object.
pub fn print(args: &InspectArgs, data: &[u8], plan: &LoadedPlan, schedule: &ExecutionSchedule) {
    let obj = build(args, data, plan, schedule);
    println!("{}", serde_json::to_string_pretty(&obj).unwrap());
}

/// Build the complete JSON value.
pub(super) fn build(
    args: &InspectArgs,
    data: &[u8],
    plan: &LoadedPlan,
    schedule: &ExecutionSchedule,
) -> Value {
    json!({
        "archive": archive_json(args, data, plan),
        "graph": graph_json(plan),
        "schedule": schedule_json(schedule),
        "sections": sections_json(plan),
        "weights": weights_json(plan),
    })
}

/// Archive metadata.
fn archive_json(args: &InspectArgs, data: &[u8], plan: &LoadedPlan) -> Value {
    let h = plan.header();
    json!({
        "file": args.file.display().to_string(),
        "size_bytes": data.len(),
        "format_version": h.version,
        "graph_offset": h.graph_offset,
        "graph_size": h.graph_size,
        "weights_offset": h.weights_offset,
        "weights_size": h.weights_size,
        "graph_checksum": format!("{:#010x}", h.graph_checksum),
        "weights_checksum": format!("{:#010x}", h.weights_checksum),
        "section_count": h.section_count,
        "total_size": h.total_size,
    })
}

/// Graph nodes and I/O.
fn graph_json(plan: &LoadedPlan) -> Value {
    let sg = plan.graph();
    let nodes: Vec<Value> = sg
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| node_json(i, n, &sg.constants))
        .collect();
    json!({
        "node_count": sg.node_count(),
        "inputs": sg.input_names,
        "outputs": sg.output_names,
        "nodes": nodes,
    })
}

/// Single node.
fn node_json(idx: usize, node: &Node, constants: &ConstantStore) -> Value {
    let inputs: Vec<Value> = node.inputs.iter().filter_map(input_json).collect();
    json!({
        "index": idx,
        "op": op_json(&node.op, constants),
        "inputs": inputs,
        "num_outputs": node.num_outputs,
    })
}

/// Operation descriptor.
fn op_json(op: &GraphOp, constants: &ConstantStore) -> Value {
    match op {
        GraphOp::Input => json!("Input"),
        GraphOp::Output => json!("Output"),
        GraphOp::Prim(p) => json!({"Prim": p.name()}),
        GraphOp::Lut(l) => json!({"Lut": l.name()}),
        GraphOp::FusedView(_) => json!("FusedView"),
        GraphOp::FusedView16(_) => json!("FusedView16"),
        GraphOp::Constant(id) => {
            let size = constants.get(*id).map_or(0, |c| c.byte_size());
            json!({"Constant": {"id": id.raw(), "byte_size": size}})
        }
        GraphOp::CallSubgraph(s) => json!({"CallSubgraph": s.raw()}),
        GraphOp::MatMulLut4(id) => json!({"MatMulLut4": id.raw()}),
        GraphOp::MatMulLut8(id) => json!({"MatMulLut8": id.raw()}),
        GraphOp::BatchMatMulLut4(id) => json!({"BatchMatMulLut4": id.raw()}),
        GraphOp::BatchMatMulLut8(id) => json!({"BatchMatMulLut8": id.raw()}),
        GraphOp::MatMulLut16(id) => json!({"MatMulLut16": id.raw()}),
        GraphOp::BatchMatMulLut16(id) => json!({"BatchMatMulLut16": id.raw()}),
        GraphOp::RingPrimUnary(p, level) => {
            json!({"RingPrimUnary": {"op": p.name(), "level": format!("{:?}", level)}})
        }
        GraphOp::RingPrimBinary(p, level) => {
            json!({"RingPrimBinary": {"op": p.name(), "level": format!("{:?}", level)}})
        }
        GraphOp::RingActivation(act, level) => {
            json!({"RingActivation": {"op": format!("{:?}", act), "level": format!("{:?}", level)}})
        }
        GraphOp::RingAccumulate(level) => {
            json!({"RingAccumulate": {"level": format!("{:?}", level)}})
        }
        GraphOp::RingReduce { op, axis, level } => {
            json!({"RingReduce": {"op": op.name(), "axis": axis, "level": format!("{:?}", level)}})
        }
        GraphOp::Float(f) => json!({"Float": f.name()}),
        GraphOp::FusedFloatChain(chain) => {
            let names: Vec<&str> = chain.iter().map(|f| f.name()).collect();
            json!({"FusedFloatChain": names})
        }
        GraphOp::Custom { id, arity } => {
            json!({"Custom": {"id": id.raw(), "arity": arity}})
        }
        GraphOp::Passthrough => json!("Passthrough"),
    }
}

/// Input edge.
fn input_json(slot: &InputSlot) -> Option<Value> {
    match slot.source {
        InputSource::Node(id) => Some(json!({
            "source": "node",
            "node_index": id.index(),
            "output_port": slot.output_port,
        })),
        InputSource::GraphInput { index } => Some(json!({
            "source": "graph_input",
            "index": index,
        })),
        InputSource::None => None,
    }
}

/// Execution schedule.
fn schedule_json(schedule: &ExecutionSchedule) -> Value {
    let levels: Vec<Value> = schedule
        .levels
        .iter()
        .map(|l| {
            let ids: Vec<u32> = l.node_ids.iter().map(|n| n.index()).collect();
            json!({"node_ids": ids})
        })
        .collect();
    json!({
        "num_levels": schedule.levels.len(),
        "critical_path": schedule.critical_path,
        "parallelism_ratio": schedule.parallelism_ratio(),
        "levels": levels,
    })
}

/// Section table entries.
fn sections_json(plan: &LoadedPlan) -> Value {
    let entries: Vec<Value> = plan
        .sections()
        .entries
        .iter()
        .map(|e| {
            json!({
                "kind": e.kind,
                "kind_name": section_kind_name(e.kind),
                "offset": e.offset,
                "size": e.size,
                "checksum": format!("{:#010x}", e.checksum),
            })
        })
        .collect();
    json!(entries)
}

/// Map section kind to name.
fn section_kind_name(kind: u32) -> &'static str {
    match kind {
        SECTION_WEIGHT_INDEX => "weight_index",
        SECTION_LAYER_HEADER => "layer_header",
        SECTION_PIPELINE => "pipeline",
        k if k >= SECTION_CUSTOM_BASE => "custom",
        _ => "unknown",
    }
}

/// Weight tensor metadata.
fn weights_json(plan: &LoadedPlan) -> Value {
    let entry = plan.sections().find(SECTION_WEIGHT_INDEX);
    let Some(entry) = entry else {
        return json!([]);
    };
    let raw = plan.weights();
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > raw.len() {
        return json!([]);
    }
    let Ok(tensors) = deserialize_tensors(&raw[start..end]) else {
        return json!([]);
    };
    let items: Vec<Value> = tensors.iter().map(tensor_json).collect();
    json!(items)
}

/// Deserialize tensor metadata.
fn deserialize_tensors(bytes: &[u8]) -> Result<Vec<TensorMetadata>, rkyv::rancor::Error> {
    rkyv::from_bytes::<Vec<TensorMetadata>, rkyv::rancor::Error>(bytes)
}

/// Single tensor metadata.
fn tensor_json(t: &TensorMetadata) -> Value {
    let mut obj = json!({
        "name": t.name,
        "shape": t.shape,
        "dtype": t.dtype.name(),
        "offset": t.offset,
        "size": t.size,
        "checksum": format!("{:#010x}", t.checksum),
    });
    if let Some(q) = &t.quantization {
        obj["quantization"] = json!({
            "scheme": format!("{:?}", q.scheme),
            "scale": q.scale,
            "zero_point": q.zero_point,
            "min_val": q.min_val,
            "max_val": q.max_val,
            "group_size": q.group_size,
        });
    }
    obj
}
