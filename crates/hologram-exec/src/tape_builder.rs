//! Tape builder: compiles a `SerializedGraph` + `ExecutionSchedule` into a `Tape`.
//!
//! This is the "compile" step that pre-resolves kernel function pointers and
//! output element sizes for every node. The resulting tape can be executed
//! repeatedly per inference call without per-node op matching or HashMap lookups.
//!
//! Built once at model load time, amortized across all inference calls.

use std::collections::HashMap;

use hologram_archive::format::graph::SerializedGraph;
use hologram_core::op::{FloatDType, FloatOp};
use hologram_graph::graph::node::{InputSource, NodeId};
use hologram_graph::graph::GraphOp;
use hologram_graph::schedule::ExecutionSchedule;

use crate::error::{ExecError, ExecResult};
use crate::tape::{BoxedInstruction, BoxedKernel, BoxedTape};

/// Build a [`BoxedTape`] from a serialized graph and its execution schedule.
///
/// For each node in schedule order:
/// - Resolves a `BoxedKernel` that captures the op's parameters
/// - Pre-computes `output_elem_size` from the node's dtype
/// - Records input indices for zero-copy gathering at execution time
///
/// Constants and graph inputs are skipped (they are seeded into the arena
/// before tape execution).
pub fn build_tape(sg: &SerializedGraph, schedule: &ExecutionSchedule) -> ExecResult<BoxedTape> {
    // Build flat lookup tables for O(1) access by node index.
    let max_idx = sg
        .nodes
        .iter()
        .map(|n| n.id.index() as usize + 1)
        .max()
        .unwrap_or(0);

    let mut node_by_idx: Vec<Option<usize>> = vec![None; max_idx];
    for (i, n) in sg.nodes.iter().enumerate() {
        let idx = n.id.index() as usize;
        if idx < max_idx {
            node_by_idx[idx] = Some(i);
        }
    }

    let dtypes: HashMap<NodeId, FloatDType> = sg.node_dtypes_map();

    let total_nodes: usize = schedule.levels.iter().map(|l| l.node_ids.len()).sum();
    let mut tape = BoxedTape::with_capacity(total_nodes, schedule.levels.len());

    for level in &schedule.levels {
        for &node_id in &level.node_ids {
            let idx = node_id.index() as usize;
            let node_pos = if idx < max_idx {
                node_by_idx[idx]
            } else {
                None
            };
            let Some(node_pos) = node_pos else {
                continue;
            };
            let node = &sg.nodes[node_pos];

            // Skip constants and inputs — they're seeded into the arena.
            match &node.op {
                GraphOp::Constant(_) | GraphOp::Input => continue,
                _ => {}
            }

            // Resolve kernel.
            let kernel = resolve_kernel(&node.op)?;

            // Pre-compute output elem_size.
            let output_elem_size = compute_elem_size(node_id, &node.op, &dtypes);

            // Collect input indices.
            let input_indices: Vec<u32> = node
                .inputs
                .iter()
                .filter_map(|slot| match slot.source {
                    InputSource::Node(id) => Some(id.index()),
                    InputSource::GraphInput { .. } => None,
                    InputSource::None => None,
                })
                .collect();

            tape.push(BoxedInstruction {
                kernel,
                output_idx: node_id.index(),
                input_indices,
                output_elem_size,
            });
        }
        tape.end_level();
    }

    Ok(tape)
}

/// Resolve a `GraphOp` to a boxed kernel that captures its parameters.
fn resolve_kernel(op: &GraphOp) -> ExecResult<BoxedKernel> {
    match op {
        GraphOp::Float(fop) => resolve_float_kernel(fop),
        GraphOp::FusedFloatChain(chain) => {
            let chain = chain.clone();
            Ok(Box::new(move |inputs, _ctx| {
                crate::float_dispatch::dispatch_fused_chain(&chain, inputs)
            }))
        }
        GraphOp::Output => Ok(Box::new(|inputs, _ctx| {
            Ok(inputs.first().map(|b| b.to_vec()).unwrap_or_default())
        })),
        GraphOp::Lut(_) | GraphOp::FusedView(_) => {
            let view = op
                .to_view()
                .ok_or_else(|| ExecError::UnsupportedOp("Lut/FusedView without view".into()))?;
            Ok(Box::new(move |inputs, _ctx| {
                Ok(crate::kv::KvStore::apply_unary(&view, inputs[0]))
            }))
        }
        GraphOp::Prim(p) => {
            let p = *p;
            if p.arity() == 1 {
                let view = op
                    .to_view()
                    .ok_or_else(|| ExecError::UnsupportedOp("Prim without view".into()))?;
                Ok(Box::new(move |inputs, _ctx| {
                    Ok(crate::kv::KvStore::apply_unary(&view, inputs[0]))
                }))
            } else {
                Ok(Box::new(move |inputs, _ctx| {
                    crate::kv::KvStore::apply_binary(p, inputs[0], inputs[1])
                }))
            }
        }
        GraphOp::MatMulLut4(cid) | GraphOp::BatchMatMulLut4(cid) => {
            let cid = *cid;
            Ok(Box::new(move |inputs, _ctx| {
                // LUT-GEMM: constants/weights not available in tape kernel context yet.
                // Will be wired via WeightCache in a future iteration.
                let _ = (inputs, cid);
                Err(ExecError::UnsupportedOp(
                    "LUT-GEMM Q4 requires WeightCache (not yet wired into tape)".into(),
                ))
            }))
        }
        GraphOp::MatMulLut8(cid) | GraphOp::BatchMatMulLut8(cid) => {
            let cid = *cid;
            Ok(Box::new(move |inputs, _ctx| {
                let _ = (inputs, cid);
                Err(ExecError::UnsupportedOp(
                    "LUT-GEMM Q8 requires WeightCache (not yet wired into tape)".into(),
                ))
            }))
        }
        _ => Err(ExecError::UnsupportedOp(format!(
            "tape builder: unsupported op {:?}",
            op
        ))),
    }
}

/// Resolve a `FloatOp` to a boxed kernel, handling dynamic size parameters.
///
/// For ops like Softmax, RmsNorm, LayerNorm, and Reduce* where the `size`
/// parameter may be stale (compiled at one seq_len but executed at another),
/// the kernel infers the correct size from the input buffer at runtime.
/// This eliminates the need for the executor's `resolve_dynamic_sizes` pass.
fn resolve_float_kernel(fop: &FloatOp) -> ExecResult<BoxedKernel> {
    use crate::float_dispatch;

    match fop {
        // Softmax/LogSoftmax: size may be stale — infer from input[0] buffer.
        // The correct size is input_floats, which for a [batch, seq, hidden]
        // tensor is the last dim (hidden_size). Since we don't have shapes,
        // we use the full element count as fallback (works for 1-D and 2-D
        // when batch*seq=1, which covers decode-step execution).
        FloatOp::Softmax { size } => {
            let compiled_size = *size;
            Ok(Box::new(move |inputs, _ctx| {
                let n_floats = inputs[0].len() / 4;
                // If compiled size divides evenly into the buffer, use it.
                // Otherwise infer: if it doesn't divide, the compiled size is
                // stale and we fall back to the full buffer (1-D softmax).
                let actual_size = if compiled_size > 0
                    && n_floats > 0
                    && n_floats % (compiled_size as usize) == 0
                {
                    compiled_size
                } else {
                    n_floats as u32
                };
                float_dispatch::dispatch_float_ctx(
                    &FloatOp::Softmax { size: actual_size },
                    inputs,
                    None,
                )
            }))
        }
        FloatOp::LogSoftmax { size } => {
            let compiled_size = *size;
            Ok(Box::new(move |inputs, _ctx| {
                let n_floats = inputs[0].len() / 4;
                let actual_size = if compiled_size > 0
                    && n_floats > 0
                    && n_floats % (compiled_size as usize) == 0
                {
                    compiled_size
                } else {
                    n_floats as u32
                };
                float_dispatch::dispatch_float_ctx(
                    &FloatOp::LogSoftmax { size: actual_size },
                    inputs,
                    None,
                )
            }))
        }
        // RmsNorm/LayerNorm with size=0 sentinel: infer from buffer.
        FloatOp::RmsNorm { size: 0, epsilon } => {
            let eps = *epsilon;
            Ok(Box::new(move |inputs, _ctx| {
                let n_floats = (inputs[0].len() / 4) as u32;
                float_dispatch::dispatch_float_ctx(
                    &FloatOp::RmsNorm {
                        size: n_floats,
                        epsilon: eps,
                    },
                    inputs,
                    None,
                )
            }))
        }
        FloatOp::LayerNorm { size: 0, epsilon } => {
            let eps = *epsilon;
            Ok(Box::new(move |inputs, _ctx| {
                let n_floats = (inputs[0].len() / 4) as u32;
                float_dispatch::dispatch_float_ctx(
                    &FloatOp::LayerNorm {
                        size: n_floats,
                        epsilon: eps,
                    },
                    inputs,
                    None,
                )
            }))
        }
        // Reduce* with size=0: infer from buffer.
        FloatOp::ReduceSum { size: 0 } => Ok(Box::new(move |inputs, _ctx| {
            let n = (inputs[0].len() / 4) as u32;
            float_dispatch::dispatch_float_ctx(&FloatOp::ReduceSum { size: n }, inputs, None)
        })),
        FloatOp::ReduceMean { size: 0 } => Ok(Box::new(move |inputs, _ctx| {
            let n = (inputs[0].len() / 4) as u32;
            float_dispatch::dispatch_float_ctx(&FloatOp::ReduceMean { size: n }, inputs, None)
        })),
        FloatOp::ReduceMax { size: 0 } => Ok(Box::new(move |inputs, _ctx| {
            let n = (inputs[0].len() / 4) as u32;
            float_dispatch::dispatch_float_ctx(&FloatOp::ReduceMax { size: n }, inputs, None)
        })),
        FloatOp::ReduceMin { size: 0 } => Ok(Box::new(move |inputs, _ctx| {
            let n = (inputs[0].len() / 4) as u32;
            float_dispatch::dispatch_float_ctx(&FloatOp::ReduceMin { size: n }, inputs, None)
        })),
        FloatOp::ReduceProd { size: 0 } => Ok(Box::new(move |inputs, _ctx| {
            let n = (inputs[0].len() / 4) as u32;
            float_dispatch::dispatch_float_ctx(&FloatOp::ReduceProd { size: n }, inputs, None)
        })),
        // Default: capture the op as-is (parameters are correct at compile time).
        _ => {
            let fop = *fop;
            Ok(Box::new(move |inputs, ctx| {
                float_dispatch::dispatch_float_ctx(&fop, inputs, ctx)
            }))
        }
    }
}

/// Pre-compute the output element size for a node.
///
/// Uses the compiled dtype when available, falling back to the op's
/// declared output dtype. Default: 4 (f32).
fn compute_elem_size(node_id: NodeId, op: &GraphOp, dtypes: &HashMap<NodeId, FloatDType>) -> u8 {
    // Try compiled dtype first (most reliable).
    if let Some(dtype) = dtypes.get(&node_id) {
        return dtype.byte_size() as u8;
    }
    // Infer from op's output dtype declaration.
    if let GraphOp::Float(fop) = op {
        // For most ops, output is f32 (4 bytes).
        // Special cases: Cast changes dtype, IsNaN outputs u8, etc.
        match fop {
            FloatOp::IsNaN => return 1,
            FloatOp::Cast { to, .. } => return to.byte_size() as u8,
            FloatOp::Shape { .. } => return 8, // i64
            _ => {}
        }
    }
    4 // f32 default
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_graph::graph::edge;
    use hologram_graph::graph::Graph;

    fn make_simple_graph() -> (SerializedGraph, ExecutionSchedule) {
        let mut graph = Graph::new();
        let input_id = graph.add_node(GraphOp::Input);
        let input_idx = graph.add_input("x");

        let relu_id = graph.add_node(GraphOp::Float(FloatOp::Relu));
        edge::connect_graph_input(&mut graph, input_idx, relu_id, 0);

        let out_id = graph.add_node(GraphOp::Output);
        edge::connect(&mut graph, relu_id, out_id, 0);
        graph.add_output("y", out_id);

        let sg = SerializedGraph::from_graph(&graph);
        let schedule = ExecutionSchedule::build(&graph).expect("schedule should build");
        (sg, schedule)
    }

    #[test]
    fn build_tape_from_simple_graph() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule).expect("build_tape should succeed");
        // Should have instructions for Relu and Output (Input is skipped).
        assert!(
            !tape.instructions.is_empty(),
            "expected at least 1 instruction, got 0",
        );
    }

    #[test]
    fn tape_elem_size_defaults_to_f32() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule).expect("build_tape should succeed");
        for instr in &tape.instructions {
            // Relu is f32 → f32, so elem_size should be 4.
            assert_eq!(instr.output_elem_size, 4);
        }
    }
}
