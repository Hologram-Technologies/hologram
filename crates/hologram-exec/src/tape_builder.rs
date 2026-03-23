//! Tape builder: compiles a `SerializedGraph` + `ExecutionSchedule` into a tape.
//!
//! This is the "compile" step that pre-resolves kernel enum variants and
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
use crate::kv::CustomOpRegistry;
use crate::tape::{EnumTape, TapeInstruction, TapeKernel};

/// Build an [`EnumTape`] from a serialized graph and its execution schedule.
///
/// For each node in schedule order:
/// - Resolves a [`TapeKernel`] enum variant (no closure, no heap allocation)
/// - Pre-computes `output_elem_size` from the node's dtype
/// - Records input indices for zero-copy gathering at execution time
///
/// Constants and graph inputs are skipped (they are seeded into the arena
/// before tape execution).
pub fn build_tape(
    sg: &SerializedGraph,
    schedule: &ExecutionSchedule,
    registry: Option<&CustomOpRegistry>,
) -> ExecResult<EnumTape> {
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
    let shapes: HashMap<NodeId, Vec<usize>> = sg.node_shapes_map();

    let total_nodes: usize = schedule.levels.iter().map(|l| l.node_ids.len()).sum();
    let mut tape = EnumTape::with_capacity(total_nodes, schedule.levels.len());

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

            // Resolve kernel enum variant.
            let kernel = resolve_kernel(&node.op, registry)?;

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

            // Pre-compute output byte size hint from compiled shapes.
            let output_byte_hint = compute_output_byte_hint(node_id, &shapes, output_elem_size);

            // Pre-compute weight offset for LUT-GEMM prefetching.
            let weight_offset_hint = compute_weight_offset(&kernel, &sg.constants);

            tape.push(TapeInstruction {
                kernel,
                output_idx: node_id.index(),
                input_indices,
                output_elem_size,
                output_byte_hint,
                weight_offset_hint,
                passthrough: false,
                can_reuse_input: false,
            });
        }
        tape.end_level();
    }

    // ── Post-pass: compute consumer counts and set optimization flags ──
    apply_reuse_flags(&mut tape);

    Ok(tape)
}

/// Scan instructions to compute per-node consumer counts, then set
/// `passthrough` (for Output ops with single-consumer inputs) and
/// `can_reuse_input` (for unary inline ops with single-consumer inputs).
fn apply_reuse_flags(tape: &mut EnumTape) {
    // Count how many instructions consume each node index.
    let max_idx = tape
        .instructions
        .iter()
        .map(|i| i.output_idx as usize)
        .max()
        .unwrap_or(0);
    let mut consumer_counts = vec![0u32; max_idx + 1];
    for instr in &tape.instructions {
        for &idx in &instr.input_indices {
            let i = idx as usize;
            if i < consumer_counts.len() {
                consumer_counts[i] += 1;
            }
        }
    }

    let is_single_consumer = |idx: u32| -> bool {
        let i = idx as usize;
        i < consumer_counts.len() && consumer_counts[i] == 1
    };

    for instr in &mut tape.instructions {
        match &instr.kernel {
            // Output passthrough: move buffer directly if input has one consumer.
            TapeKernel::Output if instr.input_indices.len() == 1 => {
                if is_single_consumer(instr.input_indices[0]) {
                    instr.passthrough = true;
                }
            }
            // Unary inline ops: reuse input buffer in-place if single consumer.
            TapeKernel::InlineRelu
            | TapeKernel::InlineNeg
            | TapeKernel::InlineAbs
            | TapeKernel::InlineSigmoid
            | TapeKernel::InlineSilu
            | TapeKernel::InlineTanh
            | TapeKernel::InlineGelu
            | TapeKernel::InlineExp
            | TapeKernel::InlineReciprocal
                if instr.input_indices.len() == 1 =>
            {
                if is_single_consumer(instr.input_indices[0]) {
                    instr.can_reuse_input = true;
                }
            }
            _ => {}
        }
    }
}

/// Resolve a `GraphOp` to a [`TapeKernel`] enum variant.
///
/// No closures, no heap allocation — just selects the right variant
/// and captures the op parameters inline.
fn resolve_kernel(op: &GraphOp, registry: Option<&CustomOpRegistry>) -> ExecResult<TapeKernel> {
    match op {
        GraphOp::Float(fop) => Ok(resolve_float_kernel(fop)),
        GraphOp::FusedFloatChain(chain) => Ok(TapeKernel::FusedFloatChain(chain.clone())),
        GraphOp::Output => Ok(TapeKernel::Output),
        GraphOp::Lut(_) | GraphOp::FusedView(_) => {
            let view = op
                .to_view()
                .ok_or_else(|| ExecError::UnsupportedOp("Lut/FusedView without view".into()))?;
            Ok(TapeKernel::LutView(view))
        }
        GraphOp::Prim(p) => {
            if p.arity() == 1 {
                let view = op
                    .to_view()
                    .ok_or_else(|| ExecError::UnsupportedOp("Prim without view".into()))?;
                Ok(TapeKernel::PrimUnary(view))
            } else {
                Ok(TapeKernel::PrimBinary(*p))
            }
        }
        GraphOp::MatMulLut4(cid) | GraphOp::BatchMatMulLut4(cid) => {
            Ok(TapeKernel::MatMulLut4(*cid))
        }
        GraphOp::MatMulLut8(cid) | GraphOp::BatchMatMulLut8(cid) => {
            Ok(TapeKernel::MatMulLut8(*cid))
        }
        GraphOp::Custom { id, arity: _ } => {
            let reg = registry.ok_or_else(|| {
                ExecError::UnsupportedOp(format!(
                    "custom op {} requires a CustomOpRegistry",
                    id.raw()
                ))
            })?;
            let handler = reg.get_handler(*id).ok_or_else(|| {
                ExecError::UnsupportedOp(format!("custom op {} not registered", id.raw()))
            })?;
            Ok(TapeKernel::Custom(handler.clone()))
        }
        _ => Err(ExecError::UnsupportedOp(format!(
            "tape builder: unsupported op {:?}",
            op
        ))),
    }
}

/// Resolve a `FloatOp` to a [`TapeKernel`] variant.
///
/// KvWrite/KvRead are intercepted and mapped to dedicated TapeKernel variants.
/// All other ops are stored as `TapeKernel::Float(op)` — size inference
/// happens at dispatch time inside `dispatch_float_into`.
fn resolve_float_kernel(fop: &FloatOp) -> TapeKernel {
    match fop {
        // Inline hot ops — skip backend + dispatch_float_into entirely.
        FloatOp::Relu => TapeKernel::InlineRelu,
        FloatOp::Neg => TapeKernel::InlineNeg,
        FloatOp::Sigmoid => TapeKernel::InlineSigmoid,
        FloatOp::Silu => TapeKernel::InlineSilu,
        FloatOp::Tanh => TapeKernel::InlineTanh,
        FloatOp::Gelu => TapeKernel::InlineGelu,
        FloatOp::Exp => TapeKernel::InlineExp,
        FloatOp::Add => TapeKernel::InlineAdd,
        FloatOp::Mul => TapeKernel::InlineMul,
        FloatOp::Sub => TapeKernel::InlineSub,
        FloatOp::Div => TapeKernel::InlineDiv,
        FloatOp::Abs => TapeKernel::InlineAbs,
        FloatOp::Reciprocal => TapeKernel::InlineReciprocal,

        // Inline custom ops — bake dimensions/parameters at build time.
        FloatOp::MatMul { m, k, n } => TapeKernel::InlineMatMul {
            m: *m,
            k: *k,
            n: *n,
        },
        FloatOp::Softmax { size } => TapeKernel::InlineSoftmax { size: *size },
        FloatOp::RmsNorm { size, epsilon } => TapeKernel::InlineRmsNorm {
            size: *size,
            epsilon: *epsilon,
        },

        FloatOp::KvWrite {
            layer,
            n_kv_heads,
            head_dim,
            is_key,
        } => TapeKernel::KvWrite {
            layer: *layer,
            n_kv_heads: *n_kv_heads,
            head_dim: *head_dim,
            is_key: *is_key,
        },
        FloatOp::KvRead {
            layer,
            n_kv_heads,
            head_dim,
        } => TapeKernel::KvRead {
            layer: *layer,
            n_kv_heads: *n_kv_heads,
            head_dim: *head_dim,
        },
        _ => TapeKernel::Float(*fop),
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
        match fop {
            FloatOp::IsNaN => return 1,
            FloatOp::Cast { to, .. } => return to.byte_size() as u8,
            FloatOp::Shape { .. } => return 8, // i64
            _ => {}
        }
    }
    4 // f32 default
}

/// Pre-compute the total output byte size for a node from compiled shapes.
///
/// Returns the product of shape dimensions × element size, or 0 if the
/// shape is unknown or contains a 0-sentinel (dynamic dimension).
fn compute_output_byte_hint(
    node_id: NodeId,
    shapes: &HashMap<NodeId, Vec<usize>>,
    elem_size: u8,
) -> u32 {
    let Some(shape) = shapes.get(&node_id) else {
        return 0;
    };
    if shape.is_empty() {
        return 0;
    }
    if shape.contains(&0) {
        return 0;
    }
    let n_elements: usize = shape.iter().product();
    let byte_size = n_elements.saturating_mul(elem_size as usize);
    if byte_size > u32::MAX as usize {
        0
    } else {
        byte_size as u32
    }
}

/// Compute the byte offset into the weight archive for LUT-GEMM constant prefetch.
///
/// Returns 0 for non-LUT-GEMM ops (no weight prefetch needed).
fn compute_weight_offset(
    kernel: &TapeKernel,
    constants: &hologram_graph::constant::ConstantStore,
) -> u32 {
    let cid = match kernel {
        TapeKernel::MatMulLut4(cid) | TapeKernel::MatMulLut8(cid) => *cid,
        _ => return 0,
    };
    match constants.get(cid) {
        Some(hologram_graph::constant::ConstantData::Deferred { source_id, .. }) => {
            *source_id as u32
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_graph::graph::edge;
    use hologram_graph::graph::Graph;

    fn make_simple_graph() -> (SerializedGraph, ExecutionSchedule) {
        let mut graph = Graph::new();
        let _input_id = graph.add_node(GraphOp::Input);
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
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");
        assert!(
            !tape.instructions.is_empty(),
            "expected at least 1 instruction, got 0",
        );
    }

    #[test]
    fn tape_elem_size_defaults_to_f32() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");
        for instr in &tape.instructions {
            assert_eq!(instr.output_elem_size, 4);
        }
    }

    #[test]
    fn tape_kernel_is_enum_not_boxed() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");
        // Verify the Relu instruction is a Float variant (not a Box).
        for instr in &tape.instructions {
            match &instr.kernel {
                TapeKernel::InlineRelu | TapeKernel::Output => {}
                other => panic!(
                    "unexpected kernel variant: {:?}",
                    std::mem::discriminant(other)
                ),
            }
        }
    }
}
