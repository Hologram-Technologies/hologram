//! Tape-based archive execution.
//!
//! Bridges `hologram-archive` loading with `hologram-exec` tape execution,
//! providing the canonical entry points for running `.holo` archives.

use hologram_archive::loader::plan::LoadedPlan;

use crate::error::{ExecError, ExecResult};
use crate::eval::executor::{GraphInputs, GraphOutputs};
use crate::eval::schedule_bridge::build_schedule;
use crate::kv::CustomOpRegistry;
use crate::kv_cache::KvCacheState;

// ── Tape-based execution ──────────────────────────────────────────────────

/// Build a pre-compiled [`EnumTape`] from a loaded plan.
///
/// The tape pre-resolves kernel enum variants and output element sizes
/// for every node, eliminating per-node op matching, HashMap lookups,
/// and vtable indirection at execution time.
/// Built once at model load time, reused per inference.
pub fn build_tape_from_plan(plan: &LoadedPlan) -> ExecResult<crate::tape::EnumTape> {
    let schedule = build_schedule(plan.graph())?;
    crate::tape_builder::build_tape(plan.graph(), &schedule, None)
}

/// Build a pre-compiled [`EnumTape`] from a loaded plan with a custom op registry.
///
/// Custom ops (`GraphOp::Custom`) are resolved at tape build time by looking up
/// handlers in the registry. The handler closures are baked into the tape as
/// `TapeKernel::Custom` variants — zero-overhead dispatch at inference time.
pub fn build_tape_from_plan_with_ops(
    plan: &LoadedPlan,
    registry: &CustomOpRegistry,
) -> ExecResult<crate::tape::EnumTape> {
    let schedule = build_schedule(plan.graph())?;
    crate::tape_builder::build_tape(plan.graph(), &schedule, Some(registry))
}

/// Execute a pre-compiled tape against a loaded plan.
///
/// Seeds the arena with constants and graph inputs, then runs the tape's
/// pre-resolved enum-dispatch kernels.
pub fn execute_tape(
    tape: &crate::tape::EnumTape,
    plan: &LoadedPlan,
    inputs: &GraphInputs,
) -> ExecResult<GraphOutputs> {
    use hologram_graph::constant::ConstantData;
    use hologram_graph::graph::GraphOp;

    let sg = plan.graph();
    let weights = plan.weights();
    let compiled_dtypes = sg.node_dtypes_map();

    // Seed arena with constants and graph inputs.
    let mut arena = crate::buffer::BufferArena::with_capacity(sg.nodes.len());
    for node in &sg.nodes {
        match &node.op {
            GraphOp::Constant(cid) => {
                let data = match sg.constants.get(*cid) {
                    Some(ConstantData::Bytes(bytes)) => bytes.as_slice(),
                    Some(ConstantData::Deferred {
                        byte_size,
                        source_id,
                    }) => {
                        let start = *source_id as usize;
                        let end = start + *byte_size as usize;
                        if end > weights.len() {
                            return Err(ExecError::ConstantNotFound(cid.raw()));
                        }
                        &weights[start..end]
                    }
                    None => return Err(ExecError::ConstantNotFound(cid.raw())),
                };
                let es = compiled_dtypes
                    .get(&node.id)
                    .map(|d| d.byte_size())
                    .unwrap_or(4);
                arena.insert_borrowed_with_elem_size(node.id, data, es);
            }
            GraphOp::Input => {
                let input_idx = sg
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.op, GraphOp::Input))
                    .position(|n| n.id == node.id);
                if let Some(idx) = input_idx {
                    if let Some(data) = inputs.get(idx as u32) {
                        let es = compiled_dtypes
                            .get(&node.id)
                            .map(|d| d.byte_size())
                            .unwrap_or(8);
                        arena.insert_borrowed_with_elem_size(node.id, data, es);
                    }
                }
            }
            _ => {
                if let Some(dtype) = compiled_dtypes.get(&node.id) {
                    arena.set_elem_size(node.id, dtype.byte_size());
                }
            }
        }
    }

    // Pre-warm arena with output slot allocations (first-inference optimization).
    tape.prewarm_arena(&mut arena);

    // Build tape context with weight access for LUT-GEMM ops.
    let tape_ctx = crate::tape::TapeContext::new(&sg.constants, weights);

    // Execute the tape.
    tape.execute(&mut arena, &tape_ctx)?;

    // Extract outputs.
    let mut outputs = Vec::with_capacity(sg.output_names.len());
    for (i, name) in sg.output_names.iter().enumerate() {
        let node_id = sg.output_node_ids[i];
        outputs.push((name.clone(), arena.take(node_id)?));
    }
    Ok(GraphOutputs::from_named(outputs))
}

/// Execute a tape with KV cache state for autoregressive generation.
///
/// Identical to [`execute_tape`] but seeds the `TapeContext` with an
/// external `KvCacheState`. KvWrite/KvRead tape instructions will
/// read/write from this cache.
pub fn execute_tape_with_kv(
    tape: &crate::tape::EnumTape,
    plan: &LoadedPlan,
    inputs: &GraphInputs,
    kv_state: &mut KvCacheState,
) -> ExecResult<GraphOutputs> {
    use hologram_graph::constant::ConstantData;
    use hologram_graph::graph::GraphOp;

    let sg = plan.graph();
    let weights = plan.weights();
    let compiled_dtypes = sg.node_dtypes_map();

    let mut arena = crate::buffer::BufferArena::with_capacity(sg.nodes.len());
    for node in &sg.nodes {
        match &node.op {
            GraphOp::Constant(cid) => {
                let data = match sg.constants.get(*cid) {
                    Some(ConstantData::Bytes(bytes)) => bytes.as_slice(),
                    Some(ConstantData::Deferred {
                        byte_size,
                        source_id,
                    }) => {
                        let start = *source_id as usize;
                        let end = start + *byte_size as usize;
                        if end > weights.len() {
                            return Err(ExecError::ConstantNotFound(cid.raw()));
                        }
                        &weights[start..end]
                    }
                    None => return Err(ExecError::ConstantNotFound(cid.raw())),
                };
                let es = compiled_dtypes
                    .get(&node.id)
                    .map(|d| d.byte_size())
                    .unwrap_or(4);
                arena.insert_borrowed_with_elem_size(node.id, data, es);
            }
            GraphOp::Input => {
                let input_idx = sg
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.op, GraphOp::Input))
                    .position(|n| n.id == node.id);
                if let Some(idx) = input_idx {
                    if let Some(data) = inputs.get(idx as u32) {
                        let es = compiled_dtypes
                            .get(&node.id)
                            .map(|d| d.byte_size())
                            .unwrap_or(8);
                        arena.insert_borrowed_with_elem_size(node.id, data, es);
                    }
                }
            }
            _ => {
                if let Some(dtype) = compiled_dtypes.get(&node.id) {
                    arena.set_elem_size(node.id, dtype.byte_size());
                }
            }
        }
    }

    tape.prewarm_arena(&mut arena);

    // Swap the KV state into the tape context (takes ownership via RefCell).
    let kv_owned = std::mem::replace(kv_state, KvCacheState::new(0, 0, 0, 0));
    let tape_ctx = crate::tape::TapeContext::with_kv_cache(&sg.constants, weights, kv_owned);

    tape.execute(&mut arena, &tape_ctx)?;

    // Swap the updated KV state back out.
    *kv_state = tape_ctx.kv_state.expect("kv_state was set").into_inner();

    let mut outputs = Vec::with_capacity(sg.output_names.len());
    for (i, name) in sg.output_names.iter().enumerate() {
        let node_id = sg.output_node_ids[i];
        outputs.push((name.clone(), arena.take(node_id)?));
    }
    Ok(GraphOutputs::from_named(outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::HoloWriter;
    use hologram_core::op::LutOp;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::graph::GraphOp;

    #[test]
    fn tape_passthrough() {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Output, &[0])
            .output("y", 1)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape = build_tape_from_plan(&plan).unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![10, 20, 30]);
        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn tape_relu() {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape = build_tape_from_plan(&plan).unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![0, 128, 255]);
        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Relu.apply(0));
        assert_eq!(y[1], LutOp::Relu.apply(128));
        assert_eq!(y[2], LutOp::Relu.apply(255));
    }

    #[test]
    fn tape_float_relu_neg() {
        use hologram_core::op::FloatOp;

        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[0])
            .node_with_inputs(GraphOp::Float(FloatOp::Neg), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .output("y", 3)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape = build_tape_from_plan(&plan).unwrap();

        let input_f32: Vec<u8> = [
            (-3.0f32).to_le_bytes(),
            (0.0f32).to_le_bytes(),
            (2.5f32).to_le_bytes(),
            (-0.1f32).to_le_bytes(),
        ]
        .concat();
        let mut inputs = GraphInputs::new();
        inputs.set(0, input_f32);

        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(result.by_name("y").unwrap());
        assert_eq!(floats[0], -0.0); // neg(relu(-3)) = neg(0) = -0
        assert_eq!(floats[2], -2.5); // neg(relu(2.5)) = -2.5
    }

    #[test]
    fn tape_softmax() {
        use hologram_core::op::FloatOp;

        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Float(FloatOp::Softmax { size: 4 }), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape = build_tape_from_plan(&plan).unwrap();

        let input_f32: Vec<u8> = [
            1.0f32.to_le_bytes(),
            2.0f32.to_le_bytes(),
            3.0f32.to_le_bytes(),
            4.0f32.to_le_bytes(),
        ]
        .concat();
        let mut inputs = GraphInputs::new();
        inputs.set(0, input_f32);

        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(result.by_name("y").unwrap());
        let sum: f32 = floats.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum = {sum}, expected 1.0"
        );
    }

    #[test]
    fn tape_lut_gemm_q4() {
        use hologram_graph::constant::ConstantData;

        let k = 4usize;
        let n = 4usize;
        let weights = vec![1.0f32; k * n];
        let qw = crate::lut_gemm::quantize_4bit(&weights, k as u32, n as u32);
        let qw_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).unwrap().to_vec();

        let g = hologram_graph::builder::GraphBuilder::new()
            .input("a")
            .node_from_graph_input(GraphOp::Input, 0)
            .matmul_lut_4bit(ConstantData::Bytes(qw_bytes), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("c", 2)
            .build();

        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape = build_tape_from_plan(&plan).unwrap();

        let activations = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
        let mut inputs = GraphInputs::new();
        inputs.set(0, act_bytes);

        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        let output: &[f32] = bytemuck::cast_slice(result.by_name("c").unwrap());
        assert_eq!(
            output.len(),
            2 * n,
            "expected 2×{n} output, got {}",
            output.len()
        );

        for &v in &output[..n] {
            assert!(
                (v - 1.0).abs() < 0.5,
                "Q4 tape matmul row0: got {v}, expected ~1.0"
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn tape_file_roundtrip() {
        use std::io::Write;

        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_tape_file.holo");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }

        let loader = hologram_archive::HoloLoader::open(&path).unwrap();
        let plan = loader.load().unwrap();
        let tape = build_tape_from_plan(&plan).unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![100]);
        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[LutOp::Sigmoid.apply(100)]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tape_custom_op_passthrough() {
        use hologram_graph::graph::CustomOpId;
        use std::sync::Arc;

        // Build a graph: Input → Custom(id=1, arity=1) → Output
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(
                GraphOp::Custom {
                    id: CustomOpId(1),
                    arity: 1,
                },
                &[0],
            )
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();

        // Register a passthrough handler
        let mut registry = CustomOpRegistry::new();
        registry.register(
            CustomOpId(1),
            1,
            Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
        );

        let tape = build_tape_from_plan_with_ops(&plan, &registry).unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![42, 43, 44]);
        let result = execute_tape(&tape, &plan, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[42, 43, 44]);
    }
}
