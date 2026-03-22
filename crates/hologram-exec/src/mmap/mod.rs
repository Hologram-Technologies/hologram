//! Convenience functions for executing archives directly.
//!
//! Bridges `hologram-archive` loading with `hologram-exec` execution,
//! providing one-call entry points for the common case. Dispatches
//! execution via the archive's `LayerHeader` entrypoints when present.

use std::collections::HashMap;

use hologram_archive::loader::plan::LoadedPlan;
use hologram_archive::LayerEntrypoint;

use crate::error::{ExecError, ExecResult};
use crate::eval::executor::{GraphInputs, GraphOutputs, KvExecutor};
use crate::eval::schedule_bridge::build_schedule;
use crate::kv::CustomOpRegistry;

/// Execute a loaded plan using its `LayerHeader` entrypoints.
///
/// If the archive contains a `LayerHeader`, walks the layer schedule
/// and dispatches each layer by entrypoint type. Falls back to direct
/// graph execution if no `LayerHeader` is present (backward compat).
pub fn execute_plan(plan: &LoadedPlan, inputs: &GraphInputs) -> ExecResult<GraphOutputs> {
    match plan.layer_header() {
        Some(lh) => dispatch_layers(lh, plan, inputs),
        None => execute_graph_entrypoint(plan, inputs),
    }
}

/// Dispatch layers according to the `LayerHeader` schedule.
fn dispatch_layers(
    lh: &hologram_archive::LayerHeader,
    plan: &LoadedPlan,
    inputs: &GraphInputs,
) -> ExecResult<GraphOutputs> {
    let mut result = None;
    for level in &lh.schedule {
        for layer_id in level {
            let desc = lh
                .find_layer(*layer_id)
                .ok_or_else(|| ExecError::UnsupportedOp(format!("layer {:?}", layer_id.0)))?;
            match &desc.entrypoint {
                LayerEntrypoint::Graph => {
                    result = Some(execute_graph_entrypoint(plan, inputs)?);
                }
                LayerEntrypoint::Subgraph(id) => {
                    return Err(ExecError::UnsupportedOp(format!("subgraph {id}")));
                }
                LayerEntrypoint::External(r) => {
                    return Err(ExecError::UnsupportedOp(format!("external {r}")));
                }
            }
        }
    }
    result.map_or_else(|| execute_graph_entrypoint(plan, inputs), Ok)
}

/// Execute the archive's embedded graph with weights.
fn execute_graph_entrypoint(plan: &LoadedPlan, inputs: &GraphInputs) -> ExecResult<GraphOutputs> {
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_with_plan(plan.graph(), &schedule, inputs, plan.weights())
}

/// Execute a loaded plan with pre-projected shape hints from `walk_shape_context()`.
///
/// `shape_hints` maps `NodeId.index() → concrete shape` for every node projected
/// by the `ShapeContextGraph` walker. Hints override compiled shapes and inferred
/// shapes, ensuring correct execution for variable-length inputs (seq>1, batch>1).
///
/// Typical call pattern:
/// ```rust,ignore
/// let hints = walk_shape_context(&ctx_graph, &input_shapes, &shape_values, &mut map);
/// let outputs = execute_plan_with_shape_hints(&plan, &inputs, &hints)?;
/// ```
pub fn execute_plan_with_shape_hints(
    plan: &LoadedPlan,
    inputs: &GraphInputs,
    shape_hints: &HashMap<u32, Vec<usize>>,
) -> ExecResult<GraphOutputs> {
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_with_shape_hints(
        plan.graph(),
        &schedule,
        inputs,
        plan.weights(),
        shape_hints,
    )
}

/// Execute a loaded plan with shape hints and a mutable KV cache state.
///
/// Like [`execute_plan_with_shape_hints`] but also threads a `KvCacheState`
/// through the dispatch loop for `FloatOp::KvWrite`/`KvRead` ops.
pub fn execute_plan_with_kv_state(
    plan: &LoadedPlan,
    inputs: &GraphInputs,
    shape_hints: &HashMap<u32, Vec<usize>>,
    kv_state: &mut crate::kv_cache::KvCacheState,
) -> ExecResult<GraphOutputs> {
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_with_kv_state(
        plan.graph(),
        &schedule,
        inputs,
        plan.weights(),
        shape_hints,
        kv_state,
    )
}

/// Execute a loaded plan and capture ALL intermediate node outputs.
///
/// Requires the `profile` feature. Returns `IntermediateCapture` containing
/// every node's output buffer, shape, and the normal graph outputs.
/// Used for node-by-node conformance testing against ORT.
#[cfg(feature = "profile")]
pub fn execute_plan_with_intermediates(
    plan: &LoadedPlan,
    inputs: &GraphInputs,
) -> ExecResult<crate::eval::executor::IntermediateCapture> {
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_with_intermediates(plan.graph(), &schedule, inputs, plan.weights())
}

/// Execute a loaded plan with shape hints and capture ALL intermediate outputs.
///
/// Combines shape-aware execution (correct for variable-length inputs) with
/// intermediate capture (for node-by-node conformance testing against ORT).
/// Requires the `profile` feature.
#[cfg(feature = "profile")]
pub fn execute_plan_with_intermediates_and_shape_hints(
    plan: &LoadedPlan,
    inputs: &GraphInputs,
    shape_hints: &std::collections::HashMap<u32, Vec<usize>>,
) -> ExecResult<crate::eval::executor::IntermediateCapture> {
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_with_intermediates_and_shape_hints(
        plan.graph(),
        &schedule,
        inputs,
        plan.weights(),
        shape_hints,
    )
}

/// Execute a .holo archive from raw bytes.
///
/// Parses the archive, dispatches via entrypoints, and runs the graph.
pub fn execute_bytes(data: &[u8], inputs: &GraphInputs) -> ExecResult<GraphOutputs> {
    let plan = hologram_archive::load_from_bytes(data)?;
    execute_plan(&plan, inputs)
}

/// Execute a .holo archive with a custom op registry.
///
/// Enables graphs containing `GraphOp::Custom` nodes.
pub fn execute_bytes_with_ops(
    data: &[u8],
    inputs: &GraphInputs,
    registry: &CustomOpRegistry,
) -> ExecResult<GraphOutputs> {
    let plan = hologram_archive::load_from_bytes(data)?;
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_with_weights(plan.graph(), &schedule, inputs, registry, plan.weights())
}

/// Execute a .holo archive with a per-level progress callback.
///
/// `on_level(level_index, nodes_executed)` fires after each schedule level completes.
pub fn execute_bytes_with_progress<F>(
    data: &[u8],
    inputs: &GraphInputs,
    on_level: F,
) -> ExecResult<GraphOutputs>
where
    F: FnMut(usize, usize),
{
    let plan = hologram_archive::load_from_bytes(data)?;
    let schedule = build_schedule(plan.graph())?;
    KvExecutor::execute_core(
        plan.graph(),
        &schedule,
        inputs,
        None,
        plan.weights(),
        on_level,
    )
}

/// Execute a .holo archive from a file path (requires `std` feature).
///
/// Memory-maps the file, parses, dispatches via entrypoints, and runs.
#[cfg(feature = "std")]
pub fn execute_file(path: &std::path::Path, inputs: &GraphInputs) -> ExecResult<GraphOutputs> {
    let loader = hologram_archive::HoloLoader::open(path)?;
    let plan = loader.load()?;
    execute_plan(&plan, inputs)
}

// ── Tape-based execution ──────────────────────────────────────────────────

/// Build a pre-compiled [`EnumTape`] from a loaded plan.
///
/// The tape pre-resolves kernel enum variants and output element sizes
/// for every node, eliminating per-node op matching, HashMap lookups,
/// and vtable indirection at execution time.
/// Built once at model load time, reused per inference.
pub fn build_tape_from_plan(plan: &LoadedPlan) -> ExecResult<crate::tape::EnumTape> {
    let schedule = build_schedule(plan.graph())?;
    crate::tape_builder::build_tape(plan.graph(), &schedule)
}

/// Execute a pre-compiled tape against a loaded plan.
///
/// Seeds the arena with constants and graph inputs, then runs the tape's
/// pre-resolved enum-dispatch kernels. Faster than [`execute_plan`] because
/// the tape avoids per-node op matching, dtype lookups, and vtable indirection.
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

/// Execute a loaded plan with zero-copy semantics.
///
/// This is functionally identical to [`execute_plan`] — the arena's
/// `insert_borrowed` path already achieves zero-copy for constant weights
/// from mmap'd memory. Constant tensor data from the archive's weight section
/// is borrowed directly into the `BufferArena` without copying, and the mmap
/// keeps the underlying pages resident for the lifetime of the `LoadedPlan`.
///
/// This function exists as an explicit entry point for callers who want to
/// document zero-copy intent.
pub fn execute_plan_zero_copy(plan: &LoadedPlan, inputs: &GraphInputs) -> ExecResult<GraphOutputs> {
    execute_plan(plan, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::HoloWriter;
    use hologram_core::op::LutOp;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::graph::GraphOp;

    #[test]
    fn execute_bytes_passthrough() {
        // Input(0) → Output(1)
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0) // 0
            .node_with_inputs(GraphOp::Output, &[0]) // 1
            .output("y", 1)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![10, 20, 30]);
        let result = execute_bytes(&archive, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn execute_bytes_relu() {
        // Input(0) → Relu(1) → Output(2)
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .node_with_inputs(GraphOp::Output, &[1]) // 2
            .output("y", 2)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![0, 128, 255]);
        let result = execute_bytes(&archive, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Relu.apply(0));
        assert_eq!(y[1], LutOp::Relu.apply(128));
        assert_eq!(y[2], LutOp::Relu.apply(255));
    }

    #[test]
    fn execute_plan_works() {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Output, &[0])
            .output("y", 1)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = hologram_archive::load_from_bytes(&archive).unwrap();

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![42]);
        let result = execute_plan(&plan, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[42]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn execute_file_works() {
        use std::io::Write;

        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 1
            .node_with_inputs(GraphOp::Output, &[1]) // 2
            .output("y", 2)
            .build();
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_exec_file.holo");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }

        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![100]);
        let result = execute_file(&path, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[LutOp::Sigmoid.apply(100)]);

        std::fs::remove_file(&path).ok();
    }
}
