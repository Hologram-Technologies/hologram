//! Convenience functions for executing archives directly.
//!
//! Bridges `hologram-archive` loading with `hologram-exec` execution,
//! providing one-call entry points for the common case. Dispatches
//! execution via the archive's `LayerHeader` entrypoints when present.

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
