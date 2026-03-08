//! KvExecutor: level-by-level graph execution using KV dispatch.
//!
//! The executor walks through each `ParallelLevel` in the schedule,
//! gathers inputs from the arena, dispatches via `KvStore`, and
//! stores outputs back. All mutation is between levels, never during.

use std::collections::HashMap;

use hologram_archive::format::graph::SerializedGraph;
use hologram_graph::constant::{ConstantData, ConstantId};
use hologram_graph::graph::node::{InputSource, Node, NodeId};
use hologram_graph::graph::GraphOp;
use hologram_graph::schedule::levels::ParallelLevel;
use hologram_graph::schedule::ExecutionSchedule;

use crate::buffer::{BufferArena, ShapeMap};
use crate::error::{ExecError, ExecResult};
use crate::float_dispatch;
use crate::kv::{CustomOpRegistry, KvStore};
use hologram_core::op::FloatOp;

/// Named graph inputs: maps input index to byte data and optional shape.
#[derive(Debug, Clone, Default)]
pub struct GraphInputs {
    inputs: HashMap<u32, Vec<u8>>,
    shapes: HashMap<u32, Vec<usize>>,
}

impl GraphInputs {
    /// Create empty inputs.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Set data for graph input at `index`.
    pub fn set(&mut self, index: u32, data: Vec<u8>) {
        self.inputs.insert(index, data);
    }

    /// Set data with an explicit N-D shape for graph input at `index`.
    pub fn set_with_shape(&mut self, index: u32, data: Vec<u8>, shape: Vec<usize>) {
        self.inputs.insert(index, data);
        self.shapes.insert(index, shape);
    }

    /// Get data for graph input at `index`.
    pub fn get(&self, index: u32) -> Option<&[u8]> {
        self.inputs.get(&index).map(|v| v.as_slice())
    }

    /// Get the shape for graph input at `index`, if set.
    pub fn shape(&self, index: u32) -> Option<&[usize]> {
        self.shapes.get(&index).map(|v| v.as_slice())
    }

    /// Create from a list of (index, data) pairs.
    #[must_use]
    pub fn from_pairs(pairs: Vec<(u32, Vec<u8>)>) -> Self {
        Self {
            inputs: pairs.into_iter().collect(),
            shapes: HashMap::new(),
        }
    }
}

/// Named graph outputs: list of (name, data) pairs.
#[derive(Debug, Clone, Default)]
pub struct GraphOutputs {
    outputs: Vec<(String, Vec<u8>)>,
}

impl GraphOutputs {
    /// Number of outputs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Whether there are no outputs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Get output by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<(&str, &[u8])> {
        self.outputs
            .get(index)
            .map(|(name, data)| (name.as_str(), data.as_slice()))
    }

    /// Get output by name.
    #[must_use]
    pub fn by_name(&self, name: &str) -> Option<&[u8]> {
        self.outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// Consume into inner vec.
    #[must_use]
    pub fn into_inner(self) -> Vec<(String, Vec<u8>)> {
        self.outputs
    }
}

/// Stateless graph executor using KV-lookup dispatch.
pub struct KvExecutor;

impl KvExecutor {
    /// Execute a serialized graph according to its schedule.
    pub fn execute(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
    ) -> ExecResult<GraphOutputs> {
        Self::execute_with_progress(sg, schedule, inputs, |_, _| {})
    }

    /// Execute with a per-level progress callback.
    ///
    /// `on_level(level_index, nodes_executed)` is called after each level completes.
    pub fn execute_with_progress<F>(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        on_level: F,
    ) -> ExecResult<GraphOutputs>
    where
        F: FnMut(usize, usize),
    {
        Self::execute_core(sg, schedule, inputs, None, &[], on_level)
    }

    /// Execute with a custom op registry (no progress callback).
    pub fn execute_with_registry(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: &CustomOpRegistry,
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core(sg, schedule, inputs, Some(registry), &[], |_, _| {})
    }

    /// Execute with archive weight data (no custom ops or progress).
    ///
    /// Primary method for archives containing `ConstantData::Deferred`
    /// references that resolve from the weight blob.
    pub fn execute_with_plan(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        weights: &[u8],
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core(sg, schedule, inputs, None, weights, |_, _| {})
    }

    /// Execute with a custom op registry and archive weight data.
    ///
    /// `weights` is the raw weight blob from the `.holo` archive.
    /// `ConstantData::Deferred` nodes are resolved by slicing into this blob
    /// using `source_id` as the byte offset.
    pub fn execute_with_weights(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: &CustomOpRegistry,
        weights: &[u8],
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core(sg, schedule, inputs, Some(registry), weights, |_, _| {})
    }

    pub(crate) fn execute_core<F>(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: Option<&CustomOpRegistry>,
        weights: &[u8],
        mut on_level: F,
    ) -> ExecResult<GraphOutputs>
    where
        F: FnMut(usize, usize),
    {
        let node_map = build_node_map(sg);
        let mut arena = seed_arena(sg, weights)?;
        let mut shape_map = seed_shape_map(sg, &arena, inputs);

        for (i, level) in schedule.levels.iter().enumerate() {
            let count = dispatch_level(
                level,
                &node_map,
                &mut arena,
                &mut shape_map,
                inputs,
                &sg.constants,
                registry,
                weights,
            )?;
            on_level(i, count);
        }

        extract_named_outputs(sg, &mut arena)
    }
}

/// Build a `NodeId → &Node` lookup map for the graph.
fn build_node_map(sg: &SerializedGraph) -> HashMap<NodeId, &Node> {
    sg.nodes.iter().map(|n| (n.id, n)).collect()
}

/// Initialize the arena and seed all constant nodes.
fn seed_arena(sg: &SerializedGraph, weights: &[u8]) -> ExecResult<BufferArena> {
    let mut arena = BufferArena::with_capacity(sg.nodes.len());
    for node in &sg.nodes {
        if let GraphOp::Constant(cid) = &node.op {
            arena.insert(node.id, resolve_constant(&sg.constants, *cid, weights)?);
        }
    }
    Ok(arena)
}

/// Initialize shape map: seed constants with 1-D shapes and graph inputs with
/// caller-provided N-D shapes (if available).
fn seed_shape_map(sg: &SerializedGraph, arena: &BufferArena, inputs: &GraphInputs) -> ShapeMap {
    let mut sm = ShapeMap::new();

    for node in &sg.nodes {
        match &node.op {
            GraphOp::Constant(_) => {
                if let Ok(buf) = arena.get(node.id) {
                    sm.insert(node.id, ShapeMap::infer_1d(buf.len()));
                }
            }
            GraphOp::Input => {
                // Use caller-provided N-D shape if available.
                if let Some(slot_idx) = find_input_slot_index(node) {
                    if let Some(shape) = inputs.shape(slot_idx) {
                        sm.insert(node.id, shape.to_vec());
                    }
                }
            }
            _ => {}
        }
    }
    sm
}

/// Find the graph input slot index that an Input node reads from.
fn find_input_slot_index(node: &Node) -> Option<u32> {
    node.inputs.iter().find_map(|slot| match slot.source {
        InputSource::GraphInput { index } => Some(index),
        _ => None,
    })
}

/// Resolve `size=0` sentinels in FloatOps using the input shape's last dimension.
///
/// When the compiler can't determine the last-dim size (symbolic shapes like seq_len),
/// it emits size=0. At runtime we resolve this from the actual input shape.
/// If the shape-based size doesn't divide the input, falls back to the full input length.
fn resolve_dynamic_sizes(
    op: &GraphOp,
    input_shapes: &[Vec<usize>],
    input_refs: &[&[u8]],
) -> Option<GraphOp> {
    let resolve = |input_idx: usize| -> u32 {
        let shape_last = input_shapes
            .get(input_idx)
            .and_then(|s| s.last().copied())
            .unwrap_or(0);
        // Validate: the last dim must divide the actual buffer size.
        let buf_floats = input_refs.get(input_idx).map(|b| b.len() / 4).unwrap_or(0);
        if shape_last > 0 && buf_floats > 0 && buf_floats.is_multiple_of(shape_last) {
            shape_last as u32
        } else if buf_floats > 0 {
            // Shape tracking failed — use full buffer (single-row softmax).
            buf_floats as u32
        } else {
            0
        }
    };

    match op {
        GraphOp::Float(fop) => {
            let resolved = match fop {
                FloatOp::Softmax { size: 0 } => FloatOp::Softmax { size: resolve(0) },
                FloatOp::LogSoftmax { size: 0 } => FloatOp::LogSoftmax { size: resolve(0) },
                FloatOp::RmsNorm { size: 0, epsilon } => FloatOp::RmsNorm {
                    size: resolve(0),
                    epsilon: *epsilon,
                },
                FloatOp::LayerNorm { size: 0, epsilon } => FloatOp::LayerNorm {
                    size: resolve(0),
                    epsilon: *epsilon,
                },
                FloatOp::ReduceSum { size: 0 } => FloatOp::ReduceSum { size: resolve(0) },
                FloatOp::ReduceMean { size: 0 } => FloatOp::ReduceMean { size: resolve(0) },
                FloatOp::ReduceMax { size: 0 } => FloatOp::ReduceMax { size: resolve(0) },
                FloatOp::ReduceMin { size: 0 } => FloatOp::ReduceMin { size: resolve(0) },
                _ => return None,
            };
            Some(GraphOp::Float(resolved))
        }
        _ => None,
    }
}

/// Execute all non-constant nodes in a single level; returns the count dispatched.
#[allow(clippy::too_many_arguments)]
fn dispatch_level(
    level: &ParallelLevel,
    node_map: &HashMap<NodeId, &Node>,
    arena: &mut BufferArena,
    shape_map: &mut ShapeMap,
    inputs: &GraphInputs,
    constants: &hologram_graph::constant::ConstantStore,
    registry: Option<&CustomOpRegistry>,
    weights: &[u8],
) -> ExecResult<usize> {
    let mut results: Vec<(NodeId, Vec<u8>, Vec<usize>)> = Vec::with_capacity(level.node_ids.len());

    for &node_id in &level.node_ids {
        let node = node_map
            .get(&node_id)
            .ok_or(ExecError::NodeNotFound(node_id))?;

        if matches!(node.op, GraphOp::Constant(_)) {
            continue;
        }

        let input_bufs = gather_inputs(node, arena, inputs)?;
        let input_refs: Vec<&[u8]> = input_bufs.iter().map(|v| v.as_slice()).collect();

        // Gather input shapes for shape-aware ops.
        let input_shapes: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .zip(input_bufs.iter())
            .map(|(slot, buf)| {
                let dep_id = match slot.source {
                    InputSource::Node(id) => Some(id),
                    _ => None,
                };
                dep_id
                    .and_then(|id| shape_map.get(id).map(|s| s.to_vec()))
                    .unwrap_or_else(|| ShapeMap::infer_1d(buf.len()))
            })
            .collect();

        // Handle shape-aware ops (Reshape, Transpose, Shape) specially.
        let (result, out_shape) = match &node.op {
            GraphOp::Float(FloatOp::Shape { dtype }) => {
                // Return the input's logical shape as an i64 tensor.
                // If shape is 1-D (inferred from bytes), recompute using the correct dtype.
                let in_shape = &input_shapes[0];
                let shape_i64: Vec<i64> = if in_shape.len() == 1 {
                    // Re-derive from byte count using the declared dtype.
                    let elem_size = dtype.byte_size().max(1);
                    vec![input_refs[0].len() as i64 / elem_size as i64]
                } else {
                    in_shape.iter().map(|&d| d as i64).collect()
                };
                let data: Vec<u8> = bytemuck::cast_slice(&shape_i64).to_vec();
                let out_shape = vec![shape_i64.len()];
                (data, out_shape)
            }
            GraphOp::Float(FloatOp::Concat { .. }) if input_refs.len() > 2 => {
                // N-input concat: concatenate all input bytes.
                let mut data = Vec::new();
                for inp in &input_refs {
                    data.extend_from_slice(inp);
                }
                let out_shape = vec![data.len() / 4];
                (data, out_shape)
            }
            GraphOp::Float(FloatOp::Reshape) => {
                let (data, shape) = float_dispatch::dispatch_reshape_with_shape(&input_refs)
                    .map_err(|e| ExecError::ShapeMismatch {
                        expected: format!("node {node_id:?} (Reshape)"),
                        actual: e.to_string(),
                    })?;
                if std::env::var("HOLO_DEBUG_SHAPES").is_ok() {
                    eprintln!(
                        "[SHAPE] {node_id:?} Reshape: in={:?} → out={:?} ({}B)",
                        &input_shapes[0],
                        &shape,
                        data.len()
                    );
                }
                (data, shape)
            }
            GraphOp::Float(FloatOp::MatMul { m, k, n })
                if input_shapes.len() >= 2
                    && input_shapes[0].len() >= 3
                    && input_shapes[1].len() >= 3 =>
            {
                // Batched matmul: both A and B have ≥3-D shapes.
                let a_shape = &input_shapes[0];
                let b_shape = &input_shapes[1];
                if std::env::var("HOLO_DEBUG_SHAPES").is_ok() {
                    eprintln!("[SHAPE] {node_id:?} MatMul(BATCHED) m={m} k={k} n={n}: A={a_shape:?} B={b_shape:?}");
                }
                float_dispatch::dispatch_batched_matmul(&input_refs, a_shape, b_shape).map_err(
                    |e| ExecError::ShapeMismatch {
                        expected: format!("node {node_id:?} (batched MatMul)"),
                        actual: e.to_string(),
                    },
                )?
            }
            GraphOp::Float(FloatOp::Transpose { perm, ndim }) => {
                let nd = *ndim as usize;
                let p = &perm[..nd];
                let in_shape = &input_shapes[0];
                // If input shape has fewer dims than perm expects,
                // reshape the 1-D shape to match ndim dims.
                let effective_shape = if in_shape.len() < nd {
                    let total: usize = in_shape.iter().copied().fold(1usize, usize::saturating_mul);
                    let mut s = vec![1usize; nd];
                    if nd > 0 {
                        s[0] = total;
                    }
                    s
                } else {
                    in_shape.clone()
                };
                let (data, shape) =
                    float_dispatch::dispatch_transpose(input_refs[0], p, &effective_shape)
                        .map_err(|e| ExecError::ShapeMismatch {
                            expected: format!("node {node_id:?} (Transpose)"),
                            actual: e.to_string(),
                        })?;
                if std::env::var("HOLO_DEBUG_SHAPES").is_ok() {
                    eprintln!("[SHAPE] {node_id:?} Transpose perm={p:?}: in={in_shape:?} eff={effective_shape:?} → out={shape:?}");
                }
                (data, shape)
            }
            _ => {
                // Resolve size=0 sentinel in FloatOps using the input's last shape dim.
                let resolved_op = resolve_dynamic_sizes(&node.op, &input_shapes, &input_refs);
                let dispatch_op = resolved_op.as_ref().unwrap_or(&node.op);
                let result = KvStore::dispatch_with_constants(
                    dispatch_op,
                    &input_refs,
                    constants,
                    registry,
                    weights,
                )
                .map_err(|e| ExecError::ShapeMismatch {
                    expected: format!("node {node_id:?} ({:?})", node.op),
                    actual: e.to_string(),
                })?;
                // Infer output shape.
                let shape = match dispatch_op {
                    GraphOp::Float(fop) => {
                        let ishape_refs: Vec<&[usize]> =
                            input_shapes.iter().map(|s| s.as_slice()).collect();
                        float_dispatch::infer_output_shape(fop, &ishape_refs, result.len())
                    }
                    // For graph inputs, preserve caller-provided N-D shape if available.
                    GraphOp::Input => shape_map
                        .get(node_id)
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| ShapeMap::infer_1d(result.len())),
                    // Output, FusedView, etc.: pass through first input shape.
                    _ => input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| ShapeMap::infer_1d(result.len())),
                };
                if std::env::var("HOLO_DEBUG_SHAPES").is_ok() {
                    if let GraphOp::Float(fop) = dispatch_op {
                        let in_sizes: Vec<usize> = input_refs.iter().map(|b| b.len() / 4).collect();
                        eprintln!("[SHAPE] {node_id:?} {fop:?}: in_sizes={in_sizes:?} in_shapes={input_shapes:?} → out={shape:?} ({}B)", result.len());
                    }
                }
                (result, shape)
            }
        };

        results.push((node_id, result, out_shape));
    }

    let dispatched = results.len();
    for (id, data, shape) in results {
        arena.insert(id, data);
        shape_map.insert(id, shape);
    }
    Ok(dispatched)
}

/// Extract the named output buffers from the arena in declaration order.
fn extract_named_outputs(
    sg: &SerializedGraph,
    arena: &mut BufferArena,
) -> ExecResult<GraphOutputs> {
    let mut outputs = Vec::with_capacity(sg.output_names.len());
    for (i, name) in sg.output_names.iter().enumerate() {
        let node_id = sg.output_node_ids[i];
        outputs.push((name.clone(), arena.take(node_id)?));
    }
    Ok(GraphOutputs { outputs })
}

/// Gather input buffers for a node from the arena and graph inputs.
fn gather_inputs<'a>(
    node: &Node,
    arena: &'a BufferArena,
    graph_inputs: &'a GraphInputs,
) -> ExecResult<Vec<Vec<u8>>> {
    let mut bufs = Vec::with_capacity(node.inputs.len());
    for (slot_idx, slot) in node.inputs.iter().enumerate() {
        match slot.source {
            InputSource::Node(dep_id) => {
                bufs.push(arena.get(dep_id)?.to_vec());
            }
            InputSource::GraphInput { index } => {
                let data = graph_inputs
                    .get(index)
                    .ok_or(ExecError::MissingGraphInput(index))?;
                bufs.push(data.to_vec());
            }
            InputSource::None => {
                return Err(ExecError::MissingInput {
                    node: node.id,
                    slot: slot_idx,
                });
            }
        }
    }
    Ok(bufs)
}

/// Resolve a constant ID to byte data.
///
/// `Deferred` constants are resolved from the `weights` blob using
/// `source_id` as the byte offset and `byte_size` as the length.
fn resolve_constant(
    store: &hologram_graph::constant::ConstantStore,
    cid: ConstantId,
    weights: &[u8],
) -> ExecResult<Vec<u8>> {
    let data = store
        .get(cid)
        .ok_or(ExecError::ConstantNotFound(cid.raw()))?;
    match data {
        ConstantData::Bytes(bytes) => Ok(bytes.clone()),
        ConstantData::Deferred {
            byte_size,
            source_id,
        } => {
            let start = *source_id as usize;
            let end = start + *byte_size as usize;
            if end > weights.len() {
                return Err(ExecError::ConstantNotFound(cid.raw()));
            }
            Ok(weights[start..end].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::build_schedule;
    use hologram_core::op::{LutOp, PrimOp};
    use hologram_graph::constant::ConstantStore;
    use hologram_graph::graph::node::InputSlot;

    fn nid(n: u32) -> NodeId {
        NodeId::new(n, 0)
    }

    fn node(id: u32, op: GraphOp, inputs: Vec<InputSlot>) -> Node {
        Node {
            id: nid(id),
            op,
            inputs,
            num_outputs: 1,
        }
    }

    fn sg_with_io(
        nodes: Vec<Node>,
        input_names: Vec<&str>,
        output_names: Vec<&str>,
        output_ids: Vec<NodeId>,
    ) -> SerializedGraph {
        SerializedGraph {
            nodes,
            input_names: input_names.into_iter().map(String::from).collect(),
            output_names: output_names.into_iter().map(String::from).collect(),
            output_node_ids: output_ids,
            constants: ConstantStore::new(),
        }
    }

    #[test]
    fn passthrough() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(1)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![42, 43, 44]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.by_name("y").unwrap(), &[42, 43, 44]);
    }

    #[test]
    fn relu_pipeline() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(2)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![0, 128, 255]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Relu.apply(0));
        assert_eq!(y[1], LutOp::Relu.apply(128));
        assert_eq!(y[2], LutOp::Relu.apply(255));
    }

    #[test]
    fn diamond_graph() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Lut(LutOp::Sigmoid),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    3,
                    GraphOp::Prim(PrimOp::Add),
                    vec![InputSlot::from_node(nid(1)), InputSlot::from_node(nid(2))],
                ),
                node(4, GraphOp::Output, vec![InputSlot::from_node(nid(3))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(4)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        let expected = LutOp::Relu
            .apply(128)
            .wrapping_add(LutOp::Sigmoid.apply(128));
        assert_eq!(y[0], expected);
    }

    #[test]
    fn two_inputs() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Input, vec![InputSlot::from_graph_input(1)]),
                node(
                    2,
                    GraphOp::Prim(PrimOp::Add),
                    vec![InputSlot::from_node(nid(0)), InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["a", "b"],
            vec!["sum"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let inputs = GraphInputs::from_pairs(vec![(0, vec![10, 20]), (1, vec![5, 100])]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.by_name("sum").unwrap(), &[15, 120]);
    }

    #[test]
    fn multiple_outputs() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["raw", "activated"],
            vec![nid(2), nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![100]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.by_name("raw").unwrap(), &[100]);
        assert_eq!(
            result.by_name("activated").unwrap(),
            &[LutOp::Relu.apply(100)]
        );
    }

    #[test]
    fn constant_node() {
        let mut constants = ConstantStore::new();
        let cid = constants.insert(ConstantData::Bytes(vec![7, 8, 9]));
        let sg = SerializedGraph {
            nodes: vec![
                node(0, GraphOp::Constant(cid), vec![]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            input_names: Vec::new(),
            output_names: vec!["out".into()],
            output_node_ids: vec![nid(1)],
            constants,
        };
        let sched = build_schedule(&sg).unwrap();
        let result = KvExecutor::execute(&sg, &sched, &GraphInputs::new()).unwrap();
        assert_eq!(result.by_name("out").unwrap(), &[7, 8, 9]);
    }

    #[test]
    fn missing_graph_input() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(1)],
        );
        let sched = build_schedule(&sg).unwrap();
        assert!(KvExecutor::execute(&sg, &sched, &GraphInputs::new()).is_err());
    }

    #[test]
    fn chain_of_unary() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Lut(LutOp::Sigmoid),
                    vec![InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Sigmoid.apply(LutOp::Relu.apply(128)));
    }

    #[test]
    fn empty_graph() {
        let sg = sg_with_io(vec![], vec![], vec![], vec![]);
        let sched = build_schedule(&sg).unwrap();
        assert!(KvExecutor::execute(&sg, &sched, &GraphInputs::new())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn output_by_index() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(1)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![1, 2, 3]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let (name, data) = result.get(0).unwrap();
        assert_eq!(name, "y");
        assert_eq!(data, &[1, 2, 3]);
    }

    #[test]
    fn fused_view_execution() {
        use hologram_core::view::ElementWiseView;
        let view = ElementWiseView::new(|x| x.wrapping_mul(3));
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::FusedView(view),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(2)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![1, 2, 3, 10]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[3, 6, 9, 30]);
    }

    #[test]
    fn xor_self_via_neg() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Prim(PrimOp::Neg),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Prim(PrimOp::Xor),
                    vec![InputSlot::from_node(nid(0)), InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![10]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap()[0], 10u8 ^ 10u8.wrapping_neg());
    }

    /// Progress callback fires once per level with sequential indices.
    #[test]
    fn progress_callback_fires_per_level() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(2)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);

        let mut events: Vec<(usize, usize)> = Vec::new();
        let result = KvExecutor::execute_with_progress(&sg, &sched, &inputs, |idx, count| {
            events.push((idx, count));
        })
        .unwrap();

        assert_eq!(result.by_name("y").unwrap(), &[LutOp::Relu.apply(128)]);
        assert!(!events.is_empty());
        for (i, (idx, _)) in events.iter().enumerate() {
            assert_eq!(*idx, i);
        }
    }

    /// Total dispatched node count across all levels equals graph size.
    #[test]
    fn progress_callback_total_count() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Lut(LutOp::Sigmoid),
                    vec![InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![64]);

        let mut total = 0usize;
        KvExecutor::execute_with_progress(&sg, &sched, &inputs, |_, count| total += count).unwrap();

        assert_eq!(total, 4); // Input, Relu, Sigmoid, Output
    }

    #[test]
    fn deferred_constant_with_weights() {
        use hologram_graph::constant::ConstantData;
        let mut constants = ConstantStore::new();
        let cid = constants.insert(ConstantData::Deferred {
            byte_size: 3,
            source_id: 0,
        });
        let sg = SerializedGraph {
            nodes: vec![
                node(0, GraphOp::Constant(cid), vec![]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            input_names: Vec::new(),
            output_names: vec!["out".into()],
            output_node_ids: vec![nid(1)],
            constants,
        };
        let sched = build_schedule(&sg).unwrap();
        let weights = vec![10, 20, 30, 99];
        let result =
            KvExecutor::execute_with_plan(&sg, &sched, &GraphInputs::new(), &weights).unwrap();
        assert_eq!(result.by_name("out").unwrap(), &[10, 20, 30]);
    }
}
