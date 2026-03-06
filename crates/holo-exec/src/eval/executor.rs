//! KvExecutor: level-by-level graph execution using KV dispatch.
//!
//! The executor walks through each `ParallelLevel` in the schedule,
//! gathers inputs from the arena, dispatches via `KvStore`, and
//! stores outputs back. All mutation is between levels, never during.

use std::collections::HashMap;

use holo_archive::format::graph::SerializedGraph;
use holo_graph::constant::{ConstantData, ConstantId};
use holo_graph::graph::node::{InputSource, Node, NodeId};
use holo_graph::graph::GraphOp;
use holo_graph::schedule::ExecutionSchedule;

use crate::buffer::BufferArena;
use crate::error::{ExecError, ExecResult};
use crate::kv::KvStore;

/// Named graph inputs: maps input index to byte data.
#[derive(Debug, Clone, Default)]
pub struct GraphInputs {
    inputs: HashMap<u32, Vec<u8>>,
}

impl GraphInputs {
    /// Create empty inputs.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
        }
    }

    /// Set data for graph input at `index`.
    pub fn set(&mut self, index: u32, data: Vec<u8>) {
        self.inputs.insert(index, data);
    }

    /// Get data for graph input at `index`.
    pub fn get(&self, index: u32) -> Option<&[u8]> {
        self.inputs.get(&index).map(|v| v.as_slice())
    }

    /// Create from a list of (index, data) pairs.
    #[must_use]
    pub fn from_pairs(pairs: Vec<(u32, Vec<u8>)>) -> Self {
        Self {
            inputs: pairs.into_iter().collect(),
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
///
/// Walks execution levels in order, dispatching each node
/// through `KvStore::dispatch()`. Nodes within a level have
/// all dependencies satisfied by prior levels.
pub struct KvExecutor;

impl KvExecutor {
    /// Execute a serialized graph according to its schedule.
    ///
    /// 1. Build node lookup map
    /// 2. Seed arena with graph inputs and constants
    /// 3. For each level: gather inputs, dispatch, store outputs
    /// 4. Extract named outputs
    pub fn execute(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
    ) -> ExecResult<GraphOutputs> {
        // Build node-id → node lookup
        let node_map: HashMap<NodeId, &Node> = sg.nodes.iter().map(|n| (n.id, n)).collect();

        let mut arena = BufferArena::with_capacity(sg.nodes.len());

        // Seed constants into the arena
        for node in &sg.nodes {
            if let GraphOp::Constant(cid) = &node.op {
                let data = resolve_constant(&sg.constants, *cid)?;
                arena.insert(node.id, data);
            }
        }

        // Execute level by level
        for level in &schedule.levels {
            let mut results: Vec<(NodeId, Vec<u8>)> = Vec::with_capacity(level.node_ids.len());

            for &node_id in &level.node_ids {
                let node = node_map
                    .get(&node_id)
                    .ok_or(ExecError::NodeNotFound(node_id))?;

                // Skip constants (already seeded)
                if matches!(node.op, GraphOp::Constant(_)) {
                    continue;
                }

                // Gather inputs from arena / graph inputs
                let input_bufs = gather_inputs(node, &arena, inputs)?;
                let input_refs: Vec<&[u8]> = input_bufs.iter().map(|v| v.as_slice()).collect();

                let output =
                    KvStore::dispatch_with_constants(&node.op, &input_refs, &sg.constants)?;
                results.push((node_id, output));
            }

            // Insert all results into arena
            for (id, data) in results {
                arena.insert(id, data);
            }
        }

        // Extract named outputs
        let mut outputs = Vec::new();
        for (i, name) in sg.output_names.iter().enumerate() {
            let node_id = sg.output_node_ids[i];
            let data = arena.take(node_id)?;
            outputs.push((name.clone(), data));
        }

        Ok(GraphOutputs { outputs })
    }
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
                let data = arena.get(dep_id)?;
                bufs.push(data.to_vec());
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
fn resolve_constant(
    store: &holo_graph::constant::ConstantStore,
    cid: ConstantId,
) -> ExecResult<Vec<u8>> {
    let data = store
        .get(cid)
        .ok_or(ExecError::ConstantNotFound(cid.raw()))?;
    match data {
        ConstantData::Bytes(bytes) => Ok(bytes.clone()),
        ConstantData::Deferred { .. } => Err(ExecError::UnsupportedOp(
            "deferred constants not yet supported".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::build_schedule;
    use holo_core::op::{LutOp, PrimOp};
    use holo_graph::constant::ConstantStore;
    use holo_graph::graph::node::InputSlot;

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

    /// Simple passthrough: Input → Output
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

    /// Input → Relu → Output
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
        assert_eq!(y.len(), 3);
        // Relu applied element-wise
        assert_eq!(y[0], LutOp::Relu.apply(0));
        assert_eq!(y[1], LutOp::Relu.apply(128));
        assert_eq!(y[2], LutOp::Relu.apply(255));
    }

    /// Input → [Relu, Sigmoid] → Add → Output  (diamond)
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

    /// Two graph inputs added together
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

    /// Multiple outputs
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

    /// Constant node
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

    /// Missing graph input
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
        let result = KvExecutor::execute(&sg, &sched, &GraphInputs::new());
        assert!(result.is_err());
    }

    /// Chain: Input → Relu → Sigmoid → Output
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
        let expected = LutOp::Sigmoid.apply(LutOp::Relu.apply(128));
        assert_eq!(y[0], expected);
    }

    /// Empty graph
    #[test]
    fn empty_graph() {
        let sg = sg_with_io(vec![], vec![], vec![], vec![]);
        let sched = build_schedule(&sg).unwrap();
        let result = KvExecutor::execute(&sg, &sched, &GraphInputs::new()).unwrap();
        assert!(result.is_empty());
    }

    /// get output by index
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

    /// Fused view (pre-composed LUT)
    #[test]
    fn fused_view_execution() {
        use holo_core::view::ElementWiseView;
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

    /// Binary XOR with two branches from same input
    #[test]
    fn xor_self_via_neg() {
        // Input → [Neg, Identity] → Xor → Output
        // Identity is achieved by Output-like passthrough via Input node
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
        let y = result.by_name("y").unwrap();
        // 10 XOR wrapping_neg(10) = 10 XOR 246 = 252
        assert_eq!(y[0], 10u8 ^ 10u8.wrapping_neg());
    }
}
