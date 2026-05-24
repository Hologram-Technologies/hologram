//! `Graph` structure (spec VI.1).

use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;
use crate::node::{Node, NodeId, InputSource, QuantAttrs, FusionAttrs};
use crate::constant::ConstantStore;
use crate::registry::ShapeRegistry;
use crate::schedule::Schedule;

#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    inputs: SmallVec<[NodeId; 8]>,
    outputs: SmallVec<[NodeId; 8]>,
    constants: ConstantStore,
    shape_registry: ShapeRegistry,
    schedule: Option<Schedule>,
    /// Sparse per-node quantization attributes (spec X-5). Keyed on
    /// `NodeId.0`. Empty for graphs with no quantized weights.
    quant_attrs: Vec<(NodeId, QuantAttrs)>,
    /// Sparse per-node fusion metadata (spec VI.3). Keyed on `NodeId`.
    /// Stores epilogue activation discriminants for fused ops.
    fusion_attrs: Vec<(NodeId, FusionAttrs)>,
}

impl Graph {
    pub fn new() -> Self { Self::default() }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0 as usize)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id.0 as usize)
    }

    pub fn nodes(&self) -> &[Node] { &self.nodes }
    pub fn node_count(&self) -> usize { self.nodes.len() }

    pub fn add_input(&mut self, id: NodeId) { self.inputs.push(id); }
    pub fn add_output(&mut self, id: NodeId) { self.outputs.push(id); }

    pub fn inputs(&self) -> &[NodeId] { &self.inputs }
    pub fn outputs(&self) -> &[NodeId] { &self.outputs }

    pub fn constants(&self) -> &ConstantStore { &self.constants }
    pub fn constants_mut(&mut self) -> &mut ConstantStore { &mut self.constants }

    pub fn shape_registry(&self) -> &ShapeRegistry { &self.shape_registry }
    pub fn shape_registry_mut(&mut self) -> &mut ShapeRegistry { &mut self.shape_registry }

    pub fn schedule(&self) -> Option<&Schedule> { self.schedule.as_ref() }
    pub fn set_schedule(&mut self, sched: Schedule) { self.schedule = Some(sched); }

    /// Attach quantization parameters to a node (spec X-5). The node's
    /// op is expected to be `OpKind::Dequantize`; the compiler reads
    /// these into `LoweredNode.quant` during lowering.
    pub fn set_quant_attrs(&mut self, id: NodeId, attrs: QuantAttrs) {
        if let Some(slot) = self.quant_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.quant_attrs.push((id, attrs));
        }
    }

    /// Retrieve quantization parameters for a node, or `None` if the node
    /// has no quantization metadata.
    pub fn quant_attrs(&self, id: NodeId) -> Option<QuantAttrs> {
        self.quant_attrs.iter().find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Attach fusion metadata to a node. Used by fusion passes to
    /// record the epilogue activation discriminant.
    pub fn set_fusion_attrs(&mut self, id: NodeId, attrs: FusionAttrs) {
        if let Some(slot) = self.fusion_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.fusion_attrs.push((id, attrs));
        }
    }

    pub fn fusion_attrs(&self, id: NodeId) -> Option<FusionAttrs> {
        self.fusion_attrs.iter().find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    // --- Mutation API (used by fusion passes) ---

    /// Replace the op of a node in-place. Used by fusion to swap an
    /// activation node's op to a fused variant.
    pub fn replace_op(&mut self, id: NodeId, new_op: crate::node::GraphOp) {
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.op = new_op;
        }
    }

    /// Mark a node as dead (removed by fusion). The compiler and
    /// scheduler skip dead nodes. Arena indices remain stable.
    pub fn kill_node(&mut self, id: NodeId) {
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.op = crate::node::GraphOp::Dead;
            node.inputs.clear();
        }
    }

    /// Set the inputs of a node. Used by fusion to rewire edges.
    pub fn set_inputs(&mut self, id: NodeId, inputs: SmallVec<[InputSource; 4]>) {
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.inputs = inputs;
        }
    }

    /// Build a reverse-edge index: `successors[i]` = list of nodes that
    /// consume node `i` as an input. O(V + E).
    pub fn build_successor_index(&self) -> Vec<SmallVec<[NodeId; 4]>> {
        let n = self.nodes.len();
        let mut succ: Vec<SmallVec<[NodeId; 4]>> = vec![SmallVec::new(); n];
        for (i, node) in self.nodes.iter().enumerate() {
            for input in &node.inputs {
                if let InputSource::Node(NodeId(parent)) = input {
                    let p = *parent as usize;
                    if p < n {
                        succ[p].push(NodeId(i as u32));
                    }
                }
            }
        }
        succ
    }

    /// Return the list of node IDs that are still alive (not Dead).
    pub fn live_node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().enumerate()
            .filter(|(_, n)| n.op != crate::node::GraphOp::Dead)
            .map(|(i, _)| NodeId(i as u32))
            .collect()
    }

    /// Topological-sort + level-grouping schedule construction.
    pub fn compute_schedule(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            self.schedule = Some(Schedule::default());
            return;
        }
        let mut depth = vec![0u32; n];
        let mut alive = vec![true; n];
        for (i, node) in self.nodes.iter().enumerate() {
            if node.op == crate::node::GraphOp::Dead {
                alive[i] = false;
                continue;
            }
            let mut d = 0u32;
            for input in &node.inputs {
                if let InputSource::Node(NodeId(parent)) = input {
                    let parent = *parent as usize;
                    if parent < i && alive[parent] {
                        d = d.max(depth[parent] + 1);
                    }
                }
            }
            depth[i] = d;
        }
        let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
        let mut sched = Schedule::default();
        for level in 0..=max_depth {
            let mut group: SmallVec<[NodeId; 16]> = SmallVec::new();
            for (i, &d) in depth.iter().enumerate() {
                if d as usize == level && alive[i] {
                    group.push(NodeId(i as u32));
                }
            }
            if !group.is_empty() {
                sched.levels.push(group);
            }
        }
        self.schedule = Some(sched);
    }
}

