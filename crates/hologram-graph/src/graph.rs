//! `Graph` structure (spec VI.1).

use crate::constant::ConstantStore;
use crate::node::{ConvAttrs, InputSource, Node, NodeId, QuantAttrs};
use crate::registry::ShapeRegistry;
use crate::schedule::Schedule;
use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

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
    /// Sparse per-node convolution attributes (stride/pad/dilation).
    /// Empty for graphs whose conv nodes use the default
    /// `(stride = 1, pad = 0)`. Same sparse-table layout as
    /// `quant_attrs` so ordinary nodes pay no per-instance overhead.
    conv_attrs: Vec<(NodeId, ConvAttrs)>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn add_input(&mut self, id: NodeId) {
        self.inputs.push(id);
    }
    pub fn add_output(&mut self, id: NodeId) {
        self.outputs.push(id);
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn constants(&self) -> &ConstantStore {
        &self.constants
    }
    pub fn constants_mut(&mut self) -> &mut ConstantStore {
        &mut self.constants
    }

    pub fn shape_registry(&self) -> &ShapeRegistry {
        &self.shape_registry
    }
    pub fn shape_registry_mut(&mut self) -> &mut ShapeRegistry {
        &mut self.shape_registry
    }

    pub fn schedule(&self) -> Option<&Schedule> {
        self.schedule.as_ref()
    }
    pub fn set_schedule(&mut self, sched: Schedule) {
        self.schedule = Some(sched);
    }

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
        self.quant_attrs
            .iter()
            .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Attach convolution attributes (stride/padding) to a node. Only
    /// meaningful for `Conv2d` / `ConvTranspose2d` ops; other ops
    /// ignore the entry.
    pub fn set_conv_attrs(&mut self, id: NodeId, attrs: ConvAttrs) {
        if let Some(slot) = self.conv_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.conv_attrs.push((id, attrs));
        }
    }

    /// Retrieve convolution attributes for a node, or `None` if the node
    /// uses defaults.
    pub fn conv_attrs(&self, id: NodeId) -> Option<ConvAttrs> {
        self.conv_attrs
            .iter()
            .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Topological-sort + level-grouping schedule construction.
    pub fn compute_schedule(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            self.schedule = Some(Schedule::default());
            return;
        }
        let mut depth = vec![0u32; n];
        for (i, node) in self.nodes.iter().enumerate() {
            let mut d = 0u32;
            for input in &node.inputs {
                if let InputSource::Node(NodeId(parent)) = input {
                    let parent = *parent as usize;
                    if parent < i {
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
                if d as usize == level {
                    group.push(NodeId(i as u32));
                }
            }
            sched.levels.push(group);
        }
        self.schedule = Some(sched);
    }
}
