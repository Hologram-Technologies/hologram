//! `Graph` structure (spec VI.1).

use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;
use crate::node::{Node, NodeId, GraphOp, InputSource};
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

/// Builder convenience: a node carrying just a `GraphOp` and inputs.
pub fn node(op: GraphOp, inputs: SmallVec<[InputSource; 4]>) -> Node {
    Node {
        op,
        inputs,
        output_dtype: crate::registry::DTypeId(0),
        output_shape: crate::registry::ShapeId(0),
    }
}
