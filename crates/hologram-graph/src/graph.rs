//! `Graph` structure (spec VI.1).

use crate::constant::ConstantStore;
use crate::node::{ConvAttrs, GraphOp, InputSource, Node, NodeId, QuantAttrs};
use crate::registry::ShapeRegistry;
use crate::schedule::Schedule;
use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

/// Remap an `InputSource::Node` through an old→new node-id table; constants
/// and graph-input ports are id-independent and pass through unchanged.
fn remap_src(src: InputSource, map: &[u32]) -> InputSource {
    match src {
        InputSource::Node(NodeId(i)) => InputSource::Node(NodeId(map[i as usize])),
        other => other,
    }
}

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

    /// **Path B — desugar composite ops into their primitive pipelines.**
    ///
    /// A composite op (e.g. `Clip`) has no single optimized kernel; its meaning
    /// *is* a composition of primitives (`Clip(x,lo,hi) = Min(Max(x,lo),hi)`).
    /// Rather than carry bolt-on parameters, we rewrite each composite node, in
    /// topological order, into the sequence of primitive nodes that computes it
    /// — reusing the already-verified primitive kernels and the ordinary
    /// node→slot model (every intermediate is a real node with its own output
    /// slot; no special intermediate-buffer machinery). This is the UOR-native
    /// "ops as PrimitiveOp pipelines" lowering.
    ///
    /// The rewrite preserves topological order (producers before consumers) and
    /// remaps every `InputSource::Node`, the input/output port lists, and the
    /// sparse per-node attribute tables to the new node ids. Constants
    /// (`ConstantId`) and shapes (`ShapeId`) are unaffected. A cached schedule
    /// is invalidated. A composite lacking the operands its expansion needs is
    /// left untouched (the backend rejects it explicitly rather than guess).
    ///
    /// Returns the number of composite nodes expanded.
    pub fn desugar_composites(&mut self) -> usize {
        use crate::OpKind as K;
        let is_composite = |n: &Node| matches!(n.op, GraphOp::Op(K::Clip) if n.inputs.len() >= 3);
        if !self.nodes.iter().any(is_composite) {
            return 0;
        }

        let old = core::mem::take(&mut self.nodes);
        let mut new: Vec<Node> = Vec::with_capacity(old.len() + 4);
        // old node id -> new node id of the value it produces.
        let mut map: Vec<u32> = vec![0u32; old.len()];
        let mut expanded = 0usize;

        for (old_idx, node) in old.iter().enumerate() {
            // Remap inputs against already-rebuilt predecessors (topological
            // order guarantees every `Node` parent has a populated `map` slot).
            let inputs: SmallVec<[InputSource; 4]> =
                node.inputs.iter().map(|s| remap_src(*s, &map)).collect();

            let out_id = match node.op {
                GraphOp::Op(K::Clip) if inputs.len() >= 3 => {
                    // Min(Max(x, lo), hi) — elementwise, so the intermediate
                    // carries the composite's own dtype/shape.
                    let max_id = new.len() as u32;
                    new.push(Node {
                        op: GraphOp::Op(K::Max),
                        inputs: SmallVec::from_iter([inputs[0], inputs[1]]),
                        output_dtype: node.output_dtype,
                        output_shape: node.output_shape,
                    });
                    let min_id = new.len() as u32;
                    new.push(Node {
                        op: GraphOp::Op(K::Min),
                        inputs: SmallVec::from_iter([InputSource::Node(NodeId(max_id)), inputs[2]]),
                        output_dtype: node.output_dtype,
                        output_shape: node.output_shape,
                    });
                    expanded += 1;
                    min_id
                }
                _ => {
                    let id = new.len() as u32;
                    let mut n = node.clone();
                    n.inputs = inputs;
                    new.push(n);
                    id
                }
            };
            map[old_idx] = out_id;
        }

        for nid in self.inputs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        for nid in self.outputs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        for (nid, _) in self.conv_attrs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        for (nid, _) in self.quant_attrs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        self.nodes = new;
        self.schedule = None;
        expanded
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

#[cfg(test)]
mod desugar_tests {
    use super::*;
    use crate::constant::ConstantEntry;
    use crate::registry::{DTypeId, ShapeId};
    use crate::OpKind;

    /// Clip(x, lo, hi) must desugar to the primitive pipeline Min(Max(x,lo),hi),
    /// in topological order, with the output port and all references rewired to
    /// the terminal Min — reusing the existing Max/Min kernels, no Clip kernel.
    #[test]
    fn clip_desugars_to_min_of_max() {
        let mut g = Graph::new();
        let (dt, sh) = (DTypeId(8), ShapeId(0));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: dt,
            output_shape: sh,
        });
        g.add_input(x);
        let lo = g.constants_mut().insert(ConstantEntry {
            bytes: vec![0u8; 4],
            dtype: dt,
            shape: sh,
        });
        let hi = g.constants_mut().insert(ConstantEntry {
            bytes: vec![0u8; 4],
            dtype: dt,
            shape: sh,
        });
        let clip = g.add_node(Node {
            op: GraphOp::Op(OpKind::Clip),
            inputs: SmallVec::from_iter([
                InputSource::Node(x),
                InputSource::Constant(lo),
                InputSource::Constant(hi),
            ]),
            output_dtype: dt,
            output_shape: sh,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(clip)]),
            output_dtype: dt,
            output_shape: sh,
        });
        g.add_output(out);

        assert_eq!(g.desugar_composites(), 1);

        // No composite remains; exactly one Max precedes one Min.
        assert!(!g
            .nodes()
            .iter()
            .any(|n| matches!(n.op, GraphOp::Op(OpKind::Clip))));
        let max_pos = g
            .nodes()
            .iter()
            .position(|n| matches!(n.op, GraphOp::Op(OpKind::Max)))
            .expect("Max node");
        let min_pos = g
            .nodes()
            .iter()
            .position(|n| matches!(n.op, GraphOp::Op(OpKind::Min)))
            .expect("Min node");
        assert!(max_pos < min_pos, "producer Max must precede consumer Min");

        // Max(x, lo); Min(Max, hi).
        let max = &g.nodes()[max_pos];
        assert_eq!(max.inputs[0], InputSource::Node(x));
        assert_eq!(max.inputs[1], InputSource::Constant(lo));
        let min = &g.nodes()[min_pos];
        assert_eq!(min.inputs[0], InputSource::Node(NodeId(max_pos as u32)));
        assert_eq!(min.inputs[1], InputSource::Constant(hi));

        // Output port + the Output node's edge both rewired to Min.
        assert_eq!(g.outputs()[0], NodeId((g.node_count() - 1) as u32));
        assert_eq!(
            g.nodes().last().unwrap().inputs[0],
            InputSource::Node(NodeId(min_pos as u32))
        );
    }

    /// A graph with no composites is returned unchanged (zero expansions).
    #[test]
    fn no_composite_is_noop() {
        let mut g = Graph::new();
        let (dt, sh) = (DTypeId(8), ShapeId(0));
        let a = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: dt,
            output_shape: sh,
        });
        let _ = g.add_node(Node {
            op: GraphOp::Op(OpKind::Relu),
            inputs: SmallVec::from_iter([InputSource::Node(a)]),
            output_dtype: dt,
            output_shape: sh,
        });
        let before = g.node_count();
        assert_eq!(g.desugar_composites(), 0);
        assert_eq!(g.node_count(), before);
    }
}
