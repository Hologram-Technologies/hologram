//! Fluent graph builder for constructing `Graph` instances.

use crate::constant::ConstantData;
use crate::error::GraphError;
use crate::graph::edge;
use crate::graph::node::NodeId;
use crate::graph::{Graph, GraphOp, SubgraphId};
use crate::subgraph::SubgraphDef;

/// Parameters for [`GraphBuilder::conv2d_lut_4bit`].
///
/// Build with [`Conv2dLut4Params::new`] (required: kernel + input spatial
/// dims) and chain the `with_*` setters for the stride / padding / dilation
/// / group knobs. Defaults mirror PyTorch: `stride=1`, `pad=0`, `dilation=1`,
/// `group=1`.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dLut4Params {
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub group: u32,
    pub input_h: u32,
    pub input_w: u32,
}

impl Conv2dLut4Params {
    /// New params for kernel `(kernel_h × kernel_w)` on a
    /// `(input_h × input_w)` feature map. Stride 1, no padding, dilation 1,
    /// ungrouped.
    #[must_use]
    pub fn new(kernel_h: u32, kernel_w: u32, input_h: u32, input_w: u32) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
            input_h,
            input_w,
        }
    }

    #[must_use]
    pub fn with_stride(mut self, stride_h: u32, stride_w: u32) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    #[must_use]
    pub fn with_padding(mut self, pad_h: u32, pad_w: u32) -> Self {
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }

    #[must_use]
    pub fn with_dilation(mut self, dilation_h: u32, dilation_w: u32) -> Self {
        self.dilation_h = dilation_h;
        self.dilation_w = dilation_w;
        self
    }

    #[must_use]
    pub fn with_group(mut self, group: u32) -> Self {
        self.group = group;
        self
    }
}

/// Fluent builder for constructing `Graph` instances.
///
/// Nodes are referenced by builder index (insertion order, 0-based).
pub struct GraphBuilder {
    graph: Graph,
    index_to_id: Vec<NodeId>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    /// Create an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            index_to_id: Vec::new(),
        }
    }

    /// Add a node, returning `self` for chaining.
    pub fn node(mut self, op: GraphOp) -> Self {
        let id = self.graph.add_node(op);
        self.index_to_id.push(id);
        self
    }

    /// Add a node and record its N-D output shape, returning `self` for chaining.
    pub fn node_with_shape(mut self, op: GraphOp, shape: Vec<usize>) -> Self {
        let id = self.graph.add_node(op);
        self.graph.set_node_shape(id, shape);
        self.index_to_id.push(id);
        self
    }

    /// Add a node with edges from the given builder indices.
    pub fn node_with_inputs(mut self, op: GraphOp, inputs: &[usize]) -> Self {
        let id = self.graph.add_node(op);
        self.index_to_id.push(id);
        for (slot, &src_idx) in inputs.iter().enumerate() {
            if let Some(&src_id) = self.index_to_id.get(src_idx) {
                edge::connect(&mut self.graph, src_id, id, slot);
            }
        }
        self
    }

    /// Add a node with edges and an N-D output shape.
    pub fn node_with_inputs_and_shape(
        mut self,
        op: GraphOp,
        inputs: &[usize],
        shape: Vec<usize>,
    ) -> Self {
        let id = self.graph.add_node(op);
        self.graph.set_node_shape(id, shape);
        self.index_to_id.push(id);
        for (slot, &src_idx) in inputs.iter().enumerate() {
            if let Some(&src_id) = self.index_to_id.get(src_idx) {
                edge::connect(&mut self.graph, src_id, id, slot);
            }
        }
        self
    }

    /// Set the N-D output shape for a node by builder index.
    pub fn set_node_shape(mut self, index: usize, shape: Vec<usize>) -> Self {
        if let Some(&id) = self.index_to_id.get(index) {
            self.graph.set_node_shape(id, shape);
        }
        self
    }

    /// Set the output dtype for a node by builder index.
    pub fn set_node_dtype(mut self, index: usize, dtype: hologram_core::op::FloatDType) -> Self {
        if let Some(&id) = self.index_to_id.get(index) {
            self.graph.set_node_dtype(id, dtype);
        }
        self
    }

    /// Add a node wired to a graph-level input.
    pub fn node_from_graph_input(mut self, op: GraphOp, input_idx: u32) -> Self {
        let id = self.graph.add_node(op);
        self.index_to_id.push(id);
        edge::connect_graph_input(&mut self.graph, input_idx, id, 0);
        self
    }

    /// Add an edge from `source` to `target` (builder indices).
    pub fn edge(mut self, source: usize, target: usize) -> Self {
        if let (Some(&src), Some(&tgt)) =
            (self.index_to_id.get(source), self.index_to_id.get(target))
        {
            self.graph.add_edge(src, tgt);
        }
        self
    }

    /// Register a named graph input.
    pub fn input(mut self, name: impl Into<String>) -> Self {
        self.graph.add_input(name);
        self
    }

    /// Register a named graph output at builder index.
    pub fn output(mut self, name: impl Into<String>, index: usize) -> Self {
        if let Some(&id) = self.index_to_id.get(index) {
            self.graph.add_output(name, id);
        }
        self
    }

    /// Add a constant and a Constant node for it.
    pub fn constant(mut self, data: ConstantData) -> Self {
        let cid = self.graph.add_constant(data);
        let id = self.graph.add_node(GraphOp::Constant(cid));
        self.index_to_id.push(id);
        self
    }

    /// Add a constant with a known N-D shape (e.g. weight matrix).
    pub fn constant_with_shape(mut self, data: ConstantData, shape: Vec<usize>) -> Self {
        let cid = self.graph.add_constant(data);
        self.graph.set_constant_shape(cid, shape);
        let id = self.graph.add_node(GraphOp::Constant(cid));
        self.index_to_id.push(id);
        self
    }

    /// Add a 4-bit LUT-GEMM matmul node with pre-serialized weights.
    pub fn matmul_lut_4bit(mut self, weight_data: ConstantData, inputs: &[usize]) -> Self {
        let cid = self.graph.add_constant(weight_data);
        let id = self.graph.add_node(GraphOp::MatMulLut4(cid));
        self.index_to_id.push(id);
        for (slot, &src_idx) in inputs.iter().enumerate() {
            if let Some(&src_id) = self.index_to_id.get(src_idx) {
                edge::connect(&mut self.graph, src_id, id, slot);
            }
        }
        self
    }

    /// Add a 2-bit LUT-GEMM matmul node with pre-serialized weights.
    pub fn matmul_lut_2bit(mut self, weight_data: ConstantData, inputs: &[usize]) -> Self {
        let cid = self.graph.add_constant(weight_data);
        let id = self.graph.add_node(GraphOp::MatMulLut2(cid));
        self.index_to_id.push(id);
        for (slot, &src_idx) in inputs.iter().enumerate() {
            if let Some(&src_id) = self.index_to_id.get(src_idx) {
                edge::connect(&mut self.graph, src_id, id, slot);
            }
        }
        self
    }

    /// Add an 8-bit LUT-GEMM matmul node with pre-serialized weights.
    pub fn matmul_lut_8bit(mut self, weight_data: ConstantData, inputs: &[usize]) -> Self {
        let cid = self.graph.add_constant(weight_data);
        let id = self.graph.add_node(GraphOp::MatMulLut8(cid));
        self.index_to_id.push(id);
        for (slot, &src_idx) in inputs.iter().enumerate() {
            if let Some(&src_id) = self.index_to_id.get(src_idx) {
                edge::connect(&mut self.graph, src_id, id, slot);
            }
        }
        self
    }

    /// Add a Conv2d node with pre-quantized 4-bit LUT-GEMM weights.
    ///
    /// `weight_data` holds the rkyv-serialized `QuantizedWeights4`.
    /// `inputs` should be `[activation_data, f32_weight_constant]`.
    pub fn conv2d_lut_4bit(
        mut self,
        weight_data: ConstantData,
        inputs: &[usize],
        params: Conv2dLut4Params,
    ) -> Self {
        let Conv2dLut4Params {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            input_h,
            input_w,
        } = params;
        let cid = self.graph.add_constant(weight_data);
        let id = self.graph.add_node(GraphOp::Conv2dLut4 {
            cid,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            input_h,
            input_w,
        });
        self.index_to_id.push(id);
        for (slot, &src_idx) in inputs.iter().enumerate() {
            if let Some(&src_id) = self.index_to_id.get(src_idx) {
                edge::connect(&mut self.graph, src_id, id, slot);
            }
        }
        self
    }

    // `custom_op` removed in v0.2.0 conformance-first refactor along with
    // `GraphOp::Custom` and `CustomOpId`. Use a Prism module to extend the
    // primitive set instead.

    /// Register a subgraph template.
    pub fn subgraph(mut self, def: SubgraphDef) -> Self {
        self.graph.register_subgraph(def);
        self
    }

    /// Register a subgraph template and return its `SubgraphId`.
    pub fn subgraph_with_id(&mut self, def: SubgraphDef) -> crate::graph::SubgraphId {
        self.graph.register_subgraph(def)
    }

    /// Flatten a registered subgraph into the builder graph.
    ///
    /// `input_bindings` maps `(subgraph_input_index, builder_index)`.
    /// Returns the builder indices of the flattened output nodes.
    pub fn flatten_registered_subgraph(
        &mut self,
        subgraph_id: SubgraphId,
        input_bindings: &[(u32, usize)],
    ) -> Result<Vec<usize>, GraphError> {
        let bindings: Vec<(u32, NodeId)> = input_bindings
            .iter()
            .map(|&(sub_idx, builder_idx)| (sub_idx, self.index_to_id[builder_idx]))
            .collect();

        let result =
            crate::subgraph::flatten::flatten_subgraph(&mut self.graph, subgraph_id, &bindings)?;

        // Build a NodeId→builder_index lookup for output nodes.
        // We add all new nodes to our index_to_id tracker.
        let base = self.index_to_id.len();
        let new_ids: Vec<NodeId> = result.id_map.values().copied().collect();
        self.index_to_id.extend(new_ids.iter());

        // Map output NodeIds to builder indices.
        let output_indices = result
            .output_ids
            .iter()
            .map(|out_id| {
                // Search from base onwards (these are the newly added nodes).
                self.index_to_id[base..]
                    .iter()
                    .position(|id| *id == *out_id)
                    .map(|p| base + p)
                    .unwrap_or_else(|| {
                        // Might already exist (e.g., if output is a pass-through of an input).
                        self.index_to_id
                            .iter()
                            .position(|id| *id == *out_id)
                            .expect("flattened output NodeId not found in builder index")
                    })
            })
            .collect();

        Ok(output_indices)
    }

    /// Get the NodeId for a builder index.
    #[must_use]
    pub fn get_id(&self, index: usize) -> Option<NodeId> {
        self.index_to_id.get(index).copied()
    }

    /// Number of nodes added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index_to_id.len()
    }

    /// Whether no nodes have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index_to_id.is_empty()
    }

    /// Consume the builder and return the constructed graph.
    #[must_use]
    pub fn build(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::{LutOp, PrimOp};

    #[test]
    fn simple_chain() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .node_with_inputs(GraphOp::Output, &[1]) // 2
            .build();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edges().len(), 2);
    }

    #[test]
    fn diamond_graph() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 2
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();
        assert_eq!(g.node_count(), 5);
        // 0→1, 0→2, 1→3, 2→3, 3→4
        assert_eq!(g.edges().len(), 5);
    }

    #[test]
    fn named_io() {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .output("y", 1)
            .build();
        assert_eq!(g.inputs().len(), 1);
        assert_eq!(g.outputs().len(), 1);
    }

    #[test]
    fn constant_node() {
        let g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![42]))
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .build();
        assert_eq!(g.node_count(), 2);
        assert!(!g.constant_store().is_empty());
    }

    #[test]
    fn get_id() {
        let b = GraphBuilder::new()
            .node(GraphOp::Input)
            .node(GraphOp::Output);
        assert!(b.get_id(0).is_some());
        assert!(b.get_id(1).is_some());
        assert!(b.get_id(99).is_none());
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn empty_builder() {
        let b = GraphBuilder::new();
        assert!(b.is_empty());
        let g = b.build();
        assert!(g.is_empty());
    }

    #[test]
    fn matmul_lut_4bit_node() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .matmul_lut_4bit(ConstantData::Bytes(vec![1, 2, 3]), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();
        assert_eq!(g.node_count(), 3);
        assert!(!g.constant_store().is_empty());
    }

    #[test]
    fn matmul_lut_8bit_node() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .matmul_lut_8bit(ConstantData::Bytes(vec![4, 5, 6]), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();
        assert_eq!(g.node_count(), 3);
        assert!(!g.constant_store().is_empty());
    }

    #[test]
    fn invalid_edge_ignored() {
        // Edges referencing out-of-bounds indices are silently ignored
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .edge(0, 99) // invalid target
            .build();
        assert_eq!(g.node_count(), 1);
    }
}
