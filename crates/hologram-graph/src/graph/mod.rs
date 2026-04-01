//! Core graph types: GraphOp, SubgraphId, and the arena-based Graph.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use std::collections::HashMap;

pub mod edge;
pub mod node;
pub mod validate;

use crate::constant::{ConstantData, ConstantId, ConstantStore};
use hologram_core::op::{FloatDType, FloatOp, LutOp, PrimOp, RingLevel};
use hologram_core::view::ElementWiseView;
use node::{InputSlot, InputSource, Node, NodeId};

/// Identifier for a consumer-registered custom op.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct CustomOpId(pub u32);

impl CustomOpId {
    /// The raw identifier value.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Identifier for a registered subgraph template.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct SubgraphId(u32);

impl SubgraphId {
    /// Create a new subgraph identifier.
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The raw index.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> u32 {
        self.0
    }
}

/// Operations in the graph.
///
/// The fusion interface is `to_view()`: any op returning `Some(view)`
/// auto-participates in view fusion.
///
/// `FusedView` is intentionally 256 bytes (cache-line aligned LUT).
#[derive(Debug, Clone, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum GraphOp {
    /// Graph input boundary.
    Input,
    /// Graph output boundary.
    Output,
    /// Primitive operation (10 ops from holo-core).
    Prim(PrimOp),
    /// LUT-backed activation/scientific function (21 ops).
    Lut(LutOp),
    /// Precomputed 256-byte lookup table (result of view fusion).
    FusedView(ElementWiseView),
    /// Reference to a stored constant.
    Constant(ConstantId),
    /// Invoke a subgraph template (flattened before scheduling).
    CallSubgraph(SubgraphId),
    /// LUT-GEMM matmul with 4-bit quantized weights (stored as constant).
    MatMulLut4(ConstantId),
    /// LUT-GEMM matmul with 8-bit quantized weights (stored as constant).
    MatMulLut8(ConstantId),
    /// Batched LUT-GEMM with 4-bit quantized weights.
    BatchMatMulLut4(ConstantId),
    /// Batched LUT-GEMM with 8-bit quantized weights.
    BatchMatMulLut8(ConstantId),
    /// LUT-GEMM matmul with 16-bit hierarchical quantized weights (stored as constant).
    MatMulLut16(ConstantId),
    /// Batched LUT-GEMM with 16-bit hierarchical quantized weights.
    BatchMatMulLut16(ConstantId),
    /// Consumer-defined op. Dispatched via `CustomOpRegistry` at execution time.
    ///
    /// The `arity` field must match the number of edges wired to this node.
    Custom { id: CustomOpId, arity: u8 },
    /// Typed f32 tensor operation for AI inference.
    ///
    /// Unlike `PrimOp`/`LutOp` (byte-domain), these operate on f32 buffers
    /// with shape-aware semantics. Serialized into archives alongside other ops.
    Float(FloatOp),
    /// Fused chain of unary element-wise f32 ops. Single input, single output.
    /// Applied sequentially: chain[0](x) → chain[1](...) → ... → chain[n](...).
    /// Produced by the float fusion pass.
    FusedFloatChain(Vec<FloatOp>),
    /// Fused matmul + activation (compile-time epilogue fusion).
    /// Two inputs (same as MatMul). Produced by the matmul+activation fusion pass.
    FusedMatMulActivation {
        m: u32,
        k: u32,
        n: u32,
        activation: FloatOp,
    },
    /// Fused 4-bit LUT-GEMM + activation (epilogue fusion).
    MatMulLut4Activation(ConstantId, FloatOp),
    /// Fused 8-bit LUT-GEMM + activation (epilogue fusion).
    MatMulLut8Activation(ConstantId, FloatOp),
    /// Fused RmsNorm + activation (epilogue fusion).
    FusedRmsNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused LayerNorm + activation (epilogue fusion).
    FusedLayerNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused GroupNorm + activation (epilogue fusion).
    FusedGroupNormActivation {
        num_groups: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused matmul + bias add + activation (full epilogue fusion).
    /// Three inputs: [activation_input, weight_constant, bias_constant].
    /// Bias is read directly from arena (zero-copy mmap'd constant).
    /// Eliminates both intermediate buffers from MatMul → Add → Activation.
    FusedMatMulBiasActivation {
        m: u32,
        k: u32,
        n: u32,
        activation: FloatOp,
    },
    /// Fused Conv2d + activation (epilogue fusion).
    /// Three inputs (same as Conv2d): [data, weight, bias].
    FusedConv2dActivation {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        group: u32,
        activation: FloatOp,
    },
    /// Fused Conv2d + bias add + activation (3-node epilogue fusion).
    /// Three inputs: [data, weight, bias_constant].
    FusedConv2dBiasActivation {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        group: u32,
        activation: FloatOp,
    },
    /// Identity passthrough — zero-copy forward of input to output.
    ///
    /// Produced by view fusion when two involutions compose to identity
    /// (e.g. neg∘neg or bnot∘bnot). No computation required at runtime;
    /// the tape builder maps this to `TapeKernel::Passthrough`.
    Passthrough,
    /// Precomputed 128KB Q1 lookup table (result of Q1 view fusion).
    ///
    /// Box because ElementWiseView16 is 128 KB heap-allocated.
    /// Produced by q1_view_fusion when adjacent RingPrimUnary(_, Q1) nodes fuse.
    FusedView16(Box<hologram_core::q1::view::ElementWiseView16>),
    /// Byte-domain unary primitive op at a specified ring level.
    ///
    /// Stays in ring domain (Z/2^nZ) with no float conversion.
    /// Q0: uses ADD_Q0/MUL_Q0 LUT tables. Q1: uses native wrapping ops.
    RingPrimUnary(PrimOp, RingLevel),
    /// Byte-domain binary primitive op at a specified ring level.
    ///
    /// Stays in ring domain (Z/2^nZ) with no float conversion.
    /// Q0: uses ADD_Q0/MUL_Q0 LUT tables. Q1: uses native wrapping ops.
    RingPrimBinary(PrimOp, RingLevel),

    // ── Ring-native ops (from hologram-ring) ─────────────────────────────
    /// Ring-native activation (21 ops, composed from ring primitives).
    /// Uses ActivationOp::apply::<Q> at the ring level specified by RingLevel.
    /// Q0/Q1: LUT path. Q3+: piecewise polynomial path.
    RingActivation(hologram_core::op::ActivationOp, RingLevel),
    /// Ring-domain fused multiply-add: acc + a * b.
    RingAccumulate(RingLevel),
    /// Ring-domain reduction along an axis.
    RingReduce {
        op: PrimOp,
        axis: u32,
        level: RingLevel,
    },
    /// Conv2d with pre-quantized 4-bit LUT-GEMM weights (compile-time quantized).
    ///
    /// Weights are already transposed to [kernel_size, oc_per_group] and quantized
    /// via k-means Q4 at compile time. At runtime, only im2col + LUT-GEMM dispatch
    /// is needed — zero quantization/transpose overhead.
    ///
    /// Two inputs: [activation_data, f32_weight_constant].
    /// The ConstantId holds the rkyv-serialized `QuantizedWeights4`.
    /// The f32 weight is kept as a graph input for bias and shape metadata.
    Conv2dLut4 {
        cid: ConstantId,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        group: u32,
        input_h: u32,
        input_w: u32,
    },
}

impl GraphOp {
    /// Arity: number of inputs this op expects.
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Input | Self::Constant(_) => 0,
            Self::Output
            | Self::Lut(_)
            | Self::FusedView(_)
            | Self::Passthrough
            | Self::CallSubgraph(_)
            | Self::MatMulLut4(_)
            | Self::MatMulLut8(_)
            | Self::MatMulLut4Activation(..)
            | Self::MatMulLut8Activation(..)
            | Self::BatchMatMulLut4(_)
            | Self::BatchMatMulLut8(_)
            | Self::MatMulLut16(_)
            | Self::BatchMatMulLut16(_) => 1,
            Self::FusedView16(_) => 1,
            Self::RingPrimUnary(_, _) => 1,
            Self::RingPrimBinary(_, _) => 2,
            Self::RingActivation(_, _) => 1,
            Self::RingAccumulate(_) => 3,
            Self::RingReduce { .. } => 1,
            Self::Conv2dLut4 { .. } => 2, // [activation_data, f32_weight] (bias wired as 3rd input if present)
            Self::Prim(p) => p.arity(),
            Self::Custom { arity, .. } => *arity,
            Self::Float(f) => f.arity(),
            Self::FusedFloatChain(_) => 1,
            Self::FusedMatMulActivation { .. } => 2,
            Self::FusedMatMulBiasActivation { .. } => 3,
            Self::FusedConv2dActivation { .. } => 3, // data, weight, bias (same as Conv2d)
            Self::FusedConv2dBiasActivation { .. } => 3,
            Self::FusedRmsNormActivation { .. } => 2,
            Self::FusedLayerNormActivation { .. } | Self::FusedGroupNormActivation { .. } => 3,
        }
    }

    /// Whether this op is pure (no side effects, safe for CSE).
    #[must_use]
    pub const fn is_pure(&self) -> bool {
        matches!(
            self,
            Self::Prim(_)
                | Self::Lut(_)
                | Self::FusedView(_)
                | Self::FusedView16(_)
                | Self::Passthrough
                | Self::MatMulLut4(_)
                | Self::MatMulLut8(_)
                | Self::MatMulLut4Activation(..)
                | Self::MatMulLut8Activation(..)
                | Self::BatchMatMulLut4(_)
                | Self::BatchMatMulLut8(_)
                | Self::MatMulLut16(_)
                | Self::BatchMatMulLut16(_)
                | Self::Custom { .. }
                | Self::Float(_)
                | Self::FusedFloatChain(_)
                | Self::FusedMatMulActivation { .. }
                | Self::FusedMatMulBiasActivation { .. }
                | Self::FusedConv2dActivation { .. }
                | Self::FusedConv2dBiasActivation { .. }
                | Self::FusedRmsNormActivation { .. }
                | Self::FusedLayerNormActivation { .. }
                | Self::FusedGroupNormActivation { .. }
                | Self::RingPrimUnary(_, _)
                | Self::RingActivation(_, _)
                | Self::RingAccumulate(_)
                | Self::RingReduce { .. }
                | Self::RingPrimBinary(_, _)
                | Self::Conv2dLut4 { .. }
        )
    }

    /// Whether this is a fusable unary op (can become a FusedView).
    #[must_use]
    pub fn is_fusable_unary(&self) -> bool {
        self.to_view().is_some()
    }

    /// Convert to an ElementWiseView if this op is a unary table lookup.
    ///
    /// For the 4 unary PrimOps, returns a reference to a cached static view —
    /// no 256-byte table is built at runtime (UOR canonical representation).
    #[must_use]
    pub fn to_view(&self) -> Option<ElementWiseView> {
        match self {
            Self::Lut(op) => Some(ElementWiseView::from_table(*op.table())),
            Self::FusedView(v) => Some(*v),
            Self::Passthrough => Some(ElementWiseView::identity()),
            Self::Prim(PrimOp::Neg) => Some(hologram_core::view::NEG_VIEW),
            Self::Prim(PrimOp::Bnot) => Some(hologram_core::view::BNOT_VIEW),
            Self::Prim(PrimOp::Succ) => Some(hologram_core::view::SUCC_VIEW),
            Self::Prim(PrimOp::Pred) => Some(hologram_core::view::PRED_VIEW),
            _ => None,
        }
    }

    /// Convert to a Q1 ElementWiseView16 if this op is a Q1-level unary lookup.
    #[must_use]
    pub fn to_view16(&self) -> Option<hologram_core::q1::view::ElementWiseView16> {
        match self {
            Self::RingPrimUnary(p, RingLevel::Q1) => {
                Some(hologram_core::q1::view::ElementWiseView16::from_fn(|x| {
                    p.apply_unary_q1(x)
                }))
            }
            Self::FusedView16(v) => Some((**v).clone()),
            _ => None,
        }
    }
}

// --- Arena slot ---

/// Arena slot: occupied with a node or free for reuse.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum Slot {
    Occupied(Node),
    Free { next_free: Option<u32> },
}

// --- Graph ---

/// Arena-based compute graph.
///
/// Single type for construction, optimization, and serialization.
/// Replaces v1's separate `OperationGraph` + `CompileGraph`.
#[derive(Debug, Clone)]
pub struct Graph {
    slots: Vec<Slot>,
    generations: Vec<u32>,
    free_head: Option<u32>,
    node_count: usize,
    graph_inputs: Vec<String>,
    graph_outputs: Vec<(String, NodeId)>,
    constants: ConstantStore,
    constant_shapes: HashMap<ConstantId, Vec<usize>>,
    /// Compiled N-D output shapes per node.
    ///
    /// Populated during lowering from the AI-level IR which has complete shape
    /// information. Dimensions that are symbolic at compile time use 0 as a
    /// sentinel. The executor uses these shapes as ground truth, resolving 0s
    /// from actual buffer sizes at runtime.
    node_shapes: HashMap<NodeId, Vec<usize>>,
    /// Compiled output dtype per node.
    ///
    /// Populated during lowering from the AI-level IR. Defaults to F32 when
    /// absent. The executor uses this to dispatch type-aware operations
    /// (e.g., i64 shape subgraphs vs f32 tensor data).
    node_dtypes: HashMap<NodeId, FloatDType>,
    subgraphs: Vec<crate::subgraph::SubgraphDef>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            free_head: None,
            node_count: 0,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            constants: ConstantStore::new(),
            constant_shapes: HashMap::new(),
            node_shapes: HashMap::new(),
            node_dtypes: HashMap::new(),
            subgraphs: Vec::new(),
        }
    }

    /// Create an empty graph with preallocated capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            generations: Vec::with_capacity(cap),
            free_head: None,
            node_count: 0,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            constants: ConstantStore::new(),
            constant_shapes: HashMap::new(),
            node_shapes: HashMap::new(),
            node_dtypes: HashMap::new(),
            subgraphs: Vec::new(),
        }
    }

    // --- Node management ---

    /// Add a node with the given operation, returning its ID.
    pub fn add_node(&mut self, op: GraphOp) -> NodeId {
        let (index, gen) = self.allocate_slot();
        let id = NodeId::new(index, gen);
        self.slots[index as usize] = Slot::Occupied(Node::new(id, op));
        self.node_count += 1;
        id
    }

    /// Remove a node, returning it if it existed.
    pub fn remove_node(&mut self, id: NodeId) -> Option<Node> {
        if !self.is_valid_id(id) {
            return None;
        }
        let idx = id.index() as usize;
        let old = core::mem::replace(
            &mut self.slots[idx],
            Slot::Free {
                next_free: self.free_head,
            },
        );
        self.free_head = Some(id.index());
        self.generations[idx] += 1;
        self.node_count -= 1;
        match old {
            Slot::Occupied(node) => Some(node),
            Slot::Free { .. } => None,
        }
    }

    /// Get an immutable reference to a node.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        if !self.is_valid_id(id) {
            return None;
        }
        match &self.slots[id.index() as usize] {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        }
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        if !self.is_valid_id(id) {
            return None;
        }
        match &mut self.slots[id.index() as usize] {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        }
    }

    /// Whether the graph contains a live node with this ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.get(id).is_some()
    }

    /// Replace a node's operation in place.
    pub fn replace_op(&mut self, id: NodeId, op: GraphOp) -> bool {
        if let Some(node) = self.get_mut(id) {
            node.op = op;
            true
        } else {
            false
        }
    }

    // --- Edges ---

    /// Add an edge: `source` output feeds `target` as a new input.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> bool {
        if !self.contains(source) || !self.contains(target) {
            return false;
        }
        if let Some(node) = self.get_mut(target) {
            node.inputs.push(InputSlot::from_node(source));
            true
        } else {
            false
        }
    }

    /// Predecessor NodeIds of a node.
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        self.get(id)
            .map(|n| n.dependencies().collect())
            .unwrap_or_default()
    }

    /// Successor NodeIds of a node (nodes that use this node as input).
    ///
    /// This performs a full graph scan — O(V+E). For multiple lookups,
    /// use [`build_successor_index`] to build the index once, then
    /// call [`successors_from_index`] for O(degree) per lookup.
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        for node in self.nodes() {
            if node.dependencies().any(|dep| dep == id) {
                result.push(node.id);
            }
        }
        result
    }

    /// Build a reverse-edge index: for each node, the list of its successors.
    ///
    /// Built in O(V+E) via a single pass over all edges. Returns a flat
    /// Vec indexed by `NodeId::index()`. Use [`successors_from_index`]
    /// for O(degree) successor lookups after building this once.
    #[must_use]
    pub fn build_successor_index(&self) -> Vec<Vec<NodeId>> {
        let len = self.slots.len();
        let mut index: Vec<Vec<NodeId>> = Vec::with_capacity(len);
        index.resize_with(len, Vec::new);
        for node in self.nodes() {
            // Deduplicate: if a node lists the same dep twice, only record
            // the successor once. Matches successors() which uses .any().
            let mut seen_deps = Vec::new();
            for dep in node.dependencies() {
                let dep_idx = dep.index() as usize;
                if dep_idx < len && !seen_deps.contains(&dep) {
                    seen_deps.push(dep);
                    index[dep_idx].push(node.id);
                }
            }
        }
        index
    }

    /// Look up successors from a pre-built reverse-edge index. O(degree).
    ///
    /// The index must have been built by [`build_successor_index`] on the
    /// same graph state. Returns an empty slice for unknown or out-of-range IDs.
    #[must_use]
    pub fn successors_from_index(id: NodeId, index: &[Vec<NodeId>]) -> &[NodeId] {
        let idx = id.index() as usize;
        if idx < index.len() {
            &index[idx]
        } else {
            &[]
        }
    }

    /// All edges as (source, target) pairs.
    pub fn edges(&self) -> Vec<(NodeId, NodeId)> {
        let mut result = Vec::new();
        for node in self.nodes() {
            for dep in node.dependencies() {
                result.push((dep, node.id));
            }
        }
        result
    }

    // --- Iteration ---

    /// Iterator over all live nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.slots.iter().filter_map(|s| match s {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        })
    }

    /// Mutable iterator over all live nodes.
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.slots.iter_mut().filter_map(|s| match s {
            Slot::Occupied(n) => Some(n),
            Slot::Free { .. } => None,
        })
    }

    /// Collect all live NodeIds.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes().map(|n| n.id).collect()
    }

    /// Number of live nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    // --- I/O ---

    /// Register a named graph input, returning its index.
    pub fn add_input(&mut self, name: impl Into<String>) -> u32 {
        let idx = self.graph_inputs.len() as u32;
        self.graph_inputs.push(name.into());
        idx
    }

    /// Register a named graph output connected to a node.
    pub fn add_output(&mut self, name: impl Into<String>, node: NodeId) {
        self.graph_outputs.push((name.into(), node));
    }

    /// Named graph inputs.
    #[must_use]
    pub fn inputs(&self) -> &[String] {
        &self.graph_inputs
    }

    /// Named graph outputs with their source nodes.
    #[must_use]
    pub fn outputs(&self) -> &[(String, NodeId)] {
        &self.graph_outputs
    }

    /// Source nodes (nodes with no predecessors).
    pub fn sources(&self) -> Vec<NodeId> {
        self.nodes()
            .filter(|n| n.dependencies().next().is_none())
            .map(|n| n.id)
            .collect()
    }

    /// Sink nodes (nodes with no successors).
    ///
    /// Uses a pre-built reverse-edge index for O(V+E) total instead of O(V^2).
    pub fn sinks(&self) -> Vec<NodeId> {
        let succ_index = self.build_successor_index();
        self.nodes()
            .filter(|n| Self::successors_from_index(n.id, &succ_index).is_empty())
            .map(|n| n.id)
            .collect()
    }

    // --- Constants ---

    /// Add a constant and return its ID.
    pub fn add_constant(&mut self, data: ConstantData) -> ConstantId {
        self.constants.insert(data)
    }

    /// Look up a constant by ID.
    #[must_use]
    pub fn get_constant(&self, id: ConstantId) -> Option<&ConstantData> {
        self.constants.get(id)
    }

    /// Reference to the constant store.
    #[must_use]
    pub fn constant_store(&self) -> &ConstantStore {
        &self.constants
    }

    /// Set the N-D shape for a constant (e.g. weight matrix shape).
    pub fn set_constant_shape(&mut self, id: ConstantId, shape: Vec<usize>) {
        self.constant_shapes.insert(id, shape);
    }

    /// Get the N-D shape for a constant, if recorded.
    #[must_use]
    pub fn constant_shape(&self, id: ConstantId) -> Option<&[usize]> {
        self.constant_shapes.get(&id).map(|v| v.as_slice())
    }

    /// All recorded constant shapes.
    #[must_use]
    pub fn constant_shapes(&self) -> &HashMap<ConstantId, Vec<usize>> {
        &self.constant_shapes
    }

    // --- Node shapes ---

    /// Set the compiled N-D output shape for a node.
    ///
    /// Use 0 for dimensions that are symbolic at compile time (batch, seq_len).
    /// The executor resolves 0s from actual buffer sizes at runtime.
    pub fn set_node_shape(&mut self, id: NodeId, shape: Vec<usize>) {
        self.node_shapes.insert(id, shape);
    }

    /// Get the compiled N-D output shape for a node, if recorded.
    #[must_use]
    pub fn node_shape(&self, id: NodeId) -> Option<&[usize]> {
        self.node_shapes.get(&id).map(|v| v.as_slice())
    }

    /// All recorded node shapes.
    #[must_use]
    pub fn node_shapes(&self) -> &HashMap<NodeId, Vec<usize>> {
        &self.node_shapes
    }

    // --- Node dtypes ---

    /// Set the compiled output dtype for a node.
    pub fn set_node_dtype(&mut self, id: NodeId, dtype: FloatDType) {
        self.node_dtypes.insert(id, dtype);
    }

    /// Get the compiled output dtype for a node, if recorded.
    #[must_use]
    pub fn node_dtype(&self, id: NodeId) -> Option<FloatDType> {
        self.node_dtypes.get(&id).copied()
    }

    /// All recorded node dtypes.
    #[must_use]
    pub fn node_dtypes(&self) -> &HashMap<NodeId, FloatDType> {
        &self.node_dtypes
    }

    // --- Subgraphs ---

    /// Register a subgraph template, returning its ID.
    pub fn register_subgraph(&mut self, def: crate::subgraph::SubgraphDef) -> SubgraphId {
        let id = SubgraphId(self.subgraphs.len() as u32);
        self.subgraphs.push(def);
        id
    }

    /// Look up a subgraph by ID.
    #[must_use]
    pub fn get_subgraph(&self, id: SubgraphId) -> Option<&crate::subgraph::SubgraphDef> {
        self.subgraphs.get(id.0 as usize)
    }

    // --- Rewire ---

    /// Rewire all successors of `old` to point to `new` instead.
    pub fn rewire_successors(&mut self, old: NodeId, new: NodeId) {
        for slot in &mut self.slots {
            if let Slot::Occupied(node) = slot {
                for input in &mut node.inputs {
                    if input.source == InputSource::Node(old) {
                        input.source = InputSource::Node(new);
                    }
                }
            }
        }
    }

    /// Rewire successors of `old` to point to `new`, using a pre-built index.
    ///
    /// Only visits actual successors from the index instead of scanning all slots.
    /// O(degree) instead of O(V×E).
    pub fn rewire_successors_indexed(
        &mut self,
        old: NodeId,
        new: NodeId,
        succ_index: &[Vec<NodeId>],
    ) {
        let successors: Vec<NodeId> = Self::successors_from_index(old, succ_index).to_vec();
        for succ_id in successors {
            if let Some(node) = self.get_mut(succ_id) {
                for input in &mut node.inputs {
                    if input.source == InputSource::Node(old) {
                        input.source = InputSource::Node(new);
                    }
                }
            }
        }
    }

    // --- Private ---

    /// Allocate or reuse a slot, returning (index, generation).
    fn allocate_slot(&mut self) -> (u32, u32) {
        if let Some(free_idx) = self.free_head {
            let idx = free_idx as usize;
            if let Slot::Free { next_free } = &self.slots[idx] {
                self.free_head = *next_free;
            }
            (free_idx, self.generations[idx])
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot::Free { next_free: None });
            self.generations.push(0);
            (idx, 0)
        }
    }

    /// Check if a NodeId is valid (correct generation).
    fn is_valid_id(&self, id: NodeId) -> bool {
        let idx = id.index() as usize;
        idx < self.slots.len() && self.generations[idx] == id.generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let g = Graph::new();
        assert!(g.is_empty());
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn add_and_get() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Input);
        assert!(g.contains(id));
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.get(id).unwrap().op, GraphOp::Input);
    }

    #[test]
    fn remove_and_stale() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Input);
        g.remove_node(id);
        assert!(!g.contains(id)); // stale
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn slot_reuse() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        g.remove_node(a);
        let b = g.add_node(GraphOp::Output);
        // Same slot index, different generation
        assert_eq!(b.index(), a.index());
        assert_ne!(b.generation(), a.generation());
        assert!(!g.contains(a));
        assert!(g.contains(b));
    }

    #[test]
    fn edges() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        assert!(g.add_edge(a, b));
        assert_eq!(g.predecessors(b), alloc::vec![a]);
        assert_eq!(g.successors(a), alloc::vec![b]);
    }

    #[test]
    fn named_io() {
        let mut g = Graph::new();
        let idx = g.add_input("x");
        assert_eq!(idx, 0);
        let node = g.add_node(GraphOp::Output);
        g.add_output("y", node);
        assert_eq!(g.inputs().len(), 1);
        assert_eq!(g.outputs().len(), 1);
    }

    #[test]
    fn constants() {
        let mut g = Graph::new();
        let cid = g.add_constant(ConstantData::Bytes(alloc::vec![42]));
        assert_eq!(
            g.get_constant(cid),
            Some(&ConstantData::Bytes(alloc::vec![42]))
        );
    }

    #[test]
    fn sources_and_sinks() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        g.add_edge(a, b);
        assert!(g.sources().contains(&a));
        assert!(g.sinks().contains(&b));
    }

    #[test]
    fn rewire_successors() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Input);
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        g.add_edge(a, c);
        g.rewire_successors(a, b);
        assert_eq!(g.predecessors(c), alloc::vec![b]);
    }

    #[test]
    fn replace_op() {
        let mut g = Graph::new();
        let id = g.add_node(GraphOp::Lut(LutOp::Relu));
        assert!(g.replace_op(id, GraphOp::Lut(LutOp::Sigmoid)));
        assert_eq!(g.get(id).unwrap().op, GraphOp::Lut(LutOp::Sigmoid));
    }

    #[test]
    fn successor_index() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);

        let index = g.build_successor_index();
        assert_eq!(Graph::successors_from_index(a, &index), &[b]);
        assert_eq!(Graph::successors_from_index(b, &index), &[c]);
        assert!(Graph::successors_from_index(c, &index).is_empty());
    }

    #[test]
    fn graphop_arity() {
        assert_eq!(GraphOp::Input.arity(), 0);
        assert_eq!(GraphOp::Output.arity(), 1);
        assert_eq!(GraphOp::Lut(LutOp::Sigmoid).arity(), 1);
        assert_eq!(GraphOp::Prim(PrimOp::Add).arity(), 2);
        assert_eq!(GraphOp::Constant(ConstantId::new(0)).arity(), 0);
    }

    #[test]
    fn graphop_is_pure() {
        assert!(GraphOp::Prim(PrimOp::Add).is_pure());
        assert!(GraphOp::Lut(LutOp::Relu).is_pure());
        assert!(!GraphOp::Input.is_pure());
        assert!(!GraphOp::Output.is_pure());
        assert!(!GraphOp::Constant(ConstantId::new(0)).is_pure());
    }

    #[test]
    fn graphop_to_view() {
        // LutOp produces a view
        let v = GraphOp::Lut(LutOp::Relu).to_view().unwrap();
        assert_eq!(v.apply(0), LutOp::Relu.apply(0));
        // Unary PrimOp produces a view
        let v = GraphOp::Prim(PrimOp::Neg).to_view().unwrap();
        assert_eq!(v.apply(1), 255); // wrapping_neg
                                     // Binary PrimOp does not
        assert!(GraphOp::Prim(PrimOp::Add).to_view().is_none());
        // Input does not
        assert!(GraphOp::Input.to_view().is_none());
    }

    #[test]
    fn graphop_fusable_unary() {
        assert!(GraphOp::Lut(LutOp::Sigmoid).is_fusable_unary());
        assert!(GraphOp::Prim(PrimOp::Neg).is_fusable_unary());
        assert!(!GraphOp::Prim(PrimOp::Add).is_fusable_unary());
        assert!(!GraphOp::Input.is_fusable_unary());
    }
}
