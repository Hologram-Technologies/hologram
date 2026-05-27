//! `Graph` structure (spec VI.1).

use crate::constant::{ConstantEntry, ConstantStore};
use crate::node::{
    ConvAttrs, GatherAttrs, GemmAttrs, GraphOp, InputSource, LrnAttrs, Node, NodeId, NormAttrs,
    QuantAttrs, ReduceAttrs,
};
use crate::registry::ShapeRegistry;
use crate::schedule::Schedule;
use alloc::string::String;
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

/// Remap an `InputSource::Node` through an old→`InputSource` table. Unlike
/// [`remap_src`] this lets a node be replaced by *any* operand kind (another
/// node, a constant, or a graph-input) — required by the elision pass, which
/// can collapse e.g. `Add(x, 0)` so that the node's consumers read `x`
/// directly, whatever kind of operand `x` is.
fn remap_is(src: InputSource, map: &[InputSource]) -> InputSource {
    match src {
        InputSource::Node(NodeId(i)) => map[i as usize],
        other => other,
    }
}

/// The additive / multiplicative identity elements, recognized as constant
/// operands by the algebraic-elision pass. A constant counts as one of these
/// only when *every* element matches the dtype's exact identity bit pattern,
/// so the elision is value-preserving within the runtime's accuracy contract
/// (the rules below — `x+0`, `x-0`, `x·1`, `x/1`, `x^1` — equal `x` for every
/// finite, infinite, and NaN input; they differ at most in the sign of a zero,
/// which is numerically immaterial). Annihilators like `x·0 → 0` are *not*
/// recognized: they are unsound under IEEE (`∞·0 = NaN`), so eliding them
/// would silently change results.
#[derive(Clone, Copy, PartialEq)]
enum IdentityConst {
    /// `+0.0` — the additive identity (`x + 0 = x`, `x - 0 = x`).
    Zero,
    /// `1.0` — the multiplicative identity (`x · 1 = x`, `x / 1 = x`, `x¹ = x`).
    One,
}

/// Resolve an operand to its [`ConstantEntry`], whether referenced inline
/// (`InputSource::Constant`) or produced by a `Constant` node in `nodes`.
fn resolve_const<'a>(
    nodes: &'a [Node],
    consts: &'a ConstantStore,
    src: InputSource,
) -> Option<&'a ConstantEntry> {
    match src {
        InputSource::Constant(cid) => consts.get(cid),
        InputSource::Node(NodeId(i)) => match nodes.get(i as usize)?.op {
            GraphOp::Constant(cid) => consts.get(cid),
            _ => None,
        },
        InputSource::GraphInput(_) => None,
    }
}

/// Classify a constant operand as the dtype's exact `+0.0` or `1.0` fill, or
/// `None`. Only the IEEE float dtypes (f16=6, bf16=7, f32=8) participate —
/// the identity rules are stated in those algebras — and an empty / partial
/// constant is treated as unknown.
fn identity_fill(
    nodes: &[Node],
    consts: &ConstantStore,
    src: InputSource,
) -> Option<IdentityConst> {
    let entry = resolve_const(nodes, consts, src)?;
    let (esize, zero, one): (usize, &[u8], &[u8]) = match entry.dtype.0 {
        8 => (4, &[0, 0, 0, 0], &[0x00, 0x00, 0x80, 0x3F]), // f32: 1.0 = 0x3F800000
        6 => (2, &[0, 0], &[0x00, 0x3C]),                   // f16: 1.0 = 0x3C00
        7 => (2, &[0, 0], &[0x80, 0x3F]),                   // bf16: 1.0 = 0x3F80
        _ => return None,
    };
    if entry.bytes.is_empty() || entry.bytes.len() % esize != 0 {
        return None;
    }
    if entry.bytes.chunks_exact(esize).all(|c| c == zero) {
        Some(IdentityConst::Zero)
    } else if entry.bytes.chunks_exact(esize).all(|c| c == one) {
        Some(IdentityConst::One)
    } else {
        None
    }
}

#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    inputs: SmallVec<[NodeId; 8]>,
    outputs: SmallVec<[NodeId; 8]>,
    /// Semantic input-port names, parallel to `inputs` by position (empty
    /// string ⇒ unnamed). Preserved through graph rewrites (the input vec's
    /// order is stable; rewrites only renumber the `NodeId` values).
    input_names: Vec<String>,
    /// Semantic output-port names, parallel to `outputs` by position.
    output_names: Vec<String>,
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
    /// Sparse per-node LRN attributes (size / α / β / bias). Same layout.
    lrn_attrs: Vec<(NodeId, LrnAttrs)>,
    /// Sparse per-node GEMM scalars (α / β). Same layout.
    gemm_attrs: Vec<(NodeId, GemmAttrs)>,
    /// Sparse per-node normalization grouping (`num_groups`). Same layout.
    norm_attrs: Vec<(NodeId, NormAttrs)>,
    /// Sparse per-node reduction axes (`axes_mask` / `keepdims`). Same layout.
    reduce_attrs: Vec<(NodeId, ReduceAttrs)>,
    /// Sparse per-node `Gather` axis. Same layout; absent ⇒ axis 0.
    gather_attrs: Vec<(NodeId, GatherAttrs)>,
    /// Open producer-defined metadata (`key`, `bytes`) to embed in the compiled
    /// archive as `Extension` sections (tokenizer, generation config, class
    /// labels, …). Carried opaquely; not part of the graph's compute semantics.
    extensions: Vec<(String, Vec<u8>)>,
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
        self.input_names.push(String::new());
    }
    pub fn add_output(&mut self, id: NodeId) {
        self.outputs.push(id);
        self.output_names.push(String::new());
    }
    /// Register an input port with a semantic `name` (e.g. `"input_ids"`), so a
    /// caller can bind a model input to this port by name.
    pub fn add_named_input(&mut self, id: NodeId, name: impl Into<String>) {
        self.inputs.push(id);
        self.input_names.push(name.into());
    }
    /// Register an output port with a semantic `name` (e.g. `"logits"`).
    pub fn add_named_output(&mut self, id: NodeId, name: impl Into<String>) {
        self.outputs.push(id);
        self.output_names.push(name.into());
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }
    /// Semantic name of input port `i` (empty string if unnamed).
    pub fn input_name(&self, i: usize) -> &str {
        self.input_names.get(i).map(|s| s.as_str()).unwrap_or("")
    }
    /// Semantic name of output port `i` (empty string if unnamed).
    pub fn output_name(&self, i: usize) -> &str {
        self.output_names.get(i).map(|s| s.as_str()).unwrap_or("")
    }

    /// Attach open producer metadata under `key` (tokenizer, generation config,
    /// class labels, …); the compiler embeds it as an archive `Extension`
    /// section, retrievable at runtime via `InferenceSession::extension`.
    pub fn add_extension(&mut self, key: impl Into<String>, bytes: Vec<u8>) {
        self.extensions.push((key.into(), bytes));
    }
    /// The producer metadata sections attached to this graph.
    pub fn extensions(&self) -> &[(String, Vec<u8>)] {
        &self.extensions
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

    /// Attach LRN attributes (size / α / β / bias) to a node.
    pub fn set_lrn_attrs(&mut self, id: NodeId, attrs: LrnAttrs) {
        if let Some(slot) = self.lrn_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.lrn_attrs.push((id, attrs));
        }
    }

    /// Retrieve LRN attributes for a node, or `None` if unset.
    pub fn lrn_attrs(&self, id: NodeId) -> Option<LrnAttrs> {
        self.lrn_attrs
            .iter()
            .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Attach GEMM scalars (α / β) to a node.
    pub fn set_gemm_attrs(&mut self, id: NodeId, attrs: GemmAttrs) {
        if let Some(slot) = self.gemm_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.gemm_attrs.push((id, attrs));
        }
    }

    /// Retrieve GEMM scalars for a node, or `None` if unset.
    pub fn gemm_attrs(&self, id: NodeId) -> Option<GemmAttrs> {
        self.gemm_attrs
            .iter()
            .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Attach normalization grouping (`num_groups`) to a node (GroupNorm).
    pub fn set_norm_attrs(&mut self, id: NodeId, attrs: NormAttrs) {
        if let Some(slot) = self.norm_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.norm_attrs.push((id, attrs));
        }
    }

    /// Retrieve normalization grouping for a node, or `None` if unset.
    pub fn norm_attrs(&self, id: NodeId) -> Option<NormAttrs> {
        self.norm_attrs
            .iter()
            .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Attach reduction axes (`axes_mask` / `keepdims`) to a node.
    pub fn set_reduce_attrs(&mut self, id: NodeId, attrs: ReduceAttrs) {
        if let Some(slot) = self.reduce_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.reduce_attrs.push((id, attrs));
        }
    }

    /// Retrieve reduction axes for a node, or `None` (⇒ reduce all axes).
    pub fn reduce_attrs(&self, id: NodeId) -> Option<ReduceAttrs> {
        self.reduce_attrs
            .iter()
            .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
    }

    /// Attach a `Gather` axis to a node.
    pub fn set_gather_attrs(&mut self, id: NodeId, attrs: GatherAttrs) {
        if let Some(slot) = self.gather_attrs.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = attrs;
        } else {
            self.gather_attrs.push((id, attrs));
        }
    }

    /// Retrieve a node's `Gather` axis, or `None` (⇒ axis 0).
    pub fn gather_attrs(&self, id: NodeId) -> Option<GatherAttrs> {
        self.gather_attrs
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
        let is_composite = |n: &Node| {
            matches!(n.op, GraphOp::Op(K::Clip) if n.inputs.len() >= 3)
                || matches!(n.op, GraphOp::Op(K::FusedSwiGlu) if n.inputs.len() >= 3)
        };
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
                GraphOp::Op(K::FusedSwiGlu) if inputs.len() >= 3 => {
                    // SwiGLU(x, W_gate, W_up) = Silu(x·W_gate) ⊙ (x·W_up).
                    // Reuses the matmul engine + verified Silu/Mul kernels; the
                    // matmul intermediates carry the composite's [m,n] output
                    // shape (ShapeArgs derives m,k,n from the operand shapes).
                    let (x, w_gate, w_up) = (inputs[0], inputs[1], inputs[2]);
                    let mk_node = |op, ins: &[InputSource]| Node {
                        op: GraphOp::Op(op),
                        inputs: SmallVec::from_iter(ins.iter().copied()),
                        output_dtype: node.output_dtype,
                        output_shape: node.output_shape,
                    };
                    let gate_id = new.len() as u32;
                    new.push(mk_node(K::MatMul, &[x, w_gate]));
                    let silu_id = new.len() as u32;
                    new.push(mk_node(K::Silu, &[InputSource::Node(NodeId(gate_id))]));
                    let up_id = new.len() as u32;
                    new.push(mk_node(K::MatMul, &[x, w_up]));
                    let mul_id = new.len() as u32;
                    new.push(mk_node(
                        K::Mul,
                        &[
                            InputSource::Node(NodeId(silu_id)),
                            InputSource::Node(NodeId(up_id)),
                        ],
                    ));
                    expanded += 1;
                    mul_id
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
        for (nid, _) in self.gemm_attrs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        for (nid, _) in self.norm_attrs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        for (nid, _) in self.reduce_attrs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        for (nid, _) in self.gather_attrs.iter_mut() {
            *nid = NodeId(map[nid.0 as usize]);
        }
        self.nodes = new;
        self.schedule = None;
        expanded
    }

    /// Algebraic-elision pass: remove computation that UOR's algebra proves
    /// unnecessary, so it is never scheduled, dispatched, or addressed.
    ///
    /// This is the "invariant facets we don't need to compute" optimization,
    /// run at compile time (after [`desugar_composites`](Self::desugar_composites),
    /// before scheduling). Two value-preserving rewrites, then dead-node
    /// elimination — every rule is exact within the runtime's accuracy
    /// contract (see `IdentityConst`); no rule is applied unless it provably
    /// preserves the result, so this never trades correctness for speed.
    ///
    /// 1. **Identity-element elimination.** `x+0`, `0+x`, `x-0`, `x·1`, `1·x`,
    ///    `x/1`, `x¹` collapse to `x` (the constant operand must be the exact
    ///    dtype identity fill). The node's consumers are redirected straight
    ///    to `x`, whatever operand kind it is.
    /// 2. **Involution cancellation.** `Neg(Neg x)` and `Bnot(Bnot x)` collapse
    ///    to `x`; `Reshape` to an identical shape is dropped and a
    ///    `Reshape(Reshape x)` chain collapses to a single relabel. (`Reshape`
    ///    is a zero-movement relabel, so this is exact.)
    /// 3. **Dead-node elimination.** Any node not reachable from a graph output
    ///    (graph inputs are always retained — they are the call ABI) is
    ///    dropped, including constants left dangling by rules 1–2.
    ///
    /// Returns the number of nodes removed. A cached schedule is invalidated.
    pub fn elide_invariants(&mut self) -> usize {
        use crate::OpKind as K;
        let before = self.nodes.len();
        if before == 0 {
            return 0;
        }

        // ── Phase 1: algebraic rewrite (forward / topological) ──
        let old = core::mem::take(&mut self.nodes);
        let mut new: Vec<Node> = Vec::with_capacity(old.len());
        // old node id -> the operand its consumers should now read.
        let mut map: Vec<InputSource> = (0..old.len() as u32)
            .map(|i| InputSource::Node(NodeId(i)))
            .collect();

        for (old_idx, node) in old.iter().enumerate() {
            let inputs: SmallVec<[InputSource; 4]> =
                node.inputs.iter().map(|s| remap_is(*s, &map)).collect();

            // `Some(operand)` ⇒ elide this node, redirecting consumers to
            // `operand` (already in new-id space). `None` ⇒ keep the node.
            let redirect: Option<InputSource> = match node.op {
                GraphOp::Op(K::Add) if inputs.len() == 2 => {
                    if identity_fill(&new, &self.constants, inputs[0]) == Some(IdentityConst::Zero)
                    {
                        Some(inputs[1])
                    } else if identity_fill(&new, &self.constants, inputs[1])
                        == Some(IdentityConst::Zero)
                    {
                        Some(inputs[0])
                    } else {
                        None
                    }
                }
                GraphOp::Op(K::Mul) if inputs.len() == 2 => {
                    if identity_fill(&new, &self.constants, inputs[0]) == Some(IdentityConst::One) {
                        Some(inputs[1])
                    } else if identity_fill(&new, &self.constants, inputs[1])
                        == Some(IdentityConst::One)
                    {
                        Some(inputs[0])
                    } else {
                        None
                    }
                }
                // Non-commutative: only the right operand may be the identity.
                GraphOp::Op(K::Sub) if inputs.len() == 2 => {
                    (identity_fill(&new, &self.constants, inputs[1]) == Some(IdentityConst::Zero))
                        .then_some(inputs[0])
                }
                GraphOp::Op(K::Div) | GraphOp::Op(K::Pow) if inputs.len() == 2 => {
                    (identity_fill(&new, &self.constants, inputs[1]) == Some(IdentityConst::One))
                        .then_some(inputs[0])
                }
                // Involutions: f(f(x)) → x.
                GraphOp::Op(K::Neg) | GraphOp::Op(K::Bnot) if inputs.len() == 1 => {
                    match inputs[0] {
                        InputSource::Node(NodeId(j)) if new[j as usize].op == node.op => {
                            Some(new[j as usize].inputs[0])
                        }
                        _ => None,
                    }
                }
                _ => None,
            };

            if let Some(operand) = redirect {
                map[old_idx] = operand;
                continue;
            }

            // Reshape relabels: drop a same-shape reshape; collapse a chain.
            if let GraphOp::Op(K::Reshape) = node.op {
                if let InputSource::Node(NodeId(j)) = inputs[0] {
                    let parent = &new[j as usize];
                    if parent.output_shape == node.output_shape {
                        // Relabel to the shape it already has → identity.
                        map[old_idx] = inputs[0];
                        continue;
                    }
                    if parent.op == GraphOp::Op(K::Reshape) {
                        // Reshape∘Reshape → one relabel to the outer shape.
                        let inner_src = parent.inputs[0];
                        let id = new.len() as u32;
                        new.push(Node {
                            op: GraphOp::Op(K::Reshape),
                            inputs: SmallVec::from_iter([inner_src]),
                            output_dtype: node.output_dtype,
                            output_shape: node.output_shape,
                        });
                        map[old_idx] = InputSource::Node(NodeId(id));
                        continue;
                    }
                }
            }

            let id = new.len() as u32;
            let mut n = node.clone();
            n.inputs = inputs;
            new.push(n);
            map[old_idx] = InputSource::Node(NodeId(id));
        }

        // Port / attribute lists only ever name op/IO nodes (never elided), so
        // each maps to a `Node` operand; extract its id.
        let to_id = |is: InputSource| match is {
            InputSource::Node(nid) => nid,
            _ => unreachable!("input/output/attr node was elided to a non-node operand"),
        };
        for nid in self.inputs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for nid in self.outputs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.conv_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.quant_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.lrn_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.gemm_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.norm_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.reduce_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        for (nid, _) in self.gather_attrs.iter_mut() {
            *nid = to_id(map[nid.0 as usize]);
        }
        self.nodes = new;

        // ── Phase 2: dead-node elimination ──
        let n = self.nodes.len();
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        for &NodeId(o) in self.outputs.iter() {
            stack.push(o as usize);
        }
        // Graph inputs are the call ABI — retain them even if unused.
        for &NodeId(i) in self.inputs.iter() {
            stack.push(i as usize);
        }
        while let Some(i) = stack.pop() {
            if i >= n || live[i] {
                continue;
            }
            live[i] = true;
            for inp in &self.nodes[i].inputs {
                if let InputSource::Node(NodeId(p)) = inp {
                    stack.push(*p as usize);
                }
            }
        }
        if live.iter().any(|&l| !l) {
            let mut dmap: Vec<u32> = vec![0u32; n];
            let kept = core::mem::take(&mut self.nodes);
            let mut compact: Vec<Node> = Vec::with_capacity(n);
            for (i, node) in kept.into_iter().enumerate() {
                if live[i] {
                    dmap[i] = compact.len() as u32;
                    compact.push(node);
                }
            }
            for node in compact.iter_mut() {
                for inp in node.inputs.iter_mut() {
                    *inp = remap_src(*inp, &dmap);
                }
            }
            for nid in self.inputs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for nid in self.outputs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.conv_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.quant_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.lrn_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.gemm_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.norm_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.reduce_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            for (nid, _) in self.gather_attrs.iter_mut() {
                *nid = NodeId(dmap[nid.0 as usize]);
            }
            self.nodes = compact;
        }

        self.schedule = None;
        before.saturating_sub(self.nodes.len())
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

#[cfg(test)]
mod elision_tests {
    use super::*;
    use crate::constant::ConstantEntry;
    use crate::registry::{DTypeId, ShapeId};
    use crate::OpKind as K;

    const F32: DTypeId = DTypeId(8);
    const SH: ShapeId = ShapeId(0);

    fn input(g: &mut Graph) -> NodeId {
        let id = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: F32,
            output_shape: SH,
        });
        g.add_input(id);
        id
    }
    fn konst(g: &mut Graph, v: f32) -> InputSource {
        let cid = g.constants_mut().insert(ConstantEntry {
            bytes: v.to_le_bytes().to_vec(),
            dtype: F32,
            shape: SH,
        });
        InputSource::Constant(cid)
    }
    fn op(g: &mut Graph, k: K, ins: &[InputSource]) -> NodeId {
        g.add_node(Node {
            op: GraphOp::Op(k),
            inputs: SmallVec::from_iter(ins.iter().copied()),
            output_dtype: F32,
            output_shape: SH,
        })
    }
    fn output(g: &mut Graph, src: NodeId) {
        let id = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(src)]),
            output_dtype: F32,
            output_shape: SH,
        });
        g.add_output(id);
    }
    fn out_feeds(g: &Graph) -> InputSource {
        g.nodes()[g.outputs()[0].0 as usize].inputs[0]
    }

    /// `x + 0`, `0 + x`, `x - 0`, `x · 1`, `1 · x`, `x / 1`, `x¹` all collapse
    /// so the output reads the live input `x` directly and the op is gone.
    #[test]
    fn identity_elements_collapse() {
        for (k, lhs_id, rhs_id) in [
            (K::Add, Some(0.0), None),
            (K::Add, None, Some(0.0)),
            (K::Sub, None, Some(0.0)),
            (K::Mul, Some(1.0), None),
            (K::Mul, None, Some(1.0)),
            (K::Div, None, Some(1.0)),
            (K::Pow, None, Some(1.0)),
        ] {
            let mut g = Graph::new();
            let x = input(&mut g);
            let a = match lhs_id {
                Some(v) => konst(&mut g, v),
                None => InputSource::Node(x),
            };
            let b = match rhs_id {
                Some(v) => konst(&mut g, v),
                None => InputSource::Node(x),
            };
            let n = op(&mut g, k, &[a, b]);
            output(&mut g, n);
            let removed = g.elide_invariants();
            assert!(removed >= 1, "{k:?} identity should be elided");
            assert!(
                !g.nodes().iter().any(|nd| nd.op == GraphOp::Op(k)),
                "{k:?} op should be gone"
            );
            assert_eq!(out_feeds(&g), InputSource::Node(g.inputs()[0]));
        }
    }

    /// Annihilators are NOT elided: `x · 0 → 0` is unsound under IEEE
    /// (`∞·0 = NaN`), so the Mul must survive.
    #[test]
    fn multiply_by_zero_is_preserved() {
        let mut g = Graph::new();
        let x = input(&mut g);
        let z = konst(&mut g, 0.0);
        let n = op(&mut g, K::Mul, &[InputSource::Node(x), z]);
        output(&mut g, n);
        g.elide_invariants();
        assert!(g.nodes().iter().any(|nd| nd.op == GraphOp::Op(K::Mul)));
    }

    /// A non-identity constant operand blocks the rule.
    #[test]
    fn non_identity_constant_blocks_elision() {
        let mut g = Graph::new();
        let x = input(&mut g);
        let two = konst(&mut g, 2.0);
        let n = op(&mut g, K::Add, &[InputSource::Node(x), two]);
        output(&mut g, n);
        g.elide_invariants();
        assert!(g.nodes().iter().any(|nd| nd.op == GraphOp::Op(K::Add)));
    }

    /// `Neg(Neg x)` and `Bnot(Bnot x)` cancel to `x`; a single one survives.
    #[test]
    fn involutions_cancel_in_pairs() {
        for k in [K::Neg, K::Bnot] {
            let mut g = Graph::new();
            let x = input(&mut g);
            let inner = op(&mut g, k, &[InputSource::Node(x)]);
            let outer = op(&mut g, k, &[InputSource::Node(inner)]);
            output(&mut g, outer);
            g.elide_invariants();
            assert!(
                !g.nodes().iter().any(|nd| nd.op == GraphOp::Op(k)),
                "{k:?}∘{k:?} should fully cancel"
            );
            assert_eq!(out_feeds(&g), InputSource::Node(g.inputs()[0]));

            // Odd count: one survives.
            let mut g = Graph::new();
            let x = input(&mut g);
            let a = op(&mut g, k, &[InputSource::Node(x)]);
            let b = op(&mut g, k, &[InputSource::Node(a)]);
            let c = op(&mut g, k, &[InputSource::Node(b)]);
            output(&mut g, c);
            g.elide_invariants();
            assert_eq!(
                g.nodes()
                    .iter()
                    .filter(|nd| nd.op == GraphOp::Op(k))
                    .count(),
                1
            );
        }
    }

    /// A `Reshape` to a shape the input already has is a no-op and is dropped.
    #[test]
    fn reshape_to_same_shape_drops() {
        let mut g = Graph::new();
        let x = input(&mut g); // shape SH
        let r = op(&mut g, K::Reshape, &[InputSource::Node(x)]); // also SH
        output(&mut g, r);
        g.elide_invariants();
        assert!(!g.nodes().iter().any(|nd| nd.op == GraphOp::Op(K::Reshape)));
        assert_eq!(out_feeds(&g), InputSource::Node(g.inputs()[0]));
    }

    /// `Reshape(Reshape x)` collapses to a single relabel to the outer shape.
    #[test]
    fn reshape_chain_collapses() {
        let mut g = Graph::new();
        let s2 = g
            .shape_registry_mut()
            .intern(crate::registry::ShapeDescriptor {
                rank: 1,
                dims: [4, 0, 0, 0, 0, 0, 0, 0],
                dims_overflow: None,
            });
        let s3 = g
            .shape_registry_mut()
            .intern(crate::registry::ShapeDescriptor {
                rank: 2,
                dims: [2, 2, 0, 0, 0, 0, 0, 0],
                dims_overflow: None,
            });
        let x = input(&mut g);
        let r1 = g.add_node(Node {
            op: GraphOp::Op(K::Reshape),
            inputs: SmallVec::from_iter([InputSource::Node(x)]),
            output_dtype: F32,
            output_shape: s2,
        });
        let r2 = g.add_node(Node {
            op: GraphOp::Op(K::Reshape),
            inputs: SmallVec::from_iter([InputSource::Node(r1)]),
            output_dtype: F32,
            output_shape: s3,
        });
        output(&mut g, r2);
        g.elide_invariants();
        let reshapes: Vec<_> = g
            .nodes()
            .iter()
            .filter(|nd| nd.op == GraphOp::Op(K::Reshape))
            .collect();
        assert_eq!(reshapes.len(), 1, "chain collapses to one relabel");
        assert_eq!(reshapes[0].output_shape, s3);
        assert_eq!(reshapes[0].inputs[0], InputSource::Node(g.inputs()[0]));
    }

    /// Dead nodes (not reachable from any output) are removed.
    #[test]
    fn dead_nodes_are_eliminated() {
        let mut g = Graph::new();
        let x = input(&mut g);
        let live = op(&mut g, K::Relu, &[InputSource::Node(x)]);
        // Dead branch: never feeds an output.
        let _dead = op(&mut g, K::Sigmoid, &[InputSource::Node(x)]);
        output(&mut g, live);
        g.elide_invariants();
        assert!(!g.nodes().iter().any(|nd| nd.op == GraphOp::Op(K::Sigmoid)));
        assert!(g.nodes().iter().any(|nd| nd.op == GraphOp::Op(K::Relu)));
    }
}
