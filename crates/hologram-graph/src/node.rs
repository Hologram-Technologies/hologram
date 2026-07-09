//! Node + GraphOp definitions (spec VI.1, VI.2).

use crate::registry::{DTypeId, ShapeId};
use smallvec::SmallVec;

/// Stable opaque handle. The compiler may also stamp a generation tag on
/// these for use-after-free protection; here the index is sufficient since
/// the graph is append-only during build and frozen during compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantId(pub u32);

/// The op slot of a graph node. Spec VI.1: a single closed enum unifies
/// all dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphOp {
    /// Reference into the closed `OpKind` catalog. The compiler routes
    /// each variant to its corresponding emitter + kernel pair.
    Op(crate::OpKind),
    /// Graph input port.
    Input,
    /// Graph output port.
    Output,
    /// Inline constant referenced by `ConstantStore`.
    Constant(ConstantId),
}

/// Where a node's input value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputSource {
    Node(NodeId),
    Constant(ConstantId),
    /// Named graph-input port; resolved by the runtime against the
    /// session's input bindings.
    GraphInput(u32),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: GraphOp,
    pub inputs: SmallVec<[InputSource; 4]>,
    pub output_dtype: DTypeId,
    pub output_shape: ShapeId,
}

/// Quantization attributes (spec X-5). Symmetric INT8/INT4 scheme:
/// `dequantized = (q − zero_point) · scale`. Stored on `Graph::quant_attrs`
/// keyed by `NodeId` rather than inlined into `Node` so that ordinary nodes
/// pay no per-instance overhead.
///
/// `axis < 0` is **per-tensor** (one scalar `scale_bits`/`zero_point`).
/// `axis >= 0` is **per-channel** along that axis (ONNX `DequantizeLinear`
/// per-axis): the dequantize node then carries the per-channel `scale` (f32)
/// and `zero_point` (i32) vectors as its 2nd and 3rd operands, and the
/// compiler derives the channel count / inner stride from the input shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantAttrs {
    /// Source quantized dtype (a [`hologram_types::DTypeId`] numeric tag).
    pub quant_dtype: u8,
    /// `f32::to_bits` of the per-tensor scale (per-tensor mode only).
    pub scale_bits: u32,
    /// Symmetric zero-point (per-tensor mode only).
    pub zero_point: i32,
    /// Quantization axis: `< 0` ⇒ per-tensor; `>= 0` ⇒ per-channel along it.
    pub axis: i32,
    /// **Layout of the weight's bytes as they are bound**
    /// ([`hologram_types::weight_layout`]). A statement of fact, not a request.
    ///
    /// A weightless compile — the graph carries a κ for the weight and the bytes
    /// arrive at materialization — has no constant bytes for the compiler to
    /// transpose, so the fast output-major decode path was unreachable no matter
    /// what the model was. Declaring `OUTPUT_MAJOR` says the binder will
    /// materialize `[n, k]`, which lets the *load-time* fusion emit the same
    /// fused call the constant-weight path emits. The backend re-validates every
    /// property it can (symmetry, per-channel, `k` bound).
    ///
    /// **Only a load-time-bound weight may declare `OUTPUT_MAJOR`.** A graph
    /// constant's bytes are already here, in `[k, n]`; claiming otherwise is
    /// false and the compiler rejects it. Nothing can honour it, and the loader
    /// would read `[k, n]` as `[n, k]` and return a plausible wrong answer. To
    /// put a constant on the fused path, set [`Self::act_quant`] alone — the
    /// compiler owns those bytes and transposes them itself.
    ///
    /// If the declaration cannot be served by any output-major kernel, the loader
    /// fails with `ExecError::UnsatisfiableWeightLayout` rather than falling back
    /// to a `[k, n]`-assuming path. See `docs/numerics/w8a8.md`.
    pub weight_layout: u8,
    /// **Activation treatment this weight opts into**
    /// ([`hologram_types::act_quant`]). Orthogonal to [`Self::weight_layout`],
    /// which describes bytes; this describes numerics. Defaults to `W8A32`,
    /// bit-identical to `dequantize → matmul`.
    ///
    /// `W8A8_TOKEN_SYM` rounds the activation, so it changes the computed value.
    /// It is therefore never an implicit upgrade of an existing path — a weight
    /// must ask for it, constant or not. E8CB has no W1A32 form and must set it.
    pub act_quant: u8,
}

impl Default for QuantAttrs {
    fn default() -> Self {
        Self {
            quant_dtype: 0,
            scale_bits: 0,
            zero_point: 0,
            axis: -1,
            // A weight declares nothing by default: `[k,n]` bytes, f32
            // activations. W8A8 is never an implicit upgrade.
            weight_layout: hologram_types::weight_layout::ROW_MAJOR,
            act_quant: hologram_types::act_quant::W8A32,
        }
    }
}

/// Per-node convolution attributes (stride / padding / dilation).
/// Stored sparsely on `Graph::conv_attrs` keyed by `NodeId` so the
/// common case (default `stride = (1, 1)`, no padding) costs nothing.
/// The compiler threads these into `LoweredNode.shape` during lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvAttrs {
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    /// Kernel window. `0` means "derive from the weight operand" (the conv /
    /// pool default). The `Im2Col` / `Col2Im` ops have no weight operand, so
    /// they carry the window explicitly here.
    pub k_h: u32,
    pub k_w: u32,
}

impl Default for ConvAttrs {
    fn default() -> Self {
        Self {
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            k_h: 0,
            k_w: 0,
        }
    }
}

/// Per-node GEMM scalars (ONNX `Gemm`: `Y = α·A·B + β·C`). Stored sparsely on
/// `Graph::gemm_attrs`; `*_bits` are `f32::to_bits`. Default is the plain
/// `A·B + C` (α = β = 1) — without this the lowered `GemmCall` carried α=β=0
/// and the op computed zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmAttrs {
    pub alpha_bits: u32,
    pub beta_bits: u32,
}

impl Default for GemmAttrs {
    fn default() -> Self {
        Self {
            alpha_bits: 1.0f32.to_bits(),
            beta_bits: 1.0f32.to_bits(),
        }
    }
}

/// Per-node LRN (local response normalization) attributes. Stored sparsely on
/// `Graph::lrn_attrs` keyed by `NodeId` (ONNX defaults: α=1e-4, β=0.75,
/// bias=1.0). `*_bits` are `f32::to_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LrnAttrs {
    pub size: u32,
    pub alpha_bits: u32,
    pub beta_bits: u32,
    pub bias_bits: u32,
}

impl Default for LrnAttrs {
    fn default() -> Self {
        Self {
            size: 1,
            alpha_bits: 0.0001f32.to_bits(),
            beta_bits: 0.75f32.to_bits(),
            bias_bits: 1.0f32.to_bits(),
        }
    }
}

/// Per-node normalization grouping attribute. Stored sparsely on
/// `Graph::norm_attrs` keyed by `NodeId`. Only `GroupNorm` reads it; the
/// compiler derives `InstanceNorm`'s effective group count (= channels) and
/// leaves `LayerNorm`/`RmsNorm` ungrouped. `num_groups = 1` is plain
/// per-sample normalization over all channels × spatial (the ONNX default
/// for GroupNorm is supplied explicitly by the frontend).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormAttrs {
    pub num_groups: u32,
}

impl Default for NormAttrs {
    fn default() -> Self {
        Self { num_groups: 1 }
    }
}

/// Per-node reduction axes (ONNX `axes` + `keepdims`). Stored sparsely on
/// `Graph::reduce_attrs` keyed by `NodeId`. `axes_mask` bit `i` set ⇒ reduce
/// axis `i`. A node with no attached `ReduceAttrs` reduces over **all** axes
/// (full reduction to a scalar — the default), so existing graphs are
/// unaffected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ReduceAttrs {
    pub axes_mask: u32,
    pub keepdims: bool,
}

/// Per-node `Gather` axis (ONNX `Gather.axis`, default 0). Stored sparsely on
/// `Graph::gather_attrs` keyed by `NodeId`. `axis < 0` counts from the end of
/// the data rank (ONNX convention), normalized against the data shape at
/// compile time. A node with no attached `GatherAttrs` gathers along axis 0
/// (the embedding-lookup default).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GatherAttrs {
    pub axis: i32,
}

/// Attention semantics that are NOT derivable from operand shapes: whether the
/// attention is causal (lower-triangular score mask) and an optional softmax
/// score multiplier. Stored sparsely on `Graph::attention_attrs` keyed by
/// `NodeId`, like the other op attributes. `scale_bits == 0` ⇒ the default
/// `1/√head_dim`. Grouped-query `kv_heads` is NOT carried here — the compiler
/// derives it from the K operand's head dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AttentionAttrs {
    /// Causal (autoregressive) attention: query `i` attends only to keys
    /// `j ≤ i`. Set by the importer/fusion for decoder LMs.
    pub causal: bool,
    /// `f32::to_bits` of the softmax score multiplier; `0` ⇒ default `1/√d`.
    pub scale_bits: u32,
}
