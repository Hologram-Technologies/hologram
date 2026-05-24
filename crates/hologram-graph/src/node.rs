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
    /// Removed by a fusion pass. The compiler and scheduler skip Dead
    /// nodes. Using a sentinel variant instead of `Option<Node>` avoids
    /// invalidating arena indices.
    Dead,
}

impl GraphOp {
    /// Returns `true` for elementwise-unary activation ops that are
    /// valid epilogue targets for MatMul/Conv/Norm fusion.
    pub fn is_fusable_activation(self) -> bool {
        match self {
            GraphOp::Op(k) => k.is_fusable_activation(),
            _ => false,
        }
    }
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

/// Fusion epilogue metadata (spec VI.3). Stored on `Graph::fusion_attrs`
/// keyed by `NodeId`. Captures the epilogue activation to apply after a
/// fused op (e.g., `FusedMatMulActivation`) or the full chain for
/// `FusedUnaryChain`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FusionAttrs {
    /// The activation op discriminant (e.g. `OpKind::Relu as u16`).
    /// For `FusedUnaryChain`, this is the first element of the chain.
    pub activation: u16,
    /// Number of ops in the chain (1..=8). 0 means single activation.
    pub chain_len: u8,
    /// Chained activation discriminants. `chain[0]` is redundant with
    /// `activation` when `chain_len > 0`.
    pub chain: [u16; 8],
}

/// Per-tensor quantization attributes (spec X-5). Symmetric INT8/INT4
/// scheme: `dequantized = (q − zero_point) · scale`. Stored on
/// `Graph::quant_attrs` keyed by `NodeId` rather than inlined into
/// `Node` so that ordinary nodes pay no per-instance overhead.
/// Per-channel quantization is a future extension layered as multiple
/// `QuantAttrs` keyed on the channel axis.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantAttrs {
    /// Source quantized dtype (DTypeI8 or DTypeI4 numeric tag).
    pub quant_dtype: u8,
    /// `f32::to_bits` of the per-tensor scale.
    pub scale_bits: u32,
    /// Symmetric zero-point.
    pub zero_point: i32,
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
