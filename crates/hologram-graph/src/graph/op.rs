//! Graph operation definitions and semantic/legacy bridging.

use crate::constant::ConstantId;
use hologram_core::op::{FloatOp, LutOp, PrimOp, RingLevel};
use hologram_core::view::ElementWiseView;
use hologram_ops::{
    AttentionAttrs, ClipAttrs, ConcatAttrs, Conv2dAttrs, ConvTransposeAttrs, CumSumAttrs,
    ExpandAttrs, GemmAttrs, GlobalAvgPoolAttrs, GroupNormAttrs, LrnAttrs, MatMulAttrs, NormAttrs,
    PadAttrs, Pool2dAttrs, ReduceAttrs, ResizeAttrs, RotaryEmbeddingAttrs, SemanticOp, SliceAttrs,
    SoftmaxAttrs, TransposeAttrs,
};

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
pub struct SubgraphId(pub(crate) u32);

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
    /// LUT-GEMM matmul with 2-bit quantized weights (4 centroids).
    /// Pure integer inner loop — no dequant to f32. Half the bandwidth of Q4.
    MatMulLut2(ConstantId),
    /// Fused 2-bit LUT-GEMM + activation (epilogue fusion).
    MatMulLut2Activation(ConstantId, FloatOp),
    /// Fused 4-bit LUT-GEMM + activation (epilogue fusion).
    MatMulLut4Activation(ConstantId, FloatOp),
    /// Fused 8-bit LUT-GEMM + activation (epilogue fusion).
    MatMulLut8Activation(ConstantId, FloatOp),
    /// Consumer-defined op. Dispatched via `CustomOpRegistry` at execution time.
    ///
    /// The `arity` field must match the number of edges wired to this node.
    Custom { id: CustomOpId, arity: u8 },
    /// Canonical semantic compute op.
    ///
    /// This is the new graph-facing op path. It captures semantic intent
    /// without committing to a legacy `FloatOp` encoding or backend dispatch.
    Compute(SemanticOp),
    /// Typed f32 tensor operation for AI inference.
    ///
    /// Unlike `PrimOp`/`LutOp` (byte-domain), these operate on f32 buffers
    /// with shape-aware semantics. Serialized into archives alongside other ops.
    ///
    /// Legacy compatibility path. New graph construction should prefer
    /// `GraphOp::Compute` wherever the canonical semantic op model is enough.
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
    /// Fused AddRmsNorm + activation (residual + normalize + activation).
    FusedAddRmsNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused InstanceNorm + activation (epilogue fusion).
    FusedInstanceNormActivation {
        size: u32,
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
        input_h: u32,
        input_w: u32,
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
        input_h: u32,
        input_w: u32,
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
    /// Ring-native activation (21 ops, composed from ring primitives).
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
    pub fn arity(&self) -> u8 {
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
            | Self::BatchMatMulLut16(_)
            | Self::MatMulLut2(_)
            | Self::MatMulLut2Activation(..) => 1,
            Self::FusedView16(_) => 1,
            Self::RingPrimUnary(_, _) => 1,
            Self::RingPrimBinary(_, _) => 2,
            Self::RingActivation(_, _) => 1,
            Self::RingAccumulate(_) => 3,
            Self::RingReduce { .. } => 1,
            Self::Conv2dLut4 { .. } => 2,
            Self::Prim(p) => p.arity(),
            Self::Custom { arity, .. } => *arity,
            Self::Compute(op) => op.arity(),
            Self::Float(f) => f.arity(),
            Self::FusedFloatChain(_) => 1,
            Self::FusedMatMulActivation { .. } => 2,
            Self::FusedMatMulBiasActivation { .. } => 3,
            Self::FusedConv2dActivation { .. } => 3,
            Self::FusedConv2dBiasActivation { .. } => 3,
            Self::FusedRmsNormActivation { .. } => 2,
            Self::FusedLayerNormActivation { .. }
            | Self::FusedGroupNormActivation { .. }
            | Self::FusedAddRmsNormActivation { .. }
            | Self::FusedInstanceNormActivation { .. } => 3,
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
                | Self::MatMulLut2(_)
                | Self::MatMulLut4Activation(..)
                | Self::MatMulLut8Activation(..)
                | Self::MatMulLut2Activation(..)
                | Self::BatchMatMulLut4(_)
                | Self::BatchMatMulLut8(_)
                | Self::MatMulLut16(_)
                | Self::BatchMatMulLut16(_)
                | Self::Custom { .. }
                | Self::Compute(_)
                | Self::Float(_)
                | Self::FusedFloatChain(_)
                | Self::FusedMatMulActivation { .. }
                | Self::FusedMatMulBiasActivation { .. }
                | Self::FusedConv2dActivation { .. }
                | Self::FusedConv2dBiasActivation { .. }
                | Self::FusedRmsNormActivation { .. }
                | Self::FusedLayerNormActivation { .. }
                | Self::FusedGroupNormActivation { .. }
                | Self::FusedAddRmsNormActivation { .. }
                | Self::FusedInstanceNormActivation { .. }
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
                    p.apply_unary_u64(x as u64, 2) as u16
                }))
            }
            Self::FusedView16(v) => Some((**v).clone()),
            _ => None,
        }
    }

    /// Build a graph op from a [`FloatOp`], preferring the canonical
    /// `GraphOp::Compute(SemanticOp)` form whenever the canonical layer
    /// covers the variant. Falls back to `GraphOp::Float(f)` for ops
    /// that haven't been migrated to `SemanticOp` yet (e.g. comparison
    /// ops, pooling, attention — Sprint 37 Phase 3.4 tracks expansion).
    ///
    /// This is the single smart-constructor used by lowering pipelines
    /// (compiler, ONNX import, hand-built graphs) to opt new code into
    /// the canonical path without forcing every caller to know which
    /// variants are covered.
    #[must_use]
    pub fn from_float(f: FloatOp) -> Self {
        match semantic_for_float(f) {
            Some(s) => Self::Compute(s),
            None => Self::Float(f),
        }
    }

    /// Resolve this graph op to a legacy `FloatOp` when possible.
    ///
    /// This is the one-way execution-side bridge: canonical
    /// `GraphOp::Compute(SemanticOp)` nodes are lowered through this
    /// adapter into `FloatOp` so they can flow through the existing
    /// tape and backend dispatch. The reverse direction
    /// (`FloatOp` → `SemanticOp`) was speculative and has been removed
    /// — no production callers ever needed it. If a future use case
    /// emerges, add it back as a focused helper.
    #[must_use]
    pub fn legacy_float_op(&self) -> Option<FloatOp> {
        match self {
            Self::Compute(op) => float_from_semantic(*op),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }
}

#[must_use]
fn float_from_semantic(op: SemanticOp) -> Option<FloatOp> {
    match op {
        SemanticOp::Add => Some(FloatOp::Add),
        SemanticOp::Sub => Some(FloatOp::Sub),
        SemanticOp::Mul => Some(FloatOp::Mul),
        SemanticOp::Div => Some(FloatOp::Div),
        SemanticOp::Neg => Some(FloatOp::Neg),
        SemanticOp::Relu => Some(FloatOp::Relu),
        SemanticOp::Gelu => Some(FloatOp::Gelu),
        SemanticOp::Silu => Some(FloatOp::Silu),
        SemanticOp::Tanh => Some(FloatOp::Tanh),
        SemanticOp::Sigmoid => Some(FloatOp::Sigmoid),
        SemanticOp::Exp => Some(FloatOp::Exp),
        SemanticOp::Log => Some(FloatOp::Log),
        SemanticOp::Sqrt => Some(FloatOp::Sqrt),
        SemanticOp::Abs => Some(FloatOp::Abs),
        SemanticOp::Reciprocal => Some(FloatOp::Reciprocal),
        SemanticOp::Cos => Some(FloatOp::Cos),
        SemanticOp::Sin => Some(FloatOp::Sin),
        SemanticOp::Sign => Some(FloatOp::Sign),
        SemanticOp::Floor => Some(FloatOp::Floor),
        SemanticOp::Ceil => Some(FloatOp::Ceil),
        SemanticOp::Round => Some(FloatOp::Round),
        SemanticOp::Erf => Some(FloatOp::Erf),
        SemanticOp::MatMul(attrs) => Some(FloatOp::MatMul {
            m: attrs.m,
            k: attrs.k,
            n: attrs.n,
        }),
        SemanticOp::Softmax(attrs) => Some(FloatOp::Softmax { size: attrs.size }),
        SemanticOp::LogSoftmax(attrs) => Some(FloatOp::LogSoftmax { size: attrs.size }),
        SemanticOp::RmsNorm(attrs) => Some(FloatOp::RmsNorm {
            size: attrs.size,
            epsilon: attrs.epsilon,
        }),
        SemanticOp::LayerNorm(attrs) => Some(FloatOp::LayerNorm {
            size: attrs.size,
            epsilon: attrs.epsilon,
        }),
        SemanticOp::InstanceNorm(attrs) => Some(FloatOp::InstanceNorm {
            size: attrs.size,
            epsilon: attrs.epsilon,
        }),
        SemanticOp::GroupNorm(attrs) => Some(FloatOp::GroupNorm {
            num_groups: attrs.num_groups,
            epsilon: attrs.epsilon,
        }),
        SemanticOp::AddRmsNorm(attrs) => Some(FloatOp::AddRmsNorm {
            size: attrs.size,
            epsilon: attrs.epsilon,
        }),
        SemanticOp::Transpose(attrs) => Some(FloatOp::Transpose {
            perm: attrs.perm,
            ndim: attrs.ndim,
        }),
        SemanticOp::Reshape => Some(FloatOp::Reshape),
        SemanticOp::Slice(attrs) => Some(FloatOp::Slice {
            axis_from_end: attrs.axis_from_end,
            start: attrs.start,
            end: attrs.end,
            axis_size: attrs.axis_size,
        }),
        SemanticOp::Concat(attrs) => Some(FloatOp::Concat {
            size_a: attrs.size_a,
            size_b: attrs.size_b,
            dtype: hologram_core::op::FloatDType::F32,
        }),
        SemanticOp::Conv2d(attrs) => Some(FloatOp::Conv2d {
            kernel_h: attrs.kernel_h,
            kernel_w: attrs.kernel_w,
            stride_h: attrs.stride_h,
            stride_w: attrs.stride_w,
            pad_h: attrs.pad_h,
            pad_w: attrs.pad_w,
            dilation_h: attrs.dilation_h,
            dilation_w: attrs.dilation_w,
            group: attrs.group,
            input_h: attrs.input_h,
            input_w: attrs.input_w,
        }),
        SemanticOp::FusedSwiGlu => Some(FloatOp::FusedSwiGLU),
        SemanticOp::Pow => Some(FloatOp::Pow),
        SemanticOp::Mod => Some(FloatOp::Mod),
        SemanticOp::Min => Some(FloatOp::Min),
        SemanticOp::Max => Some(FloatOp::Max),
        SemanticOp::Equal => Some(FloatOp::Equal),
        SemanticOp::Less => Some(FloatOp::Less),
        SemanticOp::LessOrEqual => Some(FloatOp::LessOrEqual),
        SemanticOp::Greater => Some(FloatOp::Greater),
        SemanticOp::GreaterOrEqual => Some(FloatOp::GreaterOrEqual),
        SemanticOp::And => Some(FloatOp::And),
        SemanticOp::Or => Some(FloatOp::Or),
        SemanticOp::Xor => Some(FloatOp::Xor),
        SemanticOp::Not => Some(FloatOp::Not),
        SemanticOp::IsNaN => Some(FloatOp::IsNaN),
        SemanticOp::ReduceSum(attrs) => Some(FloatOp::ReduceSum { size: attrs.size }),
        SemanticOp::ReduceMean(attrs) => Some(FloatOp::ReduceMean { size: attrs.size }),
        SemanticOp::ReduceMax(attrs) => Some(FloatOp::ReduceMax { size: attrs.size }),
        SemanticOp::ReduceMin(attrs) => Some(FloatOp::ReduceMin { size: attrs.size }),
        SemanticOp::ReduceProd(attrs) => Some(FloatOp::ReduceProd { size: attrs.size }),
        SemanticOp::MaxPool2d(attrs) => Some(FloatOp::MaxPool2d {
            kernel_h: attrs.kernel_h,
            kernel_w: attrs.kernel_w,
            stride_h: attrs.stride_h,
            stride_w: attrs.stride_w,
            pad_h: attrs.pad_h,
            pad_w: attrs.pad_w,
        }),
        SemanticOp::AvgPool2d(attrs) => Some(FloatOp::AvgPool2d {
            kernel_h: attrs.kernel_h,
            kernel_w: attrs.kernel_w,
            stride_h: attrs.stride_h,
            stride_w: attrs.stride_w,
            pad_h: attrs.pad_h,
            pad_w: attrs.pad_w,
        }),
        SemanticOp::GlobalAvgPool(attrs) => Some(FloatOp::GlobalAvgPool {
            channels: attrs.channels,
            spatial_h: attrs.spatial_h,
            spatial_w: attrs.spatial_w,
        }),
        SemanticOp::Where => Some(FloatOp::Where),
        SemanticOp::Clip(attrs) => Some(FloatOp::Clip {
            min: attrs.min,
            max: attrs.max,
        }),
        SemanticOp::CumSum(attrs) => Some(FloatOp::CumSum { axis: attrs.axis }),
        SemanticOp::Pad(attrs) => Some(FloatOp::PadOp { mode: attrs.mode }),
        SemanticOp::Resize(attrs) => Some(FloatOp::Resize { mode: attrs.mode }),
        SemanticOp::Lrn(attrs) => Some(FloatOp::LRN {
            size: attrs.size,
            alpha: attrs.alpha,
            beta: attrs.beta,
            bias: attrs.bias,
        }),
        SemanticOp::ConvTranspose2d(attrs) => Some(FloatOp::ConvTranspose {
            kernel_h: attrs.kernel_h,
            kernel_w: attrs.kernel_w,
            stride_h: attrs.stride_h,
            stride_w: attrs.stride_w,
            pad_h: attrs.pad_h,
            pad_w: attrs.pad_w,
            dilation_h: attrs.dilation_h,
            dilation_w: attrs.dilation_w,
            group: attrs.group,
            output_pad_h: attrs.output_pad_h,
            output_pad_w: attrs.output_pad_w,
            input_h: attrs.input_h,
            input_w: attrs.input_w,
        }),
        SemanticOp::Gemm(attrs) => Some(FloatOp::Gemm {
            m: attrs.m,
            k: attrs.k,
            n: attrs.n,
            alpha: attrs.alpha,
            beta: attrs.beta,
            trans_a: attrs.trans_a,
            trans_b: attrs.trans_b,
            // Quantised B is permanent-FloatOp territory (ADR-048),
            // so the canonical → legacy bridge always emits f32 (0).
            quant_b: 0,
        }),
        SemanticOp::Expand(attrs) => Some(FloatOp::Expand {
            ndim: attrs.ndim,
            target_shape: attrs.target_shape,
        }),
        SemanticOp::RotaryEmbedding(attrs) => Some(FloatOp::RotaryEmbedding {
            dim: attrs.dim,
            base: attrs.base,
            n_heads: attrs.n_heads,
        }),
        SemanticOp::Attention(attrs) => Some(FloatOp::Attention {
            head_dim: attrs.head_dim,
            num_q_heads: attrs.num_q_heads,
            num_kv_heads: attrs.num_kv_heads,
            scale: attrs.scale,
            causal: attrs.causal,
            // Canonical attention is the un-fused form (ADR-049):
            // RoPE / QK-norm / sparse-V are upstream canonical ops or
            // execution flags, not part of canonical semantics.
            heads_first: true,
            qk_norm: false,
            rope: false,
            rope_base: 0,
            sparse_v: true,
        }),
    }
}

/// Inverse of [`float_from_semantic`] for the variants the canonical
/// layer covers. Returns `None` for `FloatOp` variants that have no
/// `SemanticOp` equivalent yet — those fall through to
/// `GraphOp::Float` via [`GraphOp::from_float`].
///
/// Private helper, intentionally not exposed as a public method on
/// `GraphOp`. Per ADR-046, the public canonical bridge is one-way; this
/// inverse is only for smart construction at the graph-builder
/// boundary, where it's clear that the goal is to *promote* a legacy
/// `FloatOp` to canonical form, not to keep both forms in flight.
#[must_use]
fn semantic_for_float(op: FloatOp) -> Option<SemanticOp> {
    match op {
        FloatOp::Add => Some(SemanticOp::Add),
        FloatOp::Sub => Some(SemanticOp::Sub),
        FloatOp::Mul => Some(SemanticOp::Mul),
        FloatOp::Div => Some(SemanticOp::Div),
        FloatOp::Neg => Some(SemanticOp::Neg),
        FloatOp::Relu => Some(SemanticOp::Relu),
        FloatOp::Gelu => Some(SemanticOp::Gelu),
        FloatOp::Silu => Some(SemanticOp::Silu),
        FloatOp::Tanh => Some(SemanticOp::Tanh),
        FloatOp::Sigmoid => Some(SemanticOp::Sigmoid),
        FloatOp::Exp => Some(SemanticOp::Exp),
        FloatOp::Log => Some(SemanticOp::Log),
        FloatOp::Sqrt => Some(SemanticOp::Sqrt),
        FloatOp::Abs => Some(SemanticOp::Abs),
        FloatOp::Reciprocal => Some(SemanticOp::Reciprocal),
        FloatOp::Cos => Some(SemanticOp::Cos),
        FloatOp::Sin => Some(SemanticOp::Sin),
        FloatOp::Sign => Some(SemanticOp::Sign),
        FloatOp::Floor => Some(SemanticOp::Floor),
        FloatOp::Ceil => Some(SemanticOp::Ceil),
        FloatOp::Round => Some(SemanticOp::Round),
        FloatOp::Erf => Some(SemanticOp::Erf),
        FloatOp::MatMul { m, k, n } => Some(SemanticOp::MatMul(MatMulAttrs { m, k, n })),
        FloatOp::Softmax { size } => Some(SemanticOp::Softmax(SoftmaxAttrs { size })),
        FloatOp::LogSoftmax { size } => Some(SemanticOp::LogSoftmax(SoftmaxAttrs { size })),
        FloatOp::RmsNorm { size, epsilon } => {
            Some(SemanticOp::RmsNorm(NormAttrs { size, epsilon }))
        }
        FloatOp::LayerNorm { size, epsilon } => {
            Some(SemanticOp::LayerNorm(NormAttrs { size, epsilon }))
        }
        FloatOp::InstanceNorm { size, epsilon } => {
            Some(SemanticOp::InstanceNorm(NormAttrs { size, epsilon }))
        }
        FloatOp::GroupNorm {
            num_groups,
            epsilon,
        } => Some(SemanticOp::GroupNorm(GroupNormAttrs {
            num_groups,
            epsilon,
        })),
        FloatOp::AddRmsNorm { size, epsilon } => {
            Some(SemanticOp::AddRmsNorm(NormAttrs { size, epsilon }))
        }
        FloatOp::Transpose { perm, ndim } => {
            Some(SemanticOp::Transpose(TransposeAttrs { perm, ndim }))
        }
        FloatOp::Reshape => Some(SemanticOp::Reshape),
        FloatOp::Slice {
            axis_from_end,
            start,
            end,
            axis_size,
        } => Some(SemanticOp::Slice(SliceAttrs {
            axis_from_end,
            start,
            end,
            axis_size,
        })),
        FloatOp::Concat { size_a, size_b, .. } => {
            Some(SemanticOp::Concat(ConcatAttrs { size_a, size_b }))
        }
        FloatOp::Conv2d {
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
        } => Some(SemanticOp::Conv2d(Conv2dAttrs {
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
        })),
        FloatOp::FusedSwiGLU => Some(SemanticOp::FusedSwiGlu),
        FloatOp::Pow => Some(SemanticOp::Pow),
        FloatOp::Mod => Some(SemanticOp::Mod),
        FloatOp::Min => Some(SemanticOp::Min),
        FloatOp::Max => Some(SemanticOp::Max),
        FloatOp::Equal => Some(SemanticOp::Equal),
        FloatOp::Less => Some(SemanticOp::Less),
        FloatOp::LessOrEqual => Some(SemanticOp::LessOrEqual),
        FloatOp::Greater => Some(SemanticOp::Greater),
        FloatOp::GreaterOrEqual => Some(SemanticOp::GreaterOrEqual),
        FloatOp::And => Some(SemanticOp::And),
        FloatOp::Or => Some(SemanticOp::Or),
        FloatOp::Xor => Some(SemanticOp::Xor),
        FloatOp::Not => Some(SemanticOp::Not),
        FloatOp::IsNaN => Some(SemanticOp::IsNaN),
        FloatOp::ReduceSum { size } => Some(SemanticOp::ReduceSum(ReduceAttrs { size })),
        FloatOp::ReduceMean { size } => Some(SemanticOp::ReduceMean(ReduceAttrs { size })),
        FloatOp::ReduceMax { size } => Some(SemanticOp::ReduceMax(ReduceAttrs { size })),
        FloatOp::ReduceMin { size } => Some(SemanticOp::ReduceMin(ReduceAttrs { size })),
        FloatOp::ReduceProd { size } => Some(SemanticOp::ReduceProd(ReduceAttrs { size })),
        FloatOp::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => Some(SemanticOp::MaxPool2d(Pool2dAttrs {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        })),
        FloatOp::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => Some(SemanticOp::AvgPool2d(Pool2dAttrs {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        })),
        FloatOp::GlobalAvgPool {
            channels,
            spatial_h,
            spatial_w,
        } => Some(SemanticOp::GlobalAvgPool(GlobalAvgPoolAttrs {
            channels,
            spatial_h,
            spatial_w,
        })),
        FloatOp::Where => Some(SemanticOp::Where),
        FloatOp::Clip { min, max } => Some(SemanticOp::Clip(ClipAttrs { min, max })),
        FloatOp::CumSum { axis } => Some(SemanticOp::CumSum(CumSumAttrs { axis })),
        FloatOp::PadOp { mode } => Some(SemanticOp::Pad(PadAttrs {
            // FloatOp::PadOp doesn't carry pad amounts — the legacy
            // path infers them from input/output shapes. Canonical
            // `Pad` requires explicit attrs, so promotion fills in
            // mode + zero-padding defaults; explicit pad amounts must
            // be supplied by the constructor.
            pad_h: 0,
            pad_w: 0,
            value: 0,
            mode,
        })),
        FloatOp::Resize { mode } => Some(SemanticOp::Resize(ResizeAttrs { mode })),
        FloatOp::LRN {
            size,
            alpha,
            beta,
            bias,
        } => Some(SemanticOp::Lrn(LrnAttrs {
            size,
            alpha,
            beta,
            bias,
        })),
        FloatOp::ConvTranspose {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            output_pad_h,
            output_pad_w,
            input_h,
            input_w,
        } => Some(SemanticOp::ConvTranspose2d(ConvTransposeAttrs {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            output_pad_h,
            output_pad_w,
            input_h,
            input_w,
        })),
        FloatOp::Gemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
            quant_b,
        } => {
            // Quantised B is permanent-FloatOp territory (ADR-048).
            // Only promote to canonical when B is f32 (quant_b == 0).
            if quant_b != 0 {
                None
            } else {
                Some(SemanticOp::Gemm(GemmAttrs {
                    m,
                    k,
                    n,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                }))
            }
        }
        FloatOp::Expand { ndim, target_shape } => {
            Some(SemanticOp::Expand(ExpandAttrs { ndim, target_shape }))
        }
        FloatOp::RotaryEmbedding { dim, base, n_heads } => {
            Some(SemanticOp::RotaryEmbedding(RotaryEmbeddingAttrs {
                dim,
                base,
                n_heads,
            }))
        }
        FloatOp::Attention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
            qk_norm,
            rope,
            rope_base: _,
            sparse_v: _,
        } => {
            // Canonical attention is the un-fused form (ADR-049). The
            // legacy variant must already be in canonical layout
            // (heads_first) and have no fused RoPE / QK-norm flags.
            // Other legacy variants stay on `FloatOp::Attention`.
            if heads_first && !qk_norm && !rope {
                Some(SemanticOp::Attention(AttentionAttrs {
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    causal,
                }))
            } else {
                None
            }
        }
        // Remaining unmigrated variants (Attention, RotaryEmbedding,
        // ReverseSequence) plus the permanent-FloatOp set per ADR-048
        // (Cast, Embed, Shape, Range, Gather, GatherND, ScatterND,
        // Dequantize, TopK, NonZero, Compress, KvWrite, KvRead,
        // ArgMax, NormProjectionGemv, AddNormProjectionGemv,
        // SwiGluProjectionGemv) fall through to legacy.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphop_arity() {
        assert_eq!(GraphOp::Input.arity(), 0);
        assert_eq!(GraphOp::Output.arity(), 1);
        assert_eq!(GraphOp::Lut(LutOp::Sigmoid).arity(), 1);
        assert_eq!(GraphOp::Prim(PrimOp::Add).arity(), 2);
        assert_eq!(GraphOp::Compute(SemanticOp::Add).arity(), 2);
        assert_eq!(GraphOp::Constant(ConstantId::new(0)).arity(), 0);
    }

    #[test]
    fn graphop_is_pure() {
        assert!(GraphOp::Prim(PrimOp::Add).is_pure());
        assert!(GraphOp::Lut(LutOp::Relu).is_pure());
        assert!(GraphOp::Compute(SemanticOp::Relu).is_pure());
        assert!(!GraphOp::Input.is_pure());
        assert!(!GraphOp::Output.is_pure());
        assert!(!GraphOp::Constant(ConstantId::new(0)).is_pure());
    }

    #[test]
    fn graphop_legacy_float_bridge_covers_compute_and_legacy_float() {
        let compute = GraphOp::Compute(SemanticOp::Transpose(TransposeAttrs {
            perm: [1, 0, 2, 3, 4, 5, 6, 7],
            ndim: 2,
        }));
        assert_eq!(
            compute.legacy_float_op(),
            Some(FloatOp::Transpose {
                perm: [1, 0, 2, 3, 4, 5, 6, 7],
                ndim: 2,
            })
        );

        let legacy = GraphOp::Float(FloatOp::Add);
        assert_eq!(legacy.legacy_float_op(), Some(FloatOp::Add));
    }

    #[test]
    fn graphop_to_view() {
        let v = GraphOp::Lut(LutOp::Relu).to_view().unwrap();
        assert_eq!(v.apply(0), LutOp::Relu.apply(0));
        let v = GraphOp::Prim(PrimOp::Neg).to_view().unwrap();
        assert_eq!(v.apply(1), 255);
        assert!(GraphOp::Prim(PrimOp::Add).to_view().is_none());
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
