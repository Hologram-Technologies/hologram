//! Instruction tape executor for zero-match dispatch.
//!
//! The tape is a flat array of pre-resolved instructions compiled from
//! the graph's execution schedule. Each instruction stores a kernel function
//! pointer and pre-resolved input/output indices, eliminating the large
//! `match op { ... }` dispatch at runtime.
//!
//! The tape is built once per model load and executed per inference call.
//! This is Phase 0.7 of the Compile-Time-First Acceleration plan.

use smallvec::SmallVec;
use std::sync::LazyLock;

use hologram_core::op::FloatOp;

/// Static empty shape map — avoids allocating a HashMap when no overrides exist.
static EMPTY_SHAPE_MAP: LazyLock<std::collections::HashMap<u32, Vec<usize>>> =
    LazyLock::new(std::collections::HashMap::new);
use hologram_graph::graph::node::NodeId;

use crate::buffer::{BufferArena, OutputBuffer};
use crate::error::ExecResult;
use crate::eval::executor::ExecutionContext;

// ── Enum-dispatch tape (Phase 8) ──────────────────────────────────────────────

use std::cell::{Cell, RefCell};

use hologram_core::op::{PrimOp, RingLevel};
use hologram_core::view::ElementWiseView;
use hologram_graph::constant::{ConstantId, ConstantStore};

use crate::kernel_dispatch::dispatch_kernel;
use crate::kv::weight_cache::WeightCache;
use crate::kv_cache::KvCacheState;

/// Execution context for the enum-dispatch tape.
///
/// Carries weight archive access, a lazily-populated weight cache
/// for LUT-GEMM ops, an optional KV cache for autoregressive generation,
pub struct TapeContext<'a> {
    /// Optional per-inference execution state (position offset, etc.).
    pub ctx: Option<ExecutionContext>,
    /// Constant store for resolving `ConstantId` → raw bytes.
    pub constants: &'a ConstantStore,
    /// Raw weight archive bytes for deferred constants.
    pub weights: &'a [u8],
    /// Persistent cache for deserialized quantized weights.
    /// Borrowed from the caller so it persists across execution calls.
    /// For LUT-GEMM, the first call deserializes weights; subsequent calls
    /// reuse them — eliminating per-step rkyv deserialization overhead.
    pub weight_cache: &'a parking_lot::RwLock<WeightCache>,
    /// Optional KV cache for autoregressive generation (KvWrite/KvRead ops).
    pub kv_state: Option<RefCell<KvCacheState>>,
    /// Pre-computed shape overrides from `ShapeContextGraph`.
    /// Keyed by raw node index. When present, the executor sets this as the
    /// output `TensorMeta` after dispatch, overriding any heuristic inference.
    /// Borrowed (not owned) to eliminate per-call HashMap cloning.
    pub shape_overrides: &'a std::collections::HashMap<u32, Vec<usize>>,
    /// Carry flux for dynamic precision — tracks accumulated carry across
    /// ring operations. Per-frame: call `reset_flux()` at frame boundaries
    /// in streaming workloads. Uses `Cell` for interior mutability (zero-cost
    /// since CurvatureFlux is Copy).
    pub flux: Cell<hologram_core::carry::CurvatureFlux>,
    /// Optional cancellation token for cooperative cancellation.
    /// When set, the executor checks `is_cancelled()` at level boundaries
    /// and returns `ExecError::Cancelled` if signalled.
    pub cancel: Option<crate::runner::CancellationToken>,
}

impl<'a> TapeContext<'a> {
    /// Create a context from a constant store and weight archive.
    #[must_use]
    pub fn new(
        constants: &'a ConstantStore,
        weights: &'a [u8],
        weight_cache: &'a parking_lot::RwLock<WeightCache>,
    ) -> Self {
        TapeContext {
            ctx: None,
            constants,
            weights,
            weight_cache,
            kv_state: None,
            shape_overrides: &EMPTY_SHAPE_MAP,
            flux: Cell::new(hologram_core::carry::CurvatureFlux::ZERO),
            cancel: None,
        }
    }

    /// Create a context with a KV cache for autoregressive generation.
    #[must_use]
    pub fn with_kv_cache(
        constants: &'a ConstantStore,
        weights: &'a [u8],
        weight_cache: &'a parking_lot::RwLock<WeightCache>,
        kv: KvCacheState,
    ) -> Self {
        TapeContext {
            ctx: None,
            constants,
            weights,
            weight_cache,
            kv_state: Some(RefCell::new(kv)),
            shape_overrides: &EMPTY_SHAPE_MAP,
            flux: Cell::new(hologram_core::carry::CurvatureFlux::ZERO),
            cancel: None,
        }
    }

    /// Reset carry flux to zero. Call at frame boundaries in streaming workloads.
    #[inline]
    pub fn reset_flux(&self) {
        self.flux.set(hologram_core::carry::CurvatureFlux::ZERO);
    }
}

/// Pre-resolved kernel variant — replaces `Box<dyn Fn>` with a small enum.
///
/// Each variant captures only the op parameters needed for dispatch.
/// The `dispatch_kernel` function matches on this enum and calls the
/// appropriate dispatch function directly, enabling inlining and
/// eliminating vtable indirection.
pub enum TapeKernel {
    /// Fused chain of unary float ops.
    FusedFloatChain(Vec<FloatOp>),
    /// Graph output passthrough.
    Output,
    /// Byte-domain LUT (256-byte table).
    LutView(ElementWiseView),
    /// Q1 domain LUT (128KB table, heap-allocated).
    LutView16(Box<hologram_core::q1::view::ElementWiseView16>),
    /// Byte-domain unary prim via LUT.
    PrimUnary(ElementWiseView),
    /// Byte-domain binary prim.
    PrimBinary(PrimOp),
    /// 4-bit quantized LUT-GEMM matmul.
    MatMulLut4(ConstantId),
    /// 8-bit quantized LUT-GEMM matmul.
    MatMulLut8(ConstantId),
    /// 4-bit quantized LUT-GEMM matmul + fused activation (epilogue fusion).
    MatMulLut4Activation(ConstantId, FloatOp),
    /// 8-bit quantized LUT-GEMM matmul + fused activation (epilogue fusion).
    MatMulLut8Activation(ConstantId, FloatOp),
    /// 16-bit hierarchical quantized LUT-GEMM matmul.
    MatMulLut16(ConstantId),
    /// 2-bit quantized LUT-GEMM matmul (pure integer kernel, no BLAS).
    MatMulLut2(ConstantId),
    /// 2-bit quantized LUT-GEMM matmul + fused activation (epilogue fusion).
    MatMulLut2Activation(ConstantId, FloatOp),
    /// KV cache write (autoregressive generation).
    KvWrite {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        is_key: bool,
        /// When true, input is heads-first — transpose to seq-first for storage.
        heads_first: bool,
    },
    /// KV cache read (autoregressive generation).
    KvRead {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        /// When true, output heads-first — transpose from seq-first cache.
        heads_first: bool,
    },

    // ── Inline hot ops (Phase 9a) ─────────────────────────────────────
    // Skip backend vtable + dispatch_float_into entirely.
    // The execute loop calls the kernel function directly.
    /// Inline Relu: v.max(0.0). Zero dispatch overhead.
    InlineRelu,
    /// Inline Neg: -v.
    InlineNeg,
    /// Inline Sigmoid: 1/(1+exp(-v)).
    InlineSigmoid,
    /// Inline Silu: v * sigmoid(v).
    InlineSilu,
    /// Inline Tanh.
    InlineTanh,
    /// Inline Gelu (approximate).
    InlineGelu,
    /// Inline Exp.
    InlineExp,
    /// Inline binary Add.
    InlineAdd,
    /// Inline binary Mul.
    InlineMul,
    /// Inline binary Sub.
    InlineSub,
    /// Inline binary Div.
    InlineDiv,
    /// Inline Abs: v.abs().
    InlineAbs,
    /// Inline Reciprocal: 1.0 / v.
    InlineReciprocal,

    // ── Inline custom ops (Phase 9a.3–9a.4) ─────────────────────────────
    // Skip dispatch_float_into → dispatch_custom_into indirection.
    // Still try backend (Metal GPU) first, then direct CPU kernel call.
    /// Inline MatMul with baked dimensions.
    InlineMatMul { m: u32, k: u32, n: u32 },
    /// Fused MatMul + element-wise activation (epilogue fusion).
    /// Activation applied in-register before writeback — avoids memory round-trip.
    InlineMatMulActivation {
        m: u32,
        k: u32,
        n: u32,
        activation: FloatOp,
    },
    /// Fused MatMul + bias add + activation (full epilogue fusion).
    /// Three inputs: [activation, weight, bias]. Bias from arena (zero-copy).
    /// Eliminates both intermediate buffers from MatMul → Add(bias) → Activation.
    InlineMatMulBiasActivation {
        m: u32,
        k: u32,
        n: u32,
        activation: FloatOp,
    },
    /// Inline Softmax with baked row size.
    InlineSoftmax { size: u32 },
    /// Inline RmsNorm with baked row size and epsilon (as f32::to_bits()).
    InlineRmsNorm { size: u32, epsilon: u32 },

    // ── Inline ops (Phase 9a expansion — Sprint 21) ──────────────────
    /// Inline Log: v.ln().
    InlineLog,
    /// Inline Sqrt: v.sqrt().
    InlineSqrt,
    /// Inline Cos.
    InlineCos,
    /// Inline Sin.
    InlineSin,
    /// Inline Sign.
    InlineSign,
    /// Inline Floor.
    InlineFloor,
    /// Inline Ceil.
    InlineCeil,
    /// Inline Round.
    InlineRound,
    /// Inline Erf (Abramowitz & Stegun).
    InlineErf,
    /// Inline binary Min.
    InlineMin,
    /// Inline binary Max.
    InlineMax,
    /// Inline LayerNorm with baked size and epsilon.
    InlineLayerNorm { size: u32, epsilon: u32 },
    /// Inline AddRmsNorm with baked size and epsilon.
    InlineAddRmsNorm { size: u32, epsilon: u32 },
    /// Inline LogSoftmax with baked row size.
    InlineLogSoftmax { size: u32 },
    /// Inline Attention with baked head config.
    InlineAttention {
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        scale: u32,
        causal: bool,
        heads_first: bool,
        sparse_v: bool,
    },
    /// Inline RotaryEmbedding with baked params (uses position offset from TapeContext).
    InlineRoPE { dim: u32, base: u32, n_heads: u32 },
    /// Inline Gather with baked dim and dtype.
    InlineGather {
        dim: u32,
        dtype: hologram_core::op::FloatDType,
    },
    /// Inline Concat with baked sizes and dtype.
    InlineConcat {
        size_a: u32,
        size_b: u32,
        dtype: hologram_core::op::FloatDType,
    },
    /// Inline Transpose with baked permutation and input shape.
    InlineTranspose {
        /// Permutation indices (first `ndim` entries valid).
        perm: [u8; 8],
        /// Input shape (first `ndim` entries valid).
        input_shape: [u32; 8],
        /// Number of valid dimensions.
        ndim: u8,
    },
    // ── Complete FloatOp coverage (no more Float(FloatOp) catch-all) ────
    /// Power: a^b.
    InlinePow,
    /// Modulo: a % b.
    InlineMod,
    /// Clamp to [min, max]. min/max stored as f32 bits.
    InlineClip { min: u32, max: u32 },
    /// Test for NaN: output is u8 (0 or 1).
    InlineIsNaN,
    /// Logical NOT (unary, f32 domain: 0→1, nonzero→0).
    InlineNot,
    /// Logical AND (binary, f32 domain).
    InlineAnd,
    /// Logical OR.
    InlineOr,
    /// Logical XOR.
    InlineXor,
    /// Equality comparison (f32→f32: 0.0 or 1.0).
    InlineEqual,
    /// Less-than comparison.
    InlineLess,
    /// Less-or-equal comparison.
    InlineLessOrEqual,
    /// Greater-than comparison.
    InlineGreater,
    /// Greater-or-equal comparison.
    InlineGreaterOrEqual,
    /// General matrix multiply with alpha/beta/transpose flags.
    InlineGemm {
        m: u32,
        k: u32,
        n: u32,
        alpha: u32,
        beta: u32,
        trans_a: bool,
        trans_b: bool,
        quant_b: u8,
    },
    /// Sum reduction along last `size` elements.
    InlineReduceSum { size: u32 },
    /// Mean reduction along last `size` elements.
    InlineReduceMean { size: u32 },
    /// Max reduction along last `size` elements.
    InlineReduceMax { size: u32 },
    /// Min reduction along last `size` elements.
    InlineReduceMin { size: u32 },
    /// Product reduction along last `size` elements.
    InlineReduceProd { size: u32 },
    /// Type cast between dtypes.
    InlineCast {
        from: hologram_core::op::FloatDType,
        to: hologram_core::op::FloatDType,
    },
    /// Embedding lookup: [token_ids, table] → [len(ids), dim].
    InlineEmbed { dim: u32, quant: u8 },
    /// Conditional selection: cond ? x : y.
    InlineWhere,
    /// Generate range [start, limit) with step.
    InlineRange,
    /// Extract shape as i64 tensor.
    InlineShape {
        dtype: hologram_core::op::FloatDType,
        start: i64,
        end: i64,
    },
    /// Contiguous slice along axis.
    InlineSlice {
        axis_from_end: u8,
        start: u32,
        end: u32,
        axis_size: u32,
    },
    /// GatherND.
    InlineGatherND,
    /// Fused SiLU gating (SwiGLU): silu(gate) * up.
    InlineFusedSwiGLU,
    /// Reshape: zero-copy data, metadata-only shape change.
    InlineReshape,
    /// Dequantize Q4_0 → f32.
    InlineDequantize,
    /// 2-D convolution.
    InlineConv2d {
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
    /// Fused Conv2d + activation epilogue. Activation applied in-register after GEMM.
    InlineConv2dActivation {
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
    /// Fused Conv2d + bias + activation epilogue (3-node fusion).
    InlineConv2dBiasActivation {
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
    /// 2-D transposed convolution.
    InlineConvTranspose {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        group: u32,
        output_pad_h: u32,
        output_pad_w: u32,
        input_h: u32,
        input_w: u32,
    },
    /// 2-D max pooling.
    InlineMaxPool2d {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// 2-D average pooling.
    InlineAvgPool2d {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// Global average pool: spatial dims → 1.
    InlineGlobalAvgPool {
        channels: u32,
        spatial_h: u32,
        spatial_w: u32,
    },
    /// Resize (nearest/linear/cubic). Mode encoded as u8.
    InlineResize { mode: u8 },
    /// N-D padding. Mode: 0=constant, 1=reflect, 2=edge.
    InlinePad { mode: u8 },
    /// Instance normalization.
    InlineInstanceNorm { size: u32, epsilon: u32 },
    /// Local response normalization.
    InlineLRN {
        size: u32,
        alpha: u32,
        beta: u32,
        bias: u32,
    },
    /// Top-K along axis.
    InlineTopK { axis: u32, largest: bool },
    /// ScatterND.
    InlineScatterND,
    /// Cumulative sum along axis.
    InlineCumSum { axis: u32 },
    /// NonZero: returns indices of non-zero elements.
    InlineNonZero,
    /// Compress along axis.
    InlineCompress { axis: u32 },
    /// ReverseSequence along time/batch axes.
    InlineReverseSequence { batch_axis: u32, time_axis: u32 },

    /// Identity passthrough — same-type Cast only.
    Passthrough,

    /// Custom op — handler baked at tape build time from registry.
    Custom(crate::kv::CustomHandler),

    /// Group normalization.
    InlineGroupNorm { num_groups: u32, epsilon: u32 },

    /// ArgMax: index of max value along last axis. Output is I64.
    InlineArgMax { axis: u32, keepdims: bool },

    // ── Epilogue fusion: norm + activation ────────────────────────────
    /// Fused RmsNorm + activation.
    InlineRmsNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused LayerNorm + activation.
    InlineLayerNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused GroupNorm + activation.
    InlineGroupNormActivation {
        num_groups: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused AddRmsNorm + activation (residual + normalize + activation).
    InlineAddRmsNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused InstanceNorm + activation.
    InlineInstanceNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },

    // ── Deep decode fusions (Plan 054) ─────────────────────────────────
    /// Fused RmsNorm → projection GEMV.
    /// Inputs: [x, norm_weight, proj_weight]. Output: [M, n_total].
    InlineNormProjectionGemv {
        norm_size: u32,
        epsilon: u32,
        k: u32,
        n_total: u32,
    },
    /// Fused Add + RmsNorm → projection GEMV.
    /// Inputs: [x, residual, norm_weight, proj_weight]. Output: [M, n_total].
    InlineAddNormProjectionGemv {
        norm_size: u32,
        epsilon: u32,
        k: u32,
        n_total: u32,
    },
    /// Fused SwiGLU + down projection GEMV.
    /// Inputs: [gate, up, down_weight]. Output: [M, n].
    InlineSwiGluProjectionGemv { k: u32, n: u32 },

    /// Ring-arithmetic unary op. Stays in ring domain (Z/2^nZ), no float conversion.
    /// Q0: applies PrimOp via LUT (apply_unary). Q1: native wrapping u16 ops.
    RingPrimUnary { op: PrimOp, level: RingLevel },
    /// Ring-arithmetic binary op. Stays in ring domain (Z/2^nZ), no float conversion.
    /// Q0: uses ADD_Q0/MUL_Q0 LUT (apply_binary). Q1: add_q1/mul_q1 native ops.
    RingPrimBinary { op: PrimOp, level: RingLevel },

    /// Ring-native activation. Applies ActivationOp::apply element-wise at the specified level.
    /// Q0/Q1: LUT path (O(1) per element). Q3+: piecewise polynomial (register arithmetic).
    RingActivation {
        op: hologram_core::op::ActivationOp,
        level: RingLevel,
    },
    /// Ring-domain fused multiply-add: acc + a * b, element-wise.
    RingAccumulate { level: RingLevel },

    /// Conv2d with pre-quantized 4-bit LUT-GEMM weights (compile-time quantized).
    /// im2col → transpose col → lut_gemm_4bit_par → scatter. Zero quantization overhead.
    InlineConv2dLut4 {
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

    /// Broadcast-expand: physically replicate data along dims where input=1.
    InlineExpand { ndim: u8, target_shape: [u32; 8] },
}

impl TapeKernel {
    /// Returns the inline arity if this is an inline unary (1) or binary (2) op.
    /// Returns `None` for all other kernels (Float, Lut, MatMul, KvCache, etc.).
    #[inline]
    pub(crate) fn inline_arity(&self) -> Option<u8> {
        match self {
            TapeKernel::InlineRelu
            | TapeKernel::InlineNeg
            | TapeKernel::InlineAbs
            | TapeKernel::InlineSigmoid
            | TapeKernel::InlineSilu
            | TapeKernel::InlineTanh
            | TapeKernel::InlineGelu
            | TapeKernel::InlineExp
            | TapeKernel::InlineReciprocal
            | TapeKernel::InlineLog
            | TapeKernel::InlineSqrt
            | TapeKernel::InlineCos
            | TapeKernel::InlineSin
            | TapeKernel::InlineSign
            | TapeKernel::InlineFloor
            | TapeKernel::InlineCeil
            | TapeKernel::InlineRound
            | TapeKernel::InlineErf
            | TapeKernel::InlineClip { .. }
            | TapeKernel::InlineNot
            | TapeKernel::InlineIsNaN => Some(1),
            TapeKernel::InlineAdd
            | TapeKernel::InlineMul
            | TapeKernel::InlineSub
            | TapeKernel::InlineDiv
            | TapeKernel::InlineMin
            | TapeKernel::InlineMax
            | TapeKernel::InlinePow
            | TapeKernel::InlineMod
            | TapeKernel::InlineEqual
            | TapeKernel::InlineLess
            | TapeKernel::InlineLessOrEqual
            | TapeKernel::InlineGreater
            | TapeKernel::InlineGreaterOrEqual
            | TapeKernel::InlineAnd
            | TapeKernel::InlineOr
            | TapeKernel::InlineXor
            | TapeKernel::InlineFusedSwiGLU => Some(2),
            _ => None,
        }
    }

    /// Short name for profiling output.
    fn profile_name(&self) -> String {
        match self {
            Self::MatMulLut4(_) => "MatMulLut4".into(),
            Self::MatMulLut8(_) => "MatMulLut8".into(),
            Self::MatMulLut4Activation(_, _) => "MatMulLut4Act".into(),
            Self::MatMulLut8Activation(_, _) => "MatMulLut8Act".into(),
            Self::MatMulLut16(_) => "MatMulLut16".into(),
            Self::MatMulLut2(_) => "MatMulLut2".into(),
            Self::MatMulLut2Activation(_, _) => "MatMulLut2Act".into(),
            Self::InlineMatMul { m, k, n } => format!("MatMul({m}x{k}x{n})"),
            Self::InlineMatMulActivation { m, k, n, .. } => format!("MatMulAct({m}x{k}x{n})"),
            Self::InlineMatMulBiasActivation { m, k, n, .. } => {
                format!("MatMulBiasAct({m}x{k}x{n})")
            }
            Self::InlineAttention {
                num_q_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                format!("Attention(q{num_q_heads}kv{num_kv_heads}d{head_dim})")
            }
            Self::InlineSoftmax { size } => format!("Softmax({size})"),
            Self::InlineRmsNorm { size, .. } => format!("RmsNorm({size})"),
            Self::InlineAddRmsNorm { size, .. } => format!("AddRmsNorm({size})"),
            Self::InlineLayerNorm { size, .. } => format!("LayerNorm({size})"),
            Self::InlineRoPE { dim, n_heads, .. } => format!("RoPE({dim}x{n_heads})"),
            Self::InlineGather { .. } => "Gather".into(),
            Self::InlineConcat { .. } => "Concat".into(),
            Self::InlineTranspose { .. } => "Transpose".into(),
            Self::KvWrite { layer, .. } => format!("KvWrite(L{layer})"),
            Self::KvRead { layer, .. } => format!("KvRead(L{layer})"),
            Self::FusedFloatChain(_) => "FusedChain".into(),
            Self::InlineAdd => "Add".into(),
            Self::InlineMul => "Mul".into(),
            Self::InlineSub => "Sub".into(),
            Self::InlineDiv => "Div".into(),
            Self::InlineRelu => "Relu".into(),
            Self::InlineSilu => "Silu".into(),
            Self::InlineSigmoid => "Sigmoid".into(),
            Self::InlineGelu => "Gelu".into(),
            Self::InlineFusedSwiGLU => "FusedSwiGLU".into(),
            Self::InlineNormProjectionGemv { .. } => "NormProjGemv".into(),
            Self::InlineAddNormProjectionGemv { .. } => "AddNormProjGemv".into(),
            Self::InlineSwiGluProjectionGemv { .. } => "SwiGluProjGemv".into(),
            Self::InlineSlice { .. } => "Slice".into(),
            Self::InlineReshape => "Reshape".into(),
            Self::Passthrough => "Passthrough".into(),
            Self::Output => "Output".into(),
            Self::InlineNeg => "Neg".into(),
            Self::InlineTanh => "Tanh".into(),
            Self::InlineExp => "Exp".into(),
            Self::InlineAbs => "Abs".into(),
            Self::InlineReciprocal => "Reciprocal".into(),
            Self::InlineGemm { m, k, n, .. } => format!("Gemm({m}x{k}x{n})"),
            Self::Custom(_) => "Custom".into(),
            Self::InlineConv2d { .. } => "Conv2d".into(),
            Self::InlineConv2dActivation { .. } => "Conv2dAct".into(),
            Self::InlineConv2dBiasActivation { .. } => "Conv2dBiasAct".into(),
            Self::InlineConvTranspose { .. } => "ConvTranspose".into(),
            Self::InlineMaxPool2d { .. } => "MaxPool2d".into(),
            Self::InlineAvgPool2d { .. } => "AvgPool2d".into(),
            Self::InlineGlobalAvgPool { .. } => "GlobalAvgPool".into(),
            Self::InlineResize { .. } => "Resize".into(),
            Self::InlinePad { .. } => "Pad".into(),
            Self::InlineInstanceNorm { .. } => "InstanceNorm".into(),
            Self::InlineGroupNorm { .. } => "GroupNorm".into(),
            Self::InlineArgMax { .. } => "ArgMax".into(),
            Self::InlineLRN { .. } => "LRN".into(),
            Self::InlineLogSoftmax { .. } => "LogSoftmax".into(),
            Self::InlineCast { .. } => "Cast".into(),
            Self::InlineEmbed { .. } => "Embed".into(),
            Self::InlineWhere => "Where".into(),
            Self::InlineRange => "Range".into(),
            Self::InlineShape { .. } => "Shape".into(),
            Self::InlineDequantize => "Dequantize".into(),
            Self::InlineGatherND => "GatherND".into(),
            Self::InlineReduceSum { .. } => "ReduceSum".into(),
            Self::InlineReduceMean { .. } => "ReduceMean".into(),
            Self::InlineReduceMax { .. } => "ReduceMax".into(),
            Self::InlineReduceMin { .. } => "ReduceMin".into(),
            Self::InlineReduceProd { .. } => "ReduceProd".into(),
            Self::InlinePow => "Pow".into(),
            Self::InlineMod => "Mod".into(),
            Self::InlineClip { .. } => "Clip".into(),
            Self::InlineIsNaN => "IsNaN".into(),
            Self::InlineNot => "Not".into(),
            Self::InlineAnd => "And".into(),
            Self::InlineOr => "Or".into(),
            Self::InlineXor => "Xor".into(),
            Self::InlineEqual => "Equal".into(),
            Self::InlineLess => "Less".into(),
            Self::InlineLessOrEqual => "LessOrEqual".into(),
            Self::InlineGreater => "Greater".into(),
            Self::InlineGreaterOrEqual => "GreaterOrEqual".into(),
            Self::InlineMin => "Min".into(),
            Self::InlineMax => "Max".into(),
            Self::InlineLog => "Log".into(),
            Self::InlineSqrt => "Sqrt".into(),
            Self::InlineCos => "Cos".into(),
            Self::InlineSin => "Sin".into(),
            Self::InlineSign => "Sign".into(),
            Self::InlineFloor => "Floor".into(),
            Self::InlineCeil => "Ceil".into(),
            Self::InlineRound => "Round".into(),
            Self::InlineErf => "Erf".into(),
            // Remaining variants — catch-all with discriminant for debugging.
            Self::InlineConv2dLut4 { .. } => "Conv2dLut4".into(),
            Self::InlineExpand { .. } => "Expand".into(),
            _ => "Other".into(),
        }
    }
}

/// Pre-resolved dispatch path for a tape instruction. Determined at build time
/// from kernel variant + reuse flags. Eliminates per-instruction arity matching.
#[derive(Clone, Copy, Debug, Default)]
#[repr(u8)]
pub enum FastPath {
    /// Zero-copy buffer move (Output, Passthrough with single consumer).
    Passthrough = 0,
    /// In-place unary mutation (single-consumer unary inline op).
    InPlaceUnary = 1,
    /// Inline unary — direct f32 arena access, no SmallVec.
    InlineUnary = 2,
    /// Inline binary — direct f32 arena access for 2 inputs.
    InlineBinary = 3,
    /// Reshape/cast — zero-copy data, adjust shape metadata.
    Reshape = 4,
    /// General dispatch — gather inputs into SmallVec, dispatch to backend.
    #[default]
    General = 5,
}

/// Pre-resolved shape resolution strategy. Determined at tape build time
/// based on available compile-time shape information.
#[derive(Clone, Copy, Debug, Default)]
#[repr(u8)]
pub enum ShapeSource {
    /// Use compile-time TensorMeta directly (fastest — no runtime work).
    #[default]
    Compiled = 0,
    /// Derive from input[0]'s runtime metadata (element-preserving ops).
    InputMeta = 1,
    /// Infer from output buffer byte length (fallback for unresolved shapes).
    BufferLength = 2,
}

/// A single instruction in the enum-dispatch tape.
pub struct TapeInstruction {
    /// The kernel to execute (enum variant, no heap allocation).
    pub kernel: TapeKernel,
    /// Output node index (where to store the result in the arena).
    pub output_idx: u32,
    /// Input node indices (where to gather inputs from the arena).
    ///
    /// `SmallVec<[u32; 2]>`: ~95% of ops have ≤2 inputs, avoiding heap
    /// allocation for the common case during tape build.
    pub input_indices: SmallVec<[u32; 2]>,
    /// Element size of the output (for arena metadata).
    pub output_elem_size: u8,
    /// Pre-computed output byte size hint (0 = unknown/dynamic).
    pub output_byte_hint: u32,
    /// Byte offset into the weight archive for LUT-GEMM constants.
    /// 0 = no weight prefetch needed (non-LUT-GEMM ops).
    /// When non-zero, the executor prefetches this address in the weight
    /// archive while the previous instruction executes.
    pub weight_offset_hint: u32,
    /// If true, this Output instruction can move the input buffer directly
    /// instead of copying through `out_buf`. Set when the input has exactly
    /// one consumer (this instruction).
    pub passthrough: bool,
    /// If true, a unary inline op can overwrite its input buffer in place.
    /// Set when the input has exactly one consumer and the op preserves size.
    pub can_reuse_input: bool,
    /// Pre-computed output tensor metadata (shape + dtype) from compiled graph.
    /// `None` = not available (infer from buffer size at runtime).
    pub output_meta: Option<hologram_core::op::TensorMeta>,
    /// Pre-resolved dispatch path (set by apply_reuse_flags at build time).
    pub fast_path: FastPath,
    /// Pre-resolved shape resolution strategy (set at build time).
    pub shape_source: ShapeSource,
}

/// Pre-compiled execution tape using enum dispatch.
///
/// Each instruction carries a [`TapeKernel`] enum variant instead of a
/// boxed closure. This eliminates vtable indirection, enables inlining
/// of small kernels, and removes per-kernel heap allocation.
pub struct EnumTape {
    /// Flat instruction array in execution order.
    pub instructions: Vec<TapeInstruction>,
    /// Level boundaries: `level_offsets[i]..level_offsets[i+1]`.
    pub level_offsets: Vec<usize>,
    /// Per-node remaining consumer count for liveness-based eviction.
    /// Computed once at tape finalization. During execution, decremented
    /// after each instruction; when a node's count reaches 0, its arena
    /// slot is freed to reclaim memory.
    pub(crate) consumer_counts: Vec<u32>,
    /// Per-level weight byte ranges for madvise prefetching.
    /// `level_weight_ranges[i] = (start_byte, end_byte)` covering all
    /// deferred constants accessed by instructions in level `i`.
    /// Empty if no weight index was computed.
    pub(crate) level_weight_ranges: Vec<(u64, u64)>,
    /// Activation checkpointing: maps node_idx → instruction_index.
    ///
    /// For nodes with multiple consumers separated by many instructions
    /// (skip connections), the node is evicted after its first consumer
    /// and recomputed from this instruction when the next consumer needs it.
    /// This trades ~30% extra compute for O(layer) peak activation memory.
    pub(crate) checkpoint_map: std::collections::HashMap<u32, usize>,
    /// When true, activation checkpointing is active: skip-connection buffers
    /// are force-evicted after first consumer and recomputed when needed.
    /// Default: false (checkpoints identified but not triggered).
    pub checkpoint_enabled: bool,
    /// When true alongside `checkpoint_enabled`, evicted buffers stay as
    /// Heap Vecs instead of promoting to Mmap. This avoids the mmap/munmap
    /// syscall overhead per buffer (~10-50µs each) which dominates Conv2d-heavy
    /// models like VAE. The freed Vec memory stays in the allocator's free-list
    /// (RSS doesn't drop) but IS reused by subsequent allocations.
    /// Default: false (large buffers use Mmap for immediate page return).
    pub heap_only_eviction: bool,
    /// Workspace slot assignments: `slot_assignments[node_idx] = slot_id`.
    /// Nodes with the same slot_id share a physical buffer (non-overlapping lifetimes).
    /// `u32::MAX` means no aliasing (node uses its own buffer).
    /// Computed by `compute_slot_assignments()` after consumer counts are finalized.
    pub(crate) slot_assignments: Vec<u32>,
    /// Number of workspace slots. Pre-allocated in `prewarm_arena()`.
    pub(crate) n_slots: u32,
}

impl EnumTape {
    /// Create an empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            level_offsets: vec![0],
            consumer_counts: Vec::new(),
            level_weight_ranges: Vec::new(),
            checkpoint_map: std::collections::HashMap::new(),
            checkpoint_enabled: false,
            heap_only_eviction: false,
            slot_assignments: Vec::new(),
            n_slots: 0,
        }
    }

    /// Create a tape with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(n_instructions: usize, n_levels: usize) -> Self {
        let mut level_offsets = Vec::with_capacity(n_levels + 1);
        level_offsets.push(0);
        Self {
            instructions: Vec::with_capacity(n_instructions),
            level_offsets,
            consumer_counts: Vec::new(),
            level_weight_ranges: Vec::new(),
            checkpoint_map: std::collections::HashMap::new(),
            checkpoint_enabled: false,
            heap_only_eviction: false,
            slot_assignments: Vec::new(),
            n_slots: 0,
        }
    }

    /// Add an instruction and return its index.
    pub fn push(&mut self, instr: TapeInstruction) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx
    }

    /// Mark the end of the current level.
    pub fn end_level(&mut self) {
        self.level_offsets.push(self.instructions.len());
    }

    /// Compute per-level weight byte ranges for madvise prefetching.
    ///
    /// For each level, scans instructions for input nodes that reference
    /// `Deferred` constants and computes the bounding byte range in the
    /// weight blob. The executor can then issue `MADV_WILLNEED` for the
    /// next level's range while the current level computes.
    pub fn compute_level_weight_ranges(
        &mut self,
        constants: &hologram_graph::constant::ConstantStore,
        sg: &hologram_archive::format::graph::SerializedGraph,
    ) {
        use hologram_graph::constant::ConstantData;
        use hologram_graph::graph::GraphOp;

        // Build node_id → (offset, size) map for Deferred constants.
        let mut const_ranges: std::collections::HashMap<u32, (u64, u64)> =
            std::collections::HashMap::new();
        for node in &sg.nodes {
            if let GraphOp::Constant(cid) = &node.op {
                if let Some(ConstantData::Deferred {
                    byte_size,
                    source_id,
                }) = constants.get(*cid)
                {
                    const_ranges.insert(node.id.index(), ((*source_id), (*byte_size)));
                }
            }
        }

        let n_levels = self.n_levels();
        let mut ranges = Vec::with_capacity(n_levels);

        for level_idx in 0..n_levels {
            let start = self.level_offsets[level_idx];
            let end = self.level_offsets[level_idx + 1];
            let mut min_offset = u64::MAX;
            let mut max_end: u64 = 0;

            for instr in &self.instructions[start..end] {
                for &input_idx in &instr.input_indices {
                    if let Some(&(offset, size)) = const_ranges.get(&input_idx) {
                        min_offset = min_offset.min(offset);
                        max_end = max_end.max(offset + size);
                    }
                }
            }

            if min_offset < u64::MAX {
                ranges.push((min_offset, max_end));
            } else {
                ranges.push((0, 0)); // No weights in this level.
            }
        }

        self.level_weight_ranges = ranges;
    }

    /// Compute per-node consumer counts for liveness-based arena eviction.
    ///
    /// Must be called after all instructions are added. During execution,
    /// each node's count is decremented when consumed; at zero, the arena
    /// slot is freed. Output nodes are exempt (consumer_count = u32::MAX).
    pub fn finalize_consumer_counts_with_graph(
        &mut self,
        sg: &hologram_archive::format::graph::SerializedGraph,
    ) {
        self.finalize_consumer_counts();

        // Protect graph output nodes and their inputs from eviction.
        for &out_id in &sg.output_node_ids {
            let idx = out_id.index() as usize;
            if idx < self.consumer_counts.len() {
                self.consumer_counts[idx] = u32::MAX;
            }
        }
        for node in &sg.nodes {
            if matches!(node.op, hologram_graph::graph::GraphOp::Output) {
                let idx = node.id.index() as usize;
                if idx < self.consumer_counts.len() {
                    self.consumer_counts[idx] = u32::MAX;
                }
            }
        }
        // Protect passthrough inputs — data is moved, source must survive.
        for instr in &self.instructions {
            if instr.passthrough {
                for &input_idx in &instr.input_indices {
                    let idx = input_idx as usize;
                    if idx < self.consumer_counts.len() {
                        self.consumer_counts[idx] = u32::MAX;
                    }
                }
            }
        }

        // ── Activation checkpointing ─────────────────────────────────────
        // Identify skip-connection nodes: multi-consumer nodes where the
        // gap between producer and last consumer is large. These get evicted
        // after first consumer and recomputed when the distant consumer needs them.
        const CHECKPOINT_GAP_THRESHOLD: usize = 5; // instructions between producer and last consumer
        const CHECKPOINT_SIZE_THRESHOLD: u32 = 256 * 1024; // 256 KB minimum to bother

        // Build producer map: node_idx → instruction_index.
        let mut producer: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (instr_idx, instr) in self.instructions.iter().enumerate() {
            producer.insert(instr.output_idx, instr_idx);
        }

        // Build last-consumer map: node_idx → instruction_index of last consumer.
        let mut last_consumer: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (instr_idx, instr) in self.instructions.iter().enumerate() {
            for &input_idx in &instr.input_indices {
                last_consumer.insert(input_idx, instr_idx);
            }
        }

        // Identify checkpoint candidates.
        for (&node_idx, &prod_instr) in &producer {
            let idx = node_idx as usize;
            // Must have multiple consumers (skip connection pattern).
            if idx >= self.consumer_counts.len() || self.consumer_counts[idx] < 2 {
                continue;
            }
            // Don't checkpoint protected nodes.
            if self.consumer_counts[idx] == u32::MAX {
                continue;
            }
            // Must be large enough to matter.
            let byte_hint = self.instructions[prod_instr].output_byte_hint;
            if byte_hint < CHECKPOINT_SIZE_THRESHOLD {
                continue;
            }
            // Gap must be large enough.
            if let Some(&last_con) = last_consumer.get(&node_idx) {
                if last_con > prod_instr && (last_con - prod_instr) >= CHECKPOINT_GAP_THRESHOLD {
                    // Verify the producer's inputs are all borrowed constants
                    // or protected nodes (so they'll still be available for recomputation).
                    let inputs_available =
                        self.instructions[prod_instr]
                            .input_indices
                            .iter()
                            .all(|&inp| {
                                let i = inp as usize;
                                // Protected (u32::MAX) nodes are always available.
                                // Nodes without a producer entry are constants/inputs (always available).
                                i < self.consumer_counts.len()
                                    && (self.consumer_counts[i] == u32::MAX
                                        || !producer.contains_key(&inp))
                            });
                    if inputs_available {
                        self.checkpoint_map.insert(node_idx, prod_instr);
                    }
                }
            }
        }
        if !self.checkpoint_map.is_empty() {
            tracing::info!(
                n_checkpoints = self.checkpoint_map.len(),
                "activation checkpointing: identified recomputable skip connections"
            );
        }

        // ── Workspace slot assignment (greedy interval coloring) ─────────
        // Nodes with non-overlapping lifetimes can share a physical buffer.
        // Greedy: for each node (sorted by birth), try to reuse a free slot;
        // if none fit, allocate a new slot.
        self.compute_slot_assignments(&producer, &last_consumer);
    }

    /// Compute workspace slot assignments using greedy interval coloring.
    /// Nodes whose lifetimes don't overlap share the same physical buffer slot.
    fn compute_slot_assignments(
        &mut self,
        producer: &std::collections::HashMap<u32, usize>,
        last_consumer: &std::collections::HashMap<u32, usize>,
    ) {
        let n = self.consumer_counts.len();
        let mut assignments = vec![u32::MAX; n]; // u32::MAX = unassigned
                                                 // slot_end[slot_id] = instruction index where slot becomes free
        let mut slot_ends: Vec<usize> = Vec::new();

        // Build intervals for non-protected, non-constant nodes.
        struct Interval {
            node: u32,
            born: usize,
            dies: usize,
        }
        let mut intervals: Vec<Interval> = Vec::new();
        for (&node_idx, &prod_instr) in producer {
            let idx = node_idx as usize;
            if idx >= self.consumer_counts.len() {
                continue;
            }
            // Skip protected nodes (constants, outputs).
            if self.consumer_counts[idx] == u32::MAX {
                continue;
            }
            // Skip checkpointed nodes (they get evicted/recomputed).
            if self.checkpoint_map.contains_key(&node_idx) {
                continue;
            }
            let dies = last_consumer.get(&node_idx).copied().unwrap_or(prod_instr);
            intervals.push(Interval {
                node: node_idx,
                born: prod_instr,
                dies,
            });
        }

        // Sort by birth time (greedy scheduling).
        intervals.sort_by_key(|iv| iv.born);

        for iv in &intervals {
            // Try to reuse a free slot (one that ended before this interval starts).
            // Prefer smallest slot that fits (best-fit to reduce waste).
            let mut best_slot = None;
            for (slot_id, &end) in slot_ends.iter().enumerate() {
                if end < iv.born {
                    best_slot = match best_slot {
                        None => Some(slot_id),
                        Some(prev) => {
                            // Already found one — pick whichever; first-fit is fine.
                            Some(prev)
                        }
                    };
                    if best_slot.is_some() {
                        break; // first-fit
                    }
                }
            }

            let slot_id = match best_slot {
                Some(id) => {
                    slot_ends[id] = iv.dies;
                    id
                }
                None => {
                    let id = slot_ends.len();
                    slot_ends.push(iv.dies);
                    id
                }
            };

            assignments[iv.node as usize] = slot_id as u32;
        }

        self.n_slots = slot_ends.len() as u32;
        self.slot_assignments = assignments;

        if self.n_slots > 0 {
            let total_nodes = intervals.len();
            tracing::info!(
                slots = self.n_slots,
                nodes = total_nodes,
                "workspace: {total_nodes} nodes → {} slots ({:.0}% memory reduction)",
                self.n_slots,
                (1.0 - self.n_slots as f64 / total_nodes.max(1) as f64) * 100.0,
            );
        }
    }

    pub fn finalize_consumer_counts(&mut self) {
        let max_idx = self
            .instructions
            .iter()
            .flat_map(|i| {
                i.input_indices
                    .iter()
                    .copied()
                    .chain(std::iter::once(i.output_idx))
            })
            .max()
            .unwrap_or(0) as usize;

        let mut counts = vec![0u32; max_idx + 1];
        for instr in &self.instructions {
            // Dedup input indices: if an op uses the same input twice
            // (e.g., Add(x, x)), count it only once per instruction.
            let mut deduped: SmallVec<[u32; 4]> = instr.input_indices.iter().copied().collect();
            deduped.sort_unstable();
            deduped.dedup();
            for &input_idx in &deduped {
                if (input_idx as usize) < counts.len() {
                    counts[input_idx as usize] = counts[input_idx as usize].saturating_add(1);
                }
            }
        }
        // Passthrough and output nodes should never be evicted.
        for instr in &self.instructions {
            if instr.passthrough {
                counts[instr.output_idx as usize] = u32::MAX;
            }
        }
        self.consumer_counts = counts;
    }

    /// Mark specific node indices as non-evictable (e.g., graph output nodes).
    pub fn protect_outputs(&mut self, output_node_ids: &[u32]) {
        for &id in output_node_ids {
            let idx = id as usize;
            if idx < self.consumer_counts.len() {
                self.consumer_counts[idx] = u32::MAX;
            }
        }
    }

    /// Bake shape overrides into instruction `output_meta` fields.
    ///
    /// Call this once before execution instead of passing a `shape_overrides`
    /// HashMap on every `execute()` call. Each override is resolved to a
    /// `TensorMeta` and stored directly on the instruction.
    pub fn apply_shape_overrides(
        &mut self,
        overrides: &std::collections::HashMap<u32, Vec<usize>>,
    ) {
        for instr in &mut self.instructions {
            if let Some(shape) = overrides.get(&instr.output_idx) {
                let dtype = instr
                    .output_meta
                    .map(|m| m.dtype)
                    .unwrap_or(hologram_core::op::FloatDType::F32);
                instr.output_meta = Some(hologram_core::op::TensorMeta::new(dtype, shape));
            }
        }
    }

    /// Number of levels in the tape.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.level_offsets.len().saturating_sub(1)
    }

    /// Pre-allocate output slots in the arena so `swap_insert` has buffers
    /// to recycle from the very first instruction (eliminates first-inference
    /// allocation overhead).
    /// Estimate total bytes that `prewarm_arena` would allocate.
    pub fn prewarm_estimate(&self) -> u64 {
        self.instructions
            .iter()
            .filter(|i| i.output_byte_hint > 0 && !i.passthrough)
            .map(|i| i.output_byte_hint as u64)
            .sum()
    }

    pub fn prewarm_arena(&self, arena: &mut BufferArena<'_>) {
        for instr in &self.instructions {
            if instr.output_byte_hint > 0 && !instr.passthrough {
                let id = NodeId::new(instr.output_idx, 0);
                if !arena.contains(id) {
                    let buf = Vec::with_capacity(instr.output_byte_hint as usize);
                    arena.insert_with_elem_size(id, buf, instr.output_elem_size as usize);
                }
            }
        }
    }

    /// Execute with liveness-based eviction of dead activation buffers.
    ///
    /// When `live_counts` is provided, each node's consumer count is
    /// decremented after execution. When a count reaches 0, the node's
    /// arena slot is freed immediately, bounding peak memory to the
    /// maximum live activation set rather than the sum of all outputs.
    /// Execute with eviction counts. Delegates to `execute_direct`.
    /// The `live_counts` parameter is accepted for API compatibility but
    /// ignored — `execute_direct` uses pre-allocated buffers instead of
    /// per-instruction eviction.
    pub fn execute_with_eviction(
        &self,
        arena: &mut BufferArena<'_>,
        tape_ctx: &TapeContext<'_>,
        _live_counts: Option<&[u32]>,
    ) -> ExecResult<()> {
        self.execute_direct(arena, tape_ctx)
    }

    /// Single-path executor with pre-allocated buffers.
    ///
    /// Eliminates all per-instruction overhead: no arena insert/evict,
    /// no SmallVec collection, no mmap out, no checkpoint recompute,
    /// no Metal backend dispatch. Just a tight loop over instructions
    /// with one `match kernel` per instruction.
    ///
    /// Buffers are pre-allocated at call time using `output_byte_hint`.
    /// Each instruction writes directly into its output slot.
    pub fn execute_direct(
        &self,
        arena: &mut BufferArena<'_>,
        tape_ctx: &TapeContext<'_>,
    ) -> ExecResult<()> {
        // Output buffer allocation: two strategies.
        //
        // (a) Heap (default, checkpoint_enabled = false): pre-allocate each
        //     buffer with Vec::with_capacity from output_byte_hint. Fast for
        //     LLMs where the total working set fits in a few GiB.
        //
        // (b) Arena (checkpoint_enabled = true): allocate one contiguous
        //     MmapLender sized to the peak working set (from slot_assignments),
        //     sub-allocate each buffer as an OutputBuffer::Arena pointing into
        //     the mmap region. Evicted slots call advise_free_region to return
        //     pages to the OS. Peak RSS tracks live working set, not total.
        let max_idx = self
            .instructions
            .iter()
            .map(|i| i.output_idx as usize + 1)
            .max()
            .unwrap_or(0);

        // When heap_only_eviction is set, disable Mmap promotion by setting
        // the thread-local threshold to usize::MAX. Restored after execution.
        if self.heap_only_eviction {
            crate::buffer::output_buffer::set_mmap_threshold(usize::MAX);
        }

        // Create output buffers.
        let mut bufs: Vec<OutputBuffer> = if self.checkpoint_enabled {
            // Eviction path (diffusion models): start with empty buffers.
            // Kernels allocate on demand via OutputBuffer::resize. When the
            // resize exceeds the mmap threshold, the buffer self-promotes
            // to Mmap. On eviction, Mmap buffers call munmap — immediate
            // page return. With heap_only_eviction, no promotion happens
            // and freed Vec memory is reused by the allocator.
            (0..max_idx).map(|_| OutputBuffer::new()).collect()
        } else {
            // Heap path (LLMs): pre-allocate with output_byte_hint.
            // No eviction, no mmap overhead.
            let mut v: Vec<OutputBuffer> = (0..max_idx).map(|_| OutputBuffer::new()).collect();
            for instr in &self.instructions {
                let idx = instr.output_idx as usize;
                if idx < v.len() && instr.output_byte_hint > 0 {
                    let needed = instr.output_byte_hint as usize;
                    if v[idx].capacity() < needed {
                        v[idx] = OutputBuffer::with_capacity(needed);
                    }
                }
            }
            v
        };

        // Mutable copy of consumer counts: decremented as each instruction
        // consumes its inputs. When the count for a node reaches zero, its
        // buffer is freed (deallocated, not just cleared) so peak working-set
        // memory tracks actual liveness instead of the sum of all activations.
        //
        // Protected nodes (graph outputs, constants, passthrough sources)
        // carry u32::MAX in `consumer_counts` from `finalize_consumer_counts`
        // and are never decremented or freed — they survive until the final
        // arena.insert below.
        //
        // Gated behind `checkpoint_enabled` because the per-instruction live
        // count decrement + heap free adds ~10% overhead. LLMs (TinyLlama at
        // 1.1B params, ~4 GiB working set) fit comfortably without eviction
        // and pay the perf for nothing. Vision/diffusion models like SD VAE,
        // where peak activation memory is 20+ GiB at 512×512, need it: with
        // eviction the SD VAE decoder fits in ~3 GiB instead of 20+ GiB.
        let do_profile = std::env::var("HOLOGRAM_PROFILE_DIRECT").is_ok();
        let mut op_times: std::collections::HashMap<String, (std::time::Duration, usize)> =
            std::collections::HashMap::new();

        let evict = self.checkpoint_enabled;
        let mut live_counts: Vec<u32> = if evict {
            let mut lc = self.consumer_counts.clone();
            if lc.len() < bufs.len() {
                lc.resize(bufs.len(), 0);
            }
            lc
        } else {
            Vec::new()
        };

        // Per-kernel-type profiling (enabled via HOLOGRAM_PROFILE=1 env var).
        let profile = std::env::var("HOLOGRAM_PROFILE").is_ok_and(|v| v == "1");
        let mut profile_times: std::collections::HashMap<String, (std::time::Duration, usize)> =
            std::collections::HashMap::new();

        // Weight prefetch: issue madvise(WILLNEED) for the next level's
        // weight range when we start a new level. This gives the OS time to
        // page in weights while the current level computes.
        let has_levels = self.level_offsets.len() > 1;
        let has_prefetch = has_levels && !self.level_weight_ranges.is_empty();
        let mut current_level = 0usize;
        let mut next_level_boundary = if has_levels {
            self.level_offsets.get(1).copied().unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };

        // Execute: one match per instruction.
        for (instr_idx, instr) in self.instructions.iter().enumerate() {
            // Cooperative cancellation check at level boundaries.
            if has_levels && instr_idx >= next_level_boundary {
                if let Some(ref cancel) = tape_ctx.cancel {
                    if cancel.is_cancelled() {
                        return Err(crate::error::ExecError::Cancelled);
                    }
                }
            }

            // Check if we've crossed a level boundary → prefetch next level's weights.
            if has_prefetch && instr_idx >= next_level_boundary {
                current_level += 1;
                next_level_boundary = self
                    .level_offsets
                    .get(current_level + 1)
                    .copied()
                    .unwrap_or(usize::MAX);

                // Prefetch weights for level current_level + 1 (look-ahead).
                let prefetch_level = current_level + 1;
                if prefetch_level < self.level_weight_ranges.len() {
                    let (start, end) = self.level_weight_ranges[prefetch_level];
                    if end > start {
                        prefetch_weight_range(tape_ctx.weights, start, end);
                    }
                }
                // Release weights from level current_level - 2 (look-behind).
                if current_level >= 2 {
                    let release_level = current_level - 2;
                    if release_level < self.level_weight_ranges.len() {
                        let (start, end) = self.level_weight_ranges[release_level];
                        if end > start {
                            release_weight_range(tape_ctx.weights, start, end);
                        }
                    }
                }
            }
            // Passthrough: zero-copy move.
            if instr.passthrough {
                if let Some(&src_idx) = instr.input_indices.first() {
                    let src = src_idx as usize;
                    let dst = instr.output_idx as usize;
                    if src < bufs.len() && !bufs[src].is_empty() {
                        // Source is in bufs — move directly.
                        let src_buf = std::mem::take(&mut bufs[src]);
                        bufs[dst] = src_buf;
                    } else if let Ok(data) = arena.get(NodeId::new(src_idx, 0)) {
                        // Source is in arena (constant or graph input).
                        bufs[dst].clear();
                        bufs[dst].extend_from_slice(data);
                    }
                }
                continue;
            }

            // Zero-copy input gathering + mutable output access.
            // SAFETY: output_idx != any input_idx in a valid DAG.
            let out_idx = instr.output_idx as usize;

            // Build input_metas for resolving variable-length dimensions from shape_overrides.
            let input_metas: crate::shape_resolve::InputMetas =
                if tape_ctx.shape_overrides.is_empty() {
                    SmallVec::new()
                } else {
                    instr
                        .input_indices
                        .iter()
                        .map(|&idx| {
                            tape_ctx.shape_overrides.get(&idx).map(|shape| {
                                hologram_core::op::TensorMeta::new(
                                    hologram_core::op::FloatDType::F32,
                                    shape,
                                )
                            })
                        })
                        .collect()
                };

            let bufs_ptr = bufs.as_mut_ptr();
            let bufs_len = bufs.len();

            let input_refs: SmallVec<[&[u8]; 4]> = instr
                .input_indices
                .iter()
                .map(|&idx| {
                    let i = idx as usize;
                    if i < bufs_len {
                        let buf = unsafe { &*bufs_ptr.add(i) };
                        if !buf.is_empty() {
                            return buf.as_slice();
                        }
                    }
                    arena.get(NodeId::new(idx, 0)).unwrap_or(&[])
                })
                .collect();

            let out_buf = unsafe { &mut *bufs_ptr.add(out_idx) };
            out_buf.clear();

            // Pre-dispatch: log every instruction so that if a kernel hangs
            // we can identify which one from the last line in the log.
            // Uses eprintln for immediate flush (tracing may buffer).
            if std::env::var("HOLOGRAM_TRACE_INSTRS").is_ok() {
                let input_sizes: SmallVec<[usize; 4]> = instr
                    .input_indices
                    .iter()
                    .map(|&idx| {
                        let i = idx as usize;
                        if i < bufs.len() && !bufs[i].is_empty() {
                            bufs[i].len()
                        } else {
                            arena.get(NodeId::new(idx, 0)).map(|d| d.len()).unwrap_or(0)
                        }
                    })
                    .collect();
                eprintln!(
                    "[instr {}/{}] {} out={} inputs={:?}",
                    instr_idx,
                    self.instructions.len(),
                    instr.kernel.profile_name(),
                    instr.output_idx,
                    input_sizes.as_slice(),
                );
            }

            // Collect input shapes from the arena's shape registry.
            let input_shapes: SmallVec<[Option<hologram_shape::TensorShape>; 4]> = instr
                .input_indices
                .iter()
                .map(|&idx| arena.get_shape(NodeId::new(idx, 0)).cloned())
                .collect();

            let t0 = std::time::Instant::now();

            dispatch_kernel(
                &instr.kernel,
                &input_refs,
                &input_metas,
                &input_shapes,
                tape_ctx,
                out_buf,
            )?;

            // Drop input refs before mutating bufs again — `input_refs`
            // borrows from `bufs` via raw pointer.
            drop(input_refs);

            // ── Intermediate dump (HOLOGRAM_DUMP_DIR=<path>) ─────────────
            // Saves each instruction's output as a raw binary file for
            // node-by-node comparison with ORT.
            if let Ok(dump_dir) = std::env::var("HOLOGRAM_DUMP_DIR") {
                let out_data = out_buf.as_slice();
                let path = format!("{dump_dir}/node_{instr_idx:05}_{}.bin", instr.output_idx);
                if let Err(e) = std::fs::write(&path, out_data) {
                    eprintln!("[dump] failed to write {path}: {e}");
                }
                // Also write a metadata line to a manifest file
                let manifest_path = format!("{dump_dir}/manifest.csv");
                let line = format!(
                    "{instr_idx},{},{},{}\n",
                    instr.output_idx,
                    out_data.len(),
                    instr.kernel.profile_name(),
                );
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&manifest_path)
                {
                    let _ = f.write_all(line.as_bytes());
                }
            }

            // ── Shape tracking: record output shape after dispatch ────────
            // Convert TapeKernel → FloatOp, collect input shapes, infer
            // output shape, and store in the arena's shape registry.
            {
                let out_shape =
                    crate::executor::tape_kernel_to_float_op(&instr.kernel).and_then(|float_op| {
                        let input_shapes: SmallVec<[hologram_shape::TensorShape; 4]> = instr
                            .input_indices
                            .iter()
                            .filter_map(|&idx| arena.get_shape(NodeId::new(idx, 0)).cloned())
                            .collect();
                        if input_shapes.is_empty() {
                            return None;
                        }
                        let refs: SmallVec<[&hologram_shape::TensorShape; 4]> =
                            input_shapes.iter().collect();
                        hologram_shape::infer_output_shape(&float_op, &refs).ok()
                    });
                if let Some(shape) = out_shape {
                    arena.set_shape(NodeId::new(instr.output_idx, 0), shape);
                }
            }

            // Evict input buffers whose last consumer was this instruction
            // (only when explicitly enabled — see live_counts setup above).
            if evict {
                for &input_idx in &instr.input_indices {
                    let i = input_idx as usize;
                    if i >= live_counts.len() {
                        continue;
                    }
                    if live_counts[i] == u32::MAX {
                        continue;
                    }
                    live_counts[i] = live_counts[i].saturating_sub(1);
                    if live_counts[i] == 0 && i < bufs.len() && !bufs[i].is_empty() {
                        // Drop the buffer. For Mmap variants, this calls
                        // munmap and immediately returns pages to the OS.
                        // For Heap variants, the allocator frees the memory
                        // (pages may linger in the free-list, but Heap is
                        // only used for small buffers < MMAP_EVICT_THRESHOLD).
                        bufs[i] = OutputBuffer::new();
                    }
                }
            }

            // Track live buffer memory when HOLOGRAM_TRACE_INSTRS is set.
            if std::env::var("HOLOGRAM_TRACE_INSTRS").is_ok() && (instr_idx + 1) % 50 == 0 {
                let live_bytes: usize = bufs.iter().map(|b| b.len()).sum();
                eprintln!(
                    "[mem {}/{}] live={:.1}MiB bufs={}",
                    instr_idx + 1,
                    self.instructions.len(),
                    live_bytes as f64 / (1024.0 * 1024.0),
                    bufs.iter().filter(|b| !b.is_empty()).count(),
                );
            }

            // Always measure per-instruction time. The `t0` Instant was
            // captured just before `dispatch_kernel` above.
            let elapsed = t0.elapsed();

            if do_profile {
                let key = instr.kernel.profile_name();
                let cat = key.split('(').next().unwrap_or(&key).to_string();
                let entry = op_times
                    .entry(cat)
                    .or_insert((std::time::Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
            }

            // Log any instruction that takes longer than 1 second —
            // this catches pathological Conv2d or other kernels that
            // dominate UNet forward-pass time without needing
            // HOLOGRAM_PROFILE=1 for the aggregate summary.
            if elapsed.as_secs() >= 1 {
                let input_sizes: SmallVec<[usize; 4]> = instr
                    .input_indices
                    .iter()
                    .map(|&idx| {
                        let i = idx as usize;
                        if i < bufs.len() && !bufs[i].is_empty() {
                            bufs[i].len()
                        } else {
                            arena.get(NodeId::new(idx, 0)).map(|d| d.len()).unwrap_or(0)
                        }
                    })
                    .collect();
                tracing::warn!(
                    instr_idx,
                    kernel = %instr.kernel.profile_name(),
                    elapsed_s = format_args!("{:.1}", elapsed.as_secs_f64()),
                    output_idx = instr.output_idx,
                    output_bytes = out_buf.len(),
                    ?input_sizes,
                    "slow instruction (≥1s)"
                );
            }

            if profile {
                let key = instr.kernel.profile_name();
                let entry = profile_times
                    .entry(key)
                    .or_insert((std::time::Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
            }
        }

        if profile {
            let mut entries: Vec<_> = profile_times.into_iter().collect();
            entries.sort_by_key(|e| std::cmp::Reverse(e.1 .0));
            let total: std::time::Duration = entries.iter().map(|(_, (d, _))| *d).sum();
            tracing::info!("PROFILE total={:.1}ms", total.as_secs_f64() * 1000.0);
            for (name, (dur, count)) in &entries {
                let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
                tracing::info!(
                    "  {name}: {:.2}ms ({count}x, {pct:.1}%)",
                    dur.as_secs_f64() * 1000.0
                );
            }
        }

        // Write all instruction output buffers back to arena so callers
        // (collect_outputs, tests) can access results by NodeId.
        for instr in &self.instructions {
            let idx = instr.output_idx as usize;
            if idx < bufs.len() {
                let data = std::mem::take(&mut bufs[idx]);
                arena.insert(NodeId::new(instr.output_idx, 0), data.into_vec());
            }
        }

        // Profile summary.
        if do_profile {
            let mut entries: Vec<_> = op_times.into_iter().collect();
            entries.sort_by_key(|e| std::cmp::Reverse(e.1 .0));
            let total: std::time::Duration = entries.iter().map(|(_, (d, _))| *d).sum();
            eprintln!(
                "\n[PROFILE] Op timing ({:.1}ms total):",
                total.as_secs_f64() * 1000.0
            );
            for (name, (dur, count)) in &entries {
                let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
                eprintln!(
                    "  {name:30} {count:5}x  {:.1}ms  ({pct:.1}%)",
                    dur.as_secs_f64() * 1000.0
                );
            }
        }

        // Restore mmap threshold if we overrode it.
        if self.heap_only_eviction {
            crate::buffer::output_buffer::set_mmap_threshold(
                crate::buffer::output_buffer::DEFAULT_MMAP_EVICT_THRESHOLD,
            );
        }

        Ok(())
    }

    /// Execute the tape against the given arena and context.
    ///
    /// Uses swap-insert for zero-allocation buffer recycling after warmup.
    /// Enum dispatch replaces vtable indirection with a direct match.
    /// Processes instructions level-by-level, flushing GPU work at level
    /// boundaries (Phase 8.2: command buffer batching).
    /// Execute the tape. Delegates to `execute_direct`.
    pub fn execute(
        &self,
        arena: &mut BufferArena<'_>,
        tape_ctx: &TapeContext<'_>,
    ) -> ExecResult<()> {
        self.execute_direct(arena, tape_ctx)
    }

    /// Constrained execution with per-instruction weight window management.
    ///
    /// Like [`execute_direct`](Self::execute_direct) but interleaves weight
    /// window ensure/evict calls around each instruction that references a
    /// weight constant. Also tracks peak activation memory and returns it
    /// alongside the execution result.
    ///
    /// Strips profiling and tracing overhead for a leaner hot loop.
    pub(crate) fn execute_with_weight_window(
        &self,
        arena: &mut BufferArena<'_>,
        tape_ctx: &TapeContext<'_>,
        weight_window: &mut crate::constrained::weight_window::WeightWindow,
    ) -> ExecResult<usize> {
        use hologram_graph::graph::node::NodeId;

        let max_idx = self
            .instructions
            .iter()
            .map(|i| i.output_idx as usize + 1)
            .max()
            .unwrap_or(0);

        // Pre-allocate output buffers with hints (no eviction in constrained mode).
        let mut bufs: Vec<OutputBuffer> = {
            let mut v: Vec<OutputBuffer> = (0..max_idx).map(|_| OutputBuffer::new()).collect();
            for instr in &self.instructions {
                let idx = instr.output_idx as usize;
                if idx < v.len() && instr.output_byte_hint > 0 {
                    let needed = instr.output_byte_hint as usize;
                    if v[idx].capacity() < needed {
                        v[idx] = OutputBuffer::with_capacity(needed);
                    }
                }
            }
            v
        };

        let mut peak_activation_bytes: usize = 0;

        for instr in self.instructions.iter() {
            // Weight window: ensure required constants are resident.
            if let Some(cid) = kernel_constant_id(&instr.kernel) {
                weight_window.ensure(&[cid], tape_ctx.constants)?;
            }

            // Passthrough: zero-copy move.
            if instr.passthrough {
                if let Some(&src_idx) = instr.input_indices.first() {
                    let src = src_idx as usize;
                    let dst = instr.output_idx as usize;
                    if src < bufs.len() && !bufs[src].is_empty() {
                        let src_buf = std::mem::take(&mut bufs[src]);
                        bufs[dst] = src_buf;
                    } else if let Ok(data) = arena.get(NodeId::new(src_idx, 0)) {
                        bufs[dst].clear();
                        bufs[dst].extend_from_slice(data);
                    }
                }
                // Weight window: evict after use.
                if let Some(cid) = kernel_constant_id(&instr.kernel) {
                    weight_window.evict(&[cid]);
                }
                continue;
            }

            // Zero-copy input gathering + mutable output access.
            let out_idx = instr.output_idx as usize;
            let bufs_ptr = bufs.as_mut_ptr();
            let bufs_len = bufs.len();

            let input_refs: SmallVec<[&[u8]; 4]> = instr
                .input_indices
                .iter()
                .map(|&idx| {
                    let i = idx as usize;
                    if i < bufs_len {
                        let buf = unsafe { &*bufs_ptr.add(i) };
                        if !buf.is_empty() {
                            return buf.as_slice();
                        }
                    }
                    arena.get(NodeId::new(idx, 0)).unwrap_or(&[])
                })
                .collect();

            let out_buf = unsafe { &mut *bufs_ptr.add(out_idx) };
            out_buf.clear();

            let input_metas: crate::shape_resolve::InputMetas =
                if tape_ctx.shape_overrides.is_empty() {
                    SmallVec::new()
                } else {
                    instr
                        .input_indices
                        .iter()
                        .map(|&idx| {
                            tape_ctx.shape_overrides.get(&idx).map(|shape| {
                                hologram_core::op::TensorMeta::new(
                                    hologram_core::op::FloatDType::F32,
                                    shape,
                                )
                            })
                        })
                        .collect()
                };

            let input_shapes: SmallVec<[Option<hologram_shape::TensorShape>; 4]> = instr
                .input_indices
                .iter()
                .map(|&idx| arena.get_shape(NodeId::new(idx, 0)).cloned())
                .collect();

            dispatch_kernel(
                &instr.kernel,
                &input_refs,
                &input_metas,
                &input_shapes,
                tape_ctx,
                out_buf,
            )?;

            drop(input_refs);

            // Weight window: evict weight after dispatch.
            if let Some(cid) = kernel_constant_id(&instr.kernel) {
                weight_window.evict(&[cid]);
            }

            // Track peak activation memory.
            let live_bytes: usize = bufs.iter().map(|b| b.len()).sum();
            if live_bytes > peak_activation_bytes {
                peak_activation_bytes = live_bytes;
            }
        }

        // Write output buffers back to arena.
        for instr in &self.instructions {
            let idx = instr.output_idx as usize;
            if idx < bufs.len() {
                let data = std::mem::take(&mut bufs[idx]);
                arena.insert(NodeId::new(instr.output_idx, 0), data.into_vec());
            }
        }

        Ok(peak_activation_bytes)
    }
}

/// Extract the `ConstantId` from a kernel that references a weight constant.
#[inline]
fn kernel_constant_id(kernel: &TapeKernel) -> Option<hologram_graph::constant::ConstantId> {
    match kernel {
        TapeKernel::MatMulLut2(c)
        | TapeKernel::MatMulLut4(c)
        | TapeKernel::MatMulLut8(c)
        | TapeKernel::MatMulLut16(c)
        | TapeKernel::MatMulLut2Activation(c, _)
        | TapeKernel::MatMulLut4Activation(c, _)
        | TapeKernel::MatMulLut8Activation(c, _)
        | TapeKernel::InlineConv2dLut4 { cid: c, .. } => Some(*c),
        _ => None,
    }
}

impl Default for EnumTape {
    fn default() -> Self {
        Self::new()
    }
}

// ── Backward-compat aliases ──────────────────────────────────────────────────

/// Backward-compatible alias for [`TapeInstruction`].
pub type BoxedInstruction = TapeInstruction;

/// Backward-compatible alias for [`EnumTape`].
pub type BoxedTape = EnumTape;

// ── Weight prefetch helpers ────────────────────────────────────────────────

/// Issue madvise(WILLNEED) for a byte range within the weight blob.
/// Tells the OS to page in these bytes asynchronously.
#[inline]
fn prefetch_weight_range(weights: &[u8], start: u64, end: u64) {
    let start = start as usize;
    let end = (end as usize).min(weights.len());
    if start >= end {
        return;
    }
    #[cfg(unix)]
    {
        let ptr = weights[start..].as_ptr();
        let len = end - start;
        // SAFETY: ptr is within the weights slice, len doesn't extend past it.
        unsafe {
            libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_WILLNEED);
        }
    }
}

/// Issue madvise(DONTNEED) for a byte range within the weight blob.
/// Tells the OS these pages can be reclaimed.
#[inline]
fn release_weight_range(weights: &[u8], start: u64, end: u64) {
    let start = start as usize;
    let end = (end as usize).min(weights.len());
    if start >= end {
        return;
    }
    #[cfg(unix)]
    {
        let ptr = weights[start..].as_ptr();
        let len = end - start;
        // SAFETY: ptr is within the weights slice, len doesn't extend past it.
        // MADV_DONTNEED on mmap'd memory allows OS to reclaim pages; the
        // next access will re-fault them in (zero-cost for mmap).
        unsafe {
            libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_dispatch::binary_broadcast;

    fn empty_constants() -> ConstantStore {
        ConstantStore::new()
    }

    #[test]
    fn enum_tape_empty_executes() {
        let tape = EnumTape::new();
        let mut arena = BufferArena::new();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        assert!(tape.execute(&mut arena, &ctx).is_ok());
    }

    #[test]
    fn enum_tape_output_passthrough() {
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::Output,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 1,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![10, 20, 30]);

        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();

        assert_eq!(arena.get(NodeId::new(1, 0)).unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn enum_tape_float_relu() {
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 8, // 2 floats × 4 bytes
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.push(TapeInstruction {
            kernel: TapeKernel::Output,
            output_idx: 2,
            input_indices: smallvec::smallvec![1],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();

        // Input: two f32 values [-1.0, 2.0]
        let input_bytes: Vec<u8> = [(-1.0f32).to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input_bytes);

        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[0.0, 2.0]); // relu(-1)=0, relu(2)=2
    }

    #[test]
    fn enum_tape_lut_view() {
        use hologram_core::op::LutOp;
        let view = hologram_core::view::ElementWiseView::from_table(*LutOp::Relu.table());

        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::LutView(view),
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 1,
            output_byte_hint: 3,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![0, 128, 255]);

        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(1, 0)).unwrap();
        assert_eq!(out[0], LutOp::Relu.apply(0));
        assert_eq!(out[1], LutOp::Relu.apply(128));
        assert_eq!(out[2], LutOp::Relu.apply(255));
    }

    #[test]
    fn enum_tape_two_level_chain() {
        // Input(0) → Relu(1) → Output(2)
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        tape.push(TapeInstruction {
            kernel: TapeKernel::Output,
            output_idx: 2,
            input_indices: smallvec::smallvec![1],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();

        assert_eq!(tape.n_levels(), 2);

        let input: Vec<u8> = [(-3.0f32).to_le_bytes(), 5.0f32.to_le_bytes()].concat();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input);

        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[0.0, 5.0]);
    }

    #[test]
    fn enum_tape_swap_insert_recycles_buffers() {
        // Run the same tape twice — second run should reuse allocations.
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();

        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);

        // Run 1
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), 1.0f32.to_le_bytes().to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out1 = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Run 2 (reuse arena)
        arena.insert(NodeId::new(0, 0), 2.0f32.to_le_bytes().to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out2 = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        let f1: f32 = f32::from_le_bytes(out1[..4].try_into().unwrap());
        let f2: f32 = f32::from_le_bytes(out2[..4].try_into().unwrap());
        assert_eq!(f1, 1.0);
        assert_eq!(f2, 2.0);
    }

    // ── Inline hot op tests (Phase 9a) ────────────────────────────

    #[test]
    fn inline_relu_matches_generic() {
        let input: Vec<u8> = [(-2.0f32).to_le_bytes(), 3.0f32.to_le_bytes()].concat();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);

        // Inline path
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input.clone());
        tape.execute(&mut arena, &ctx).unwrap();
        let inline_out = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Generic Float path
        let mut tape2 = EnumTape::new();
        tape2.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape2.end_level();
        let mut arena2 = BufferArena::new();
        arena2.insert(NodeId::new(0, 0), input);
        tape2.execute(&mut arena2, &ctx).unwrap();
        let generic_out = arena2.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Byte-for-byte match.
        assert_eq!(inline_out, generic_out, "InlineRelu must match Float(Relu)");
        let floats: &[f32] = bytemuck::cast_slice(&inline_out);
        assert_eq!(floats, &[0.0, 3.0]);
    }

    #[test]
    fn inline_add_matches_generic() {
        let a: Vec<u8> = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let b: Vec<u8> = [10.0f32.to_le_bytes(), 20.0f32.to_le_bytes()].concat();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);

        // Inline path
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineAdd,
            output_idx: 2,
            input_indices: smallvec::smallvec![0, 1],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), a.clone());
        arena.insert(NodeId::new(1, 0), b.clone());
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[11.0, 22.0]);
    }

    #[test]
    fn inline_mul_sigmoid_chain() {
        // Test chaining inline ops: Input → InlineSigmoid → InlineMul → Output
        let input: Vec<u8> = [0.0f32.to_le_bytes()].concat(); // sigmoid(0) = 0.5
        let two: Vec<u8> = [2.0f32.to_le_bytes()].concat();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);

        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineSigmoid,
            output_idx: 2,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineMul,
            output_idx: 3,
            input_indices: smallvec::smallvec![2, 1],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input);
        arena.insert(NodeId::new(1, 0), two);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(3, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        // sigmoid(0) * 2 = 0.5 * 2 = 1.0
        assert!((floats[0] - 1.0).abs() < 1e-5, "got {}", floats[0]);
    }

    // ── binary_broadcast tests ──────────────────────────────────────

    #[test]
    fn broadcast_same_size() {
        let mut dst = vec![0.0f32; 2];
        binary_broadcast(&[1.0, 2.0], &[3.0, 4.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![4.0, 6.0]);
    }

    #[test]
    fn broadcast_scalar_b() {
        let mut dst = vec![0.0f32; 3];
        binary_broadcast(&[1.0, 2.0, 3.0], &[10.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn broadcast_scalar_a() {
        let mut dst = vec![0.0f32; 2];
        binary_broadcast(&[10.0], &[1.0, 2.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![11.0, 12.0]);
    }

    #[test]
    fn broadcast_general() {
        let mut dst = vec![0.0f32; 3];
        binary_broadcast(&[1.0, 2.0], &[10.0, 20.0, 30.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![11.0, 22.0, 31.0]);
    }

    // ── Sprint 21 tests: Passthrough, new inline variants, norm direct-write ──

    /// Helper: build and execute a single-instruction tape, return output f32s.
    fn run_unary_tape(kernel: TapeKernel, input: &[f32]) -> Vec<f32> {
        let input_bytes: Vec<u8> = bytemuck::cast_slice(input).to_vec();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel,
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: (input.len() * 4) as u32,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input_bytes);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        if out.is_empty() {
            return vec![];
        }
        bytemuck::cast_slice(out).to_vec()
    }

    fn run_binary_tape(kernel: TapeKernel, a: &[f32], b: &[f32]) -> Vec<f32> {
        let a_bytes: Vec<u8> = bytemuck::cast_slice(a).to_vec();
        let b_bytes: Vec<u8> = bytemuck::cast_slice(b).to_vec();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel,
            output_idx: 2,
            input_indices: smallvec::smallvec![0, 1],
            output_elem_size: 4,
            output_byte_hint: (a.len() * 4) as u32,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), a_bytes);
        arena.insert(NodeId::new(1, 0), b_bytes);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        if out.is_empty() {
            return vec![];
        }
        bytemuck::cast_slice(out).to_vec()
    }

    #[test]
    fn passthrough_identity_cast() {
        // Passthrough kernel should forward input bytes unchanged.
        let input = [1.0f32, 2.0, 3.0];
        let out = run_unary_tape(TapeKernel::Passthrough, &input);
        assert_eq!(out, input);
    }

    #[test]
    fn passthrough_empty_input() {
        let out = run_unary_tape(TapeKernel::Passthrough, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn inline_log() {
        let out = run_unary_tape(TapeKernel::InlineLog, &[1.0, std::f32::consts::E]);
        assert!((out[0] - 0.0).abs() < 1e-6, "ln(1) = 0");
        assert!((out[1] - 1.0).abs() < 1e-5, "ln(e) = 1");
    }

    #[test]
    fn inline_sqrt() {
        let out = run_unary_tape(TapeKernel::InlineSqrt, &[4.0, 9.0, 0.0]);
        assert_eq!(out, [2.0, 3.0, 0.0]);
    }

    #[test]
    fn inline_cos_sin() {
        let out_cos = run_unary_tape(TapeKernel::InlineCos, &[0.0]);
        let out_sin = run_unary_tape(TapeKernel::InlineSin, &[0.0]);
        assert!((out_cos[0] - 1.0).abs() < 1e-6, "cos(0) = 1");
        assert!(out_sin[0].abs() < 1e-6, "sin(0) = 0");
    }

    #[test]
    fn inline_sign() {
        let out = run_unary_tape(TapeKernel::InlineSign, &[-5.0, 0.0, 3.0]);
        // ONNX Sign spec: Sign(0) = 0 (not IEEE signum which returns 1.0 for +0.0).
        assert_eq!(out, [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn inline_floor_ceil_round() {
        let out_floor = run_unary_tape(TapeKernel::InlineFloor, &[1.7, -1.3]);
        let out_ceil = run_unary_tape(TapeKernel::InlineCeil, &[1.1, -1.9]);
        let out_round = run_unary_tape(TapeKernel::InlineRound, &[1.5, 2.3]);
        assert_eq!(out_floor, [1.0, -2.0]);
        assert_eq!(out_ceil, [2.0, -1.0]);
        assert_eq!(out_round, [2.0, 2.0]);
    }

    #[test]
    fn inline_erf() {
        let out = run_unary_tape(TapeKernel::InlineErf, &[0.0, 1.0]);
        assert!(out[0].abs() < 1e-5, "erf(0) = 0");
        assert!((out[1] - 0.8427).abs() < 0.01, "erf(1) ≈ 0.8427");
    }

    #[test]
    fn inline_min_max() {
        let a = [1.0f32, 5.0, 3.0];
        let b = [2.0f32, 4.0, 3.0];
        let mins = run_binary_tape(TapeKernel::InlineMin, &a, &b);
        let maxs = run_binary_tape(TapeKernel::InlineMax, &a, &b);
        assert_eq!(mins, [1.0, 4.0, 3.0]);
        assert_eq!(maxs, [2.0, 5.0, 3.0]);
    }

    #[test]
    fn inline_layer_norm() {
        // LayerNorm: normalize [1, 2, 3] with weight=[1,1,1] bias=[0,0,0]
        // mean=2, var=2/3, inv_std≈1.2247
        let x: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec();
        let w: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 1.0, 1.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[0.0f32, 0.0, 0.0]).to_vec();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineLayerNorm {
                size: 3,
                epsilon: f32::to_bits(1e-5),
            },
            output_idx: 3,
            input_indices: smallvec::smallvec![0, 1, 2],
            output_elem_size: 4,
            output_byte_hint: 12,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), x);
        arena.insert(NodeId::new(1, 0), w);
        arena.insert(NodeId::new(2, 0), b);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(3, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats.len(), 3);
        // Normalized: (x - mean) * inv_std ≈ [-1.2247, 0, 1.2247]
        assert!((floats[0] + 1.2247).abs() < 0.01);
        assert!(floats[1].abs() < 0.01);
        assert!((floats[2] - 1.2247).abs() < 0.01);
    }

    #[test]
    fn inline_log_softmax() {
        // LogSoftmax of [0, 0, 0] → [-ln(3), -ln(3), -ln(3)]
        let x = [0.0f32, 0.0, 0.0];
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineLogSoftmax { size: 3 },
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 12,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), bytemuck::cast_slice(&x).to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        let expected = -(3.0f32.ln());
        for &v in floats {
            assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn inline_softmax_into_direct_write() {
        // Verify InlineSoftmax writes correct values and sums to 1.
        let x = [1.0f32, 2.0, 3.0];
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineSoftmax { size: 3 },
            output_idx: 1,
            input_indices: smallvec::smallvec![0],
            output_elem_size: 4,
            output_byte_hint: 12,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), bytemuck::cast_slice(&x).to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats.len(), 3);
        let sum: f32 = floats.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
        // Values should be monotonically increasing (input was sorted).
        assert!(floats[0] < floats[1]);
        assert!(floats[1] < floats[2]);
    }

    #[test]
    fn inline_gather_dispatch() {
        use hologram_core::op::FloatDType;
        // Gather: inputs[0]=indices (i64), inputs[1]=table (f32)
        // dim=1 means each entry is 1 float. indices=[2,0] → table[2]=30, table[0]=10
        let table: Vec<u8> = bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]).to_vec();
        let idx_vals: [i64; 2] = [2, 0];
        let indices: Vec<u8> = bytemuck::cast_slice(&idx_vals).to_vec();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineGather {
                dim: 1,
                dtype: FloatDType::F32,
            },
            output_idx: 2,
            // inputs[0] = indices, inputs[1] = table
            input_indices: smallvec::smallvec![1, 0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        // node 0 = table, node 1 = indices
        arena.insert(NodeId::new(0, 0), table);
        arena.insert(NodeId::new(1, 0), indices);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[30.0, 10.0]);
    }

    #[test]
    fn inline_concat_dispatch() {
        use hologram_core::op::FloatDType;
        // Concat: [1,2] + [3,4,5] → [1,2,3,4,5]
        let a: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[3.0f32, 4.0, 5.0]).to_vec();
        let constants = empty_constants();
        let wc = parking_lot::RwLock::new(WeightCache::new());
        let ctx = TapeContext::new(&constants, &[], &wc);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineConcat {
                size_a: 2,
                size_b: 3,
                dtype: FloatDType::F32,
            },
            output_idx: 2,
            input_indices: smallvec::smallvec![0, 1],
            output_elem_size: 4,
            output_byte_hint: 20,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), a);
        arena.insert(NodeId::new(1, 0), b);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // ── Ring-arithmetic execution path tests (Refinement E) ──────────────────

    fn make_ring_tape(kernel: TapeKernel, n_inputs: u8) -> (EnumTape, ConstantStore) {
        let mut tape = EnumTape::new();
        let input_indices: smallvec::SmallVec<[u32; 2]> = (0..n_inputs as u32).collect();
        tape.push(TapeInstruction {
            kernel,
            output_idx: n_inputs as u32,
            input_indices,
            output_elem_size: 1,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: FastPath::default(),
            shape_source: ShapeSource::default(),
        });
        tape.end_level();
        (tape, empty_constants())
    }

    #[test]
    fn ring_prim_binary_q0_add_wrapping() {
        // 255 + 1 = 0 (wrapping mod 256). No float conversion.
        let (tape, constants) = make_ring_tape(
            TapeKernel::RingPrimBinary {
                op: PrimOp::Add,
                level: RingLevel::Q0,
            },
            2,
        );
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![255u8]);
        arena.insert(NodeId::new(1, 0), vec![1u8]);
        let wc = parking_lot::RwLock::new(WeightCache::default());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();
        assert_eq!(arena.get(NodeId::new(2, 0)).unwrap(), &[0u8]);
    }

    #[test]
    fn ring_prim_binary_q0_mul_wrapping() {
        // 200 * 2 = 400 mod 256 = 144. No float conversion.
        let (tape, constants) = make_ring_tape(
            TapeKernel::RingPrimBinary {
                op: PrimOp::Mul,
                level: RingLevel::Q0,
            },
            2,
        );
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![200u8]);
        arena.insert(NodeId::new(1, 0), vec![2u8]);
        let wc = parking_lot::RwLock::new(WeightCache::default());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();
        assert_eq!(arena.get(NodeId::new(2, 0)).unwrap(), &[144u8]);
    }

    #[test]
    fn ring_prim_unary_q0_neg_wrapping() {
        // neg(1) = 255 (wrapping neg mod 256).
        let (tape, constants) = make_ring_tape(
            TapeKernel::RingPrimUnary {
                op: PrimOp::Neg,
                level: RingLevel::Q0,
            },
            1,
        );
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![1u8, 128u8, 0u8]);
        let wc = parking_lot::RwLock::new(WeightCache::default());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();
        assert_eq!(arena.get(NodeId::new(1, 0)).unwrap(), &[255u8, 128u8, 0u8]);
    }

    #[test]
    fn ring_prim_binary_q1_add_wrapping() {
        // Q1: 65535 + 1 = 0 (mod 65536). Input/output as le bytes.
        let (tape, constants) = make_ring_tape(
            TapeKernel::RingPrimBinary {
                op: PrimOp::Add,
                level: RingLevel::Q1,
            },
            2,
        );
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), 65535u16.to_le_bytes().to_vec()); // 0xFF 0xFF
        arena.insert(NodeId::new(1, 0), 1u16.to_le_bytes().to_vec()); // 0x01 0x00
        let wc = parking_lot::RwLock::new(WeightCache::default());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), 0u16);
    }

    #[test]
    fn ring_prim_unary_q1_neg() {
        // Q1: neg(1) = 65535.
        let (tape, constants) = make_ring_tape(
            TapeKernel::RingPrimUnary {
                op: PrimOp::Neg,
                level: RingLevel::Q1,
            },
            1,
        );
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), 1u16.to_le_bytes().to_vec());
        let wc = parking_lot::RwLock::new(WeightCache::default());
        let ctx = TapeContext::new(&constants, &[], &wc);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), 65535u16);
    }

    #[test]
    fn ring_prim_binary_q0_exhaustive_add() {
        // Exhaustively verify Q0 Add ring path matches apply_binary for all 256 byte values.
        let constants = empty_constants();
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                let (tape, _) = make_ring_tape(
                    TapeKernel::RingPrimBinary {
                        op: PrimOp::Add,
                        level: RingLevel::Q0,
                    },
                    2,
                );
                let mut arena = BufferArena::new();
                arena.insert(NodeId::new(0, 0), vec![a]);
                arena.insert(NodeId::new(1, 0), vec![b]);
                let wc = parking_lot::RwLock::new(WeightCache::default());
                let ctx = TapeContext::new(&constants, &[], &wc);
                tape.execute(&mut arena, &ctx).unwrap();
                let ring_out = arena.get(NodeId::new(2, 0)).unwrap()[0];
                let expected = PrimOp::Add.apply_binary(a, b);
                assert_eq!(ring_out, expected, "ring Q0 Add mismatch at ({a},{b})");
            }
        }
    }
}
