//! Constrained execution profile configuration.
//!
//! Defines memory limits, weight residency policies, and kernel allowlists
//! for deterministic, bounded-memory tape execution. Workload-agnostic:
//! applies to AI inference, numeric pipelines, rendering, and signal processing.

use std::collections::HashSet;

use crate::tape::TapeKernel;

/// Weight residency policy for constrained execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPolicy {
    /// Load all weights into memory once at startup.
    FullResident,
    /// Existing lazy-cache behavior (load on first access, keep forever).
    LazyCache,
    /// Bounded sliding window — evict LRU weights when cap is exceeded.
    BoundedWindow,
    /// Stream weights per-op with no caching.
    NoCacheStream,
}

/// Fieldless discriminant mirroring [`TapeKernel`] variants.
///
/// Used by [`KernelAllowlist`] to specify which kernel types are permitted
/// without carrying kernel-specific data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelDiscriminant {
    FusedFloatChain,
    Output,
    LutView,
    LutView16,
    PrimUnary,
    PrimBinary,
    MatMulLut4,
    MatMulLut8,
    MatMulLut4Activation,
    MatMulLut8Activation,
    MatMulLut16,
    MatMulLut2,
    MatMulLut2Activation,
    KvWrite,
    KvRead,
    InlineRelu,
    InlineNeg,
    InlineSigmoid,
    InlineSilu,
    InlineTanh,
    InlineGelu,
    InlineExp,
    InlineAdd,
    InlineMul,
    InlineSub,
    InlineDiv,
    InlineAbs,
    InlineReciprocal,
    InlineMatMul,
    InlineMatMulActivation,
    InlineMatMulBiasActivation,
    InlineSoftmax,
    InlineRmsNorm,
    InlineLog,
    InlineSqrt,
    InlineCos,
    InlineSin,
    InlineSign,
    InlineFloor,
    InlineCeil,
    InlineRound,
    InlineErf,
    InlineMin,
    InlineMax,
    InlineLayerNorm,
    InlineAddRmsNorm,
    InlineLogSoftmax,
    InlineAttention,
    InlineRoPE,
    InlineGather,
    InlineConcat,
    InlineTranspose,
    InlinePow,
    InlineMod,
    InlineClip,
    InlineIsNaN,
    InlineNot,
    InlineAnd,
    InlineOr,
    InlineXor,
    InlineEqual,
    InlineLess,
    InlineLessOrEqual,
    InlineGreater,
    InlineGreaterOrEqual,
    InlineGemm,
    InlineReduceSum,
    InlineReduceMean,
    InlineReduceMax,
    InlineReduceMin,
    InlineReduceProd,
    InlineCast,
    InlineEmbed,
    InlineWhere,
    InlineRange,
    InlineShape,
    InlineSlice,
    InlineGatherND,
    InlineFusedSwiGLU,
    InlineReshape,
    InlineDequantize,
    InlineConv2d,
    InlineConv2dActivation,
    InlineConv2dBiasActivation,
    InlineConvTranspose,
    InlineMaxPool2d,
    InlineAvgPool2d,
    InlineGlobalAvgPool,
    InlineResize,
    InlinePad,
    InlineInstanceNorm,
    InlineLRN,
    InlineTopK,
    InlineScatterND,
    InlineCumSum,
    InlineNonZero,
    InlineCompress,
    InlineReverseSequence,
    Passthrough,
    Custom,
    InlineGroupNorm,
    InlineArgMax,
    InlineRmsNormActivation,
    InlineLayerNormActivation,
    InlineGroupNormActivation,
    InlineAddRmsNormActivation,
    InlineInstanceNormActivation,
    InlineNormProjectionGemv,
    InlineAddNormProjectionGemv,
    InlineSwiGluProjectionGemv,
    RingPrimUnary,
    RingPrimBinary,
    RingActivation,
    RingAccumulate,
    InlineConv2dLut4,
    InlineExpand,
}

impl KernelDiscriminant {
    /// Extract the discriminant from a concrete [`TapeKernel`].
    #[must_use]
    pub fn from_kernel(kernel: &TapeKernel) -> Self {
        match kernel {
            TapeKernel::FusedFloatChain(_) => Self::FusedFloatChain,
            TapeKernel::Output => Self::Output,
            TapeKernel::LutView(_) => Self::LutView,
            TapeKernel::LutView16(_) => Self::LutView16,
            TapeKernel::PrimUnary(_) => Self::PrimUnary,
            TapeKernel::PrimBinary(_) => Self::PrimBinary,
            TapeKernel::MatMulLut4(_) => Self::MatMulLut4,
            TapeKernel::MatMulLut8(_) => Self::MatMulLut8,
            TapeKernel::MatMulLut4Activation(_, _) => Self::MatMulLut4Activation,
            TapeKernel::MatMulLut8Activation(_, _) => Self::MatMulLut8Activation,
            TapeKernel::MatMulLut16(_) => Self::MatMulLut16,
            TapeKernel::MatMulLut2(_) => Self::MatMulLut2,
            TapeKernel::MatMulLut2Activation(_, _) => Self::MatMulLut2Activation,
            TapeKernel::KvWrite { .. } => Self::KvWrite,
            TapeKernel::KvRead { .. } => Self::KvRead,
            TapeKernel::InlineRelu => Self::InlineRelu,
            TapeKernel::InlineNeg => Self::InlineNeg,
            TapeKernel::InlineSigmoid => Self::InlineSigmoid,
            TapeKernel::InlineSilu => Self::InlineSilu,
            TapeKernel::InlineTanh => Self::InlineTanh,
            TapeKernel::InlineGelu => Self::InlineGelu,
            TapeKernel::InlineExp => Self::InlineExp,
            TapeKernel::InlineAdd => Self::InlineAdd,
            TapeKernel::InlineMul => Self::InlineMul,
            TapeKernel::InlineSub => Self::InlineSub,
            TapeKernel::InlineDiv => Self::InlineDiv,
            TapeKernel::InlineAbs => Self::InlineAbs,
            TapeKernel::InlineReciprocal => Self::InlineReciprocal,
            TapeKernel::InlineMatMul { .. } => Self::InlineMatMul,
            TapeKernel::InlineMatMulActivation { .. } => Self::InlineMatMulActivation,
            TapeKernel::InlineMatMulBiasActivation { .. } => Self::InlineMatMulBiasActivation,
            TapeKernel::InlineSoftmax { .. } => Self::InlineSoftmax,
            TapeKernel::InlineRmsNorm { .. } => Self::InlineRmsNorm,
            TapeKernel::InlineLog => Self::InlineLog,
            TapeKernel::InlineSqrt => Self::InlineSqrt,
            TapeKernel::InlineCos => Self::InlineCos,
            TapeKernel::InlineSin => Self::InlineSin,
            TapeKernel::InlineSign => Self::InlineSign,
            TapeKernel::InlineFloor => Self::InlineFloor,
            TapeKernel::InlineCeil => Self::InlineCeil,
            TapeKernel::InlineRound => Self::InlineRound,
            TapeKernel::InlineErf => Self::InlineErf,
            TapeKernel::InlineMin => Self::InlineMin,
            TapeKernel::InlineMax => Self::InlineMax,
            TapeKernel::InlineLayerNorm { .. } => Self::InlineLayerNorm,
            TapeKernel::InlineAddRmsNorm { .. } => Self::InlineAddRmsNorm,
            TapeKernel::InlineLogSoftmax { .. } => Self::InlineLogSoftmax,
            TapeKernel::InlineAttention { .. } => Self::InlineAttention,
            TapeKernel::InlineRoPE { .. } => Self::InlineRoPE,
            TapeKernel::InlineGather { .. } => Self::InlineGather,
            TapeKernel::InlineConcat { .. } => Self::InlineConcat,
            TapeKernel::InlineTranspose { .. } => Self::InlineTranspose,
            TapeKernel::InlinePow => Self::InlinePow,
            TapeKernel::InlineMod => Self::InlineMod,
            TapeKernel::InlineClip { .. } => Self::InlineClip,
            TapeKernel::InlineIsNaN => Self::InlineIsNaN,
            TapeKernel::InlineNot => Self::InlineNot,
            TapeKernel::InlineAnd => Self::InlineAnd,
            TapeKernel::InlineOr => Self::InlineOr,
            TapeKernel::InlineXor => Self::InlineXor,
            TapeKernel::InlineEqual => Self::InlineEqual,
            TapeKernel::InlineLess => Self::InlineLess,
            TapeKernel::InlineLessOrEqual => Self::InlineLessOrEqual,
            TapeKernel::InlineGreater => Self::InlineGreater,
            TapeKernel::InlineGreaterOrEqual => Self::InlineGreaterOrEqual,
            TapeKernel::InlineGemm { .. } => Self::InlineGemm,
            TapeKernel::InlineReduceSum { .. } => Self::InlineReduceSum,
            TapeKernel::InlineReduceMean { .. } => Self::InlineReduceMean,
            TapeKernel::InlineReduceMax { .. } => Self::InlineReduceMax,
            TapeKernel::InlineReduceMin { .. } => Self::InlineReduceMin,
            TapeKernel::InlineReduceProd { .. } => Self::InlineReduceProd,
            TapeKernel::InlineCast { .. } => Self::InlineCast,
            TapeKernel::InlineEmbed { .. } => Self::InlineEmbed,
            TapeKernel::InlineWhere => Self::InlineWhere,
            TapeKernel::InlineRange => Self::InlineRange,
            TapeKernel::InlineShape { .. } => Self::InlineShape,
            TapeKernel::InlineSlice { .. } => Self::InlineSlice,
            TapeKernel::InlineGatherND => Self::InlineGatherND,
            TapeKernel::InlineFusedSwiGLU => Self::InlineFusedSwiGLU,
            TapeKernel::InlineReshape => Self::InlineReshape,
            TapeKernel::InlineDequantize => Self::InlineDequantize,
            TapeKernel::InlineConv2d { .. } => Self::InlineConv2d,
            TapeKernel::InlineConv2dActivation { .. } => Self::InlineConv2dActivation,
            TapeKernel::InlineConv2dBiasActivation { .. } => Self::InlineConv2dBiasActivation,
            TapeKernel::InlineConvTranspose { .. } => Self::InlineConvTranspose,
            TapeKernel::InlineMaxPool2d { .. } => Self::InlineMaxPool2d,
            TapeKernel::InlineAvgPool2d { .. } => Self::InlineAvgPool2d,
            TapeKernel::InlineGlobalAvgPool { .. } => Self::InlineGlobalAvgPool,
            TapeKernel::InlineResize { .. } => Self::InlineResize,
            TapeKernel::InlinePad { .. } => Self::InlinePad,
            TapeKernel::InlineInstanceNorm { .. } => Self::InlineInstanceNorm,
            TapeKernel::InlineLRN { .. } => Self::InlineLRN,
            TapeKernel::InlineTopK { .. } => Self::InlineTopK,
            TapeKernel::InlineScatterND => Self::InlineScatterND,
            TapeKernel::InlineCumSum { .. } => Self::InlineCumSum,
            TapeKernel::InlineNonZero => Self::InlineNonZero,
            TapeKernel::InlineCompress { .. } => Self::InlineCompress,
            TapeKernel::InlineReverseSequence { .. } => Self::InlineReverseSequence,
            TapeKernel::Passthrough => Self::Passthrough,
            TapeKernel::Custom(_) => Self::Custom,
            TapeKernel::InlineGroupNorm { .. } => Self::InlineGroupNorm,
            TapeKernel::InlineArgMax { .. } => Self::InlineArgMax,
            TapeKernel::InlineRmsNormActivation { .. } => Self::InlineRmsNormActivation,
            TapeKernel::InlineLayerNormActivation { .. } => Self::InlineLayerNormActivation,
            TapeKernel::InlineGroupNormActivation { .. } => Self::InlineGroupNormActivation,
            TapeKernel::InlineAddRmsNormActivation { .. } => Self::InlineAddRmsNormActivation,
            TapeKernel::InlineInstanceNormActivation { .. } => Self::InlineInstanceNormActivation,
            TapeKernel::InlineNormProjectionGemv { .. } => Self::InlineNormProjectionGemv,
            TapeKernel::InlineAddNormProjectionGemv { .. } => Self::InlineAddNormProjectionGemv,
            TapeKernel::InlineSwiGluProjectionGemv { .. } => Self::InlineSwiGluProjectionGemv,
            TapeKernel::RingPrimUnary { .. } => Self::RingPrimUnary,
            TapeKernel::RingPrimBinary { .. } => Self::RingPrimBinary,
            TapeKernel::RingActivation { .. } => Self::RingActivation,
            TapeKernel::RingAccumulate { .. } => Self::RingAccumulate,
            TapeKernel::InlineConv2dLut4 { .. } => Self::InlineConv2dLut4,
            TapeKernel::InlineExpand { .. } => Self::InlineExpand,
        }
    }
}

/// Configurable kernel filter for constrained execution.
///
/// Determines which [`TapeKernel`] variants are allowed in a constrained tape.
/// Provides preset configurations for common workloads plus custom construction.
#[derive(Debug, Clone)]
pub struct KernelAllowlist {
    allowed: HashSet<KernelDiscriminant>,
}

impl KernelAllowlist {
    /// Create an allowlist from an explicit set of discriminants.
    #[must_use]
    pub fn from_discriminants(allowed: HashSet<KernelDiscriminant>) -> Self {
        Self { allowed }
    }

    /// Check whether a kernel is permitted.
    #[must_use]
    pub fn is_allowed(&self, kernel: &TapeKernel) -> bool {
        self.allowed
            .contains(&KernelDiscriminant::from_kernel(kernel))
    }

    /// Preset: AI inference workload (transformers, vision models).
    ///
    /// Includes: elementwise ops, matmul (inline + LUT-quantized), norms,
    /// softmax, attention, rotary, KV cache, conv, pooling, gather, concat,
    /// reshape, transpose, output, passthrough.
    #[must_use]
    pub fn inference() -> Self {
        use KernelDiscriminant::*;
        Self::from_discriminants(HashSet::from([
            // Structural
            Output,
            Passthrough,
            FusedFloatChain,
            // Elementwise
            InlineRelu,
            InlineNeg,
            InlineSigmoid,
            InlineSilu,
            InlineTanh,
            InlineGelu,
            InlineExp,
            InlineAdd,
            InlineMul,
            InlineSub,
            InlineDiv,
            InlineAbs,
            InlineReciprocal,
            InlineLog,
            InlineSqrt,
            InlineErf,
            InlineMin,
            InlineMax,
            InlineClip,
            InlineWhere,
            InlineFusedSwiGLU,
            InlinePow,
            // MatMul
            InlineMatMul,
            InlineMatMulActivation,
            InlineMatMulBiasActivation,
            InlineGemm,
            MatMulLut2,
            MatMulLut4,
            MatMulLut8,
            MatMulLut16,
            MatMulLut2Activation,
            MatMulLut4Activation,
            MatMulLut8Activation,
            InlineDequantize,
            // Norms
            InlineRmsNorm,
            InlineLayerNorm,
            InlineAddRmsNorm,
            InlineGroupNorm,
            InlineInstanceNorm,
            InlineRmsNormActivation,
            InlineLayerNormActivation,
            InlineGroupNormActivation,
            InlineAddRmsNormActivation,
            InlineInstanceNormActivation,
            // Fused norm+projection
            InlineNormProjectionGemv,
            InlineAddNormProjectionGemv,
            InlineSwiGluProjectionGemv,
            // Attention
            InlineSoftmax,
            InlineLogSoftmax,
            InlineAttention,
            InlineRoPE,
            KvWrite,
            KvRead,
            // Shape ops
            InlineGather,
            InlineConcat,
            InlineTranspose,
            InlineReshape,
            InlineSlice,
            InlineShape,
            InlineEmbed,
            InlineCast,
            // Reduction
            InlineReduceSum,
            InlineReduceMean,
            InlineReduceMax,
            InlineReduceMin,
            InlineReduceProd,
            InlineArgMax,
            // Conv + pooling
            InlineConv2d,
            InlineConv2dActivation,
            InlineConv2dBiasActivation,
            InlineConvTranspose,
            InlineMaxPool2d,
            InlineAvgPool2d,
            InlineGlobalAvgPool,
            InlineResize,
            InlinePad,
            InlineExpand,
            InlineConv2dLut4,
        ]))
    }

    /// Preset: general numeric computation (numpy-like pipelines, rendering).
    ///
    /// Includes: elementwise ops, matmul, shape ops, reductions, conv, pooling.
    /// Excludes: KV cache, attention, rotary, quantized LUT-GEMM, norms.
    #[must_use]
    pub fn compute() -> Self {
        use KernelDiscriminant::*;
        Self::from_discriminants(HashSet::from([
            // Structural
            Output,
            Passthrough,
            FusedFloatChain,
            // Elementwise
            InlineRelu,
            InlineNeg,
            InlineSigmoid,
            InlineSilu,
            InlineTanh,
            InlineGelu,
            InlineExp,
            InlineAdd,
            InlineMul,
            InlineSub,
            InlineDiv,
            InlineAbs,
            InlineReciprocal,
            InlineLog,
            InlineSqrt,
            InlineCos,
            InlineSin,
            InlineSign,
            InlineFloor,
            InlineCeil,
            InlineRound,
            InlineErf,
            InlineMin,
            InlineMax,
            InlineClip,
            InlineIsNaN,
            InlineNot,
            InlineAnd,
            InlineOr,
            InlineXor,
            InlineEqual,
            InlineLess,
            InlineLessOrEqual,
            InlineGreater,
            InlineGreaterOrEqual,
            InlinePow,
            InlineMod,
            InlineWhere,
            InlineRange,
            // MatMul
            InlineMatMul,
            InlineMatMulActivation,
            InlineMatMulBiasActivation,
            InlineGemm,
            // Shape ops
            InlineGather,
            InlineGatherND,
            InlineConcat,
            InlineTranspose,
            InlineReshape,
            InlineSlice,
            InlineShape,
            InlineCast,
            InlineScatterND,
            InlinePad,
            InlineResize,
            InlineTopK,
            InlineCumSum,
            InlineNonZero,
            InlineCompress,
            InlineReverseSequence,
            // Reduction
            InlineReduceSum,
            InlineReduceMean,
            InlineReduceMax,
            InlineReduceMin,
            InlineReduceProd,
            InlineArgMax,
            // Conv + pooling
            InlineConv2d,
            InlineConv2dActivation,
            InlineConv2dBiasActivation,
            InlineConvTranspose,
            InlineMaxPool2d,
            InlineAvgPool2d,
            InlineGlobalAvgPool,
            InlineExpand,
            // Ring-arithmetic ops
            RingPrimUnary,
            RingPrimBinary,
            RingActivation,
            RingAccumulate,
        ]))
    }
}

/// Constrained execution profile.
///
/// Defines memory limits, weight residency policy, and kernel restrictions
/// for deterministic, bounded-memory tape execution. Workload-agnostic.
#[derive(Debug, Clone)]
pub struct ConstrainedProfile {
    /// Maximum bytes for weight residency (weight window cap).
    pub max_weight_bytes: usize,
    /// Maximum bytes for activation buffers.
    pub max_activation_bytes: usize,
    /// Weight loading/eviction policy.
    pub weight_policy: WeightPolicy,
    /// Optional kernel allowlist. `None` means all kernels are allowed.
    pub kernel_allowlist: Option<KernelAllowlist>,
    /// Whether custom ops are permitted.
    pub allow_custom_ops: bool,
    /// Whether fallback (non-inline) kernels are permitted.
    pub allow_fallback_kernels: bool,
}

impl Default for ConstrainedProfile {
    fn default() -> Self {
        Self {
            max_weight_bytes: 256 * 1024 * 1024,    // 256 MB
            max_activation_bytes: 64 * 1024 * 1024, // 64 MB
            weight_policy: WeightPolicy::BoundedWindow,
            kernel_allowlist: None,
            allow_custom_ops: false,
            allow_fallback_kernels: false,
        }
    }
}
