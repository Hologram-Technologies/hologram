//! `KernelCall` enum — one variant per `OpKind` (spec IX.1).
//!
//! Each variant carries the resolved buffer references and op-specific
//! parameters. The hot path matches on this enum and dispatches to the
//! corresponding kernel function.

use crate::workspace::BufferRef;

/// Direct PrimitiveOp wrapper kernels.
#[derive(Debug, Clone, Copy)]
pub struct UnaryCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    pub witt_bits: u16,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    pub witt_bits: u16,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct GemmCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub c: BufferRef,
    pub output: BufferRef,
    pub m: u32, pub k: u32, pub n: u32,
    pub alpha_bits: u64, pub beta_bits: u64,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dCall {
    pub x: BufferRef,
    pub w: BufferRef,
    pub output: BufferRef,
    pub batch: u32, pub channels_in: u32, pub channels_out: u32,
    pub h_in: u32, pub w_in: u32,
    pub h_out: u32, pub w_out: u32,
    pub k_h: u32, pub k_w: u32,
    pub stride_h: u32, pub stride_w: u32,
    pub pad_h: u32, pub pad_w: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct NormCall {
    pub x: BufferRef,
    pub gamma: BufferRef,
    pub beta: BufferRef,
    /// Optional residual buffer for fused-add normalizations (e.g. AddRmsNorm).
    /// `slot == u32::MAX` indicates no residual (plain norm).
    pub residual: BufferRef,
    pub output: BufferRef,
    pub batch: u32, pub feature: u32,
    pub epsilon_bits: u64,
    pub dtype: u8,
}

impl NormCall {
    /// Sentinel for an unused residual buffer.
    pub const NO_RESIDUAL: BufferRef = BufferRef { slot: u32::MAX, offset: 0, length: 0 };

    pub const fn has_residual(&self) -> bool {
        self.residual.slot != u32::MAX
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReduceCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    pub axis_count: u32,
    pub keepdims: bool,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct SoftmaxCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub feature: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct PoolCall {
    pub x: BufferRef,
    pub output: BufferRef,
    pub batch: u32, pub channels: u32,
    pub h_in: u32, pub w_in: u32,
    pub h_out: u32, pub w_out: u32,
    pub k_h: u32, pub k_w: u32,
    pub stride_h: u32, pub stride_w: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionCall {
    pub q: BufferRef, pub k: BufferRef, pub v: BufferRef,
    pub output: BufferRef,
    pub batch: u32, pub heads: u32, pub seq: u32, pub head_dim: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct WhereCall {
    pub cond: BufferRef,
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    pub dtype: u8,
}

/// Dequantize kernel payload (spec X-5). Reads `element_count` quantized
/// values from `input` (interpreted per `quant_dtype` — `DTYPE_I8` or
/// `DTYPE_I4`), applies `output = (q − zero_point) · scale`, and writes
/// the result into `output` at `dtype` (typically `DTYPE_F32` or
/// `DTYPE_BF16`).
///
/// `scale_bits` and `zero_point` are passed by value rather than via a
/// separate buffer since they are per-tensor scalars resolved at compile
/// time. Per-channel quantization (one scale per output channel) is left
/// for a future fused matmul-with-dequant kernel.
#[derive(Debug, Clone, Copy)]
pub struct DequantizeCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    /// Source quantized dtype: `DTYPE_I8` or `DTYPE_I4`.
    pub quant_dtype: u8,
    /// Destination float dtype: `DTYPE_F32`, `DTYPE_BF16`, etc.
    pub dtype: u8,
    /// `f32::to_bits` of the per-tensor scale.
    pub scale_bits: u32,
    /// Symmetric zero-point (i32, conventional INT8/INT4 range).
    pub zero_point: i32,
}

/// Fused MatMul + activation epilogue call. The activation is applied
/// element-wise to each output of the matmul without writing an
/// intermediate buffer.
#[derive(Debug, Clone, Copy)]
pub struct FusedMatMulActivationCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub dtype: u8,
    /// The activation op to apply as epilogue. Encoded as the `OpKind`
    /// discriminant (e.g. Relu=10, Silu=14, etc.).
    pub activation: u16,
}

/// Fused Conv2d + activation epilogue call.
#[derive(Debug, Clone, Copy)]
pub struct FusedConv2dActivationCall {
    pub x: BufferRef,
    pub w: BufferRef,
    pub output: BufferRef,
    pub batch: u32, pub channels_in: u32, pub channels_out: u32,
    pub h_in: u32, pub w_in: u32,
    pub h_out: u32, pub w_out: u32,
    pub k_h: u32, pub k_w: u32,
    pub stride_h: u32, pub stride_w: u32,
    pub pad_h: u32, pub pad_w: u32,
    pub dtype: u8,
    pub activation: u16,
}

/// Fused Norm + activation epilogue call.
#[derive(Debug, Clone, Copy)]
pub struct FusedNormActivationCall {
    pub x: BufferRef,
    pub gamma: BufferRef,
    pub beta: BufferRef,
    pub residual: BufferRef,
    pub output: BufferRef,
    pub batch: u32, pub feature: u32,
    pub epsilon_bits: u64,
    pub dtype: u8,
    pub activation: u16,
}

/// Fused unary chain call. Applies up to 8 elementwise-unary
/// activations sequentially with a single buffer read/write.
#[derive(Debug, Clone, Copy)]
pub struct FusedUnaryChainCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u32,
    pub dtype: u8,
    /// Number of activations in the chain (1..=8).
    pub chain_len: u8,
    /// Activation discriminants in application order. Unused slots
    /// are 0 (identity). Encoded as `OpKind as u16`.
    pub chain: [u16; 8],
}

/// Closed kernel-call surface. One variant per OpKind.
#[derive(Debug, Clone, Copy)]
pub enum KernelCall {
    // Direct primitives
    Neg(UnaryCall), Bnot(UnaryCall), Succ(UnaryCall), Pred(UnaryCall),
    Add(BinaryCall), Sub(BinaryCall), Mul(BinaryCall),
    Xor(BinaryCall), And(BinaryCall), Or(BinaryCall),

    // Elementwise unary
    Relu(UnaryCall), Sigmoid(UnaryCall), Tanh(UnaryCall),
    Gelu(UnaryCall), Silu(UnaryCall), Elu(UnaryCall), Selu(UnaryCall),
    Exp(UnaryCall), Log(UnaryCall), Log1p(UnaryCall),
    Sqrt(UnaryCall), Reciprocal(UnaryCall),
    Sin(UnaryCall), Cos(UnaryCall), Tan(UnaryCall),
    Asin(UnaryCall), Acos(UnaryCall), Atan(UnaryCall),
    Ceil(UnaryCall), Floor(UnaryCall), Round(UnaryCall), Erf(UnaryCall),
    IsNaN(UnaryCall), Sign(UnaryCall), Abs(UnaryCall),

    // Elementwise binary
    Div(BinaryCall), Pow(BinaryCall), Mod(BinaryCall),
    Min(BinaryCall), Max(BinaryCall),
    Equal(BinaryCall), Less(BinaryCall), LessOrEqual(BinaryCall),
    Greater(BinaryCall), GreaterOrEqual(BinaryCall),

    // Linear algebra / convolution
    MatMul(MatMulCall), Gemm(GemmCall),
    Conv2d(Conv2dCall), ConvTranspose2d(Conv2dCall),

    // Normalization
    LayerNorm(NormCall), RmsNorm(NormCall),
    GroupNorm(NormCall), InstanceNorm(NormCall), AddRmsNorm(NormCall),

    // Reduction
    ReduceSum(ReduceCall), ReduceMean(ReduceCall),
    ReduceProd(ReduceCall), ReduceMin(ReduceCall), ReduceMax(ReduceCall),

    // Layout
    Reshape(LayoutCall), Transpose(LayoutCall),
    Concat(LayoutCall), Slice(LayoutCall),

    // Activation+reduce
    Softmax(SoftmaxCall), LogSoftmax(SoftmaxCall),

    // Pooling
    MaxPool2d(PoolCall), AvgPool2d(PoolCall), GlobalAvgPool(PoolCall),

    // Structured
    Attention(AttentionCall),
    FusedSwiGlu(MatMulCall),
    FusedMatMulActivation(FusedMatMulActivationCall),
    FusedConv2dActivation(FusedConv2dActivationCall),
    FusedNormActivation(FusedNormActivationCall),
    FusedUnaryChain(FusedUnaryChainCall),

    // Utility
    Pad(LayoutCall), Expand(LayoutCall), Resize(LayoutCall),
    CumSum(ReduceCall), RotaryEmbedding(UnaryCall),
    Clip(UnaryCall), Lrn(UnaryCall),
    Where(WhereCall),

    // Backward variants — same payload shapes as their forward counterparts.
    MatMulGradA(MatMulCall), MatMulGradB(MatMulCall),
    Conv2dGradX(Conv2dCall), Conv2dGradW(Conv2dCall),
    SoftmaxGrad(SoftmaxCall), LogSoftmaxGrad(SoftmaxCall),
    LayerNormGrad(NormCall), RmsNormGrad(NormCall), GroupNormGrad(NormCall),
    ReduceSumGrad(ReduceCall), ReduceMeanGrad(ReduceCall), ReduceProdGrad(ReduceCall),
    SubGrad(BinaryCall), MulGrad(BinaryCall), DivGrad(BinaryCall), PowGrad(BinaryCall),
    MinGrad(BinaryCall), MaxGrad(BinaryCall),
    ConcatGrad(LayoutCall), SliceGrad(LayoutCall),
    AvgPool2dGrad(PoolCall), GlobalAvgPoolGrad(PoolCall),
    PadGrad(LayoutCall),
    AttentionGrad(AttentionCall),
    FusedSwiGluGrad(MatMulCall),
    UnaryGrad(UnaryCall),

    // Quantization (spec X-5)
    Dequantize(DequantizeCall),
}
