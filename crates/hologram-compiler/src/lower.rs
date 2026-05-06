//! OpKind -> KernelCall lowering (spec VII.2 step 8 + IX.1).

use hologram_backend::{
    KernelCall, BufferRef, UnaryCall, BinaryCall, MatMulCall, GemmCall,
    Conv2dCall, NormCall, ReduceCall, LayoutCall, SoftmaxCall, PoolCall,
    AttentionCall, WhereCall,
};
use hologram_graph::OpKind;
use crate::error::CompileError;

/// Resolved per-node lowering inputs.
pub struct LoweredNode {
    pub kind: OpKind,
    pub inputs: Vec<BufferRef>,
    pub output: BufferRef,
    pub element_count: u32,
    pub witt_bits: u16,
    pub dtype: u8,
}

pub fn lower(node: &LoweredNode) -> Result<KernelCall, CompileError> {
    use OpKind as K;
    let inp0 = || node.inputs.first().copied().unwrap_or(BufferRef { slot: 0, offset: 0, length: 0 });
    let inp1 = || node.inputs.get(1).copied().unwrap_or(BufferRef { slot: 0, offset: 0, length: 0 });
    let inp2 = || node.inputs.get(2).copied().unwrap_or(BufferRef { slot: 0, offset: 0, length: 0 });
    let unary = UnaryCall {
        input: inp0(),
        output: node.output,
        element_count: node.element_count,
        witt_bits: node.witt_bits,
        dtype: node.dtype,
    };
    let binary = BinaryCall {
        a: inp0(), b: inp1(), output: node.output,
        element_count: node.element_count,
        witt_bits: node.witt_bits,
        dtype: node.dtype,
    };
    let layout = LayoutCall {
        input: inp0(), output: node.output,
        element_count: node.element_count, dtype: node.dtype,
    };
    let where_call = WhereCall {
        cond: inp0(), a: inp1(), b: inp2(), output: node.output,
        element_count: node.element_count, dtype: node.dtype,
    };
    let zero_norm = NormCall {
        x: inp0(), gamma: inp1(), beta: inp2(),
        residual: NormCall::NO_RESIDUAL,
        output: node.output,
        batch: 0, feature: 0, epsilon_bits: 0, dtype: node.dtype,
    };
    let add_rms_norm_call = NormCall {
        x: inp0(), gamma: inp1(), beta: NormCall::NO_RESIDUAL,
        residual: inp2(),
        output: node.output,
        batch: 0, feature: 0, epsilon_bits: 0, dtype: node.dtype,
    };
    let zero_reduce = ReduceCall {
        input: inp0(), output: node.output,
        element_count: node.element_count, axis_count: 0, keepdims: false,
        dtype: node.dtype,
    };
    let zero_softmax = SoftmaxCall {
        input: inp0(), output: node.output,
        batch: 0, feature: 0, dtype: node.dtype,
    };
    let zero_pool = PoolCall {
        x: inp0(), output: node.output,
        batch: 0, channels: 0, h_in: 0, w_in: 0, h_out: 0, w_out: 0,
        k_h: 0, k_w: 0, stride_h: 1, stride_w: 1, dtype: node.dtype,
    };
    let zero_matmul = MatMulCall {
        a: inp0(), b: inp1(), output: node.output,
        m: 0, k: 0, n: 0, dtype: node.dtype,
    };
    let zero_gemm = GemmCall {
        a: inp0(), b: inp1(), c: inp2(), output: node.output,
        m: 0, k: 0, n: 0, alpha_bits: 0, beta_bits: 0, dtype: node.dtype,
    };
    let zero_conv = Conv2dCall {
        x: inp0(), w: inp1(), output: node.output,
        batch: 0, channels_in: 0, channels_out: 0,
        h_in: 0, w_in: 0, h_out: 0, w_out: 0,
        k_h: 0, k_w: 0, stride_h: 1, stride_w: 1,
        pad_h: 0, pad_w: 0, dtype: node.dtype,
    };
    let zero_attn = AttentionCall {
        q: inp0(), k: inp1(), v: inp2(), output: node.output,
        batch: 0, heads: 0, seq: 0, head_dim: 0, dtype: node.dtype,
    };

    Ok(match node.kind {
        K::Neg => KernelCall::Neg(unary),  K::Bnot => KernelCall::Bnot(unary),
        K::Succ => KernelCall::Succ(unary), K::Pred => KernelCall::Pred(unary),
        K::Add => KernelCall::Add(binary), K::Sub => KernelCall::Sub(binary),
        K::Mul => KernelCall::Mul(binary), K::Xor => KernelCall::Xor(binary),
        K::And => KernelCall::And(binary), K::Or  => KernelCall::Or(binary),

        K::Relu => KernelCall::Relu(unary), K::Sigmoid => KernelCall::Sigmoid(unary),
        K::Tanh => KernelCall::Tanh(unary), K::Gelu => KernelCall::Gelu(unary),
        K::Silu => KernelCall::Silu(unary), K::Elu => KernelCall::Elu(unary),
        K::Selu => KernelCall::Selu(unary),
        K::Exp => KernelCall::Exp(unary), K::Log => KernelCall::Log(unary),
        K::Log1p => KernelCall::Log1p(unary), K::Sqrt => KernelCall::Sqrt(unary),
        K::Reciprocal => KernelCall::Reciprocal(unary),
        K::Sin => KernelCall::Sin(unary), K::Cos => KernelCall::Cos(unary),
        K::Tan => KernelCall::Tan(unary), K::Asin => KernelCall::Asin(unary),
        K::Acos => KernelCall::Acos(unary), K::Atan => KernelCall::Atan(unary),
        K::Ceil => KernelCall::Ceil(unary), K::Floor => KernelCall::Floor(unary),
        K::Round => KernelCall::Round(unary), K::Erf => KernelCall::Erf(unary),
        K::IsNaN => KernelCall::IsNaN(unary), K::Sign => KernelCall::Sign(unary),
        K::Abs => KernelCall::Abs(unary),

        K::Div => KernelCall::Div(binary), K::Pow => KernelCall::Pow(binary),
        K::Mod => KernelCall::Mod(binary), K::Min => KernelCall::Min(binary),
        K::Max => KernelCall::Max(binary),
        K::Equal => KernelCall::Equal(binary), K::Less => KernelCall::Less(binary),
        K::LessOrEqual => KernelCall::LessOrEqual(binary),
        K::Greater => KernelCall::Greater(binary),
        K::GreaterOrEqual => KernelCall::GreaterOrEqual(binary),

        K::MatMul => KernelCall::MatMul(zero_matmul),
        K::Gemm => KernelCall::Gemm(zero_gemm),
        K::Conv2d => KernelCall::Conv2d(zero_conv),
        K::ConvTranspose2d => KernelCall::ConvTranspose2d(zero_conv),

        K::LayerNorm => KernelCall::LayerNorm(zero_norm),
        K::RmsNorm => KernelCall::RmsNorm(zero_norm),
        K::GroupNorm => KernelCall::GroupNorm(zero_norm),
        K::InstanceNorm => KernelCall::InstanceNorm(zero_norm),
        K::AddRmsNorm => KernelCall::AddRmsNorm(add_rms_norm_call),

        K::ReduceSum => KernelCall::ReduceSum(zero_reduce),
        K::ReduceMean => KernelCall::ReduceMean(zero_reduce),
        K::ReduceProd => KernelCall::ReduceProd(zero_reduce),
        K::ReduceMin => KernelCall::ReduceMin(zero_reduce),
        K::ReduceMax => KernelCall::ReduceMax(zero_reduce),

        K::Reshape => KernelCall::Reshape(layout),
        K::Transpose => KernelCall::Transpose(layout),
        K::Concat => KernelCall::Concat(layout),
        K::Slice => KernelCall::Slice(layout),

        K::Softmax => KernelCall::Softmax(zero_softmax),
        K::LogSoftmax => KernelCall::LogSoftmax(zero_softmax),

        K::MaxPool2d => KernelCall::MaxPool2d(zero_pool),
        K::AvgPool2d => KernelCall::AvgPool2d(zero_pool),
        K::GlobalAvgPool => KernelCall::GlobalAvgPool(zero_pool),

        K::Attention => KernelCall::Attention(zero_attn),
        K::FusedSwiGlu => KernelCall::FusedSwiGlu(zero_matmul),

        K::Pad => KernelCall::Pad(layout), K::Expand => KernelCall::Expand(layout),
        K::Resize => KernelCall::Resize(layout),
        K::CumSum => KernelCall::CumSum(zero_reduce),
        K::RotaryEmbedding => KernelCall::RotaryEmbedding(unary),
        K::Clip => KernelCall::Clip(unary),
        K::Lrn => KernelCall::Lrn(unary),
        K::Where => KernelCall::Where(where_call),

        // Backward
        K::MatMulGradA => KernelCall::MatMulGradA(zero_matmul),
        K::MatMulGradB => KernelCall::MatMulGradB(zero_matmul),
        K::Conv2dGradX => KernelCall::Conv2dGradX(zero_conv),
        K::Conv2dGradW => KernelCall::Conv2dGradW(zero_conv),
        K::SoftmaxGrad => KernelCall::SoftmaxGrad(zero_softmax),
        K::LogSoftmaxGrad => KernelCall::LogSoftmaxGrad(zero_softmax),
        K::LayerNormGrad => KernelCall::LayerNormGrad(zero_norm),
        K::RmsNormGrad => KernelCall::RmsNormGrad(zero_norm),
        K::GroupNormGrad => KernelCall::GroupNormGrad(zero_norm),
        K::ReduceSumGrad => KernelCall::ReduceSumGrad(zero_reduce),
        K::ReduceMeanGrad => KernelCall::ReduceMeanGrad(zero_reduce),
        K::ReduceProdGrad => KernelCall::ReduceProdGrad(zero_reduce),
        K::SubGrad => KernelCall::SubGrad(binary),
        K::MulGrad => KernelCall::MulGrad(binary),
        K::DivGrad => KernelCall::DivGrad(binary),
        K::PowGrad => KernelCall::PowGrad(binary),
        K::MinGrad => KernelCall::MinGrad(binary),
        K::MaxGrad => KernelCall::MaxGrad(binary),
        K::ConcatGrad => KernelCall::ConcatGrad(layout),
        K::SliceGrad => KernelCall::SliceGrad(layout),
        K::AvgPool2dGrad => KernelCall::AvgPool2dGrad(zero_pool),
        K::GlobalAvgPoolGrad => KernelCall::GlobalAvgPoolGrad(zero_pool),
        K::PadGrad => KernelCall::PadGrad(layout),
        K::AttentionGrad => KernelCall::AttentionGrad(zero_attn),
        K::FusedSwiGluGrad => KernelCall::FusedSwiGluGrad(zero_matmul),
        K::UnaryGrad => KernelCall::UnaryGrad(unary),
    })
}
