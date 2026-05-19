//! Single OpKind→emit_term dispatch table (spec V.3 / VII.2 step 3).
//!
//! The compiler calls one function per node; this module routes each
//! `OpKind` to its marker's `emit_term`, returning the root index of the
//! emitted Term tree. Per spec I-9 the tree IS the formal specification.
//!
//! Variable layout: the caller pushes one `Term::Variable` per argument
//! contiguously starting at `args_start`. Arity is recovered from
//! `OpKind::primary_arity()`. The emitter pulls operand indices off
//! `(args_start + i)` as needed.

use prism::vocabulary::WittLevel;
use uor_foundation::enforcement::TermArena;

use crate::emit::EmitResult;
use crate::kind::OpKind;
use crate::{
    activation_reduce, backward, conv, direct, elementwise_binary, elementwise_unary, layout,
    linalg, normalization, pooling, quantization, reduction, structured, utility,
};

/// Emit the Term tree for `kind` into `arena`. Variables for the op's
/// arguments are assumed pushed contiguously beginning at `args_start`.
/// Returns the root index of the emitted tree.
pub fn emit_op_term<const CAP: usize>(
    kind: OpKind,
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    args_start: u32,
) -> EmitResult {
    use OpKind as K;
    let a0 = args_start;
    let a1 = args_start.saturating_add(1);
    let a2 = args_start.saturating_add(2);

    match kind {
        // Direct PrimitiveOp wrappers (spec V.3): single Application.
        K::Neg => direct::NegOp::emit_term(arena, level, a0),
        K::Bnot => direct::BnotOp::emit_term(arena, level, a0),
        K::Succ => direct::SuccOp::emit_term(arena, level, a0),
        K::Pred => direct::PredOp::emit_term(arena, level, a0),
        K::Add => direct::AddOp::emit_term(arena, level, a0),
        K::Sub => direct::SubOp::emit_term(arena, level, a0),
        K::Mul => direct::MulOp::emit_term(arena, level, a0),
        K::Xor => direct::XorOp::emit_term(arena, level, a0),
        K::And => direct::AndOp::emit_term(arena, level, a0),
        K::Or => direct::OrOp::emit_term(arena, level, a0),

        // Elementwise unary.
        K::Relu => elementwise_unary::ReluOp::emit_term(arena, level, a0),
        K::Sigmoid => elementwise_unary::SigmoidOp::emit_term(arena, level, a0),
        K::Tanh => elementwise_unary::TanhOp::emit_term(arena, level, a0),
        K::Gelu => elementwise_unary::GeluOp::emit_term(arena, level, a0),
        K::Silu => elementwise_unary::SiluOp::emit_term(arena, level, a0),
        K::Elu => elementwise_unary::EluOp::emit_term(arena, level, a0),
        K::Selu => elementwise_unary::SeluOp::emit_term(arena, level, a0),
        K::Exp => elementwise_unary::ExpOp::emit_term(arena, level, a0),
        K::Log => elementwise_unary::LogOp::emit_term(arena, level, a0),
        K::Log1p => elementwise_unary::Log1pOp::emit_term(arena, level, a0),
        K::Sqrt => elementwise_unary::SqrtOp::emit_term(arena, level, a0),
        K::Reciprocal => elementwise_unary::ReciprocalOp::emit_term(arena, level, a0),
        K::Sin => elementwise_unary::SinOp::emit_term(arena, level, a0),
        K::Cos => elementwise_unary::CosOp::emit_term(arena, level, a0),
        K::Tan => elementwise_unary::TanOp::emit_term(arena, level, a0),
        K::Asin => elementwise_unary::AsinOp::emit_term(arena, level, a0),
        K::Acos => elementwise_unary::AcosOp::emit_term(arena, level, a0),
        K::Atan => elementwise_unary::AtanOp::emit_term(arena, level, a0),
        K::Ceil => elementwise_unary::CeilOp::emit_term(arena, level, a0),
        K::Floor => elementwise_unary::FloorOp::emit_term(arena, level, a0),
        K::Round => elementwise_unary::RoundOp::emit_term(arena, level, a0),
        K::Erf => elementwise_unary::ErfOp::emit_term(arena, level, a0),
        K::IsNaN => elementwise_unary::IsNaNOp::emit_term(arena, level, a0),
        K::Sign => elementwise_unary::SignOp::emit_term(arena, level, a0),
        K::Abs => elementwise_unary::AbsOp::emit_term(arena, level, a0),

        // Elementwise binary (non-primitive).
        K::Div => elementwise_binary::DivOp::emit_term(arena, level, a0),
        K::Pow => elementwise_binary::PowOp::emit_term(arena, level, a0),
        K::Mod => elementwise_binary::ModOp::emit_term(arena, level, a0),
        K::Min => elementwise_binary::MinOp::emit_term(arena, level, a0),
        K::Max => elementwise_binary::MaxOp::emit_term(arena, level, a0),
        K::Equal => elementwise_binary::EqualOp::emit_term(arena, level, a0),
        K::Less => elementwise_binary::LessOp::emit_term(arena, level, a0),
        K::LessOrEqual => elementwise_binary::LessOrEqualOp::emit_term(arena, level, a0),
        K::Greater => elementwise_binary::GreaterOp::emit_term(arena, level, a0),
        K::GreaterOrEqual => elementwise_binary::GreaterOrEqualOp::emit_term(arena, level, a0),

        // Linear algebra: nested Recurse over (i,j,k) → Add(acc, Mul(a,b)).
        K::MatMul => linalg::emit_matmul(arena, level, a0, a1),
        K::Gemm => linalg::emit_gemm(arena, level, a0, a1, a2),

        // Convolution: 4-deep Recurse (out_h, out_w, k_h, k_w) → Add(acc, Mul(x,w)).
        K::Conv2d => conv::emit_conv2d(arena, level, a0, a1),
        K::ConvTranspose2d => conv::emit_conv_transpose_2d(arena, level, a0, a1),

        // Normalization: ReduceMean → Sub → Mul → ReduceMean → Sqrt → Div → Mul → Add.
        K::LayerNorm => normalization::emit_layer_norm(arena, level, a0, a1, a2),
        K::RmsNorm => normalization::emit_rms_norm(arena, level, a0, a1, a2),
        K::GroupNorm => normalization::emit_group_norm(arena, level, a0, a1, a2),
        K::InstanceNorm => normalization::emit_instance_norm(arena, level, a0, a1, a2),
        K::AddRmsNorm => normalization::emit_add_rms_norm(arena, level, a0, a1),

        // Reductions: single Recurse over reduction axes.
        K::ReduceSum => reduction::emit_reduce_sum(arena, level, a0),
        K::ReduceMean => reduction::emit_reduce_mean(arena, level, a0),
        K::ReduceProd => reduction::emit_reduce_prod(arena, level, a0),
        K::ReduceMin => reduction::emit_reduce_min(arena, level, a0),
        K::ReduceMax => reduction::emit_reduce_max(arena, level, a0),

        // Layout: bijective relabel; emit a single Variable referencing the remap.
        K::Reshape => layout::emit_layout_relabel(arena, level, a0),
        K::Transpose => layout::emit_layout_relabel(arena, level, a0),
        K::Concat => layout::emit_layout_relabel(arena, level, a0),
        K::Slice => layout::emit_layout_relabel(arena, level, a0),

        // Activation+reduce: ReduceMax → Sub → Exp → ReduceSum → Div.
        K::Softmax => activation_reduce::emit_softmax(arena, level, a0),
        K::LogSoftmax => activation_reduce::emit_log_softmax(arena, level, a0),

        // Pooling: windowed Recurse.
        K::MaxPool2d => pooling::emit_max_pool_2d(arena, level, a0),
        K::AvgPool2d => pooling::emit_avg_pool_2d(arena, level, a0),
        K::GlobalAvgPool => pooling::emit_global_avg_pool(arena, level, a0),

        // Structured: Attention = MatMul(Q,Kᵀ) → Mul(scale) → Softmax → MatMul(_,V).
        K::Attention => structured::emit_attention(arena, level, a0, a1, a2),
        K::FusedSwiGlu => structured::emit_fused_swiglu(arena, level, a0, a1),

        // Utility.
        K::Pad => utility::emit_layout_relabel(arena, level, a0),
        K::Expand => utility::emit_layout_relabel(arena, level, a0),
        K::Resize => utility::emit_resize(arena, level, a0),
        K::CumSum => utility::emit_cumsum(arena, level, a0),
        K::RotaryEmbedding => utility::emit_rotary_embedding(arena, level, a0),
        K::Clip => utility::emit_clip(arena, level, a0),
        K::Lrn => utility::emit_lrn(arena, level, a0),
        K::Where => utility::emit_where(arena, level, a0, a1, a2),

        // Quantization (spec X-5).
        K::Dequantize => quantization::emit_dequantize(arena, level, a0),

        // Backward variants (spec V.4).
        K::MatMulGradA => backward::emit_matmul_grad_a(arena, level, a0, a1),
        K::MatMulGradB => backward::emit_matmul_grad_b(arena, level, a0, a1),
        K::Conv2dGradX => backward::emit_conv2d_grad_x(arena, level, a0, a1),
        K::Conv2dGradW => backward::emit_conv2d_grad_w(arena, level, a0, a1),
        K::SoftmaxGrad => backward::emit_softmax_grad(arena, level, a0),
        K::LogSoftmaxGrad => backward::emit_log_softmax_grad(arena, level, a0),
        K::LayerNormGrad => backward::emit_layer_norm_grad(arena, level, a0, a1, a2),
        K::RmsNormGrad => backward::emit_rms_norm_grad(arena, level, a0, a1, a2),
        K::GroupNormGrad => backward::emit_group_norm_grad(arena, level, a0, a1, a2),
        K::ReduceSumGrad => backward::emit_reduce_sum_grad(arena, level, a0),
        K::ReduceMeanGrad => backward::emit_reduce_mean_grad(arena, level, a0),
        K::ReduceProdGrad => backward::emit_reduce_prod_grad(arena, level, a0),
        K::SubGrad => backward::emit_sub_grad(arena, level, a0),
        K::MulGrad => backward::emit_mul_grad(arena, level, a0),
        K::DivGrad => backward::emit_div_grad(arena, level, a0),
        K::PowGrad => backward::emit_pow_grad(arena, level, a0),
        K::MinGrad => backward::emit_min_grad(arena, level, a0),
        K::MaxGrad => backward::emit_max_grad(arena, level, a0),
        K::ConcatGrad => backward::emit_concat_grad(arena, level, a0),
        K::SliceGrad => backward::emit_slice_grad(arena, level, a0),
        K::AvgPool2dGrad => backward::emit_avg_pool_2d_grad(arena, level, a0),
        K::GlobalAvgPoolGrad => backward::emit_global_avg_pool_grad(arena, level, a0),
        K::PadGrad => backward::emit_pad_grad(arena, level, a0),
        K::AttentionGrad => backward::emit_attention_grad(arena, level, a0, a1, a2),
        K::FusedSwiGluGrad => backward::emit_fused_swiglu_grad(arena, level, a0, a1, a2),
        K::UnaryGrad => backward::emit_unary_grad(arena, level, a0),
    }
}
