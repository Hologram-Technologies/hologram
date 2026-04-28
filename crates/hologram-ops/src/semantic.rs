//! `SemanticOp` enum — the closed serialisation and dispatch surface.
//!
//! Per-op marker structs live in [`crate::kernels`] alongside their
//! kernels; this file owns only the variant enum and the macro that
//! routes from a `SemanticOp` value to the matching marker struct's
//! [`Op`] trait methods.

use crate::attrs::{
    AttentionAttrs, ClipAttrs, ConcatAttrs, Conv2dAttrs, ConvTransposeAttrs, CumSumAttrs,
    ExpandAttrs, GemmAttrs, GlobalAvgPoolAttrs, GroupNormAttrs, LrnAttrs, MatMulAttrs, NormAttrs,
    PadAttrs, Pool2dAttrs, ReduceAttrs, ResizeAttrs, RotaryEmbeddingAttrs, SliceAttrs,
    SoftmaxAttrs, TransposeAttrs,
};
use crate::kernels::{
    add::Add,
    attention::Attention,
    binary::{
        And, Div, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Max, Min, Mod, Mul, Or, Pow,
        Sub, Xor,
    },
    clip::Clip,
    conv::Conv2d,
    conv_transpose::ConvTranspose2d,
    cumsum::CumSum,
    expand::Expand,
    fused::FusedSwiGlu,
    gemm::Gemm,
    lrn::Lrn,
    matmul::MatMul,
    norm::{AddRmsNorm, GroupNorm, InstanceNorm, LayerNorm, RmsNorm},
    pad::Pad,
    pool::{AvgPool2d, GlobalAvgPool, MaxPool2d},
    reduce::{ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum},
    reshape::Reshape,
    resize::Resize,
    rotary::RotaryEmbedding,
    select::Where,
    shape::{Concat, Slice, Transpose},
    softmax::{LogSoftmax, Softmax},
    unary::{
        Abs, Ceil, Cos, Erf, Exp, Floor, Gelu, IsNaN, Log, Neg, Not, Reciprocal, Relu, Round,
        Sigmoid, Sign, Silu, Sin, Sqrt, Tanh,
    },
};
use crate::trait_def::{BackwardRule, Op, OpCategory, OpSignature};

/// Macro: dispatch a method call across every `SemanticOp` variant by
/// constructing the matching marker struct and invoking `$method` on it.
///
/// This is the **single** place that names every variant, so adding a
/// new `SemanticOp` variant produces one compile error here pointing at
/// every per-op fact that needs filling in.
macro_rules! semantic_dispatch {
    ($self:expr, |$op:ident| $body:expr) => {
        match $self {
            SemanticOp::Add => {
                let $op = Add;
                $body
            }
            SemanticOp::Sub => {
                let $op = Sub;
                $body
            }
            SemanticOp::Mul => {
                let $op = Mul;
                $body
            }
            SemanticOp::Div => {
                let $op = Div;
                $body
            }
            SemanticOp::Pow => {
                let $op = Pow;
                $body
            }
            SemanticOp::Mod => {
                let $op = Mod;
                $body
            }
            SemanticOp::Min => {
                let $op = Min;
                $body
            }
            SemanticOp::Max => {
                let $op = Max;
                $body
            }
            SemanticOp::Equal => {
                let $op = Equal;
                $body
            }
            SemanticOp::Less => {
                let $op = Less;
                $body
            }
            SemanticOp::LessOrEqual => {
                let $op = LessOrEqual;
                $body
            }
            SemanticOp::Greater => {
                let $op = Greater;
                $body
            }
            SemanticOp::GreaterOrEqual => {
                let $op = GreaterOrEqual;
                $body
            }
            SemanticOp::And => {
                let $op = And;
                $body
            }
            SemanticOp::Or => {
                let $op = Or;
                $body
            }
            SemanticOp::Xor => {
                let $op = Xor;
                $body
            }
            SemanticOp::Not => {
                let $op = Not;
                $body
            }
            SemanticOp::IsNaN => {
                let $op = IsNaN;
                $body
            }
            SemanticOp::Neg => {
                let $op = Neg;
                $body
            }
            SemanticOp::Relu => {
                let $op = Relu;
                $body
            }
            SemanticOp::Gelu => {
                let $op = Gelu;
                $body
            }
            SemanticOp::Silu => {
                let $op = Silu;
                $body
            }
            SemanticOp::Tanh => {
                let $op = Tanh;
                $body
            }
            SemanticOp::Sigmoid => {
                let $op = Sigmoid;
                $body
            }
            SemanticOp::Exp => {
                let $op = Exp;
                $body
            }
            SemanticOp::Log => {
                let $op = Log;
                $body
            }
            SemanticOp::Sqrt => {
                let $op = Sqrt;
                $body
            }
            SemanticOp::Abs => {
                let $op = Abs;
                $body
            }
            SemanticOp::Reciprocal => {
                let $op = Reciprocal;
                $body
            }
            SemanticOp::Cos => {
                let $op = Cos;
                $body
            }
            SemanticOp::Sin => {
                let $op = Sin;
                $body
            }
            SemanticOp::Sign => {
                let $op = Sign;
                $body
            }
            SemanticOp::Floor => {
                let $op = Floor;
                $body
            }
            SemanticOp::Ceil => {
                let $op = Ceil;
                $body
            }
            SemanticOp::Round => {
                let $op = Round;
                $body
            }
            SemanticOp::Erf => {
                let $op = Erf;
                $body
            }
            SemanticOp::MatMul(attrs) => {
                let $op = MatMul(*attrs);
                $body
            }
            SemanticOp::Softmax(attrs) => {
                let $op = Softmax(*attrs);
                $body
            }
            SemanticOp::LogSoftmax(attrs) => {
                let $op = LogSoftmax(*attrs);
                $body
            }
            SemanticOp::RmsNorm(attrs) => {
                let $op = RmsNorm(*attrs);
                $body
            }
            SemanticOp::LayerNorm(attrs) => {
                let $op = LayerNorm(*attrs);
                $body
            }
            SemanticOp::InstanceNorm(attrs) => {
                let $op = InstanceNorm(*attrs);
                $body
            }
            SemanticOp::GroupNorm(attrs) => {
                let $op = GroupNorm(*attrs);
                $body
            }
            SemanticOp::AddRmsNorm(attrs) => {
                let $op = AddRmsNorm(*attrs);
                $body
            }
            SemanticOp::Transpose(attrs) => {
                let $op = Transpose(*attrs);
                $body
            }
            SemanticOp::Reshape => {
                let $op = Reshape;
                $body
            }
            SemanticOp::Slice(attrs) => {
                let $op = Slice(*attrs);
                $body
            }
            SemanticOp::Concat(attrs) => {
                let $op = Concat(*attrs);
                $body
            }
            SemanticOp::Conv2d(attrs) => {
                let $op = Conv2d(*attrs);
                $body
            }
            SemanticOp::FusedSwiGlu => {
                let $op = FusedSwiGlu;
                $body
            }
            SemanticOp::ReduceSum(attrs) => {
                let $op = ReduceSum(*attrs);
                $body
            }
            SemanticOp::ReduceMean(attrs) => {
                let $op = ReduceMean(*attrs);
                $body
            }
            SemanticOp::ReduceMax(attrs) => {
                let $op = ReduceMax(*attrs);
                $body
            }
            SemanticOp::ReduceMin(attrs) => {
                let $op = ReduceMin(*attrs);
                $body
            }
            SemanticOp::ReduceProd(attrs) => {
                let $op = ReduceProd(*attrs);
                $body
            }
            SemanticOp::MaxPool2d(attrs) => {
                let $op = MaxPool2d(*attrs);
                $body
            }
            SemanticOp::AvgPool2d(attrs) => {
                let $op = AvgPool2d(*attrs);
                $body
            }
            SemanticOp::GlobalAvgPool(attrs) => {
                let $op = GlobalAvgPool(*attrs);
                $body
            }
            SemanticOp::Where => {
                let $op = Where;
                $body
            }
            SemanticOp::Clip(attrs) => {
                let $op = Clip(*attrs);
                $body
            }
            SemanticOp::CumSum(attrs) => {
                let $op = CumSum(*attrs);
                $body
            }
            SemanticOp::Pad(attrs) => {
                let $op = Pad(*attrs);
                $body
            }
            SemanticOp::Resize(attrs) => {
                let $op = Resize(*attrs);
                $body
            }
            SemanticOp::Lrn(attrs) => {
                let $op = Lrn(*attrs);
                $body
            }
            SemanticOp::ConvTranspose2d(attrs) => {
                let $op = ConvTranspose2d(*attrs);
                $body
            }
            SemanticOp::Gemm(attrs) => {
                let $op = Gemm(*attrs);
                $body
            }
            SemanticOp::Expand(attrs) => {
                let $op = Expand(*attrs);
                $body
            }
            SemanticOp::RotaryEmbedding(attrs) => {
                let $op = RotaryEmbedding(*attrs);
                $body
            }
            SemanticOp::Attention(attrs) => {
                let $op = Attention(*attrs);
                $body
            }
        }
    };
}

/// Canonical semantic compute operation.
///
/// This is the graph-facing semantic payload: a compute node can carry
/// one of these regardless of how later layers lower it into tape
/// kernels or backend dispatch.
///
/// The closed enum is the serialisation and exhaustive-match surface;
/// per-op semantic facts live on the [`Op`] trait implementations
/// alongside each op's kernel. The methods below forward to those trait
/// impls via `semantic_dispatch!`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum SemanticOp {
    /// Elementwise addition.
    Add,
    /// Elementwise subtraction.
    Sub,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise division.
    Div,
    /// Elementwise power: `c = a^b`.
    Pow,
    /// Elementwise IEEE remainder (`fmodf`).
    Mod,
    /// Elementwise min.
    Min,
    /// Elementwise max.
    Max,
    /// Elementwise equality (1.0 / 0.0).
    Equal,
    /// Elementwise less-than.
    Less,
    /// Elementwise less-or-equal.
    LessOrEqual,
    /// Elementwise greater-than.
    Greater,
    /// Elementwise greater-or-equal.
    GreaterOrEqual,
    /// Elementwise logical AND on f32 truthiness.
    And,
    /// Elementwise logical OR.
    Or,
    /// Elementwise logical XOR.
    Xor,
    /// Logical NOT on f32 truthiness.
    Not,
    /// `1.0` when input is NaN, `0.0` otherwise.
    IsNaN,
    /// Unary negation.
    Neg,
    /// ReLU activation.
    Relu,
    /// GELU activation.
    Gelu,
    /// SiLU activation.
    Silu,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid activation.
    Sigmoid,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Reciprocal.
    Reciprocal,
    /// Cosine.
    Cos,
    /// Sine.
    Sin,
    /// Sign.
    Sign,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round.
    Round,
    /// Error function.
    Erf,
    /// Matrix multiply.
    MatMul(MatMulAttrs),
    /// Softmax along the last axis.
    Softmax(SoftmaxAttrs),
    /// LogSoftmax along the last axis.
    LogSoftmax(SoftmaxAttrs),
    /// RMS normalization.
    RmsNorm(NormAttrs),
    /// Layer normalization.
    LayerNorm(NormAttrs),
    /// Instance normalization.
    InstanceNorm(NormAttrs),
    /// Group normalization.
    GroupNorm(GroupNormAttrs),
    /// Residual add + RMSNorm.
    AddRmsNorm(NormAttrs),
    /// Physical transpose.
    Transpose(TransposeAttrs),
    /// Metadata-only reshape.
    Reshape,
    /// Contiguous single-axis slice.
    Slice(SliceAttrs),
    /// Concatenation along an axis.
    Concat(ConcatAttrs),
    /// 2-D convolution.
    Conv2d(Conv2dAttrs),
    /// Fused SiLU gating semantic.
    FusedSwiGlu,
    /// Sum reduction along the last `size` elements per row.
    ReduceSum(ReduceAttrs),
    /// Mean reduction along the last `size` elements per row.
    ReduceMean(ReduceAttrs),
    /// Max reduction along the last `size` elements per row.
    ReduceMax(ReduceAttrs),
    /// Min reduction along the last `size` elements per row.
    ReduceMin(ReduceAttrs),
    /// Product reduction along the last `size` elements per row.
    ReduceProd(ReduceAttrs),
    /// 2-D max pool.
    MaxPool2d(Pool2dAttrs),
    /// 2-D average pool.
    AvgPool2d(Pool2dAttrs),
    /// Global average pool (collapses spatial dims to `1×1`).
    GlobalAvgPool(GlobalAvgPoolAttrs),
    /// Ternary select: `out = condition ? x : y`.
    Where,
    /// Elementwise clamp.
    Clip(ClipAttrs),
    /// Cumulative sum along the last axis.
    CumSum(CumSumAttrs),
    /// Constant-mode padding (NCHW symmetric).
    Pad(PadAttrs),
    /// Spatial resize (NCHW). Mode on the attrs picks
    /// nearest / linear / cubic; canonical layer ships nearest.
    Resize(ResizeAttrs),
    /// Local Response Normalization (cross-channel).
    Lrn(LrnAttrs),
    /// 2-D transposed convolution.
    ConvTranspose2d(ConvTransposeAttrs),
    /// Generalised matmul: `Y = α·op(A)@op(B) + β·C`.
    Gemm(GemmAttrs),
    /// Broadcast input to a target shape.
    Expand(ExpandAttrs),
    /// Half-rotation rotary embedding.
    RotaryEmbedding(RotaryEmbeddingAttrs),
    /// Scaled dot-product attention (ADR-049).
    Attention(AttentionAttrs),
}

impl SemanticOp {
    /// Number of consumed inputs.
    #[inline]
    #[must_use]
    pub fn arity(&self) -> u8 {
        semantic_dispatch!(self, |op| op.arity())
    }

    /// Number of produced outputs.
    #[inline]
    #[must_use]
    pub fn n_outputs(&self) -> u8 {
        semantic_dispatch!(self, |op| op.n_outputs())
    }

    /// Stable machine-readable name.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &'static str {
        semantic_dispatch!(self, |op| op.name())
    }

    /// Broad semantic category.
    #[inline]
    #[must_use]
    pub fn category(&self) -> OpCategory {
        semantic_dispatch!(self, |op| op.category())
    }

    /// Default backward rule, if any.
    #[inline]
    #[must_use]
    pub fn backward(&self) -> Option<BackwardRule> {
        semantic_dispatch!(self, |op| op.backward())
    }

    /// Whether the op is differentiable.
    #[inline]
    #[must_use]
    pub fn differentiable(&self) -> bool {
        semantic_dispatch!(self, |op| op.differentiable())
    }

    /// Whether the op is layout-only (changes metadata, not values).
    #[inline]
    #[must_use]
    pub fn layout_only(&self) -> bool {
        semantic_dispatch!(self, |op| op.layout_only())
    }

    /// Planner-visible semantic signature.
    #[inline]
    #[must_use]
    pub fn signature(&self) -> OpSignature {
        semantic_dispatch!(self, |op| op.signature())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_op_reports_basic_metadata() {
        assert_eq!(SemanticOp::Add.arity(), 2);
        assert_eq!(SemanticOp::Relu.arity(), 1);
        assert_eq!(SemanticOp::Reshape.name(), "reshape");
        assert_eq!(
            SemanticOp::MatMul(MatMulAttrs { m: 1, k: 2, n: 3 }).name(),
            "matmul"
        );
    }

    #[test]
    fn semantic_op_forwards_to_op_trait() {
        assert_eq!(SemanticOp::Add.arity(), Add.arity());
        assert_eq!(SemanticOp::Add.name(), Add.name());
        assert_eq!(SemanticOp::Add.backward(), Add.backward());

        let attrs = MatMulAttrs { m: 5, k: 7, n: 9 };
        let semantic = SemanticOp::MatMul(attrs);
        let direct = MatMul(attrs);
        assert_eq!(semantic.arity(), direct.arity());
        assert_eq!(semantic.category(), direct.category());
        assert_eq!(semantic.signature(), direct.signature());
        assert_eq!(semantic.backward(), direct.backward());
    }

    #[test]
    fn semantic_op_categories_partition_the_op_set() {
        assert_eq!(SemanticOp::Add.category(), OpCategory::Elementwise);
        assert_eq!(
            SemanticOp::MatMul(MatMulAttrs { m: 1, k: 1, n: 1 }).category(),
            OpCategory::LinearAlgebra
        );
        assert_eq!(
            SemanticOp::Softmax(SoftmaxAttrs { size: 4 }).category(),
            OpCategory::Reduction
        );
        assert_eq!(
            SemanticOp::LayerNorm(NormAttrs {
                size: 4,
                epsilon: 0
            })
            .category(),
            OpCategory::Normalisation
        );
        assert_eq!(SemanticOp::Reshape.category(), OpCategory::Layout);
        assert_eq!(
            SemanticOp::Conv2d(Conv2dAttrs {
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                dilation_h: 1,
                dilation_w: 1,
                group: 1,
                input_h: 8,
                input_w: 8,
            })
            .category(),
            OpCategory::Convolution
        );
        assert_eq!(SemanticOp::FusedSwiGlu.category(), OpCategory::Fused);
    }
}
