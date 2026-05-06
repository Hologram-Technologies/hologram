//! `OpKind` — the closed catalog of canonical hologram operations.
//!
//! One variant per `Grounding`-equivalent op marker in the rest of this
//! crate. Adding an op = (a) define a marker type in the right module,
//! (b) add a variant here, (c) wire the compiler dispatch arm.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum OpKind {
    // Direct PrimitiveOp wrappers (spec V.3)
    Neg, Bnot, Succ, Pred, Add, Sub, Mul, Xor, And, Or,

    // Elementwise unary
    Relu, Sigmoid, Tanh, Gelu, Silu, Elu, Selu,
    Exp, Log, Log1p, Sqrt, Reciprocal,
    Sin, Cos, Tan, Asin, Acos, Atan,
    Ceil, Floor, Round, Erf,
    IsNaN, Sign, Abs,

    // Elementwise binary (non-primitive)
    Div, Pow, Mod, Min, Max,
    Equal, Less, LessOrEqual, Greater, GreaterOrEqual,

    // Linear algebra
    MatMul, Gemm,

    // Convolution
    Conv2d, ConvTranspose2d,

    // Normalization
    LayerNorm, RmsNorm, GroupNorm, InstanceNorm, AddRmsNorm,

    // Reduction
    ReduceSum, ReduceMean, ReduceProd, ReduceMin, ReduceMax,

    // Layout (no compute)
    Reshape, Transpose, Concat, Slice,

    // Activation+reduce
    Softmax, LogSoftmax,

    // Pooling
    MaxPool2d, AvgPool2d, GlobalAvgPool,

    // Structured
    Attention, FusedSwiGlu,

    // Utility
    Pad, Expand, Resize, CumSum, RotaryEmbedding, Clip, Lrn, Where,

    // Backward variants (spec V.4) — one per differentiable op.
    MatMulGradA, MatMulGradB,
    Conv2dGradX, Conv2dGradW,
    SoftmaxGrad, LogSoftmaxGrad,
    LayerNormGrad, RmsNormGrad, GroupNormGrad,
    ReduceSumGrad, ReduceMeanGrad, ReduceProdGrad,
    SubGrad, MulGrad, DivGrad, PowGrad,
    MinGrad, MaxGrad,
    ConcatGrad, SliceGrad,
    AvgPool2dGrad, GlobalAvgPoolGrad,
    PadGrad,
    AttentionGrad,
    FusedSwiGluGrad,
    UnaryGrad,
}

impl OpKind {
    /// Stable human-readable name (lowercase snake_case).
    pub const fn name(self) -> &'static str {
        use OpKind::*;
        match self {
            Neg => "neg", Bnot => "bnot", Succ => "succ", Pred => "pred",
            Add => "add", Sub => "sub", Mul => "mul", Xor => "xor",
            And => "and", Or => "or",
            Relu => "relu", Sigmoid => "sigmoid", Tanh => "tanh",
            Gelu => "gelu", Silu => "silu", Elu => "elu", Selu => "selu",
            Exp => "exp", Log => "log", Log1p => "log1p", Sqrt => "sqrt",
            Reciprocal => "reciprocal",
            Sin => "sin", Cos => "cos", Tan => "tan",
            Asin => "asin", Acos => "acos", Atan => "atan",
            Ceil => "ceil", Floor => "floor", Round => "round", Erf => "erf",
            IsNaN => "is_nan", Sign => "sign", Abs => "abs",
            Div => "div", Pow => "pow", Mod => "mod", Min => "min", Max => "max",
            Equal => "equal", Less => "less", LessOrEqual => "less_or_equal",
            Greater => "greater", GreaterOrEqual => "greater_or_equal",
            MatMul => "matmul", Gemm => "gemm",
            Conv2d => "conv2d", ConvTranspose2d => "conv_transpose_2d",
            LayerNorm => "layer_norm", RmsNorm => "rms_norm",
            GroupNorm => "group_norm", InstanceNorm => "instance_norm",
            AddRmsNorm => "add_rms_norm",
            ReduceSum => "reduce_sum", ReduceMean => "reduce_mean",
            ReduceProd => "reduce_prod", ReduceMin => "reduce_min",
            ReduceMax => "reduce_max",
            Reshape => "reshape", Transpose => "transpose",
            Concat => "concat", Slice => "slice",
            Softmax => "softmax", LogSoftmax => "log_softmax",
            MaxPool2d => "max_pool_2d", AvgPool2d => "avg_pool_2d",
            GlobalAvgPool => "global_avg_pool",
            Attention => "attention", FusedSwiGlu => "fused_swiglu",
            Pad => "pad", Expand => "expand", Resize => "resize",
            CumSum => "cumsum", RotaryEmbedding => "rotary_embedding",
            Clip => "clip", Lrn => "lrn", Where => "where",
            MatMulGradA => "matmul_grad_a", MatMulGradB => "matmul_grad_b",
            Conv2dGradX => "conv2d_grad_x", Conv2dGradW => "conv2d_grad_w",
            SoftmaxGrad => "softmax_grad", LogSoftmaxGrad => "log_softmax_grad",
            LayerNormGrad => "layer_norm_grad", RmsNormGrad => "rms_norm_grad",
            GroupNormGrad => "group_norm_grad",
            ReduceSumGrad => "reduce_sum_grad", ReduceMeanGrad => "reduce_mean_grad",
            ReduceProdGrad => "reduce_prod_grad",
            SubGrad => "sub_grad", MulGrad => "mul_grad", DivGrad => "div_grad",
            PowGrad => "pow_grad",
            MinGrad => "min_grad", MaxGrad => "max_grad",
            ConcatGrad => "concat_grad", SliceGrad => "slice_grad",
            AvgPool2dGrad => "avg_pool_2d_grad", GlobalAvgPoolGrad => "global_avg_pool_grad",
            PadGrad => "pad_grad",
            AttentionGrad => "attention_grad",
            FusedSwiGluGrad => "fused_swiglu_grad",
            UnaryGrad => "unary_grad",
        }
    }

    /// Whether this op is a layout-only operation (no compute Term).
    pub const fn is_layout_only(self) -> bool {
        matches!(self,
            OpKind::Reshape | OpKind::Transpose | OpKind::Concat | OpKind::Slice
            | OpKind::Pad | OpKind::Expand
        )
    }

    /// Whether this op is a direct `PrimitiveOp` wrapper.
    pub const fn is_direct(self) -> bool {
        matches!(self,
            OpKind::Neg | OpKind::Bnot | OpKind::Succ | OpKind::Pred
            | OpKind::Add | OpKind::Sub | OpKind::Mul
            | OpKind::Xor | OpKind::And | OpKind::Or
        )
    }

    /// Anchoring `PrimitiveOp` of this op's Term-tree decomposition.
    /// Per spec V.3, every op marker exposes a `PRIMARY_OP` (or `PRIMITIVE`
    /// for direct wrappers); this function consolidates those values into a
    /// single OpKind-keyed table consumed by `hologram-compiler`.
    pub const fn primary_primitive(self) -> uor_foundation::PrimitiveOp {
        use OpKind as K;
        use uor_foundation::PrimitiveOp as P;
        match self {
            // Direct PrimitiveOp wrappers — anchor is the op itself.
            K::Neg => P::Neg, K::Bnot => P::Bnot, K::Succ => P::Succ, K::Pred => P::Pred,
            K::Add => P::Add, K::Sub => P::Sub, K::Mul => P::Mul,
            K::Xor => P::Xor, K::And => P::And, K::Or => P::Or,

            // Elementwise unary anchors (mirrors hologram_ops::elementwise_unary
            // declarations).
            K::Relu | K::Abs | K::Sign | K::IsNaN | K::Ceil | K::Floor => P::And,
            K::Round | K::Sin | K::Cos | K::Log1p => P::Add,
            K::Sigmoid | K::Tanh | K::Gelu | K::Silu | K::Elu | K::Selu
                | K::Exp | K::Log | K::Sqrt | K::Reciprocal
                | K::Tan | K::Asin | K::Acos | K::Atan | K::Erf => P::Mul,

            // Elementwise binary.
            K::Div | K::Pow | K::Equal => P::Mul,
            K::Mod | K::Min | K::Max
                | K::Less | K::LessOrEqual | K::Greater | K::GreaterOrEqual => P::Sub,

            // Linear algebra / convolution.
            K::MatMul | K::Gemm | K::Conv2d | K::ConvTranspose2d
                | K::Attention | K::FusedSwiGlu => P::Mul,

            // Normalization / softmax / structured.
            K::LayerNorm | K::RmsNorm | K::GroupNorm | K::InstanceNorm | K::AddRmsNorm
                | K::Softmax | K::LogSoftmax
                | K::Lrn | K::RotaryEmbedding | K::Resize => P::Mul,

            // Reductions.
            K::ReduceSum | K::ReduceMean | K::CumSum | K::AvgPool2d | K::GlobalAvgPool
                | K::ConcatGrad | K::SliceGrad | K::PadGrad
                | K::AvgPool2dGrad | K::GlobalAvgPoolGrad
                | K::ReduceSumGrad => P::Add,
            K::ReduceProd | K::ReduceMin | K::ReduceMax | K::MaxPool2d
                | K::ReduceMeanGrad | K::ReduceProdGrad => P::Mul,

            K::Clip => P::And,
            K::Where => P::Or,

            // Layout (no-compute) — anchor at the identity-equivalent And.
            K::Reshape | K::Transpose | K::Concat | K::Slice
                | K::Pad | K::Expand => P::And,

            // Backward variants.
            K::MatMulGradA | K::MatMulGradB
                | K::Conv2dGradX | K::Conv2dGradW
                | K::SoftmaxGrad | K::LayerNormGrad | K::RmsNormGrad | K::GroupNormGrad
                | K::MulGrad | K::DivGrad | K::PowGrad
                | K::AttentionGrad | K::FusedSwiGluGrad | K::UnaryGrad => P::Mul,
            K::LogSoftmaxGrad | K::SubGrad => P::Sub,
            K::MinGrad | K::MaxGrad => P::And,
        }
    }

    /// Arity of this op's primary application (1 or 2 or 3).
    pub const fn primary_arity(self) -> u8 {
        use OpKind as K;
        match self {
            // Unary forms.
            K::Neg | K::Bnot | K::Succ | K::Pred
                | K::Relu | K::Sigmoid | K::Tanh | K::Gelu | K::Silu | K::Elu | K::Selu
                | K::Exp | K::Log | K::Log1p | K::Sqrt | K::Reciprocal
                | K::Sin | K::Cos | K::Tan | K::Asin | K::Acos | K::Atan
                | K::Ceil | K::Floor | K::Round | K::Erf
                | K::IsNaN | K::Sign | K::Abs
                | K::ReduceSum | K::ReduceMean | K::ReduceProd | K::ReduceMin | K::ReduceMax
                | K::Softmax | K::LogSoftmax
                | K::MaxPool2d | K::AvgPool2d | K::GlobalAvgPool
                | K::Resize | K::CumSum | K::RotaryEmbedding | K::Clip | K::Lrn
                | K::Reshape | K::Transpose | K::Slice | K::Pad | K::Expand
                | K::SoftmaxGrad | K::LogSoftmaxGrad
                | K::ReduceSumGrad | K::ReduceMeanGrad | K::ReduceProdGrad
                | K::AvgPool2dGrad | K::GlobalAvgPoolGrad
                | K::SliceGrad | K::PadGrad | K::UnaryGrad => 1,

            // Binary forms.
            K::Add | K::Sub | K::Mul | K::Xor | K::And | K::Or
                | K::Div | K::Pow | K::Mod | K::Min | K::Max
                | K::Equal | K::Less | K::LessOrEqual | K::Greater | K::GreaterOrEqual
                | K::SubGrad | K::MulGrad | K::DivGrad | K::PowGrad
                | K::MinGrad | K::MaxGrad
                | K::ConcatGrad | K::Concat
                | K::MatMul | K::Conv2d | K::ConvTranspose2d
                | K::FusedSwiGlu | K::AddRmsNorm
                | K::MatMulGradA | K::MatMulGradB
                | K::Conv2dGradX | K::Conv2dGradW => 2,

            // Ternary forms.
            K::Gemm | K::LayerNorm | K::RmsNorm | K::GroupNorm | K::InstanceNorm
                | K::Attention | K::AttentionGrad | K::FusedSwiGluGrad
                | K::LayerNormGrad | K::RmsNormGrad | K::GroupNormGrad
                | K::Where => 3,
        }
    }
}
