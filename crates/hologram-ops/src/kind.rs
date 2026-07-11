//! `OpKind` — the closed catalog of canonical hologram operations.
//!
//! One variant per `Grounding`-equivalent op marker in the rest of this
//! crate. Adding an op = (a) define a marker type in the right module,
//! (b) add one catalog entry here, (c) wire the compiler dispatch arm.

macro_rules! op_kind_catalog {
    ($($(#[$meta:meta])* $variant:ident => $name:literal,)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u16)]
        pub enum OpKind {
            $(
                $(#[$meta])*
                $variant,
            )+
        }

        impl OpKind {
            /// Closed catalog of canonical operation kinds.
            ///
            /// Keep this list in enum declaration order. Parser frontends,
            /// dispatch coverage, and catalog-size tests should use this
            /// constant instead of carrying local copies.
            pub const ALL: &'static [Self] = &[
                $(Self::$variant,)+
            ];

            /// Stable human-readable name (lowercase snake_case).
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }
        }
    };
}

op_kind_catalog! {
    // Direct PrimitiveOp wrappers (spec V.3)
    Neg => "neg",
    Bnot => "bnot",
    Succ => "succ",
    Pred => "pred",
    Add => "add",
    Sub => "sub",
    Mul => "mul",
    Xor => "xor",
    And => "and",
    Or => "or",

    // Elementwise unary
    Relu => "relu",
    Sigmoid => "sigmoid",
    Tanh => "tanh",
    Gelu => "gelu",
    Silu => "silu",
    Elu => "elu",
    Selu => "selu",
    Exp => "exp",
    Log => "log",
    Log1p => "log1p",
    Sqrt => "sqrt",
    Reciprocal => "reciprocal",
    Sin => "sin",
    Cos => "cos",
    Tan => "tan",
    Asin => "asin",
    Acos => "acos",
    Atan => "atan",
    Ceil => "ceil",
    Floor => "floor",
    Round => "round",
    Erf => "erf",
    IsNaN => "is_nan",
    Sign => "sign",
    Abs => "abs",

    // Elementwise binary (non-primitive)
    Div => "div",
    Pow => "pow",
    Mod => "mod",
    Min => "min",
    Max => "max",
    Equal => "equal",
    Less => "less",
    LessOrEqual => "less_or_equal",
    Greater => "greater",
    GreaterOrEqual => "greater_or_equal",

    // Linear algebra
    MatMul => "matmul",
    Gemm => "gemm",

    // Convolution
    Conv2d => "conv2d",
    ConvTranspose2d => "conv_transpose_2d",
    /// im2col: gather a conv's receptive-field patches into a `[Cin·kh·kw,
    /// Hout·Wout]` matrix (a pure layout gather). Lets a convolution be
    /// expressed as `W · im2col(x)`, and its gradients composed from the
    /// matmul VJP. Single-instance (rank-3 `[Cin,Hin,Win]` → rank-2).
    Im2Col => "im2col",
    /// col2im: the adjoint of [`Im2Col`](Self::Im2Col) — scatter-add a `[Cin·kh·kw,
    /// Hout·Wout]` patch matrix back into a `[Cin,Hin,Win]` image (overlapping
    /// windows accumulate). The input-gradient half of conv composition.
    Col2Im => "col2im",

    // Normalization
    LayerNorm => "layer_norm",
    RmsNorm => "rms_norm",
    GroupNorm => "group_norm",
    InstanceNorm => "instance_norm",
    AddRmsNorm => "add_rms_norm",

    // Reduction
    ReduceSum => "reduce_sum",
    ReduceMean => "reduce_mean",
    ReduceProd => "reduce_prod",
    ReduceMin => "reduce_min",
    ReduceMax => "reduce_max",

    // Layout (no compute)
    Reshape => "reshape",
    Transpose => "transpose",
    Concat => "concat",
    Slice => "slice",

    // Activation+reduce
    Softmax => "softmax",
    LogSoftmax => "log_softmax",

    // Pooling
    MaxPool2d => "max_pool_2d",
    AvgPool2d => "avg_pool_2d",
    GlobalAvgPool => "global_avg_pool",

    // Structured
    Attention => "attention",
    FusedSwiGlu => "fused_swiglu",

    // Utility
    Pad => "pad",
    Expand => "expand",
    Resize => "resize",
    CumSum => "cumsum",
    RotaryEmbedding => "rotary_embedding",
    Clip => "clip",
    Lrn => "lrn",
    Where => "where",
    /// Gather rows of `data` along `axis` selected by a runtime integer
    /// `indices` operand (ONNX Gather / embedding lookup): `out[…,i,…] =
    /// data[…,indices[i],…]`. A pure data-movement map like [`Im2Col`](Self::Im2Col),
    /// but the index permutation is a runtime operand rather than fixed
    /// geometry — so it is layout-only (no arithmetic Term) and its numeric
    /// contract is the kernel's, V&V'd against the ONNX spec and the
    /// `OneHot·MatMul` reference it replaces (`O(outer·idx·inner)` indexed
    /// copy vs the one-hot matmul's `O(outer·idx·axis·inner)`).
    Gather => "gather",

    // Numeric conversion
    /// Numeric dtype conversion (ONNX `Cast`): the abstract value is preserved
    /// while the representation changes — int↔float, float↔float width, and
    /// int↔int width (float→int truncates toward zero). The formal spec is the
    /// identity on values (`y = x`); the per-dtype byte conversion is the
    /// kernel's contract (V&V'd against ONNX Cast), exactly as `Dequantize`'s
    /// spec is the affine chain while the widening is the kernel's. This is the
    /// general int→float primitive — distinct from `Dequantize`, which decodes
    /// a *quantized* value with scale/zero-point.
    Cast => "cast",

    // Quantization (spec X-5)
    Dequantize => "dequantize",

    // KV-cache movement (decode)
    /// Write the rows of a `new` operand into a fixed-bucket KV cache at a
    /// runtime write position (ring wrap): `out[p][(pos+j) % bucket] =
    /// new[p][j]`, all other rows unchanged. Pure data movement like
    /// [`Gather`](Self::Gather) — the position is a runtime operand, not
    /// fixed geometry, so it is layout-only (no arithmetic Term) and the
    /// byte contract is the kernel's. The executor may realize it as an
    /// in-place move on a resident cache (the old cache label is *consumed*);
    /// the fallback is an honest copy — bit-identical either way.
    KvCacheWrite => "kv_cache_write",
}

impl OpKind {
    /// Whether this op is a layout-only operation (no compute Term).
    pub const fn is_layout_only(self) -> bool {
        matches!(
            self,
            OpKind::Reshape
                | OpKind::Transpose
                | OpKind::Concat
                | OpKind::Slice
                | OpKind::Pad
                | OpKind::Expand
                | OpKind::Im2Col
                | OpKind::Col2Im
                | OpKind::Gather
                | OpKind::KvCacheWrite
        )
    }

    /// Whether this op is a direct `PrimitiveOp` wrapper.
    pub const fn is_direct(self) -> bool {
        matches!(
            self,
            OpKind::Neg
                | OpKind::Bnot
                | OpKind::Succ
                | OpKind::Pred
                | OpKind::Add
                | OpKind::Sub
                | OpKind::Mul
                | OpKind::Xor
                | OpKind::And
                | OpKind::Or
        )
    }

    /// Anchoring `PrimitiveOp` of this op's Term-tree decomposition.
    /// Per spec V.3, every op marker exposes a `PRIMARY_OP` (or `PRIMITIVE`
    /// for direct wrappers); this function consolidates those values into a
    /// single OpKind-keyed table consumed by `hologram-compiler`.
    pub const fn primary_primitive(self) -> uor_foundation::PrimitiveOp {
        use uor_foundation::PrimitiveOp as P;
        use OpKind as K;
        match self {
            // Direct PrimitiveOp wrappers — anchor is the op itself.
            K::Neg => P::Neg,
            K::Bnot => P::Bnot,
            K::Succ => P::Succ,
            K::Pred => P::Pred,
            K::Add => P::Add,
            K::Sub => P::Sub,
            K::Mul => P::Mul,
            K::Xor => P::Xor,
            K::And => P::And,
            K::Or => P::Or,

            // Elementwise unary anchors (mirrors hologram_ops::elementwise_unary
            // declarations).
            K::Relu | K::Abs | K::Sign | K::IsNaN | K::Ceil | K::Floor => P::And,
            K::Round | K::Sin | K::Cos | K::Log1p => P::Add,
            K::Sigmoid
            | K::Tanh
            | K::Gelu
            | K::Silu
            | K::Elu
            | K::Selu
            | K::Exp
            | K::Log
            | K::Sqrt
            | K::Reciprocal
            | K::Tan
            | K::Asin
            | K::Acos
            | K::Atan
            | K::Erf => P::Mul,

            // Elementwise binary.
            K::Div | K::Pow | K::Equal => P::Mul,
            K::Mod
            | K::Min
            | K::Max
            | K::Less
            | K::LessOrEqual
            | K::Greater
            | K::GreaterOrEqual => P::Sub,

            // Linear algebra / convolution.
            K::MatMul
            | K::Gemm
            | K::Conv2d
            | K::ConvTranspose2d
            | K::Attention
            | K::FusedSwiGlu => P::Mul,

            // Normalization / softmax / structured.
            K::LayerNorm
            | K::RmsNorm
            | K::GroupNorm
            | K::InstanceNorm
            | K::AddRmsNorm
            | K::Softmax
            | K::LogSoftmax
            | K::Lrn
            | K::RotaryEmbedding
            | K::Resize => P::Mul,

            // Reductions.
            K::ReduceSum | K::ReduceMean | K::CumSum | K::AvgPool2d | K::GlobalAvgPool => P::Add,
            K::ReduceProd | K::ReduceMin | K::ReduceMax | K::MaxPool2d => P::Mul,

            K::Clip => P::And,
            K::Where => P::Or,
            K::Cast => P::Mul,
            K::Dequantize => P::Mul,

            // Layout (no-compute) — anchor at the identity-equivalent And.
            K::Reshape | K::Transpose | K::Concat | K::Slice | K::Pad | K::Expand => P::And,
            K::Im2Col | K::Col2Im | K::Gather | K::KvCacheWrite => P::And,
        }
    }

    /// Maximum Term-arena slot count this op's `emit_term` may occupy
    /// (spec V.5). The compiler-side arena is sized at the maximum CAP
    /// across the catalog (currently 96 for `Attention`); this function
    /// exposes the per-op upper bound for verification tests
    /// (`tests/dispatch_coverage.rs`).
    pub const fn cap(self) -> usize {
        use OpKind as K;
        match self {
            K::Neg
            | K::Bnot
            | K::Succ
            | K::Pred
            | K::Add
            | K::Sub
            | K::Mul
            | K::Xor
            | K::And
            | K::Or => 4,

            K::Reshape | K::Transpose | K::Concat | K::Slice | K::Pad | K::Expand => 2,
            K::Im2Col | K::Col2Im | K::Gather | K::KvCacheWrite => 2,

            K::Equal
            | K::Less
            | K::LessOrEqual
            | K::Greater
            | K::GreaterOrEqual
            | K::Mod
            | K::Min
            | K::Max
            | K::IsNaN
            | K::Sign
            | K::Abs => 16,

            K::ReduceSum
            | K::ReduceMean
            | K::ReduceProd
            | K::ReduceMin
            | K::ReduceMax
            | K::Clip
            | K::Where => 16,

            K::Relu
            | K::Sigmoid
            | K::Tanh
            | K::Silu
            | K::Elu
            | K::Selu
            | K::Ceil
            | K::Floor
            | K::Round
            | K::Softmax
            | K::LogSoftmax
            | K::MaxPool2d
            | K::AvgPool2d
            | K::GlobalAvgPool
            | K::CumSum
            | K::Resize
            | K::Div
            | K::MatMul
            | K::Gemm => 32,

            K::Gelu
            | K::Exp
            | K::Log
            | K::Log1p
            | K::Sqrt
            | K::Reciprocal
            | K::Sin
            | K::Cos
            | K::Tan
            | K::Asin
            | K::Acos
            | K::Atan
            | K::Erf
            | K::Pow
            | K::Conv2d
            | K::ConvTranspose2d
            | K::LayerNorm
            | K::RmsNorm
            | K::GroupNorm
            | K::InstanceNorm
            | K::AddRmsNorm
            | K::FusedSwiGlu
            | K::Lrn
            | K::RotaryEmbedding => 64,

            K::Attention => 96,

            K::Dequantize => 8,

            K::Cast => 4,
        }
    }

    /// Arity of this op's primary application (1 or 2 or 3).
    pub const fn primary_arity(self) -> u8 {
        use OpKind as K;
        match self {
            // Unary forms.
            K::Neg
            | K::Bnot
            | K::Succ
            | K::Pred
            | K::Relu
            | K::Sigmoid
            | K::Tanh
            | K::Gelu
            | K::Silu
            | K::Elu
            | K::Selu
            | K::Exp
            | K::Log
            | K::Log1p
            | K::Sqrt
            | K::Reciprocal
            | K::Sin
            | K::Cos
            | K::Tan
            | K::Asin
            | K::Acos
            | K::Atan
            | K::Ceil
            | K::Floor
            | K::Round
            | K::Erf
            | K::IsNaN
            | K::Sign
            | K::Abs
            | K::ReduceSum
            | K::ReduceMean
            | K::ReduceProd
            | K::ReduceMin
            | K::ReduceMax
            | K::Softmax
            | K::LogSoftmax
            | K::MaxPool2d
            | K::AvgPool2d
            | K::GlobalAvgPool
            | K::Resize
            | K::CumSum
            | K::Clip
            | K::Lrn
            | K::Reshape
            | K::Transpose
            | K::Slice
            | K::Pad
            | K::Expand
            | K::Im2Col
            | K::Col2Im
            | K::Cast
            | K::Dequantize => 1,

            // Binary forms.
            K::Add
            | K::Sub
            | K::Mul
            | K::Xor
            | K::And
            | K::Or
            | K::Div
            | K::Pow
            | K::Mod
            | K::Min
            | K::Max
            | K::Equal
            | K::Less
            | K::LessOrEqual
            | K::Greater
            | K::GreaterOrEqual
            | K::Concat
            | K::MatMul
            | K::Conv2d
            | K::ConvTranspose2d
            | K::FusedSwiGlu
            | K::AddRmsNorm
            // Gather(data, indices): the runtime index operand is the 2nd arg.
            | K::Gather => 2,

            // Ternary forms.
            K::Gemm
            | K::LayerNorm
            | K::RmsNorm
            | K::GroupNorm
            | K::InstanceNorm
            | K::Attention
            | K::Where
            // RoPE(x, cos, sin): the rotation tables are operands.
            | K::RotaryEmbedding
            // KvCacheWrite(cache, new, pos): the write position is an operand.
            | K::KvCacheWrite => 3,
        }
    }
}
