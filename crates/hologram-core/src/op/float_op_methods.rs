//! Method implementations for `FloatOp`.
//!
//! Split from `float_op.rs` to keep the enum definition file manageable.
//! This file contains query methods (`arity`, `name`, `category`, etc.)
//! and per-element apply functions (`apply_unary`, `apply_binary`, etc.).

use super::float_op::{bits_to_f32, FloatDType, FloatOp, FloatOpShape};

impl FloatOp {
    /// Number of inputs this operation expects.
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            // Unary
            Self::Neg
            | Self::Relu
            | Self::Gelu
            | Self::Silu
            | Self::Tanh
            | Self::Sigmoid
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Abs
            | Self::Reciprocal
            | Self::Cos
            | Self::Sin
            | Self::Sign
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Erf
            | Self::Clip { .. }
            | Self::IsNaN
            | Self::Not
            | Self::Softmax { .. }
            | Self::LogSoftmax { .. }
            | Self::ReduceSum { .. }
            | Self::ReduceMean { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. }
            | Self::Reshape
            | Self::Transpose { .. }
            | Self::Cast { .. }
            | Self::Shape { .. }
            | Self::Slice { .. }
            | Self::GatherND
            | Self::Dequantize
            | Self::Expand { .. } => 1,

            // Binary
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Mod
            | Self::Min
            | Self::Max
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Equal
            | Self::Less
            | Self::LessOrEqual
            | Self::Greater
            | Self::GreaterOrEqual
            | Self::MatMul { .. }
            | Self::RmsNorm { .. }
            | Self::Gather { .. }
            | Self::Embed { .. }
            | Self::Concat { .. }
            | Self::FusedSwiGLU
            | Self::RotaryEmbedding { .. }
            | Self::Gemm { .. } => 2, // Gemm C bias is optional (arity 2 or 3)

            // Ternary
            Self::AddRmsNorm { .. }
            | Self::LayerNorm { .. }
            | Self::Attention { .. }
            | Self::Where
            | Self::Range
            | Self::ScatterND => 3,

            // Vision ops
            Self::Conv2d { .. } => 3, // data, weight, bias (bias can be zero-length)
            Self::ConvTranspose { .. } => 3,
            Self::MaxPool2d { .. } => 1,
            Self::AvgPool2d { .. } => 1,
            Self::GlobalAvgPool { .. } => 1,
            Self::Resize { .. } => 2,       // data, scales/sizes
            Self::PadOp { .. } => 2,        // data, pads
            Self::InstanceNorm { .. } => 3, // data, scale, bias
            Self::GroupNorm { .. } => 3,    // data, scale, bias
            Self::ArgMax { .. } => 1,
            Self::LRN { .. } => 1,

            // Utility ops
            Self::ReduceProd { .. } => 1,
            Self::TopK { .. } => 2, // data, K
            Self::CumSum { .. } => 1,
            Self::NonZero => 1,
            Self::Compress { .. } => 2, // data, condition
            Self::ReverseSequence { .. } => 1,

            // KV cache
            Self::KvWrite { .. } => 2, // K, V
            Self::KvRead { .. } => 0,  // reads from state, no tensor inputs

            // Deep decode fusions (Plan 054)
            Self::NormProjectionGemv { .. } => 3, // x, norm_weight, proj_weight
            Self::AddNormProjectionGemv { .. } => 4, // x, residual, norm_weight, proj_weight
            Self::SwiGluProjectionGemv { .. } => 3, // gate, up, down_weight
        }
    }

    /// Human-readable name for diagnostics.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Add => "float.add",
            Self::Sub => "float.sub",
            Self::Mul => "float.mul",
            Self::Div => "float.div",
            Self::Pow => "float.pow",
            Self::Mod => "float.mod",
            Self::Min => "float.min",
            Self::Max => "float.max",
            Self::Neg => "float.neg",
            Self::Relu => "float.relu",
            Self::Gelu => "float.gelu",
            Self::Silu => "float.silu",
            Self::Tanh => "float.tanh",
            Self::Sigmoid => "float.sigmoid",
            Self::Exp => "float.exp",
            Self::Log => "float.log",
            Self::Sqrt => "float.sqrt",
            Self::Abs => "float.abs",
            Self::Reciprocal => "float.reciprocal",
            Self::Cos => "float.cos",
            Self::Sin => "float.sin",
            Self::Sign => "float.sign",
            Self::Floor => "float.floor",
            Self::Ceil => "float.ceil",
            Self::Round => "float.round",
            Self::Erf => "float.erf",
            Self::Clip { .. } => "float.clip",
            Self::IsNaN => "float.isnan",
            Self::And => "float.and",
            Self::Or => "float.or",
            Self::Xor => "float.xor",
            Self::Not => "float.not",
            Self::Equal => "float.equal",
            Self::Less => "float.less",
            Self::LessOrEqual => "float.less_or_equal",
            Self::Greater => "float.greater",
            Self::GreaterOrEqual => "float.greater_or_equal",
            Self::MatMul { .. } => "float.matmul",
            Self::Gemm { .. } => "float.gemm",
            Self::Softmax { .. } => "float.softmax",
            Self::LogSoftmax { .. } => "float.log_softmax",
            Self::RmsNorm { .. } => "float.rms_norm",
            Self::AddRmsNorm { .. } => "float.add_rms_norm",
            Self::LayerNorm { .. } => "float.layer_norm",
            Self::ReduceSum { .. } => "float.reduce_sum",
            Self::ReduceMean { .. } => "float.reduce_mean",
            Self::ReduceMax { .. } => "float.reduce_max",
            Self::ReduceMin { .. } => "float.reduce_min",
            Self::Gather { .. } => "float.gather",
            Self::Concat { .. } => "float.concat",
            Self::Reshape => "float.reshape",
            Self::Transpose { .. } => "float.transpose",
            Self::Cast { .. } => "float.cast",
            Self::Embed { .. } => "float.embed",
            Self::Where => "float.where",
            Self::Range => "float.range",
            Self::Shape { .. } => "float.shape",
            Self::Slice { .. } => "float.slice",
            Self::GatherND => "float.gather_nd",
            Self::FusedSwiGLU => "float.fused_swiglu",
            Self::RotaryEmbedding { .. } => "float.rope",
            Self::Attention { .. } => "float.attention",
            Self::Dequantize => "float.dequantize",
            Self::Conv2d { .. } => "float.conv2d",
            Self::ConvTranspose { .. } => "float.conv_transpose",
            Self::MaxPool2d { .. } => "float.max_pool_2d",
            Self::AvgPool2d { .. } => "float.avg_pool_2d",
            Self::GlobalAvgPool { .. } => "float.global_avg_pool",
            Self::Resize { .. } => "float.resize",
            Self::PadOp { .. } => "float.pad",
            Self::InstanceNorm { .. } => "float.instance_norm",
            Self::GroupNorm { .. } => "float.group_norm",
            Self::ArgMax { .. } => "float.argmax",
            Self::LRN { .. } => "float.lrn",
            Self::ReduceProd { .. } => "float.reduce_prod",
            Self::TopK { .. } => "float.top_k",
            Self::ScatterND => "float.scatter_nd",
            Self::CumSum { .. } => "float.cumsum",
            Self::NonZero => "float.nonzero",
            Self::Compress { .. } => "float.compress",
            Self::ReverseSequence { .. } => "float.reverse_sequence",
            Self::KvWrite { .. } => "float.kv_write",
            Self::KvRead { .. } => "float.kv_read",
            Self::NormProjectionGemv { .. } => "float.norm_projection_gemv",
            Self::AddNormProjectionGemv { .. } => "float.add_norm_projection_gemv",
            Self::SwiGluProjectionGemv { .. } => "float.swiglu_projection_gemv",
            Self::Expand { .. } => "float.expand",
        }
    }
}

impl FloatOp {
    /// Symbolic output shape specification.
    ///
    /// Declares how this op's output shape relates to its input shapes.
    /// The executor resolves these specs against actual runtime shapes,
    /// replacing scattered per-op shape logic with a single source of truth.
    ///
    /// IMPORTANT: No `_ =>` catch-all. Adding a new `FloatOp` variant
    /// produces a compiler error until its shape behavior is declared here.
    #[must_use]
    pub fn output_shape_spec(&self) -> super::ShapeSpec {
        use super::ShapeSpec;
        match self {
            // Unary elementwise: output = input[0] shape
            Self::Neg
            | Self::Relu
            | Self::Gelu
            | Self::Silu
            | Self::Tanh
            | Self::Sigmoid
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Abs
            | Self::Reciprocal
            | Self::Cos
            | Self::Sin
            | Self::Sign
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Erf
            | Self::Clip { .. }
            | Self::Not
            | Self::IsNaN
            | Self::Cast { .. } => ShapeSpec::SameAs(0),

            // Dequantize: expands Q4_0 blocks (18 bytes → 32 f32s), output
            // size differs from input. Requires custom shape logic.
            Self::Dequantize => ShapeSpec::Custom,

            // Shape-preserving: output = input[0] shape
            Self::Softmax { .. }
            | Self::LogSoftmax { .. }
            | Self::RmsNorm { .. }
            | Self::AddRmsNorm { .. }
            | Self::LayerNorm { .. }
            | Self::RotaryEmbedding { .. } => ShapeSpec::SameAs(0),

            // Binary elementwise + comparisons: broadcast
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Mod
            | Self::Min
            | Self::Max
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Equal
            | Self::Less
            | Self::LessOrEqual
            | Self::Greater
            | Self::GreaterOrEqual
            | Self::FusedSwiGLU => ShapeSpec::Broadcast(0, 1),

            // Reductions: drop last dim
            Self::ReduceSum { .. }
            | Self::ReduceMean { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => ShapeSpec::DropLastDim(0),

            // Where: output = broadcast of all three inputs (cond, x, y).
            Self::Where => ShapeSpec::BroadcastAll,

            // Range: 1-D output, length inferred from start/limit/delta
            Self::Range => ShapeSpec::inferred_1d(),

            // Ops requiring dedicated logic:
            // Embed/Gather: output = indices_shape ++ [dim], not always 2D.
            Self::Embed { .. }
            | Self::Gather { .. }
            | Self::MatMul { .. }
            | Self::Gemm { .. }
            | Self::Reshape
            | Self::Transpose { .. }
            | Self::Concat { .. }
            | Self::Shape { .. }
            | Self::Slice { .. }
            | Self::Attention { .. }
            | Self::GatherND => ShapeSpec::Custom,

            // Vision/spatial: all need custom shape logic
            Self::Conv2d { .. }
            | Self::ConvTranspose { .. }
            | Self::MaxPool2d { .. }
            | Self::AvgPool2d { .. }
            | Self::GlobalAvgPool { .. }
            | Self::Resize { .. }
            | Self::PadOp { .. }
            | Self::LRN { .. } => ShapeSpec::Custom,

            // Shape-preserving vision ops
            Self::InstanceNorm { .. } | Self::GroupNorm { .. } => ShapeSpec::SameAs(0),
            Self::ArgMax { .. } => ShapeSpec::DropLastDim(0),

            // Utility: custom shape logic
            Self::ReduceProd { .. } => ShapeSpec::DropLastDim(0),
            Self::TopK { .. }
            | Self::ScatterND
            | Self::CumSum { .. }
            | Self::NonZero
            | Self::Compress { .. }
            | Self::ReverseSequence { .. } => ShapeSpec::Custom,

            // KV cache: KvWrite passes K through, KvRead shape is runtime-determined.
            Self::KvWrite { .. } => ShapeSpec::SameAs(0),
            Self::KvRead { .. } => ShapeSpec::Custom,

            // Deep decode fusions: output = [M, n_total] or [M, n], custom shape.
            Self::NormProjectionGemv { .. }
            | Self::AddNormProjectionGemv { .. }
            | Self::SwiGluProjectionGemv { .. } => ShapeSpec::Custom,

            // Expand: output shape is the target_shape (custom).
            Self::Expand { .. } => ShapeSpec::Custom,
        }
    }

    /// Determine the output element type given the input dtypes.
    ///
    /// This is the single source of truth for dtype propagation. The executor
    /// uses it for element-size calculations in shape validation, replacing
    /// the scattered `/ 4` and `* 4` assumptions that caused cascading shape
    /// corruption when non-f32 types (i64, bool) flowed through the graph.
    ///
    /// Rules:
    /// - Unary/binary elementwise f32 ops preserve input dtype (always f32 in practice).
    /// - Comparisons and boolean ops produce `Bool` (1 byte/element).
    /// - Cast explicitly declares its target dtype.
    /// - Gather/Concat carry their element dtype.
    /// - Most other ops (MatMul, Softmax, norms, etc.) produce F32.
    #[must_use]
    pub fn output_dtype(&self, input_dtypes: &[FloatDType]) -> FloatDType {
        let input0 = input_dtypes.first().copied().unwrap_or(FloatDType::F32);
        match self {
            // ── Type-preserving: output dtype = input[0] dtype ──
            Self::Neg
            | Self::Relu
            | Self::Gelu
            | Self::Silu
            | Self::Tanh
            | Self::Sigmoid
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Abs
            | Self::Reciprocal
            | Self::Cos
            | Self::Sin
            | Self::Sign
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Erf
            | Self::Clip { .. }
            | Self::Reshape => input0,

            // ── Binary elementwise: preserve input dtype (broadcast doesn't change type) ──
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Mod
            | Self::Min
            | Self::Max
            | Self::FusedSwiGLU => input0,

            // ── Boolean output: comparisons and logical ops ──
            Self::Equal
            | Self::Less
            | Self::LessOrEqual
            | Self::Greater
            | Self::GreaterOrEqual
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Not
            | Self::IsNaN => FloatDType::Bool,

            // ── Index output (I64) ──
            Self::ArgMax { .. } => FloatDType::I64,

            // ── Explicit type change ──
            Self::Cast { to, .. } => *to,

            // ── Ops that carry their dtype ──
            Self::Gather { dtype, .. } | Self::Concat { dtype, .. } => *dtype,
            Self::Shape { .. } => FloatDType::I64,

            // ── Type-preserving structural ops: output dtype = input[0] dtype ──
            Self::GatherND | Self::Transpose { .. } | Self::Slice { .. } | Self::Expand { .. } => {
                input0
            }

            // Where(condition, true_val, false_val): output dtype = true_val dtype
            Self::Where => input_dtypes.get(1).copied().unwrap_or(input0),

            // ── Reduce ops: preserve input dtype ──
            Self::ReduceSum { .. }
            | Self::ReduceMean { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => input0,

            // ── Float-producing ops ──
            Self::MatMul { .. }
            | Self::Gemm { .. }
            | Self::Softmax { .. }
            | Self::LogSoftmax { .. }
            | Self::RmsNorm { .. }
            | Self::AddRmsNorm { .. }
            | Self::LayerNorm { .. }
            | Self::Embed { .. }
            | Self::Range
            | Self::RotaryEmbedding { .. }
            | Self::Attention { .. }
            | Self::Dequantize
            | Self::Conv2d { .. }
            | Self::ConvTranspose { .. }
            | Self::MaxPool2d { .. }
            | Self::AvgPool2d { .. }
            | Self::GlobalAvgPool { .. }
            | Self::Resize { .. }
            | Self::PadOp { .. }
            | Self::InstanceNorm { .. }
            | Self::GroupNorm { .. }
            | Self::LRN { .. }
            | Self::ReduceProd { .. }
            | Self::CumSum { .. }
            | Self::Compress { .. }
            | Self::ReverseSequence { .. } => FloatDType::F32,

            // ── Type-preserving utility ops ──
            Self::ScatterND => input0,

            // TopK produces F32 values (and I64 indices, but multi-output handled separately)
            Self::TopK { .. } => FloatDType::F32,

            // NonZero produces I64 indices
            Self::NonZero => FloatDType::I64,

            // KV cache: passes through f32 K/V data.
            Self::KvWrite { .. } | Self::KvRead { .. } => FloatDType::F32,

            // Deep decode fusions: always produce F32.
            Self::NormProjectionGemv { .. }
            | Self::AddNormProjectionGemv { .. }
            | Self::SwiGluProjectionGemv { .. } => FloatDType::F32,
        }
    }

    /// Whether this op is a unary element-wise f32→f32 operation
    /// that can be fused into a chain.
    #[must_use]
    pub const fn is_elementwise_unary(&self) -> bool {
        matches!(
            self,
            Self::Neg
                | Self::Relu
                | Self::Gelu
                | Self::Silu
                | Self::Tanh
                | Self::Sigmoid
                | Self::Exp
                | Self::Log
                | Self::Sqrt
                | Self::Abs
                | Self::Reciprocal
                | Self::Cos
                | Self::Sin
                | Self::Sign
                | Self::Floor
                | Self::Ceil
                | Self::Round
                | Self::Erf
                | Self::Clip { .. }
        )
    }

    /// Apply this unary element-wise op to a single f32 value.
    ///
    /// Only valid when `is_elementwise_unary()` returns true.
    /// Panics on non-unary-elementwise ops.
    #[must_use]
    pub fn apply_unary(&self, x: f32) -> f32 {
        match self {
            Self::Neg => -x,
            Self::Relu => x.max(0.0),
            Self::Gelu => {
                0.5 * x
                    * (1.0
                        + libm::tanhf(
                            core::f32::consts::FRAC_2_SQRT_PI
                                * (x + 0.044715 * x * x * x)
                                * core::f32::consts::FRAC_1_SQRT_2,
                        ))
            }
            Self::Silu => x * (1.0 / (1.0 + libm::expf(-x))),
            Self::Tanh => libm::tanhf(x),
            Self::Sigmoid => 1.0 / (1.0 + libm::expf(-x)),
            Self::Exp => libm::expf(x),
            Self::Log => libm::logf(x),
            Self::Sqrt => libm::sqrtf(x),
            Self::Abs => libm::fabsf(x),
            Self::Reciprocal => 1.0 / x,
            Self::Cos => libm::cosf(x),
            Self::Sin => libm::sinf(x),
            Self::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => libm::floorf(x),
            Self::Ceil => libm::ceilf(x),
            Self::Round => libm::roundf(x),
            Self::Erf => {
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                let a = libm::fabsf(x);
                let t = 1.0 / (1.0 + 0.327_591_1 * a);
                let y = 1.0
                    - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74)
                        * t
                        + 0.254_829_6)
                        * t
                        * libm::expf(-a * a);
                sign * y
            }
            Self::Clip { min, max } => x.clamp(bits_to_f32(*min), bits_to_f32(*max)),
            _ => panic!(
                "apply_unary called on non-elementwise-unary op: {}",
                self.name()
            ),
        }
    }
}

impl FloatOp {
    /// Dispatch category — which generic kernel pattern this op uses.
    #[must_use]
    pub const fn category(&self) -> FloatOpShape {
        match self {
            Self::Neg
            | Self::Relu
            | Self::Gelu
            | Self::Silu
            | Self::Tanh
            | Self::Sigmoid
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Abs
            | Self::Reciprocal
            | Self::Cos
            | Self::Sin
            | Self::Sign
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Erf
            | Self::Clip { .. } => FloatOpShape::UnaryElementwise,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Mod
            | Self::Min
            | Self::Max
            | Self::FusedSwiGLU => FloatOpShape::BinaryElementwise,

            Self::Equal | Self::Less | Self::LessOrEqual | Self::Greater | Self::GreaterOrEqual => {
                FloatOpShape::BinaryCompare
            }

            Self::And | Self::Or | Self::Xor => FloatOpShape::BinaryByteBool,
            Self::Not => FloatOpShape::UnaryByteBool,
            Self::IsNaN => FloatOpShape::UnaryToU8,

            Self::Conv2d { .. }
            | Self::ConvTranspose { .. }
            | Self::MaxPool2d { .. }
            | Self::AvgPool2d { .. }
            | Self::GlobalAvgPool { .. }
            | Self::Resize { .. }
            | Self::PadOp { .. }
            | Self::InstanceNorm { .. }
            | Self::GroupNorm { .. }
            | Self::LRN { .. }
            | Self::ReduceProd { .. }
            | Self::TopK { .. }
            | Self::ScatterND
            | Self::CumSum { .. }
            | Self::NonZero
            | Self::Compress { .. }
            | Self::ReverseSequence { .. } => FloatOpShape::Custom,

            _ => FloatOpShape::Custom,
        }
    }

    /// Apply this binary element-wise op to two f32 values.
    ///
    /// Only valid when `category()` returns `BinaryElementwise`.
    /// Panics on other ops.
    #[must_use]
    pub fn apply_binary(&self, a: f32, b: f32) -> f32 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Pow => libm::powf(a, b),
            Self::Mod => a % b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::FusedSwiGLU => {
                // silu(gate) * up  where gate=a, up=b
                let silu_a = a * (1.0 / (1.0 + libm::expf(-a)));
                silu_a * b
            }
            _ => panic!(
                "apply_binary called on non-binary-elementwise op: {}",
                self.name()
            ),
        }
    }

    /// Apply this comparison op to two f32 values.
    ///
    /// Only valid when `category()` returns `BinaryCompare`.
    /// Panics on other ops.
    #[must_use]
    pub fn apply_compare(&self, a: f32, b: f32) -> bool {
        match self {
            Self::Equal => a == b,
            Self::Less => a < b,
            Self::LessOrEqual => a <= b,
            Self::Greater => a > b,
            Self::GreaterOrEqual => a >= b,
            _ => panic!("apply_compare called on non-compare op: {}", self.name()),
        }
    }

    /// Apply this byte-domain boolean op to two bytes.
    ///
    /// Only valid when `category()` returns `BinaryByteBool`.
    #[must_use]
    pub const fn apply_byte_bool(&self, a: u8, b: u8) -> u8 {
        match self {
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
            _ => panic!("apply_byte_bool called on non-byte-bool op"),
        }
    }

    /// Short display name for profiling/diagnostics (e.g. "MatMul", "Relu").
    #[must_use]
    pub const fn short_name(&self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Pow => "Pow",
            Self::Mod => "Mod",
            Self::Min => "Min",
            Self::Max => "Max",
            Self::Neg => "Neg",
            Self::Relu => "Relu",
            Self::Gelu => "Gelu",
            Self::Silu => "Silu",
            Self::Tanh => "Tanh",
            Self::Sigmoid => "Sigmoid",
            Self::Exp => "Exp",
            Self::Log => "Log",
            Self::Sqrt => "Sqrt",
            Self::Abs => "Abs",
            Self::Reciprocal => "Reciprocal",
            Self::Cos => "Cos",
            Self::Sin => "Sin",
            Self::Sign => "Sign",
            Self::Floor => "Floor",
            Self::Ceil => "Ceil",
            Self::Round => "Round",
            Self::Erf => "Erf",
            Self::Clip { .. } => "Clip",
            Self::IsNaN => "IsNaN",
            Self::And => "And",
            Self::Or => "Or",
            Self::Xor => "Xor",
            Self::Not => "Not",
            Self::Equal => "Equal",
            Self::Less => "Less",
            Self::LessOrEqual => "LessOrEqual",
            Self::Greater => "Greater",
            Self::GreaterOrEqual => "GreaterOrEqual",
            Self::MatMul { .. } => "MatMul",
            Self::Gemm { .. } => "Gemm",
            Self::Softmax { .. } => "Softmax",
            Self::LogSoftmax { .. } => "LogSoftmax",
            Self::RmsNorm { .. } => "RmsNorm",
            Self::AddRmsNorm { .. } => "AddRmsNorm",
            Self::LayerNorm { .. } => "LayerNorm",
            Self::ReduceSum { .. } => "ReduceSum",
            Self::ReduceMean { .. } => "ReduceMean",
            Self::ReduceMax { .. } => "ReduceMax",
            Self::ReduceMin { .. } => "ReduceMin",
            Self::Gather { .. } => "Gather",
            Self::Concat { .. } => "Concat",
            Self::Reshape => "Reshape",
            Self::Transpose { .. } => "Transpose",
            Self::Cast { .. } => "Cast",
            Self::Embed { .. } => "Embed",
            Self::Where => "Where",
            Self::Range => "Range",
            Self::Shape { .. } => "Shape",
            Self::Slice { .. } => "Slice",
            Self::GatherND => "GatherND",
            Self::FusedSwiGLU => "SwiGLU",
            Self::RotaryEmbedding { .. } => "RoPE",
            Self::Attention { .. } => "Attention",
            Self::Dequantize => "Dequantize",
            Self::Conv2d { .. } => "Conv2d",
            Self::ConvTranspose { .. } => "ConvTranspose",
            Self::MaxPool2d { .. } => "MaxPool2d",
            Self::AvgPool2d { .. } => "AvgPool2d",
            Self::GlobalAvgPool { .. } => "GlobalAvgPool",
            Self::Resize { .. } => "Resize",
            Self::PadOp { .. } => "Pad",
            Self::InstanceNorm { .. } => "InstanceNorm",
            Self::GroupNorm { .. } => "GroupNorm",
            Self::LRN { .. } => "LRN",
            Self::ReduceProd { .. } => "ReduceProd",
            Self::TopK { .. } => "TopK",
            Self::ScatterND => "ScatterND",
            Self::CumSum { .. } => "CumSum",
            Self::NonZero => "NonZero",
            Self::Compress { .. } => "Compress",
            Self::ReverseSequence { .. } => "ReverseSequence",
            Self::KvWrite { .. } => "KvWrite",
            Self::KvRead { .. } => "KvRead",
            Self::ArgMax { .. } => "ArgMax",
            Self::NormProjectionGemv { .. } => "NormProjGemv",
            Self::AddNormProjectionGemv { .. } => "AddNormProjGemv",
            Self::SwiGluProjectionGemv { .. } => "SwiGluProjGemv",
            Self::Expand { .. } => "Expand",
        }
    }
}
