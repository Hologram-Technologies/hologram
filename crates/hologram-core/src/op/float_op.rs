//! FloatOp: typed tensor operations for AI model inference.
//!
//! Unlike `PrimOp` (Z/256Z ring arithmetic) and `LutOp` (byte-domain activation
//! tables), `FloatOp` operates on f32 buffers with shape-aware semantics.
//! Each variant carries the shape parameters needed for dispatch, since the
//! graph IR has no per-edge shape metadata.

/// Element data type for dtype-aware float ops (Cast, Shape).
///
/// Stored in `.holo` archives — must remain rkyv-serializable and `#[repr(u8)]`
/// for compact encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[repr(u8)]
pub enum FloatDType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
    F16 = 4,
    BF16 = 5,
    U8 = 6,
    Bool = 7,
    I8 = 8,
}

impl FloatDType {
    /// Number of bytes per element.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::U8 | Self::Bool | Self::I8 => 1,
        }
    }
}

/// Float-domain tensor operations for AI inference.
///
/// Serialized into `.holo` archives alongside `PrimOp`/`LutOp` ops.
/// Dispatched by `KvStore` at execution time — no `CustomOpRegistry` needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub enum FloatOp {
    // ── Arithmetic (binary, element-wise with broadcast) ──────────────────
    /// f32 addition with broadcast: out[i] = a[i] + b[i % b.len()]
    Add,
    /// f32 subtraction with broadcast.
    Sub,
    /// f32 multiplication with broadcast.
    Mul,
    /// f32 division with broadcast.
    Div,
    /// f32 power with broadcast: out[i] = a[i] ^ b[i % b.len()]
    Pow,
    /// f32 modulo with broadcast.
    Mod,
    /// Element-wise minimum with broadcast.
    Min,
    /// Element-wise maximum with broadcast.
    Max,

    // ── Unary activations ─────────────────────────────────────────────────
    /// Negation: out[i] = -x[i]
    Neg,
    /// ReLU: out[i] = max(0, x[i])
    Relu,
    /// GELU (approximate): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    Gelu,
    /// SiLU (Swish): x * sigmoid(x)
    Silu,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Reciprocal: 1/x.
    Reciprocal,

    // ── Unary math ────────────────────────────────────────────────────────
    /// Cosine.
    Cos,
    /// Sine.
    Sin,
    /// Sign: -1, 0, or 1.
    Sign,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round to nearest.
    Round,
    /// Gauss error function (Abramowitz & Stegun approximation).
    Erf,
    /// Clamp to [min, max]. min/max stored as f32 bits.
    Clip { min: u32, max: u32 },
    /// Test for NaN: output is u8 (0 or 1).
    IsNaN,

    // ── Boolean / comparison ops ──────────────────────────────────────────
    // These operate on byte buffers. Boolean ops treat nonzero as true.
    // Comparisons interpret inputs as f32 and produce u8 output.
    /// Logical AND (byte-wise, nonzero = true).
    And,
    /// Logical OR.
    Or,
    /// Logical XOR.
    Xor,
    /// Logical NOT (unary).
    Not,
    /// f32 equality comparison → u8.
    Equal,
    /// f32 less-than → u8.
    Less,
    /// f32 less-or-equal → u8.
    LessOrEqual,
    /// f32 greater-than → u8.
    Greater,
    /// f32 greater-or-equal → u8.
    GreaterOrEqual,

    // ── Linear algebra ────────────────────────────────────────────────────
    /// Matrix multiply: [m, k] × [k, n] → [m, n].
    /// Inputs: [a (f32), b (f32)]. Both row-major.
    MatMul { m: u32, k: u32, n: u32 },

    /// General matrix multiply with alpha, beta, transpose flags.
    /// out = alpha * op(A) × op(B) + beta * C.
    /// Inputs: [A (f32), B (f32), C (f32)].
    Gemm {
        m: u32,
        k: u32,
        n: u32,
        alpha: u32,
        beta: u32,
        trans_a: bool,
        trans_b: bool,
    },

    // ── Softmax ───────────────────────────────────────────────────────────
    /// Softmax along last `size` elements of each row.
    Softmax { size: u32 },

    /// LogSoftmax along last `size` elements of each row.
    LogSoftmax { size: u32 },

    // ── Normalization ─────────────────────────────────────────────────────
    /// RMS normalization. Inputs: [x (f32), weight (f32)].
    RmsNorm { size: u32, epsilon: u32 },

    /// Layer normalization. Inputs: [x (f32), weight (f32), bias (f32)].
    LayerNorm { size: u32, epsilon: u32 },

    // ── Reductions ────────────────────────────────────────────────────────
    /// Sum reduction along last `size` elements of each row.
    ReduceSum { size: u32 },
    /// Mean reduction along last `size` elements of each row.
    ReduceMean { size: u32 },
    /// Max reduction along last `size` elements of each row.
    ReduceMax { size: u32 },
    /// Min reduction along last `size` elements of each row.
    ReduceMin { size: u32 },

    // ── Shape manipulation ───────────────────────────────────────────────
    /// Gather rows: indices (i64) index into a weight table.
    /// `dtype` indicates the element type of the table (F32 for embeddings, I64 for shape data).
    Gather { dim: u32, dtype: FloatDType },

    /// Concatenate along an axis. Inputs: [a, b].
    /// `dtype` indicates the element type (F32 for tensor data, I64 for shape data).
    Concat {
        size_a: u32,
        size_b: u32,
        dtype: FloatDType,
    },

    /// Reshape / flatten / squeeze / unsqueeze: pass-through.
    /// Shape is metadata only — bytes are unchanged.
    Reshape,

    /// Physical data permutation (reorders elements by dimension).
    /// First `ndim` entries of `perm` are valid permutation indices.
    Transpose {
        /// Permutation indices; first `ndim` entries are valid.
        perm: [u8; 8],
        /// Number of valid entries in `perm`.
        ndim: u8,
    },

    /// Type cast with source and target dtype.
    Cast { from: FloatDType, to: FloatDType },

    /// Embedding lookup. Inputs: [token_ids (u32), table (f32)].
    /// table is [vocab, dim]. Output: [len(ids), dim].
    Embed { dim: u32 },

    /// Conditional selection. Inputs: [cond (u8), x (f32), y (f32)].
    Where,

    /// Generate range [start, limit) with step. Inputs: [start, limit, delta (f32)].
    Range,

    /// Extract shape as i64 tensor (returns [n_elements] based on dtype byte size).
    Shape { dtype: FloatDType },

    /// GatherND (stub: pass-through, full N-D gather later).
    GatherND,

    // ── Fused ops ─────────────────────────────────────────────────────────
    /// Fused SiLU gating (SwiGLU): out = silu(gate) * up.
    FusedSwiGLU,

    /// Rotary position embedding (RoPE).
    RotaryEmbedding { dim: u32, base: u32 },

    /// Scaled dot-product attention (multi-head / grouped-query).
    /// Inputs: [Q (f32), K (f32), V (f32)].
    /// Q is [num_q_heads, seq, head_dim], K/V are [num_kv_heads, seq, head_dim].
    /// scale stored as f32 bits.
    Attention {
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        scale: u32,
        causal: bool,
    },

    // ── Quantization ─────────────────────────────────────────────────────
    /// Dequantize Q4_0 → f32.
    Dequantize,
}

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
            | Self::GatherND
            | Self::Dequantize => 1,

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
            | Self::RotaryEmbedding { .. } => 2,

            // Ternary
            Self::LayerNorm { .. }
            | Self::Gemm { .. }
            | Self::Attention { .. }
            | Self::Where
            | Self::Range => 3,
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
            Self::GatherND => "float.gather_nd",
            Self::FusedSwiGLU => "float.fused_swiglu",
            Self::RotaryEmbedding { .. } => "float.rope",
            Self::Attention { .. } => "float.attention",
            Self::Dequantize => "float.dequantize",
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
            | Self::Dequantize
            | Self::Cast { .. } => ShapeSpec::SameAs(0),

            // Shape-preserving: output = input[0] shape
            Self::Softmax { .. }
            | Self::LogSoftmax { .. }
            | Self::RmsNorm { .. }
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

            // Where: output = input[1] shape (x tensor)
            Self::Where => ShapeSpec::SameAs(1),

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
            | Self::Attention { .. }
            | Self::GatherND => ShapeSpec::Custom,
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
                        + (std::f32::consts::FRAC_2_SQRT_PI
                            * (x + 0.044715 * x * x * x)
                            * std::f32::consts::FRAC_1_SQRT_2)
                            .tanh())
            }
            Self::Silu => x * (1.0 / (1.0 + (-x).exp())),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Exp => x.exp(),
            Self::Log => x.ln(),
            Self::Sqrt => x.sqrt(),
            Self::Abs => x.abs(),
            Self::Reciprocal => 1.0 / x,
            Self::Cos => x.cos(),
            Self::Sin => x.sin(),
            Self::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Round => x.round(),
            Self::Erf => {
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                let a = x.abs();
                let t = 1.0 / (1.0 + 0.327_591_1 * a);
                let y = 1.0
                    - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74)
                        * t
                        + 0.254_829_6)
                        * t
                        * (-a * a).exp();
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

/// Dispatch category for `FloatOp`.
///
/// Groups ops by their execution pattern so dispatch can match on the
/// category instead of every individual variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCategory {
    /// Unary f32→f32 element-wise (use `apply_unary`).
    UnaryElementwise,
    /// Binary f32→f32 element-wise with broadcast (use `apply_binary`).
    BinaryElementwise,
    /// Binary f32→bool comparison with broadcast (use `apply_compare`).
    BinaryCompare,
    /// Binary byte→byte boolean logic (use `apply_byte_bool`).
    BinaryByteBool,
    /// Unary byte→byte boolean (NOT).
    UnaryByteBool,
    /// Unary producing u8 output (IsNaN).
    UnaryToU8,
    /// Op needs dedicated dispatch logic.
    Custom,
}

impl FloatOp {
    /// Dispatch category — which generic kernel pattern this op uses.
    #[must_use]
    pub const fn category(&self) -> OpCategory {
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
            | Self::Clip { .. } => OpCategory::UnaryElementwise,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Pow
            | Self::Mod
            | Self::Min
            | Self::Max
            | Self::FusedSwiGLU => OpCategory::BinaryElementwise,

            Self::Equal | Self::Less | Self::LessOrEqual | Self::Greater | Self::GreaterOrEqual => {
                OpCategory::BinaryCompare
            }

            Self::And | Self::Or | Self::Xor => OpCategory::BinaryByteBool,
            Self::Not => OpCategory::UnaryByteBool,
            Self::IsNaN => OpCategory::UnaryToU8,

            _ => OpCategory::Custom,
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
            Self::Pow => a.powf(b),
            Self::Mod => a % b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::FusedSwiGLU => {
                // silu(gate) * up  where gate=a, up=b
                let silu_a = a * (1.0 / (1.0 + (-a).exp()));
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
            Self::GatherND => "GatherND",
            Self::FusedSwiGLU => "SwiGLU",
            Self::RotaryEmbedding { .. } => "RoPE",
            Self::Attention { .. } => "Attention",
            Self::Dequantize => "Dequantize",
        }
    }
}

/// Encode an f32 as u32 bits for storage in `Eq`/`Hash`-compatible enum fields.
#[inline]
#[must_use]
pub const fn f32_to_bits(f: f32) -> u32 {
    f.to_bits()
}

/// Decode u32 bits back to f32.
#[inline]
#[must_use]
pub const fn bits_to_f32(bits: u32) -> f32 {
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity() {
        assert_eq!(FloatOp::Add.arity(), 2);
        assert_eq!(FloatOp::Relu.arity(), 1);
        assert_eq!(FloatOp::MatMul { m: 1, k: 4, n: 2 }.arity(), 2);
        assert_eq!(FloatOp::Softmax { size: 10 }.arity(), 1);
        assert_eq!(
            FloatOp::RmsNorm {
                size: 128,
                epsilon: f32_to_bits(1e-5)
            }
            .arity(),
            2
        );
        assert_eq!(
            FloatOp::LayerNorm {
                size: 128,
                epsilon: f32_to_bits(1e-5)
            }
            .arity(),
            3
        );
        assert_eq!(FloatOp::Cos.arity(), 1);
        assert_eq!(FloatOp::Pow.arity(), 2);
        assert_eq!(FloatOp::Equal.arity(), 2);
        assert_eq!(FloatOp::Not.arity(), 1);
        assert_eq!(FloatOp::Where.arity(), 3);
        assert_eq!(
            FloatOp::Attention {
                head_dim: 64,
                num_q_heads: 8,
                num_kv_heads: 2,
                scale: f32_to_bits(0.125),
                causal: true
            }
            .arity(),
            3
        );
        assert_eq!(FloatOp::Embed { dim: 128 }.arity(), 2);
        assert_eq!(FloatOp::ReduceMin { size: 10 }.arity(), 1);
        assert_eq!(FloatOp::Dequantize.arity(), 1);
    }

    #[test]
    fn name() {
        assert_eq!(FloatOp::Add.name(), "float.add");
        assert_eq!(FloatOp::MatMul { m: 1, k: 4, n: 2 }.name(), "float.matmul");
        assert_eq!(FloatOp::FusedSwiGLU.name(), "float.fused_swiglu");
        assert_eq!(FloatOp::Cos.name(), "float.cos");
        assert_eq!(FloatOp::Equal.name(), "float.equal");
        assert_eq!(
            FloatOp::Attention {
                head_dim: 64,
                num_q_heads: 8,
                num_kv_heads: 2,
                scale: f32_to_bits(0.125),
                causal: true
            }
            .name(),
            "float.attention"
        );
    }

    #[test]
    fn output_shape_spec() {
        use super::super::ShapeSpec;
        // Unary elementwise
        assert_eq!(FloatOp::Relu.output_shape_spec(), ShapeSpec::SameAs(0));
        assert_eq!(FloatOp::Neg.output_shape_spec(), ShapeSpec::SameAs(0));
        assert_eq!(
            FloatOp::Cast {
                from: FloatDType::F32,
                to: FloatDType::I64
            }
            .output_shape_spec(),
            ShapeSpec::SameAs(0)
        );
        // Binary broadcast
        assert_eq!(FloatOp::Add.output_shape_spec(), ShapeSpec::Broadcast(0, 1));
        assert_eq!(
            FloatOp::FusedSwiGLU.output_shape_spec(),
            ShapeSpec::Broadcast(0, 1)
        );
        // Shape-preserving
        assert_eq!(
            FloatOp::Softmax { size: 128 }.output_shape_spec(),
            ShapeSpec::SameAs(0)
        );
        assert_eq!(
            FloatOp::RmsNorm {
                size: 128,
                epsilon: f32_to_bits(1e-5)
            }
            .output_shape_spec(),
            ShapeSpec::SameAs(0)
        );
        // Reductions
        assert_eq!(
            FloatOp::ReduceSum { size: 64 }.output_shape_spec(),
            ShapeSpec::DropLastDim(0)
        );
        // Where
        assert_eq!(FloatOp::Where.output_shape_spec(), ShapeSpec::SameAs(1));
        // Embed (Custom — output = indices_shape ++ [dim])
        assert_eq!(
            FloatOp::Embed { dim: 256 }.output_shape_spec(),
            ShapeSpec::Custom
        );
        // Gather (Custom — output = indices_shape ++ [dim])
        assert_eq!(
            FloatOp::Gather {
                dim: 128,
                dtype: FloatDType::F32
            }
            .output_shape_spec(),
            ShapeSpec::Custom
        );
        // Range
        assert_eq!(FloatOp::Range.output_shape_spec(), ShapeSpec::inferred_1d());
        // Custom
        assert_eq!(
            FloatOp::MatMul { m: 1, k: 4, n: 2 }.output_shape_spec(),
            ShapeSpec::Custom
        );
        assert_eq!(FloatOp::Reshape.output_shape_spec(), ShapeSpec::Custom);
        assert_eq!(
            FloatOp::Transpose {
                perm: [0, 0, 0, 0, 0, 0, 0, 0],
                ndim: 2
            }
            .output_shape_spec(),
            ShapeSpec::Custom
        );
    }

    #[test]
    fn f32_roundtrip() {
        let eps = 1e-5f32;
        assert_eq!(bits_to_f32(f32_to_bits(eps)), eps);
    }
}
