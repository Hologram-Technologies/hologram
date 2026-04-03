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

    /// Infer a dtype from its byte size. Used when only the element size
    /// is known (e.g., from arena tracking). Picks the most common type
    /// for each size: 4→F32, 8→I64, 2→F16, 1→Bool.
    #[must_use]
    pub const fn from_byte_size(bytes: usize) -> Self {
        match bytes {
            8 => Self::I64,
            4 => Self::F32,
            2 => Self::F16,
            1 => Self::Bool,
            _ => Self::F32, // safe default
        }
    }
}

/// Sentinel value for a dimension that is resolved at runtime.
///
/// Used in TensorMeta and shape specifications to indicate a dimension
/// whose size is determined by the actual input data, not by compiled shapes.
/// Analogous to ONNX Reshape's `-1` dimension inference.
pub const RUNTIME: u32 = u32::MAX;

/// Lightweight metadata attached to every buffer in the arena.
///
/// Carries shape, dtype, and dimensionality — making each tensor buffer
/// self-describing. Stored in a parallel `Vec` alongside arena buffers
/// (NOT embedded in buffer bytes — zero-copy preserved for mmap'd data).
///
/// Fixed-size (40 bytes), `Copy`, no heap allocation per tensor.
/// O(1) access by NodeId index.
#[derive(Clone, Copy, Debug)]
pub struct TensorMeta {
    /// Number of valid entries in `dims` (0 = scalar, max 8).
    pub ndim: u8,
    /// Element data type.
    pub dtype: FloatDType,
    /// Actual shape at runtime. First `ndim` entries are valid.
    pub dims: [u32; 8],
}

impl TensorMeta {
    /// Create metadata from a dtype and shape slice.
    pub fn new(dtype: FloatDType, shape: &[usize]) -> Self {
        let ndim = shape.len().min(8) as u8;
        let mut dims = [0u32; 8];
        for (i, &d) in shape.iter().take(8).enumerate() {
            dims[i] = d as u32;
        }
        Self { ndim, dtype, dims }
    }

    /// Create 1-D metadata inferred from buffer byte length and element size.
    pub fn infer_1d(byte_len: usize, elem_size: usize) -> Self {
        let n_elems = if elem_size > 0 {
            byte_len / elem_size
        } else {
            byte_len
        };
        let dtype = FloatDType::from_byte_size(elem_size);
        Self {
            ndim: 1,
            dtype,
            dims: [n_elems as u32, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    /// Total number of elements.
    pub fn n_elems(&self) -> usize {
        if self.ndim == 0 {
            return 1;
        }
        self.dims[..self.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Shape as a slice of u32.
    pub fn shape(&self) -> &[u32] {
        &self.dims[..self.ndim as usize]
    }

    /// Last dimension (normalization/reduction axis for most ops).
    pub fn last_dim(&self) -> Option<u32> {
        if self.ndim > 0 {
            Some(self.dims[self.ndim as usize - 1])
        } else {
            None
        }
    }

    /// Second-to-last dimension (e.g., sequence length for MatMul).
    pub fn second_last_dim(&self) -> Option<u32> {
        if self.ndim >= 2 {
            Some(self.dims[self.ndim as usize - 2])
        } else {
            None
        }
    }

    /// Spatial (H, W) — last two dims. For NCHW tensors these are spatial dims.
    pub fn spatial_hw(&self) -> Option<(u32, u32)> {
        if self.ndim >= 2 {
            Some((
                self.dims[self.ndim as usize - 2],
                self.dims[self.ndim as usize - 1],
            ))
        } else {
            None
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
    /// Inputs: [A (f32), B (f32/quantized), C (f32 optional)].
    /// quant_b: 0=none(f32), 1=Q4_0, 2=Q8_0 — applied to B before multiply.
    Gemm {
        m: u32,
        k: u32,
        n: u32,
        alpha: u32,
        beta: u32,
        trans_a: bool,
        trans_b: bool,
        quant_b: u8,
    },

    // ── Softmax ───────────────────────────────────────────────────────────
    /// Softmax along last `size` elements of each row.
    Softmax { size: u32 },

    /// LogSoftmax along last `size` elements of each row.
    LogSoftmax { size: u32 },

    // ── Normalization ─────────────────────────────────────────────────────
    /// RMS normalization. Inputs: [x (f32), weight (f32)].
    RmsNorm { size: u32, epsilon: u32 },

    /// Fused Add + RMS normalization. Inputs: [x (f32), residual (f32), weight (f32)].
    /// Computes: rmsnorm(x + residual, weight, epsilon).
    /// Eliminates intermediate residual buffer vs separate Add + RmsNorm.
    AddRmsNorm { size: u32, epsilon: u32 },

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

    /// Embedding lookup. Inputs: [token_ids (u32), table (f32 or quantized)].
    /// table is [vocab, dim]. Output: [len(ids), dim].
    /// quant: 0=none(f32), 1=Q4_0, 2=Q8_0.
    Embed { dim: u32, quant: u8 },

    /// Conditional selection. Inputs: [cond (u8), x (f32), y (f32)].
    Where,

    /// Generate range [start, limit) with step. Inputs: [start, limit, delta (f32)].
    Range,

    /// Extract shape as i64 tensor.
    ///
    /// Opset 15+ `start`/`end` attributes slice the output to `dims[start..end]`.
    /// Use `start = 0` and `end = i64::MAX` to return all dims (no-op defaults).
    /// Negative values count from the end of the shape (-1 = last dim).
    Shape {
        dtype: FloatDType,
        /// First dim to include (inclusive). 0 = no start clamp.
        start: i64,
        /// One past the last dim to include. `i64::MAX` = no end clamp.
        end: i64,
    },

    /// Contiguous slice along a single axis.
    /// Extracts elements [start..end) along the specified axis.
    /// `axis_from_end` counts backward: 1 = last axis, 2 = second-to-last, etc.
    /// `axis_size` is the full dimension of the sliced axis (0 = infer at runtime).
    /// Inputs: [data].
    Slice {
        axis_from_end: u8,
        start: u32,
        end: u32,
        axis_size: u32,
    },

    /// GatherND (stub: pass-through, full N-D gather later).
    GatherND,

    // ── Fused ops ─────────────────────────────────────────────────────────
    /// Fused SiLU gating (SwiGLU): out = silu(gate) * up.
    FusedSwiGLU,

    /// Rotary position embedding (RoPE).
    /// n_heads: number of heads per token (hidden_dim / dim).
    /// Used to compute position = chunk_index / n_heads.
    RotaryEmbedding { dim: u32, base: u32, n_heads: u32 },

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
        /// If true, inputs are `[n_heads, seq, head_dim]` (ONNX: already transposed).
        /// If false, inputs are `[seq, n_heads, head_dim]` (GGUF: needs transpose).
        heads_first: bool,
        /// If true, apply RMSNorm to Q and K before computing attention scores.
        /// Used by Qwen-style models with QK normalization.
        qk_norm: bool,
        /// If true, apply RoPE to Q and K before attention. Fuses the separate
        /// RotaryEmbedding nodes into the attention kernel.
        rope: bool,
        /// RoPE base frequency stored as f32 bits. Only used when `rope` is true.
        rope_base: u32,
        /// If true (default), skip V accumulation for positions with negligible
        /// attention weight (< 1e-6). Yields significant decode speedup at long
        /// context with zero quality loss.
        sparse_v: bool,
    },

    // ── Quantization ─────────────────────────────────────────────────────
    /// Dequantize Q4_0 → f32.
    Dequantize,

    // ── Vision / spatial ops ────────────────────────────────────────────
    /// 2-D convolution. Inputs: [data (f32), weight (f32), bias (f32, optional)].
    /// Strides, pads, dilations, group packed into u32 fields.
    Conv2d {
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
    /// 2-D transposed convolution.
    ConvTranspose {
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
    MaxPool2d {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// 2-D average pooling.
    AvgPool2d {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// Global average pool: spatial dims → 1.
    /// `channels`, `spatial_h`, `spatial_w` encode the input [N,C,H,W] layout
    /// so the dispatcher doesn't need to guess from buffer length.
    GlobalAvgPool {
        channels: u32,
        spatial_h: u32,
        spatial_w: u32,
    },
    /// Resize (nearest/linear/cubic). Mode encoded as u8.
    Resize { mode: u8 },
    /// N-D padding. Mode: 0=constant, 1=reflect, 2=edge.
    PadOp { mode: u8 },
    /// Instance normalization.
    InstanceNorm { size: u32, epsilon: u32 },
    /// Local response normalization.
    LRN {
        size: u32,
        alpha: u32,
        beta: u32,
        bias: u32,
    },

    // ── Utility ops ─────────────────────────────────────────────────────
    /// Product reduction along last `size` elements.
    ReduceProd { size: u32 },
    /// Top-K along an axis. Inputs: [data, K (i64)].
    TopK { axis: u32, largest: bool },
    /// ScatterND. Inputs: [data, indices, updates].
    ScatterND,
    /// Cumulative sum along an axis.
    CumSum { axis: u32 },
    /// NonZero: returns indices of non-zero elements.
    NonZero,
    /// Compress along an axis. Inputs: [data, condition].
    Compress { axis: u32 },
    /// ReverseSequence along time/batch axes.
    ReverseSequence { batch_axis: u32, time_axis: u32 },

    // ── KV cache ─────────────────────────────────────────────────────────
    /// Write a K or V tensor into the KV cache for a transformer layer.
    /// Input: [tensor (f32)]. Output: pass-through (or full cached tensor in decode).
    /// `is_key`: true for K tensor, false for V tensor.
    KvWrite {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        is_key: bool,
        /// When true, input is `[heads, seq, dim]` — transpose to seq-first for storage.
        /// When false, input is `[seq, heads, dim]` — store directly.
        heads_first: bool,
    },
    /// Read cached K and V tensors from the KV cache for a transformer layer.
    /// Inputs: none (state-only). Outputs: [K_cached (f32), V_cached (f32)].
    /// Returns the full cached K/V from position 0 to the current write position.
    KvRead {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        /// When true, output in `[heads, seq, dim]` — transpose from seq-first cache.
        /// When false, output in `[seq, heads, dim]` — return seq-first directly.
        heads_first: bool,
    },

    // ── New ops (append-only to preserve discriminant ordering) ──────────
    /// Group normalization: normalize over groups of channels.
    /// `num_groups` groups, `epsilon` stored as `f32::to_bits()`.
    GroupNorm { num_groups: u32, epsilon: u32 },
    /// ArgMax: index of the maximum value along an axis.
    /// Output dtype is I64. `axis` is the reduction axis.
    ArgMax { axis: u32, keepdims: bool },

    // ── Deep decode fusions (Plan 054) ──────────────────────────────────
    /// Fused RmsNorm → multi-output projection (GEMV at M=1, decomposed at M>1).
    /// Inputs: [x, norm_weight, projection_weight].
    /// Output: concatenated projection [M, n_total] where n_total = sum(split_sizes).
    /// Caller slices output via Slice nodes (zero-copy at M=1).
    NormProjectionGemv {
        norm_size: u32,
        epsilon: u32,
        k: u32,
        n_total: u32,
    },
    /// Fused Add + RmsNorm → multi-output projection.
    /// Inputs: [x, residual, norm_weight, projection_weight].
    /// Output: concatenated projection [M, n_total].
    AddNormProjectionGemv {
        norm_size: u32,
        epsilon: u32,
        k: u32,
        n_total: u32,
    },
    /// Fused SwiGLU (silu(gate)*up) → down projection.
    /// Inputs: [gate, up, down_weight].
    /// Output: [M, n] down-projected result. Activation computed in-register.
    SwiGluProjectionGemv { k: u32, n: u32 },
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
            | Self::Slice { .. }
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
            Self::GatherND | Self::Transpose { .. } | Self::Slice { .. } => input0,

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
            | Self::ReverseSequence { .. } => OpCategory::Custom,

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

/// Fast approximation of `exp(-a*a)` using the Schraudolph bit trick.
///
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
                causal: true,
                heads_first: true,
                qk_norm: false,
                rope: false,
                rope_base: 0,
                sparse_v: true,
            }
            .arity(),
            3
        );
        assert_eq!(FloatOp::Embed { dim: 128, quant: 0 }.arity(), 2);
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
                causal: true,
                heads_first: true,
                qk_norm: false,
                rope: false,
                rope_base: 0,
                sparse_v: true,
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
        assert_eq!(FloatOp::Where.output_shape_spec(), ShapeSpec::BroadcastAll);
        // Embed (Custom — output = indices_shape ++ [dim])
        assert_eq!(
            FloatOp::Embed { dim: 256, quant: 0 }.output_shape_spec(),
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
