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
        let n_elems = byte_len.checked_div(elem_size).unwrap_or(byte_len);
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

    /// Broadcast-expand: physically replicate data along dimensions where input=1.
    /// `target_shape` is the output shape (max 8 dims, `ndim` entries valid).
    Expand { ndim: u8, target_shape: [u32; 8] },
}

/// Dispatch category for `FloatOp`.
///
/// Execution-shape tag for `FloatOp` dispatch.
///
/// Groups float ops by the dispatcher routine they need (unary
/// elementwise, binary-with-broadcast, byte-bool, etc.). Distinct from
/// [`hologram_ops::OpCategory`], which is the *semantic* category
/// (Elementwise, LinearAlgebra, Normalisation, …) used by the
/// canonical-op layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatOpShape {
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
