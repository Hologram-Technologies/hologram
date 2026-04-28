//! Semantic attributes carried by op variants that need them.
//!
//! These are pure data types — the `SemanticOp` enum's tuple-style
//! variants embed them by value, and per-op marker structs in
//! [`crate::kernels`] wrap them. They are `rkyv`-archived because they
//! are part of the on-disk graph format.

/// MatMul semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct MatMulAttrs {
    /// Rows of A / C.
    pub m: u32,
    /// Inner dimension.
    pub k: u32,
    /// Cols of B / C.
    pub n: u32,
}

/// Softmax-family semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct SoftmaxAttrs {
    /// Size of the normalized axis.
    pub size: u32,
}

/// Norm-family semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct NormAttrs {
    /// Size of the normalized axis.
    pub size: u32,
    /// Epsilon represented as `f32::to_bits()`.
    pub epsilon: u32,
}

/// GroupNorm semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct GroupNormAttrs {
    /// Number of groups.
    pub num_groups: u32,
    /// Epsilon represented as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Transpose semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct TransposeAttrs {
    /// Permutation vector; first `ndim` entries are valid.
    pub perm: [u8; 8],
    /// Number of valid permutation entries.
    pub ndim: u8,
}

/// Slice semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct SliceAttrs {
    /// Axis counted from the end.
    pub axis_from_end: u8,
    /// Start offset.
    pub start: u32,
    /// End offset.
    pub end: u32,
    /// Compile-time axis size when known.
    pub axis_size: u32,
}

/// Concat semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ConcatAttrs {
    /// Size of A along the concatenated axis.
    pub size_a: u32,
    /// Size of B along the concatenated axis.
    pub size_b: u32,
}

/// Reduction semantic attributes (`ReduceSum`, `ReduceMean`,
/// `ReduceMax`, `ReduceMin`, `ReduceProd`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ReduceAttrs {
    /// Length of the reduced (last) axis.
    pub size: u32,
}

/// 2-D pooling attributes (`MaxPool2d`, `AvgPool2d`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct Pool2dAttrs {
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Vertical padding.
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
}

/// `GlobalAvgPool` attributes — input layout is `[N, C, H, W]`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct GlobalAvgPoolAttrs {
    /// Channel count.
    pub channels: u32,
    /// Spatial height.
    pub spatial_h: u32,
    /// Spatial width.
    pub spatial_w: u32,
}

/// `CumSum` attributes — pinned to the last axis in canonical, so
/// `axis` here is informational (source-language axis label preserved
/// for round-trip).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct CumSumAttrs {
    /// Source-language axis (round-trip metadata; canonical kernel
    /// always operates on the last axis).
    pub axis: u32,
}

/// `Pad` attributes (constant mode, NCHW symmetric padding).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct PadAttrs {
    /// Vertical padding (top + bottom each get this many rows).
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
    /// Constant fill value, encoded as `f32::to_bits()`.
    pub value: u32,
    /// Mode: 0 = constant. Other modes deferred — see ADR-048.
    pub mode: u8,
}

/// `Resize` attributes — interpolation mode is the only attribute on
/// `FloatOp::Resize`, but the canonical kernel needs the input and
/// output `H`/`W` to know what to sample, so the planner derives
/// those from the chain's tensor shapes (no attrs needed beyond
/// mode here).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ResizeAttrs {
    /// Mode: 0 = nearest. Linear / cubic deferred (ADR-048).
    pub mode: u8,
}

/// `Clip` attributes (min/max stored as `f32::to_bits()` so the attrs
/// struct stays `Copy + Eq + Hash`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ClipAttrs {
    /// Minimum bound, encoded as `f32::to_bits()`.
    pub min: u32,
    /// Maximum bound, encoded as `f32::to_bits()`.
    pub max: u32,
}

/// `LRN` (Local Response Normalization) attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct LrnAttrs {
    /// Window size in channels (typically odd).
    pub size: u32,
    /// `alpha`, encoded as `f32::to_bits()`.
    pub alpha: u32,
    /// `beta`, encoded as `f32::to_bits()`.
    pub beta: u32,
    /// `bias`, encoded as `f32::to_bits()`.
    pub bias: u32,
}

/// `ConvTranspose2d` semantic attributes (mirrors `Conv2dAttrs` plus
/// `output_pad_h`/`w` which capture the output-shape choice that's
/// underdetermined by the input + kernel + stride alone).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ConvTransposeAttrs {
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Vertical padding.
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
    /// Vertical dilation.
    pub dilation_h: u32,
    /// Horizontal dilation.
    pub dilation_w: u32,
    /// Group count.
    pub group: u32,
    /// Output padding (vertical).
    pub output_pad_h: u32,
    /// Output padding (horizontal).
    pub output_pad_w: u32,
    /// Input height (compile-time when known; for round-trip with
    /// the legacy `FloatOp::ConvTranspose`).
    pub input_h: u32,
    /// Input width.
    pub input_w: u32,
}

/// `Gemm` semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct GemmAttrs {
    /// Rows of `Y`.
    pub m: u32,
    /// Inner dimension.
    pub k: u32,
    /// Cols of `Y`.
    pub n: u32,
    /// Scalar applied to `A @ B`, encoded as `f32::to_bits()`.
    pub alpha: u32,
    /// Scalar applied to `C` before adding, encoded as `f32::to_bits()`.
    pub beta: u32,
    /// Whether `A` is provided transposed.
    pub trans_a: bool,
    /// Whether `B` is provided transposed.
    pub trans_b: bool,
}

/// Canonical scaled-dot-product `Attention` attributes (ADR-049).
///
/// Layout-flag, RoPE, QK-norm, sparse-V, and KV-cache concerns are
/// intentionally *not* on the canonical op — those are upstream
/// canonical ops (`RotaryEmbedding`, `RmsNorm`, `Transpose`) or
/// execution-side optimisations.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct AttentionAttrs {
    /// Per-head dimension.
    pub head_dim: u32,
    /// Number of Q heads.
    pub num_q_heads: u32,
    /// Number of KV heads (`num_q_heads % num_kv_heads == 0`).
    pub num_kv_heads: u32,
    /// `scale` (typically 1/√head_dim), encoded as `f32::to_bits()`.
    pub scale: u32,
    /// Causal mask flag.
    pub causal: bool,
}

/// `RotaryEmbedding` attributes (half-rotation form).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct RotaryEmbeddingAttrs {
    /// Per-head rotation dimension (must equal head_dim).
    pub dim: u32,
    /// Theta base (`f32::to_bits()`; typically 10000.0).
    pub base: u32,
    /// Number of attention heads per position.
    pub n_heads: u32,
}

/// `Expand` (broadcast to target shape) attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ExpandAttrs {
    /// Number of valid dimensions in `target_shape`.
    pub ndim: u8,
    /// Target shape.
    pub target_shape: [u32; 8],
}

/// Conv2d semantic attributes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct Conv2dAttrs {
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Vertical padding.
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
    /// Vertical dilation.
    pub dilation_h: u32,
    /// Horizontal dilation.
    pub dilation_w: u32,
    /// Group count.
    pub group: u32,
    /// Input height.
    pub input_h: u32,
    /// Input width.
    pub input_w: u32,
}
