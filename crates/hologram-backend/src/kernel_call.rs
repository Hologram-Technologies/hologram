//! `KernelCall` enum — one variant per `OpKind` (spec IX.1).
//!
//! Each variant carries the resolved buffer references and op-specific
//! parameters. The hot path matches on this enum and dispatches to the
//! corresponding kernel function.

use crate::workspace::BufferRef;
use alloc::{vec, vec::Vec};

/// Direct PrimitiveOp wrapper kernels.
#[derive(Debug, Clone, Copy)]
pub struct UnaryCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u64,
    pub witt_bits: u16,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub element_count: u64,
    pub witt_bits: u16,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub dtype: u8,
    /// When `true`, the `b` operand holds a **panel-packed** weight
    /// (`crate::layout::pack_b_panels`) rather than a row-major `k×n` matrix — the
    /// compile-time weight-layout monomorphism. Set by the compiler for
    /// constant f32 weights; the kernel then streams B contiguously. The
    /// produced value is identical to the unpacked product (layout-only), so
    /// it is excluded from `op_signature` — the operand's own κ-label already
    /// reflects its (packed) bytes.
    pub b_packed: bool,
}

/// Fused dequantize-then-matmul: `output = A · dequant(Bq)`. Produced by the
/// runtime `Dequantize → MatMul` fusion (the dequant feeds the matmul's B
/// operand and has no other consumer). The dense f32 weight is **never
/// materialized in the pool** — `Bq` stays quantized and is dequantized into a
/// transient scratch panel inside the kernel. `A` is row-major f32 `[m,k]`;
/// `Bq` is the quantized `[k,n]` weight (i8/i4) with per-tensor or per-channel
/// scale/zero-point (same layout as [`DequantizeCall`]).
#[derive(Debug, Clone, Copy)]
pub struct MatMulDequantCall {
    pub a: BufferRef,
    pub bq: BufferRef,
    pub scales: BufferRef,
    pub zero_points: BufferRef,
    pub output: BufferRef,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub channels: u32,
    pub inner: u32,
    pub quant_dtype: u8,
    pub dtype: u8,
    pub scale_bits: u32,
    pub zero_point: i32,
}

impl MatMulDequantCall {
    #[inline]
    pub const fn per_channel(&self) -> bool {
        self.channels > 0 && self.scales.slot != u32::MAX
    }
}

/// Binary op selector for [`BroadcastBinaryCall`].
pub mod broadcast_op {
    pub const ADD: u8 = 0;
    pub const SUB: u8 = 1;
    pub const MUL: u8 = 2;
}

/// Fused `Expand → elementwise-binary`: `out[o] = op(small[bcast(o)], other[o])`
/// (operands swapped when `small_is_lhs == false`). The `small` operand is the
/// **pre-Expand** tensor (`in_dims`, with 1 on the broadcast axes); it is read
/// with stride-0 broadcast indexing directly, so the full broadcasted tensor is
/// never materialized. Produced by the runtime `Expand → {Add,Sub,Mul}` fusion.
#[derive(Debug, Clone, Copy)]
pub struct BroadcastBinaryCall {
    pub small: BufferRef,
    pub other: BufferRef,
    pub output: BufferRef,
    pub rank: u8,
    pub in_dims: [u32; 8],
    pub out_dims: [u32; 8],
    /// One of [`broadcast_op`].
    pub op: u8,
    /// `true` ⇒ `op(small, other)`; `false` ⇒ `op(other, small)`.
    pub small_is_lhs: bool,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct GemmCall {
    pub a: BufferRef,
    pub b: BufferRef,
    pub c: BufferRef,
    pub output: BufferRef,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub alpha_bits: u64,
    pub beta_bits: u64,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dCall {
    pub x: BufferRef,
    pub w: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub channels_in: u32,
    pub channels_out: u32,
    pub h_in: u32,
    pub w_in: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub k_h: u32,
    pub k_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub dtype: u8,
}

/// im2col / col2im patch-matrix geometry (single instance, no batch). `Im2Col`
/// gathers `input [Cin,Hin,Win]` into `output [Cin·kh·kw, Hout·Wout]`; `Col2Im`
/// scatter-adds the patch matrix back into the image (the same fields, inverse
/// direction). Valid convolution (no padding); `Hout=(Hin−kh)/sh+1`.
#[derive(Debug, Clone, Copy)]
pub struct Im2ColCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub channels: u32,
    pub h_in: u32,
    pub w_in: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub k_h: u32,
    pub k_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct NormCall {
    pub x: BufferRef,
    pub gamma: BufferRef,
    pub beta: BufferRef,
    /// Optional residual buffer for fused-add normalizations (e.g. AddRmsNorm).
    /// `slot == u32::MAX` indicates no residual (plain norm).
    pub residual: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub feature: u32,
    /// Channel count for grouped norms (GroupNorm/InstanceNorm). 0 for plain
    /// LayerNorm/RmsNorm, where `gamma`/`beta` are indexed per-`feature` and
    /// normalization spans the whole `feature` row.
    pub channels: u32,
    /// Number of normalization groups for GroupNorm (= `channels` for
    /// InstanceNorm). 0 ⇒ ungrouped: normalize over the whole `feature` row
    /// (plain LayerNorm/RmsNorm). When > 0, each of `batch` samples is split
    /// into `num_groups` contiguous groups of `feature/num_groups` elements
    /// normalized independently, then scaled per-channel by `gamma`/`beta`
    /// (length `channels`).
    pub num_groups: u32,
    pub epsilon_bits: u64,
    pub dtype: u8,
}

impl NormCall {
    /// Sentinel for an unused residual buffer.
    pub const NO_RESIDUAL: BufferRef = BufferRef {
        slot: u32::MAX,
        offset: 0,
        length: 0,
    };

    pub const fn has_residual(&self) -> bool {
        self.residual.slot != u32::MAX
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReduceCall {
    pub input: BufferRef,
    pub output: BufferRef,
    /// Input element count (the kernel folds over these).
    pub element_count: u64,
    /// Input rank; `dims[..rank]` is the row-major input shape.
    pub rank: u8,
    pub dims: [u32; 8],
    /// Bit `i` set ⇒ axis `i` is reduced. The output is the input shape with
    /// every reduced axis collapsed to 1 (keepdims layout — byte-identical to
    /// the keepdims=false layout, which only drops the size-1 axes). A mask of
    /// `(1<<rank)-1` (or `rank == 0`) is full reduction to a scalar.
    pub axes_mask: u32,
    pub keepdims: bool,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub element_count: u64,
    pub dtype: u8,
}

/// Transpose (axis permutation). A genuine re-indexing, so it carries the
/// input `dims` and the `perm` (output axis `i` reads input axis `perm[i]`),
/// up to rank 8. The kernel gathers each output element from its permuted
/// input position. (Not a relabel; this is the irreducible re-indexing op.)
#[derive(Debug, Clone, Copy)]
pub struct TransposeCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub rank: u8,
    pub dims: [u32; 8],
    pub perm: [u8; 8],
    pub dtype: u8,
}

/// Rotary positional embedding (RoPE). `x` rotated by the per-position
/// `cos`/`sin` tables (same element layout as `x`): for a pair within a head of
/// width `head_dim` (halves at `head_dim/2`), the first half maps to
/// `x·cos − x₂·sin` and the second to `x·cos + x₁·sin` (the rotate-half form).
#[derive(Debug, Clone, Copy)]
pub struct RoPECall {
    pub x: BufferRef,
    pub cos: BufferRef,
    pub sin: BufferRef,
    pub output: BufferRef,
    pub head_dim: u32,
    pub element_count: u64,
    pub dtype: u8,
}

/// LRN (local response normalization) over the channel axis of an
/// `[batch, channels, inner]` tensor: `out = x / (bias + (α/size)·Σ_window x²)^β`,
/// the window spanning `size` neighbouring channels centred on each channel.
#[derive(Debug, Clone, Copy)]
pub struct LrnCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub channels: u32,
    pub inner: u32,
    pub size: u32,
    pub alpha_bits: u32,
    pub beta_bits: u32,
    pub bias_bits: u32,
    pub dtype: u8,
}

/// Expand (broadcast). Replicates `input` to the broadcast `out_dims`: an axis
/// with `in_dims[i] == 1` is read at index 0 (stride-0), every other axis maps
/// 1:1. Rank ≤ 8. When the sole consumer is an elementwise `{Add,Sub,Mul}` the
/// runtime fuses this into a [`BroadcastBinaryCall`] that reads the operand with
/// stride-0 indexing in place (no materialized broadcast); this call's kernel
/// is the materializing gather for the remaining consumers (matmul, concat, …).
#[derive(Debug, Clone, Copy)]
pub struct ExpandCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub rank: u8,
    pub in_dims: [u32; 8],
    pub out_dims: [u32; 8],
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct SoftmaxCall {
    pub input: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub feature: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct PoolCall {
    pub x: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub channels: u32,
    pub h_in: u32,
    pub w_in: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub k_h: u32,
    pub k_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionCall {
    pub q: BufferRef,
    pub k: BufferRef,
    pub v: BufferRef,
    pub output: BufferRef,
    pub batch: u32,
    pub heads: u32,
    pub seq: u32,
    pub head_dim: u32,
    pub dtype: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct WhereCall {
    pub cond: BufferRef,
    pub a: BufferRef,
    pub b: BufferRef,
    pub output: BufferRef,
    pub element_count: u64,
    pub dtype: u8,
}

/// Dequantize kernel payload (spec X-5). Reads `element_count` quantized
/// values from `input` (interpreted per `quant_dtype` — `DTYPE_I8` or
/// `DTYPE_I4`), applies `output = (q − zero_point) · scale`, and writes
/// the result into `output` at `dtype` (typically `DTYPE_F32` or
/// `DTYPE_BF16`).
///
/// `scale_bits` and `zero_point` are the per-tensor scalars (used when
/// `channels == 0`). Per-channel quantization (one scale/zero-point per
/// channel along an axis) reads the `scales`/`zero_points` vector operands
/// instead — see the `channels`/`inner` fields.
#[derive(Debug, Clone, Copy)]
pub struct DequantizeCall {
    pub input: BufferRef,
    /// Per-channel scale vector (f32, length `channels`). `slot == u32::MAX`
    /// ⇒ per-tensor: use the scalar `scale_bits` instead.
    pub scales: BufferRef,
    /// Per-channel zero-point vector (i32, length `channels`). `slot ==
    /// u32::MAX` ⇒ per-tensor: use the scalar `zero_point` instead.
    pub zero_points: BufferRef,
    pub output: BufferRef,
    pub element_count: u64,
    /// Per-channel: number of channels along the quant axis (0 ⇒ per-tensor).
    pub channels: u32,
    /// Per-channel: elements per channel step (product of dims after the axis),
    /// so element `i`'s channel is `(i / inner) % channels`.
    pub inner: u32,
    /// Source quantized dtype: `DTYPE_I8` or `DTYPE_I4`.
    pub quant_dtype: u8,
    /// Destination float dtype: `DTYPE_F32`, `DTYPE_BF16`, etc.
    pub dtype: u8,
    /// `f32::to_bits` of the per-tensor scale (per-tensor mode).
    pub scale_bits: u32,
    /// Symmetric zero-point (per-tensor mode).
    pub zero_point: i32,
}

impl DequantizeCall {
    /// Sentinel for an absent per-channel scale/zero-point operand (per-tensor).
    pub const NO_VEC: BufferRef = BufferRef {
        slot: u32::MAX,
        offset: 0,
        length: 0,
    };

    /// True when scale/zero-point are per-channel vectors (vs. per-tensor scalars).
    #[inline]
    pub const fn per_channel(&self) -> bool {
        self.channels > 0 && self.scales.slot != u32::MAX
    }
}

/// Activation selectors applied in a **fused matmul epilogue**
/// (content-addressed fusion): the activation runs over each matmul output
/// element while it is still hot in cache, so the activation's intermediate
/// is never separately materialized or addressed.
pub mod fused_activation {
    pub const RELU: u8 = 1;
    pub const GELU: u8 = 2;
    pub const SILU: u8 = 3;
    pub const SIGMOID: u8 = 4;
    pub const TANH: u8 = 5;
    pub const ELU: u8 = 6;
    pub const SELU: u8 = 7;
    pub const EXP: u8 = 8;
}

/// A matmul with a fused elementwise-activation epilogue — the result of
/// fusing `matmul → activation` into one content-addressed operation. The
/// matmul output is never written back as a distinct intermediate; the
/// activation is applied in place over the result.
#[derive(Debug, Clone, Copy)]
pub struct MatMulActivationCall {
    pub mm: MatMulCall,
    /// One of [`fused_activation`].
    pub act: u8,
}

/// A matmul with a fused **residual-add** epilogue — the result of fusing
/// `matmul → add(matmul_out, residual)` into one content-addressed operation
/// (the transformer skip connection `y = A·B + residual`). The matmul output
/// is never materialized as a distinct intermediate; the residual is added in
/// place over the result while it is still hot in cache, eliminating the
/// separate bandwidth-bound add pass. The matmul may itself carry a
/// panel-packed weight (`mm.b_packed`), so packing and residual fusion compose.
#[derive(Debug, Clone, Copy)]
pub struct MatMulAddCall {
    pub mm: MatMulCall,
    /// The residual/skip tensor added elementwise to the matmul result
    /// (`mm.m × mm.n`, same dtype).
    pub residual: BufferRef,
}

/// A matmul with a fused **residual-add then activation** epilogue — the
/// result of fusing the three-op chain `matmul → add(matmul_out, residual) →
/// activation` into one content-addressed operation (the transformer MLP
/// `y = act(A·B + bias)` / a residual block whose sum is immediately
/// activated). Neither the matmul product nor the post-add sum is materialized
/// as a distinct addressed intermediate: the residual is added and the
/// activation applied in place over the result while it is still hot in cache,
/// eliding two bandwidth-bound passes. Composes with panel-packing
/// (`mm.b_packed`).
#[derive(Debug, Clone, Copy)]
pub struct MatMulAddActivationCall {
    pub mm: MatMulCall,
    /// Residual/bias tensor added before the activation (`mm.m × mm.n`).
    pub residual: BufferRef,
    /// One of [`fused_activation`].
    pub act: u8,
}

/// A node's semantic operation signature: a stable `opcode` (one per
/// `KernelCall` variant) plus the LE-encoded scalar attributes that, with
/// the operands' content addresses, fully determine the result. Buffer
/// slots/offsets are deliberately excluded — they are physical placement,
/// not computation identity. Used by the content-addressed executor to
/// derive a node's output κ-label (`derive_label_witnessed`) so an
/// identical computation (same op, params, operand addresses) is
/// recognized and its compute elided.
#[derive(Debug, Clone, Copy)]
pub struct OpSignature {
    pub opcode: u16,
    params: [u8; 64],
    len: u8,
}

impl OpSignature {
    /// The op-defining scalar bytes (shape, dtype, attrs) in LE order.
    #[must_use]
    pub fn params(&self) -> &[u8] {
        &self.params[..self.len as usize]
    }
}

/// Fixed-capacity LE byte accumulator for an [`OpSignature`]'s params.
/// 64 bytes covers the widest variant (Conv2d: 13 × u32 + dtype = 53 B).
struct Pb {
    buf: [u8; 64],
    len: usize,
}
impl Pb {
    fn new() -> Self {
        Self {
            buf: [0; 64],
            len: 0,
        }
    }
    fn raw(mut self, b: &[u8]) -> Self {
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        self
    }
    fn u8(self, v: u8) -> Self {
        self.raw(&[v])
    }
    fn u16(self, v: u16) -> Self {
        self.raw(&v.to_le_bytes())
    }
    fn u32(self, v: u32) -> Self {
        self.raw(&v.to_le_bytes())
    }
    fn u64(self, v: u64) -> Self {
        self.raw(&v.to_le_bytes())
    }
    fn i32(self, v: i32) -> Self {
        self.raw(&v.to_le_bytes())
    }
    fn done(self, opcode: u16) -> OpSignature {
        OpSignature {
            opcode,
            params: self.buf,
            len: self.len as u8,
        }
    }
}

fn p_unary(c: &UnaryCall) -> Pb {
    Pb::new().u64(c.element_count).u16(c.witt_bits).u8(c.dtype)
}
fn p_binary(c: &BinaryCall) -> Pb {
    Pb::new().u64(c.element_count).u16(c.witt_bits).u8(c.dtype)
}
fn p_matmul(c: &MatMulCall) -> Pb {
    Pb::new().u32(c.m).u32(c.k).u32(c.n).u8(c.dtype)
}
fn p_gemm(c: &GemmCall) -> Pb {
    Pb::new()
        .u32(c.m)
        .u32(c.k)
        .u32(c.n)
        .u64(c.alpha_bits)
        .u64(c.beta_bits)
        .u8(c.dtype)
}
fn p_conv(c: &Conv2dCall) -> Pb {
    Pb::new()
        .u32(c.batch)
        .u32(c.channels_in)
        .u32(c.channels_out)
        .u32(c.h_in)
        .u32(c.w_in)
        .u32(c.h_out)
        .u32(c.w_out)
        .u32(c.k_h)
        .u32(c.k_w)
        .u32(c.stride_h)
        .u32(c.stride_w)
        .u32(c.pad_h)
        .u32(c.pad_w)
        .u8(c.dtype)
}
fn p_im2col(c: &Im2ColCall) -> Pb {
    Pb::new()
        .u32(c.channels)
        .u32(c.h_in)
        .u32(c.w_in)
        .u32(c.h_out)
        .u32(c.w_out)
        .u32(c.k_h)
        .u32(c.k_w)
        .u32(c.stride_h)
        .u32(c.stride_w)
        .u8(c.dtype)
}
fn p_norm(c: &NormCall) -> Pb {
    Pb::new()
        .u32(c.batch)
        .u32(c.feature)
        .u32(c.channels)
        .u32(c.num_groups)
        .u64(c.epsilon_bits)
        .u8(c.dtype)
        .u8(c.has_residual() as u8)
}
fn p_reduce(c: &ReduceCall) -> Pb {
    let mut b = Pb::new()
        .u64(c.element_count)
        .u8(c.rank)
        .u32(c.axes_mask)
        .u8(c.keepdims as u8)
        .u8(c.dtype);
    for i in 0..c.rank as usize {
        b = b.u32(c.dims[i]);
    }
    b
}
fn p_layout(c: &LayoutCall) -> Pb {
    Pb::new().u64(c.element_count).u8(c.dtype)
}

fn p_transpose(c: &TransposeCall) -> Pb {
    let mut b = Pb::new().u8(c.rank).u8(c.dtype);
    for i in 0..c.rank as usize {
        b = b.u32(c.dims[i]).u8(c.perm[i]);
    }
    b
}

fn p_expand(c: &ExpandCall) -> Pb {
    let mut b = Pb::new().u8(c.rank).u8(c.dtype);
    for i in 0..c.rank as usize {
        b = b.u32(c.in_dims[i]).u32(c.out_dims[i]);
    }
    b
}

fn p_rope(c: &RoPECall) -> Pb {
    Pb::new().u64(c.element_count).u32(c.head_dim).u8(c.dtype)
}

fn p_lrn(c: &LrnCall) -> Pb {
    Pb::new()
        .u32(c.batch)
        .u32(c.channels)
        .u32(c.inner)
        .u32(c.size)
        .u32(c.alpha_bits)
        .u32(c.beta_bits)
        .u32(c.bias_bits)
        .u8(c.dtype)
}
fn p_softmax(c: &SoftmaxCall) -> Pb {
    Pb::new().u32(c.batch).u32(c.feature).u8(c.dtype)
}
fn p_pool(c: &PoolCall) -> Pb {
    Pb::new()
        .u32(c.batch)
        .u32(c.channels)
        .u32(c.h_in)
        .u32(c.w_in)
        .u32(c.h_out)
        .u32(c.w_out)
        .u32(c.k_h)
        .u32(c.k_w)
        .u32(c.stride_h)
        .u32(c.stride_w)
        .u8(c.dtype)
}
fn p_attention(c: &AttentionCall) -> Pb {
    Pb::new()
        .u32(c.batch)
        .u32(c.heads)
        .u32(c.seq)
        .u32(c.head_dim)
        .u8(c.dtype)
}
fn p_where(c: &WhereCall) -> Pb {
    Pb::new().u64(c.element_count).u8(c.dtype)
}
fn p_dequant(c: &DequantizeCall) -> Pb {
    Pb::new()
        .u64(c.element_count)
        .u32(c.channels)
        .u32(c.inner)
        .u8(c.quant_dtype)
        .u8(c.dtype)
        .u32(c.scale_bits)
        .i32(c.zero_point)
        .u8(c.per_channel() as u8)
}

/// Closed kernel-call surface. One variant per OpKind.
#[derive(Debug, Clone, Copy)]
pub enum KernelCall {
    // Direct primitives
    Neg(UnaryCall),
    Bnot(UnaryCall),
    Succ(UnaryCall),
    Pred(UnaryCall),
    Add(BinaryCall),
    Sub(BinaryCall),
    Mul(BinaryCall),
    Xor(BinaryCall),
    And(BinaryCall),
    Or(BinaryCall),

    // Elementwise unary
    Relu(UnaryCall),
    Sigmoid(UnaryCall),
    Tanh(UnaryCall),
    Gelu(UnaryCall),
    Silu(UnaryCall),
    Elu(UnaryCall),
    Selu(UnaryCall),
    Exp(UnaryCall),
    Log(UnaryCall),
    Log1p(UnaryCall),
    Sqrt(UnaryCall),
    Reciprocal(UnaryCall),
    Sin(UnaryCall),
    Cos(UnaryCall),
    Tan(UnaryCall),
    Asin(UnaryCall),
    Acos(UnaryCall),
    Atan(UnaryCall),
    Ceil(UnaryCall),
    Floor(UnaryCall),
    Round(UnaryCall),
    Erf(UnaryCall),
    IsNaN(UnaryCall),
    Sign(UnaryCall),
    Abs(UnaryCall),

    // Elementwise binary
    Div(BinaryCall),
    Pow(BinaryCall),
    Mod(BinaryCall),
    Min(BinaryCall),
    Max(BinaryCall),
    Equal(BinaryCall),
    Less(BinaryCall),
    LessOrEqual(BinaryCall),
    Greater(BinaryCall),
    GreaterOrEqual(BinaryCall),

    // Linear algebra / convolution
    MatMul(MatMulCall),
    Gemm(GemmCall),
    Conv2d(Conv2dCall),
    ConvTranspose2d(Conv2dCall),
    Im2Col(Im2ColCall),
    Col2Im(Im2ColCall),

    // Normalization
    LayerNorm(NormCall),
    RmsNorm(NormCall),
    GroupNorm(NormCall),
    InstanceNorm(NormCall),
    AddRmsNorm(NormCall),

    // Reduction
    ReduceSum(ReduceCall),
    ReduceMean(ReduceCall),
    ReduceProd(ReduceCall),
    ReduceMin(ReduceCall),
    ReduceMax(ReduceCall),

    // Layout
    Reshape(LayoutCall),
    Transpose(TransposeCall),
    /// Concatenation — the closed `PrimitiveOp::Concat` (ADR-053). A binary
    /// placement constructor `out = a ∥ b` (n-ary concat folds as a chain);
    /// uses `BinaryCall` since it genuinely has two operands (unlike the
    /// single-input layout relabels).
    Concat(BinaryCall),
    Slice(LayoutCall),

    // Activation+reduce
    Softmax(SoftmaxCall),
    LogSoftmax(SoftmaxCall),

    // Pooling
    MaxPool2d(PoolCall),
    AvgPool2d(PoolCall),
    GlobalAvgPool(PoolCall),

    // Structured
    Attention(AttentionCall),
    FusedSwiGlu(MatMulCall),

    // Utility
    Pad(LayoutCall),
    Expand(ExpandCall),
    // Reuses `ExpandCall`'s {in_dims, out_dims} shape; the resize kernel maps
    // each output index to the nearest input index (vs broadcast for Expand).
    Resize(ExpandCall),
    CumSum(ReduceCall),
    RotaryEmbedding(RoPECall),
    Clip(UnaryCall),
    Lrn(LrnCall),
    Where(WhereCall),

    // Quantization (spec X-5)
    Dequantize(DequantizeCall),

    // Content-addressed fusion: matmul with a fused activation epilogue.
    // Constructed by the executor's fusion pass, not by the archive.
    MatMulActivation(MatMulActivationCall),
    MatMulAdd(MatMulAddCall),
    MatMulAddActivation(MatMulAddActivationCall),
    /// Fused dequantize → matmul (the dequant feeds B; dense f32 weight elided).
    MatMulDequant(MatMulDequantCall),
    /// Fused `Expand → elementwise-binary`: the broadcast operand is read with
    /// stride-0 indexing in place — the materialized broadcast tensor is elided.
    BroadcastBinary(BroadcastBinaryCall),
}

impl KernelCall {
    /// If this call is an elementwise unary activation that can be fused
    /// into a preceding matmul's epilogue, its [`fused_activation`]
    /// selector; `None` otherwise. Used by the executor's fusion pass.
    pub fn fused_activation(&self) -> Option<u8> {
        use fused_activation as fa;
        use KernelCall as K;
        Some(match self {
            K::Relu(_) => fa::RELU,
            K::Gelu(_) => fa::GELU,
            K::Silu(_) => fa::SILU,
            K::Sigmoid(_) => fa::SIGMOID,
            K::Tanh(_) => fa::TANH,
            K::Elu(_) => fa::ELU,
            K::Selu(_) => fa::SELU,
            K::Exp(_) => fa::EXP,
            _ => return None,
        })
    }
    /// Whether this op is **commutative** in its operands (`f(a,b) = f(b,a)`).
    /// The executor canonicalizes the operand order of commutative ops before
    /// deriving their content address, so `a+b` and `b+a` collapse to one
    /// κ-label and reuse each other's computation. Only ops whose algebra is
    /// genuinely order-independent qualify (never Sub/Div/Pow/comparisons).
    pub fn is_commutative(&self) -> bool {
        use KernelCall as K;
        matches!(
            self,
            K::Add(_)
                | K::Mul(_)
                | K::Xor(_)
                | K::And(_)
                | K::Or(_)
                | K::Min(_)
                | K::Max(_)
                | K::Equal(_)
        )
    }

    /// The node's content-addressing signature: a per-variant `opcode`
    /// and its op-defining scalar params. Stable across runs of the same
    /// compiled graph, so `derive_label_witnessed(opcode, params, operand
    /// labels)` is a sound key for eliding an identical computation.
    pub fn op_signature(&self) -> OpSignature {
        use KernelCall as K;
        match self {
            K::Neg(c) => p_unary(c).done(0),
            K::Bnot(c) => p_unary(c).done(1),
            K::Succ(c) => p_unary(c).done(2),
            K::Pred(c) => p_unary(c).done(3),
            K::Add(c) => p_binary(c).done(4),
            K::Sub(c) => p_binary(c).done(5),
            K::Mul(c) => p_binary(c).done(6),
            K::Xor(c) => p_binary(c).done(7),
            K::And(c) => p_binary(c).done(8),
            K::Or(c) => p_binary(c).done(9),
            K::Relu(c) => p_unary(c).done(10),
            K::Sigmoid(c) => p_unary(c).done(11),
            K::Tanh(c) => p_unary(c).done(12),
            K::Gelu(c) => p_unary(c).done(13),
            K::Silu(c) => p_unary(c).done(14),
            K::Elu(c) => p_unary(c).done(15),
            K::Selu(c) => p_unary(c).done(16),
            K::Exp(c) => p_unary(c).done(17),
            K::Log(c) => p_unary(c).done(18),
            K::Log1p(c) => p_unary(c).done(19),
            K::Sqrt(c) => p_unary(c).done(20),
            K::Reciprocal(c) => p_unary(c).done(21),
            K::Sin(c) => p_unary(c).done(22),
            K::Cos(c) => p_unary(c).done(23),
            K::Tan(c) => p_unary(c).done(24),
            K::Asin(c) => p_unary(c).done(25),
            K::Acos(c) => p_unary(c).done(26),
            K::Atan(c) => p_unary(c).done(27),
            K::Ceil(c) => p_unary(c).done(28),
            K::Floor(c) => p_unary(c).done(29),
            K::Round(c) => p_unary(c).done(30),
            K::Erf(c) => p_unary(c).done(31),
            K::IsNaN(c) => p_unary(c).done(32),
            K::Sign(c) => p_unary(c).done(33),
            K::Abs(c) => p_unary(c).done(34),
            K::Div(c) => p_binary(c).done(35),
            K::Pow(c) => p_binary(c).done(36),
            K::Mod(c) => p_binary(c).done(37),
            K::Min(c) => p_binary(c).done(38),
            K::Max(c) => p_binary(c).done(39),
            K::Equal(c) => p_binary(c).done(40),
            K::Less(c) => p_binary(c).done(41),
            K::LessOrEqual(c) => p_binary(c).done(42),
            K::Greater(c) => p_binary(c).done(43),
            K::GreaterOrEqual(c) => p_binary(c).done(44),
            K::MatMul(c) => p_matmul(c).done(45),
            K::Gemm(c) => p_gemm(c).done(46),
            K::Conv2d(c) => p_conv(c).done(47),
            K::ConvTranspose2d(c) => p_conv(c).done(48),
            K::Im2Col(c) => p_im2col(c).done(108),
            K::Col2Im(c) => p_im2col(c).done(109),
            K::LayerNorm(c) => p_norm(c).done(49),
            K::RmsNorm(c) => p_norm(c).done(50),
            K::GroupNorm(c) => p_norm(c).done(51),
            K::InstanceNorm(c) => p_norm(c).done(52),
            K::AddRmsNorm(c) => p_norm(c).done(53),
            K::ReduceSum(c) => p_reduce(c).done(54),
            K::ReduceMean(c) => p_reduce(c).done(55),
            K::ReduceProd(c) => p_reduce(c).done(56),
            K::ReduceMin(c) => p_reduce(c).done(57),
            K::ReduceMax(c) => p_reduce(c).done(58),
            K::Reshape(c) => p_layout(c).done(59),
            K::Transpose(c) => p_transpose(c).done(60),
            K::Concat(c) => p_binary(c).done(61),
            K::Slice(c) => p_layout(c).done(62),
            K::Softmax(c) => p_softmax(c).done(63),
            K::LogSoftmax(c) => p_softmax(c).done(64),
            K::MaxPool2d(c) => p_pool(c).done(65),
            K::AvgPool2d(c) => p_pool(c).done(66),
            K::GlobalAvgPool(c) => p_pool(c).done(67),
            K::Attention(c) => p_attention(c).done(68),
            K::FusedSwiGlu(c) => p_matmul(c).done(69),
            K::Pad(c) => p_layout(c).done(70),
            K::Expand(c) => p_expand(c).done(71),
            K::Resize(c) => p_expand(c).done(72),
            K::CumSum(c) => p_reduce(c).done(73),
            K::RotaryEmbedding(c) => p_rope(c).done(74),
            K::Clip(c) => p_unary(c).done(75),
            K::Lrn(c) => p_lrn(c).done(76),
            K::Where(c) => p_where(c).done(77),
            K::Dequantize(c) => p_dequant(c).done(104),
            K::MatMulActivation(c) => p_matmul(&c.mm).u8(c.act).done(105),
            K::MatMulAdd(c) => p_matmul(&c.mm).done(106),
            K::MatMulAddActivation(c) => p_matmul(&c.mm).u8(c.act).done(107),
            K::MatMulDequant(c) => Pb::new()
                .u32(c.m)
                .u32(c.k)
                .u32(c.n)
                .u32(c.channels)
                .u32(c.inner)
                .u8(c.quant_dtype)
                .u8(c.dtype)
                .u32(c.scale_bits)
                .i32(c.zero_point)
                .done(108),
            K::BroadcastBinary(c) => {
                let mut b = Pb::new()
                    .u8(c.rank)
                    .u8(c.op)
                    .u8(c.small_is_lhs as u8)
                    .u8(c.dtype);
                for i in 0..c.rank as usize {
                    b = b.u32(c.in_dims[i]).u32(c.out_dims[i]);
                }
                b.done(112)
            }
        }
    }
}

/// All buffer references of a kernel call, in **deterministic operand
/// order with the output last** — `[inputs.., output]`. This is the order
/// the content-addressed executor folds operand labels in (the
/// ordered-composition order) and the load-time slot-sizing /
/// producer-census passes consume; centralised here so the runtime and the
/// compiler's warm-start lattice extract operands identically.
#[must_use]
pub fn buffers(call: &KernelCall) -> Vec<BufferRef> {
    use KernelCall as K;
    match call {
        K::Neg(c)
        | K::Bnot(c)
        | K::Succ(c)
        | K::Pred(c)
        | K::Relu(c)
        | K::Sigmoid(c)
        | K::Tanh(c)
        | K::Gelu(c)
        | K::Silu(c)
        | K::Elu(c)
        | K::Selu(c)
        | K::Exp(c)
        | K::Log(c)
        | K::Log1p(c)
        | K::Sqrt(c)
        | K::Reciprocal(c)
        | K::Sin(c)
        | K::Cos(c)
        | K::Tan(c)
        | K::Asin(c)
        | K::Acos(c)
        | K::Atan(c)
        | K::Ceil(c)
        | K::Floor(c)
        | K::Round(c)
        | K::Erf(c)
        | K::IsNaN(c)
        | K::Sign(c)
        | K::Abs(c)
        | K::Clip(c) => vec![c.input, c.output],

        K::RotaryEmbedding(c) => vec![c.x, c.cos, c.sin, c.output],
        K::Lrn(c) => vec![c.input, c.output],

        K::Add(c)
        | K::Sub(c)
        | K::Mul(c)
        | K::Xor(c)
        | K::And(c)
        | K::Or(c)
        | K::Div(c)
        | K::Pow(c)
        | K::Mod(c)
        | K::Min(c)
        | K::Max(c)
        | K::Equal(c)
        | K::Less(c)
        | K::LessOrEqual(c)
        | K::Greater(c)
        | K::GreaterOrEqual(c)
        | K::Concat(c) => vec![c.a, c.b, c.output],

        K::MatMul(c) | K::FusedSwiGlu(c) => vec![c.a, c.b, c.output],

        K::MatMulDequant(c) if c.per_channel() => {
            vec![c.a, c.bq, c.scales, c.zero_points, c.output]
        }
        K::MatMulDequant(c) => vec![c.a, c.bq, c.output],
        K::BroadcastBinary(c) => vec![c.small, c.other, c.output],
        K::MatMulActivation(c) => vec![c.mm.a, c.mm.b, c.mm.output],
        K::MatMulAdd(c) => vec![c.mm.a, c.mm.b, c.residual, c.mm.output],
        K::MatMulAddActivation(c) => vec![c.mm.a, c.mm.b, c.residual, c.mm.output],

        K::Gemm(c) => vec![c.a, c.b, c.c, c.output],

        K::Conv2d(c) | K::ConvTranspose2d(c) => vec![c.x, c.w, c.output],

        K::Im2Col(c) | K::Col2Im(c) => vec![c.input, c.output],

        K::LayerNorm(c)
        | K::RmsNorm(c)
        | K::GroupNorm(c)
        | K::InstanceNorm(c)
        | K::AddRmsNorm(c) => vec![c.x, c.gamma, c.beta, c.output],

        K::ReduceSum(c)
        | K::ReduceMean(c)
        | K::ReduceProd(c)
        | K::ReduceMin(c)
        | K::ReduceMax(c)
        | K::CumSum(c) => vec![c.input, c.output],

        K::Reshape(c) | K::Slice(c) | K::Pad(c) => vec![c.input, c.output],

        K::Transpose(c) => vec![c.input, c.output],
        K::Expand(c) | K::Resize(c) => vec![c.input, c.output],

        K::Softmax(c) | K::LogSoftmax(c) => vec![c.input, c.output],

        K::MaxPool2d(c) | K::AvgPool2d(c) | K::GlobalAvgPool(c) => vec![c.x, c.output],

        K::Attention(c) => vec![c.q, c.k, c.v, c.output],

        K::Where(c) => vec![c.cond, c.a, c.b, c.output],

        K::Dequantize(c) if c.per_channel() => vec![c.input, c.scales, c.zero_points, c.output],
        K::Dequantize(c) => vec![c.input, c.output],
    }
}

/// The element dtype the kernel operates on (the `dtype` tag every call carries;
/// fused calls expose their inner matmul's). Centralised so the backend can
/// enforce a single dtype-support policy at dispatch instead of each kernel
/// re-checking. Exhaustive match — adding a `KernelCall` variant forces an
/// update here.
#[must_use]
pub fn call_dtype(call: &KernelCall) -> u8 {
    use KernelCall as K;
    match call {
        K::Neg(c)
        | K::Bnot(c)
        | K::Succ(c)
        | K::Pred(c)
        | K::Relu(c)
        | K::Sigmoid(c)
        | K::Tanh(c)
        | K::Gelu(c)
        | K::Silu(c)
        | K::Elu(c)
        | K::Selu(c)
        | K::Exp(c)
        | K::Log(c)
        | K::Log1p(c)
        | K::Sqrt(c)
        | K::Reciprocal(c)
        | K::Sin(c)
        | K::Cos(c)
        | K::Tan(c)
        | K::Asin(c)
        | K::Acos(c)
        | K::Atan(c)
        | K::Ceil(c)
        | K::Floor(c)
        | K::Round(c)
        | K::Erf(c)
        | K::IsNaN(c)
        | K::Sign(c)
        | K::Abs(c)
        | K::Clip(c) => c.dtype,

        K::RotaryEmbedding(c) => c.dtype,
        K::Lrn(c) => c.dtype,

        K::Add(c)
        | K::Sub(c)
        | K::Mul(c)
        | K::Xor(c)
        | K::And(c)
        | K::Or(c)
        | K::Div(c)
        | K::Pow(c)
        | K::Mod(c)
        | K::Min(c)
        | K::Max(c)
        | K::Equal(c)
        | K::Less(c)
        | K::LessOrEqual(c)
        | K::Greater(c)
        | K::GreaterOrEqual(c)
        | K::Concat(c) => c.dtype,

        K::MatMul(c) | K::FusedSwiGlu(c) => c.dtype,

        K::MatMulDequant(c) => c.dtype,
        K::BroadcastBinary(c) => c.dtype,
        K::MatMulActivation(c) => c.mm.dtype,
        K::MatMulAdd(c) => c.mm.dtype,
        K::MatMulAddActivation(c) => c.mm.dtype,

        K::Gemm(c) => c.dtype,

        K::Conv2d(c) | K::ConvTranspose2d(c) => c.dtype,

        K::Im2Col(c) | K::Col2Im(c) => c.dtype,

        K::LayerNorm(c)
        | K::RmsNorm(c)
        | K::GroupNorm(c)
        | K::InstanceNorm(c)
        | K::AddRmsNorm(c) => c.dtype,

        K::ReduceSum(c)
        | K::ReduceMean(c)
        | K::ReduceProd(c)
        | K::ReduceMin(c)
        | K::ReduceMax(c)
        | K::CumSum(c) => c.dtype,

        K::Reshape(c) | K::Slice(c) | K::Pad(c) => c.dtype,

        K::Transpose(c) => c.dtype,
        K::Expand(c) | K::Resize(c) => c.dtype,

        K::Softmax(c) | K::LogSoftmax(c) => c.dtype,

        K::MaxPool2d(c) | K::AvgPool2d(c) | K::GlobalAvgPool(c) => c.dtype,

        K::Attention(c) => c.dtype,

        K::Where(c) => c.dtype,

        K::Dequantize(c) => c.dtype,
    }
}
