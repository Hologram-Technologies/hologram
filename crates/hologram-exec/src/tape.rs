//! Instruction tape executor for zero-match dispatch.
//!
//! The tape is a flat array of pre-resolved instructions compiled from
//! the graph's execution schedule. Each instruction stores a kernel function
//! pointer and pre-resolved input/output indices, eliminating the large
//! `match op { ... }` dispatch at runtime.
//!
//! The tape is built once per model load and executed per inference call.
//! This is Phase 0.7 of the Compile-Time-First Acceleration plan.

use smallvec::SmallVec;

use hologram_core::op::FloatOp;
use hologram_graph::graph::node::NodeId;

use crate::buffer::BufferArena;
use crate::error::ExecResult;
use crate::eval::executor::ExecutionContext;

/// Non-blocking prefetch of a cache line into L1 for reading.
///
/// Uses platform-specific intrinsics where available:
/// - x86_64: `_mm_prefetch(..., _MM_HINT_T0)` (L1 temporal)
/// - aarch64: `PRFM PLDL1KEEP` via inline asm
/// - Other: no-op (rely on hardware prefetcher)
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "sse")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

// ── Enum-dispatch tape (Phase 8) ──────────────────────────────────────────────

use std::cell::RefCell;

use hologram_core::op::PrimOp;
use hologram_core::view::ElementWiseView;
use hologram_graph::constant::{ConstantId, ConstantStore};

use crate::backend::BackendSelector;
use crate::kv::weight_cache::WeightCache;
use crate::kv_cache::KvCacheState;

/// Execution context for the enum-dispatch tape.
///
/// Carries weight archive access, a lazily-populated weight cache
/// for LUT-GEMM ops, an optional KV cache for autoregressive generation,
/// and a backend selector for multi-backend dispatch (CPU/Metal/CUDA/WebGPU).
pub struct TapeContext<'a> {
    /// Optional per-inference execution state (position offset, etc.).
    pub ctx: Option<ExecutionContext>,
    /// Constant store for resolving `ConstantId` → raw bytes.
    pub constants: &'a ConstantStore,
    /// Raw weight archive bytes for deferred constants.
    pub weights: &'a [u8],
    /// Lazily-populated cache for deserialized quantized weights.
    pub weight_cache: RefCell<WeightCache>,
    /// Optional KV cache for autoregressive generation (KvWrite/KvRead ops).
    pub kv_state: Option<RefCell<KvCacheState>>,
    /// Backend selector (Auto/Cpu/Metal/Cuda/WebGpu).
    /// Resolved to a concrete `&dyn ComputeBackend` once at execute start.
    pub backend: BackendSelector,
    /// Pre-computed shape overrides from `ShapeContextGraph`.
    /// Keyed by raw node index. When present, the executor sets this as the
    /// output `TensorMeta` after dispatch, overriding any heuristic inference.
    pub shape_overrides: std::collections::HashMap<u32, Vec<usize>>,
}

impl<'a> TapeContext<'a> {
    /// Create a context from a constant store and weight archive.
    /// Uses `BackendSelector::Auto` (best available backend).
    #[must_use]
    pub fn new(constants: &'a ConstantStore, weights: &'a [u8]) -> Self {
        TapeContext {
            ctx: None,
            constants,
            weights,
            weight_cache: RefCell::new(WeightCache::new()),
            kv_state: None,
            backend: BackendSelector::Auto,
            shape_overrides: std::collections::HashMap::new(),
        }
    }

    /// Create a context with a KV cache for autoregressive generation.
    #[must_use]
    pub fn with_kv_cache(
        constants: &'a ConstantStore,
        weights: &'a [u8],
        kv: KvCacheState,
    ) -> Self {
        TapeContext {
            ctx: None,
            constants,
            weights,
            weight_cache: RefCell::new(WeightCache::new()),
            kv_state: Some(RefCell::new(kv)),
            backend: BackendSelector::Auto,
            shape_overrides: std::collections::HashMap::new(),
        }
    }
}

/// Pre-resolved kernel variant — replaces `Box<dyn Fn>` with a small enum.
///
/// Each variant captures only the op parameters needed for dispatch.
/// The `dispatch_kernel` function matches on this enum and calls the
/// appropriate dispatch function directly, enabling inlining and
/// eliminating vtable indirection.
pub enum TapeKernel {
    /// Fused chain of unary float ops.
    FusedFloatChain(Vec<FloatOp>),
    /// Graph output passthrough.
    Output,
    /// Byte-domain LUT (256-byte table).
    LutView(ElementWiseView),
    /// Byte-domain unary prim via LUT.
    PrimUnary(ElementWiseView),
    /// Byte-domain binary prim.
    PrimBinary(PrimOp),
    /// 4-bit quantized LUT-GEMM matmul.
    MatMulLut4(ConstantId),
    /// 8-bit quantized LUT-GEMM matmul.
    MatMulLut8(ConstantId),
    /// 4-bit quantized LUT-GEMM matmul + fused activation (epilogue fusion).
    MatMulLut4Activation(ConstantId, FloatOp),
    /// 8-bit quantized LUT-GEMM matmul + fused activation (epilogue fusion).
    MatMulLut8Activation(ConstantId, FloatOp),
    /// KV cache write (autoregressive generation).
    KvWrite {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        is_key: bool,
        /// When true, input is heads-first — transpose to seq-first for storage.
        heads_first: bool,
    },
    /// KV cache read (autoregressive generation).
    KvRead {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        /// When true, output heads-first — transpose from seq-first cache.
        heads_first: bool,
    },

    // ── Inline hot ops (Phase 9a) ─────────────────────────────────────
    // Skip backend vtable + dispatch_float_into entirely.
    // The execute loop calls the kernel function directly.
    /// Inline Relu: v.max(0.0). Zero dispatch overhead.
    InlineRelu,
    /// Inline Neg: -v.
    InlineNeg,
    /// Inline Sigmoid: 1/(1+exp(-v)).
    InlineSigmoid,
    /// Inline Silu: v * sigmoid(v).
    InlineSilu,
    /// Inline Tanh.
    InlineTanh,
    /// Inline Gelu (approximate).
    InlineGelu,
    /// Inline Exp.
    InlineExp,
    /// Inline binary Add.
    InlineAdd,
    /// Inline binary Mul.
    InlineMul,
    /// Inline binary Sub.
    InlineSub,
    /// Inline binary Div.
    InlineDiv,
    /// Inline Abs: v.abs().
    InlineAbs,
    /// Inline Reciprocal: 1.0 / v.
    InlineReciprocal,

    // ── Inline custom ops (Phase 9a.3–9a.4) ─────────────────────────────
    // Skip dispatch_float_into → dispatch_custom_into indirection.
    // Still try backend (Metal GPU) first, then direct CPU kernel call.
    /// Inline MatMul with baked dimensions.
    InlineMatMul { m: u32, k: u32, n: u32 },
    /// Fused MatMul + element-wise activation (epilogue fusion).
    /// Activation applied in-register before writeback — avoids memory round-trip.
    InlineMatMulActivation {
        m: u32,
        k: u32,
        n: u32,
        activation: FloatOp,
    },
    /// Fused MatMul + bias add + activation (full epilogue fusion).
    /// Three inputs: [activation, weight, bias]. Bias from arena (zero-copy).
    /// Eliminates both intermediate buffers from MatMul → Add(bias) → Activation.
    InlineMatMulBiasActivation {
        m: u32,
        k: u32,
        n: u32,
        activation: FloatOp,
    },
    /// Inline Softmax with baked row size.
    InlineSoftmax { size: u32 },
    /// Inline RmsNorm with baked row size and epsilon (as f32::to_bits()).
    InlineRmsNorm { size: u32, epsilon: u32 },

    // ── Inline ops (Phase 9a expansion — Sprint 21) ──────────────────
    /// Inline Log: v.ln().
    InlineLog,
    /// Inline Sqrt: v.sqrt().
    InlineSqrt,
    /// Inline Cos.
    InlineCos,
    /// Inline Sin.
    InlineSin,
    /// Inline Sign.
    InlineSign,
    /// Inline Floor.
    InlineFloor,
    /// Inline Ceil.
    InlineCeil,
    /// Inline Round.
    InlineRound,
    /// Inline Erf (Abramowitz & Stegun).
    InlineErf,
    /// Inline binary Min.
    InlineMin,
    /// Inline binary Max.
    InlineMax,
    /// Inline LayerNorm with baked size and epsilon.
    InlineLayerNorm { size: u32, epsilon: u32 },
    /// Inline AddRmsNorm with baked size and epsilon.
    InlineAddRmsNorm { size: u32, epsilon: u32 },
    /// Inline LogSoftmax with baked row size.
    InlineLogSoftmax { size: u32 },
    /// Inline Attention with baked head config.
    InlineAttention {
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        scale: u32,
        causal: bool,
        heads_first: bool,
    },
    /// Inline RotaryEmbedding with baked params (uses position offset from TapeContext).
    InlineRoPE { dim: u32, base: u32, n_heads: u32 },
    /// Inline Gather with baked dim and dtype.
    InlineGather {
        dim: u32,
        dtype: hologram_core::op::FloatDType,
    },
    /// Inline Concat with baked sizes and dtype.
    InlineConcat {
        size_a: u32,
        size_b: u32,
        dtype: hologram_core::op::FloatDType,
    },
    /// Inline Transpose with baked permutation and input shape.
    InlineTranspose {
        /// Permutation indices (first `ndim` entries valid).
        perm: [u8; 8],
        /// Input shape (first `ndim` entries valid).
        input_shape: [u32; 8],
        /// Number of valid dimensions.
        ndim: u8,
    },
    // ── Complete FloatOp coverage (no more Float(FloatOp) catch-all) ────
    /// Power: a^b.
    InlinePow,
    /// Modulo: a % b.
    InlineMod,
    /// Clamp to [min, max]. min/max stored as f32 bits.
    InlineClip { min: u32, max: u32 },
    /// Test for NaN: output is u8 (0 or 1).
    InlineIsNaN,
    /// Logical NOT (unary, f32 domain: 0→1, nonzero→0).
    InlineNot,
    /// Logical AND (binary, f32 domain).
    InlineAnd,
    /// Logical OR.
    InlineOr,
    /// Logical XOR.
    InlineXor,
    /// Equality comparison (f32→f32: 0.0 or 1.0).
    InlineEqual,
    /// Less-than comparison.
    InlineLess,
    /// Less-or-equal comparison.
    InlineLessOrEqual,
    /// Greater-than comparison.
    InlineGreater,
    /// Greater-or-equal comparison.
    InlineGreaterOrEqual,
    /// General matrix multiply with alpha/beta/transpose flags.
    InlineGemm {
        m: u32,
        k: u32,
        n: u32,
        alpha: u32,
        beta: u32,
        trans_a: bool,
        trans_b: bool,
        quant_b: u8,
    },
    /// Sum reduction along last `size` elements.
    InlineReduceSum { size: u32 },
    /// Mean reduction along last `size` elements.
    InlineReduceMean { size: u32 },
    /// Max reduction along last `size` elements.
    InlineReduceMax { size: u32 },
    /// Min reduction along last `size` elements.
    InlineReduceMin { size: u32 },
    /// Product reduction along last `size` elements.
    InlineReduceProd { size: u32 },
    /// Type cast between dtypes.
    InlineCast {
        from: hologram_core::op::FloatDType,
        to: hologram_core::op::FloatDType,
    },
    /// Embedding lookup: [token_ids, table] → [len(ids), dim].
    InlineEmbed { dim: u32, quant: u8 },
    /// Conditional selection: cond ? x : y.
    InlineWhere,
    /// Generate range [start, limit) with step.
    InlineRange,
    /// Extract shape as i64 tensor.
    InlineShape {
        dtype: hologram_core::op::FloatDType,
        start: i64,
        end: i64,
    },
    /// Contiguous slice along axis.
    InlineSlice {
        axis_from_end: u8,
        start: u32,
        end: u32,
        axis_size: u32,
    },
    /// GatherND.
    InlineGatherND,
    /// Fused SiLU gating (SwiGLU): silu(gate) * up.
    InlineFusedSwiGLU,
    /// Reshape: zero-copy data, metadata-only shape change.
    InlineReshape,
    /// Dequantize Q4_0 → f32.
    InlineDequantize,
    /// 2-D convolution.
    InlineConv2d {
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
    InlineConvTranspose {
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
    InlineMaxPool2d {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// 2-D average pooling.
    InlineAvgPool2d {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    },
    /// Global average pool: spatial dims → 1.
    InlineGlobalAvgPool {
        channels: u32,
        spatial_h: u32,
        spatial_w: u32,
    },
    /// Resize (nearest/linear/cubic). Mode encoded as u8.
    InlineResize { mode: u8 },
    /// N-D padding. Mode: 0=constant, 1=reflect, 2=edge.
    InlinePad { mode: u8 },
    /// Instance normalization.
    InlineInstanceNorm { size: u32, epsilon: u32 },
    /// Local response normalization.
    InlineLRN {
        size: u32,
        alpha: u32,
        beta: u32,
        bias: u32,
    },
    /// Top-K along axis.
    InlineTopK { axis: u32, largest: bool },
    /// ScatterND.
    InlineScatterND,
    /// Cumulative sum along axis.
    InlineCumSum { axis: u32 },
    /// NonZero: returns indices of non-zero elements.
    InlineNonZero,
    /// Compress along axis.
    InlineCompress { axis: u32 },
    /// ReverseSequence along time/batch axes.
    InlineReverseSequence { batch_axis: u32, time_axis: u32 },

    /// Identity passthrough — same-type Cast only.
    Passthrough,

    /// Custom op — handler baked at tape build time from registry.
    Custom(crate::kv::CustomHandler),

    /// Group normalization.
    InlineGroupNorm { num_groups: u32, epsilon: u32 },

    // ── Epilogue fusion: norm + activation ────────────────────────────
    /// Fused RmsNorm + activation.
    InlineRmsNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused LayerNorm + activation.
    InlineLayerNormActivation {
        size: u32,
        epsilon: u32,
        activation: FloatOp,
    },
    /// Fused GroupNorm + activation.
    InlineGroupNormActivation {
        num_groups: u32,
        epsilon: u32,
        activation: FloatOp,
    },
}

impl TapeKernel {
    /// Returns the inline arity if this is an inline unary (1) or binary (2) op.
    /// Returns `None` for all other kernels (Float, Lut, MatMul, KvCache, etc.).
    #[inline]
    fn inline_arity(&self) -> Option<u8> {
        match self {
            TapeKernel::InlineRelu
            | TapeKernel::InlineNeg
            | TapeKernel::InlineAbs
            | TapeKernel::InlineSigmoid
            | TapeKernel::InlineSilu
            | TapeKernel::InlineTanh
            | TapeKernel::InlineGelu
            | TapeKernel::InlineExp
            | TapeKernel::InlineReciprocal
            | TapeKernel::InlineLog
            | TapeKernel::InlineSqrt
            | TapeKernel::InlineCos
            | TapeKernel::InlineSin
            | TapeKernel::InlineSign
            | TapeKernel::InlineFloor
            | TapeKernel::InlineCeil
            | TapeKernel::InlineRound
            | TapeKernel::InlineErf
            | TapeKernel::InlineClip { .. }
            | TapeKernel::InlineNot
            | TapeKernel::InlineIsNaN => Some(1),
            TapeKernel::InlineAdd
            | TapeKernel::InlineMul
            | TapeKernel::InlineSub
            | TapeKernel::InlineDiv
            | TapeKernel::InlineMin
            | TapeKernel::InlineMax
            | TapeKernel::InlinePow
            | TapeKernel::InlineMod
            | TapeKernel::InlineEqual
            | TapeKernel::InlineLess
            | TapeKernel::InlineLessOrEqual
            | TapeKernel::InlineGreater
            | TapeKernel::InlineGreaterOrEqual
            | TapeKernel::InlineAnd
            | TapeKernel::InlineOr
            | TapeKernel::InlineXor
            | TapeKernel::InlineFusedSwiGLU => Some(2),
            _ => None,
        }
    }
}

/// Result of kernel dispatch — tells the execute loop how to store the output.
enum DispatchResult {
    /// Output written to `out_buf`. Store via swap_insert.
    InOutBuf,
    /// Output written to `out_buf` with runtime-computed metadata that
    /// overrides the compiled output_meta (e.g., KV cache decode produces
    /// different shapes than compiled).
    InOutBufWithMeta(hologram_core::op::TensorMeta),
    /// Output stored in a Metal GPU buffer. Insert directly into arena.
    #[cfg(has_metal)]
    MetalBuffer(metal::Buffer),
    /// Output deferred to `flush_deferred()`. Skip swap_insert for now.
    #[cfg(has_webgpu)]
    WgpuDeferred,
}

/// Dispatch a `TapeKernel`, returning how the output should be stored.
///
/// For `Float` and `MatMul` ops, tries the selected backend first.
/// Falls back to CPU dispatch if the backend returns `Skipped`.
#[inline]
fn dispatch_kernel(
    kernel: &TapeKernel,
    inputs: &[&[u8]],
    input_metas: &crate::shape_resolve::InputMetas,
    tape_ctx: &TapeContext<'_>,
    backend: &dyn crate::backend::ComputeBackend,
    out_buf: &mut Vec<u8>,
) -> ExecResult<DispatchResult> {
    use crate::backend::KernelOutput;
    use crate::float_dispatch;
    use crate::kv::KvStore;
    use crate::shape_resolve;

    match kernel {
        TapeKernel::FusedFloatChain(chain) => {
            float_dispatch::dispatch_fused_chain_into(chain, inputs, out_buf)?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::Output => {
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::LutView(view) | TapeKernel::PrimUnary(view) => {
            out_buf.extend_from_slice(&KvStore::apply_unary(view, inputs[0]));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::PrimBinary(p) => {
            let r = KvStore::apply_binary(*p, inputs[0], inputs[1])?;
            out_buf.extend_from_slice(&r);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::MatMulLut4(cid) => {
            dispatch_lut_gemm_4(inputs, *cid, tape_ctx, out_buf)?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::MatMulLut8(cid) => {
            dispatch_lut_gemm_8(inputs, *cid, tape_ctx, out_buf)?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::MatMulLut4Activation(cid, activation) => {
            dispatch_lut_gemm_4(inputs, *cid, tape_ctx, out_buf)?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::MatMulLut8Activation(cid, activation) => {
            dispatch_lut_gemm_8(inputs, *cid, tape_ctx, out_buf)?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::KvWrite {
            layer,
            n_kv_heads,
            head_dim,
            is_key,
            heads_first,
        } => {
            let kv_meta = dispatch_kv_write(
                inputs,
                *layer,
                *n_kv_heads,
                *head_dim,
                *is_key,
                *heads_first,
                tape_ctx,
                out_buf,
            )?;
            Ok(DispatchResult::InOutBufWithMeta(kv_meta))
        }
        TapeKernel::KvRead {
            layer,
            n_kv_heads,
            head_dim,
            heads_first,
        } => {
            dispatch_kv_read(
                *layer,
                *n_kv_heads,
                *head_dim,
                *heads_first,
                tape_ctx,
                out_buf,
            )?;
            Ok(DispatchResult::InOutBuf)
        }

        // ── Inline hot ops (Phase 9a) ─────────────────────────────────
        // Direct kernel call — no backend, no dispatch_float_into, no category match.
        TapeKernel::InlineRelu => {
            inline_unary(inputs[0], out_buf, |v| v.max(0.0));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineNeg => {
            inline_unary(inputs[0], out_buf, |v| -v);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSigmoid => {
            inline_unary(inputs[0], out_buf, |v| 1.0 / (1.0 + (-v).exp()));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSilu => {
            inline_unary(inputs[0], out_buf, |v| v * (1.0 / (1.0 + (-v).exp())));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineTanh => {
            inline_unary(inputs[0], out_buf, |v| v.tanh());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGelu => {
            inline_unary(inputs[0], out_buf, |v| {
                0.5 * v
                    * (1.0
                        + (std::f32::consts::FRAC_2_SQRT_PI
                            * std::f32::consts::FRAC_1_SQRT_2
                            * (v + 0.044715 * v * v * v))
                            .tanh())
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineExp => {
            inline_unary(inputs[0], out_buf, |v| v.exp());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineAdd => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a + b);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineMul => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a * b);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSub => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a - b);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineDiv => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a / b);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineAbs => {
            inline_unary(inputs[0], out_buf, |v| v.abs());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReciprocal => {
            inline_unary(inputs[0], out_buf, |v| 1.0 / v);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineLog => {
            inline_unary(inputs[0], out_buf, |v| v.ln());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSqrt => {
            inline_unary(inputs[0], out_buf, |v| v.sqrt());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineCos => {
            inline_unary(inputs[0], out_buf, |v| v.cos());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSin => {
            inline_unary(inputs[0], out_buf, |v| v.sin());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSign => {
            inline_unary(inputs[0], out_buf, |v| v.signum());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineFloor => {
            inline_unary(inputs[0], out_buf, |v| v.floor());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineCeil => {
            inline_unary(inputs[0], out_buf, |v| v.ceil());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineRound => {
            inline_unary(inputs[0], out_buf, |v| v.round());
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineErf => {
            inline_unary(inputs[0], out_buf, |v| {
                // Abramowitz & Stegun approximation
                #[allow(clippy::excessive_precision)]
                const A1: f32 = 0.254_829_592;
                #[allow(clippy::excessive_precision)]
                const A2: f32 = -0.284_496_736;
                #[allow(clippy::excessive_precision)]
                const A3: f32 = 1.421_413_741;
                #[allow(clippy::excessive_precision)]
                const A4: f32 = -1.453_152_027;
                #[allow(clippy::excessive_precision)]
                const A5: f32 = 1.061_405_429;
                #[allow(clippy::excessive_precision)]
                const P: f32 = 0.327_591_1;
                let sign = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
                let x = v.abs();
                let t = 1.0 / (1.0 + P * x);
                let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
                sign * y
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineMin => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a.min(b));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineMax => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a.max(b));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineLayerNorm { size, epsilon } => {
            let actual = shape_resolve::resolve_last_dim_with_weight(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
                inputs.get(1).map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_layer_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineAddRmsNorm { size, epsilon } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_add_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineLogSoftmax { size } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_log_softmax_into(inputs, actual, out_buf)?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineAttention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
        } => {
            let result = crate::float_dispatch::attention::dispatch_attention(
                inputs,
                *head_dim as usize,
                *num_q_heads as usize,
                *num_kv_heads as usize,
                f32::from_bits(*scale),
                *causal,
                *heads_first,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineRoPE { dim, base, n_heads } => {
            let start_pos = tape_ctx
                .ctx
                .as_ref()
                .map(|c| c.position_offset as usize)
                .unwrap_or(0);
            let result = crate::float_dispatch::attention::dispatch_rope(
                inputs,
                *dim as usize,
                f32::from_bits(*base),
                *n_heads as usize,
                start_pos,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGather { dim, dtype } => {
            let result = crate::float_dispatch::gather_concat::dispatch_gather(
                inputs,
                *dim as usize,
                *dtype,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineConcat {
            size_a,
            size_b,
            dtype,
        } => {
            let result = crate::float_dispatch::gather_concat::dispatch_concat(
                inputs,
                *size_a as usize,
                *size_b as usize,
                *dtype,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineTranspose {
            perm,
            input_shape,
            ndim,
        } => {
            let n = *ndim as usize;
            let compiled_shape: Vec<usize> = input_shape[..n].iter().map(|&d| d as usize).collect();
            let perm_slice: &[u8] = &perm[..n];

            // Verify baked shape matches actual input size. If the input
            // is a different size (e.g., KV cache produced a runtime-sized
            // tensor), infer the actual shape by scaling the variable dim.
            let input_elems = inputs[0].len() / 4; // f32 elements
            let compiled_elems: usize = compiled_shape.iter().product();
            let shape = if compiled_elems > 0 && compiled_elems == input_elems {
                compiled_shape
            } else if compiled_elems > 0 && input_elems > 0 {
                // Find the dim that changed (variable-length dim like seq)
                // and scale it to match the actual input size.
                let mut adjusted = compiled_shape.clone();
                let ratio = input_elems as f64 / compiled_elems as f64;
                // Find the dim most likely to be variable (not head_dim, not n_heads).
                // Heuristic: the dim that, when scaled by ratio, gives an integer.
                for i in 0..adjusted.len() {
                    let scaled = (adjusted[i] as f64 * ratio).round() as usize;
                    let check: usize = adjusted
                        .iter()
                        .enumerate()
                        .map(|(j, &d)| if j == i { scaled } else { d })
                        .product();
                    if check == input_elems {
                        adjusted[i] = scaled;
                        break;
                    }
                }
                adjusted
            } else {
                // Can't determine shape — passthrough (identity).
                out_buf.extend_from_slice(inputs[0]);
                return Ok(DispatchResult::InOutBuf);
            };

            let (result, out_shape) =
                crate::float_dispatch::dispatch_transpose(inputs[0], perm_slice, &shape)?;
            out_buf.extend_from_slice(&result);
            // Propagate permuted shape as output meta.
            let meta =
                hologram_core::op::TensorMeta::new(hologram_core::op::FloatDType::F32, &out_shape);
            Ok(DispatchResult::InOutBufWithMeta(meta))
        }
        TapeKernel::Passthrough => {
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReshape => {
            // Zero-copy: bytes unchanged, only metadata changes.
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchResult::InOutBuf)
        }

        // ── New inline simple ops (Phase 10: complete TapeKernel coverage) ──
        TapeKernel::InlinePow => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a.powf(b));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineMod => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a % b);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineClip { min, max } => {
            let min_f = hologram_core::op::bits_to_f32(*min);
            let max_f = hologram_core::op::bits_to_f32(*max);
            inline_unary(inputs[0], out_buf, |v| v.max(min_f).min(max_f));
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineIsNaN => {
            inline_unary(inputs[0], out_buf, |v| if v.is_nan() { 1.0 } else { 0.0 });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineNot => {
            inline_unary(inputs[0], out_buf, |v| if v == 0.0 { 1.0 } else { 0.0 });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineAnd => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| {
                if a != 0.0 && b != 0.0 {
                    1.0
                } else {
                    0.0
                }
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineOr => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| {
                if a != 0.0 || b != 0.0 {
                    1.0
                } else {
                    0.0
                }
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineXor => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| {
                if (a != 0.0) ^ (b != 0.0) {
                    1.0
                } else {
                    0.0
                }
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineEqual => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| {
                if a == b {
                    1.0
                } else {
                    0.0
                }
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineLess => {
            inline_binary(
                inputs[0],
                inputs[1],
                out_buf,
                |a, b| if a < b { 1.0 } else { 0.0 },
            );
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineLessOrEqual => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| {
                if a <= b {
                    1.0
                } else {
                    0.0
                }
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGreater => {
            inline_binary(
                inputs[0],
                inputs[1],
                out_buf,
                |a, b| if a > b { 1.0 } else { 0.0 },
            );
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGreaterOrEqual => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| {
                if a >= b {
                    1.0
                } else {
                    0.0
                }
            });
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineFusedSwiGLU => {
            inline_binary(inputs[0], inputs[1], out_buf, |g, u| {
                g * (1.0 / (1.0 + (-g).exp())) * u
            });
            Ok(DispatchResult::InOutBuf)
        }

        // ── Complex ops (call existing handlers) ──────────────────────────
        TapeKernel::InlineGemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
            quant_b,
        } => {
            let (actual_m, actual_k, actual_n) = shape_resolve::resolve_matmul_dims(
                *m,
                *k,
                *n,
                input_metas.first().and_then(|m| m.as_ref()),
                input_metas.get(1).and_then(|m| m.as_ref()),
                inputs[0].len(),
                inputs.get(1).map(|b| b.len()).unwrap_or(0),
            );
            let result = float_dispatch::matmul::dispatch_gemm(
                inputs,
                float_dispatch::matmul::GemmParams {
                    m: actual_m,
                    n: actual_n,
                    k: actual_k,
                    alpha: hologram_core::op::bits_to_f32(*alpha),
                    beta: hologram_core::op::bits_to_f32(*beta),
                    trans_a: *trans_a,
                    trans_b: *trans_b,
                },
                *quant_b,
            )?;
            out_buf.extend_from_slice(&result);
            // Gemm output: [M, N]
            let meta = hologram_core::op::TensorMeta::new(
                hologram_core::op::FloatDType::F32,
                &[actual_m, actual_n],
            );
            Ok(DispatchResult::InOutBufWithMeta(meta))
        }
        TapeKernel::InlineReduceSum { size } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_sum,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReduceMean { size } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_mean,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReduceMax { size } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_max,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReduceMin { size } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_min,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReduceProd { size } => {
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                *size as usize,
                float_dispatch::reduce::reduce_prod,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineCast { from, to } => {
            let result = float_dispatch::cast::dispatch_cast(inputs, *from, *to)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineEmbed { dim, quant } => {
            let result = float_dispatch::cast::dispatch_embed(inputs, *dim as usize, *quant)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineWhere => {
            let result = float_dispatch::gather_concat::dispatch_where(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineRange => {
            let result = float_dispatch::gather_concat::dispatch_range(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineShape { dtype, start, end } => {
            let result =
                float_dispatch::gather_concat::dispatch_shape(inputs, *dtype, *start, *end)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineSlice {
            axis_from_end,
            start,
            end,
            axis_size,
        } => {
            // Resolve actual axis_size from input meta when available.
            // axis_from_end encoding: ndim - original_axis (set during lowering).
            // Recover: axis = ndim - axis_from_end.
            let resolved_axis_size = input_metas
                .first()
                .and_then(|m| m.as_ref())
                .and_then(|meta| {
                    let n = meta.ndim as usize;
                    let afe = *axis_from_end as usize;
                    (afe > 0 && afe <= n).then(|| meta.dims[n - afe])
                })
                .filter(|&d| d > 0)
                .unwrap_or(*axis_size);

            let result = float_dispatch::dispatch_float_ctx(
                &FloatOp::Slice {
                    axis_from_end: *axis_from_end,
                    start: *start,
                    end: *end,
                    axis_size: resolved_axis_size,
                },
                inputs,
                tape_ctx.ctx.as_ref(),
            )?;
            out_buf.extend_from_slice(&result);

            // Output meta: adjust the sliced axis dimension.
            if let Some(in_meta) = input_metas.first().and_then(|m| m.as_ref()) {
                let n = in_meta.ndim as usize;
                let afe = *axis_from_end as usize;
                if afe > 0 && afe <= n {
                    let axis = n - afe;
                    let effective_end = (*end).min(resolved_axis_size);
                    let slice_len = effective_end.saturating_sub(*start);
                    let mut out_meta = *in_meta;
                    out_meta.dims[axis] = slice_len;
                    return Ok(DispatchResult::InOutBufWithMeta(out_meta));
                }
            }
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGatherND => {
            // GatherND: pass-through (same as Reshape — data unchanged).
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineDequantize => {
            let result = float_dispatch::cast::dispatch_dequantize(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineConv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            input_h,
            input_w,
        } => {
            let (actual_h, actual_w) = shape_resolve::resolve_spatial_dims(
                *input_h,
                *input_w,
                input_metas.first().and_then(|m| m.as_ref()),
            );
            let result = float_dispatch::conv::dispatch_conv2d_direct(
                inputs,
                *kernel_h as usize,
                *kernel_w as usize,
                *stride_h as usize,
                *stride_w as usize,
                *pad_h as usize,
                *pad_w as usize,
                *dilation_h as usize,
                *dilation_w as usize,
                *group as usize,
                actual_h,
                actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            // Compute output meta: [N, C_out, H_out, W_out] from input + weight shapes.
            if let (Some(in_meta), Some(w_meta)) = (
                input_metas.first().and_then(|m| m.as_ref()),
                input_metas.get(1).and_then(|m| m.as_ref()),
            ) {
                if in_meta.ndim >= 4 && w_meta.ndim >= 1 {
                    let n = in_meta.dims[0] as usize;
                    let c_out = w_meta.dims[0] as usize;
                    let sh = (*stride_h).max(1) as usize;
                    let sw = (*stride_w).max(1) as usize;
                    let dh = (*dilation_h).max(1) as usize;
                    let dw = (*dilation_w).max(1) as usize;
                    let h_out =
                        (actual_h + 2 * (*pad_h as usize) - dh * (*kernel_h as usize - 1) - 1) / sh
                            + 1;
                    let w_out =
                        (actual_w + 2 * (*pad_w as usize) - dw * (*kernel_w as usize - 1) - 1) / sw
                            + 1;
                    let meta = hologram_core::op::TensorMeta::new(
                        hologram_core::op::FloatDType::F32,
                        &[n, c_out, h_out, w_out],
                    );
                    return Ok(DispatchResult::InOutBufWithMeta(meta));
                }
            }
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineConvTranspose {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            output_pad_h,
            output_pad_w,
            input_h,
            input_w,
        } => {
            let (actual_h, actual_w) = shape_resolve::resolve_spatial_dims(
                *input_h,
                *input_w,
                input_metas.first().and_then(|m| m.as_ref()),
            );
            let result = float_dispatch::conv::dispatch_conv_transpose(
                inputs,
                *kernel_h as usize,
                *kernel_w as usize,
                *stride_h as usize,
                *stride_w as usize,
                *pad_h as usize,
                *pad_w as usize,
                *dilation_h as usize,
                *dilation_w as usize,
                *group as usize,
                *output_pad_h as usize,
                *output_pad_w as usize,
                actual_h,
                actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineMaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            let result = float_dispatch::pool::dispatch_max_pool_2d(
                inputs,
                *kernel_h as usize,
                *kernel_w as usize,
                *stride_h as usize,
                *stride_w as usize,
                *pad_h as usize,
                *pad_w as usize,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineAvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            let result = float_dispatch::pool::dispatch_avg_pool_2d(
                inputs,
                *kernel_h as usize,
                *kernel_w as usize,
                *stride_h as usize,
                *stride_w as usize,
                *pad_h as usize,
                *pad_w as usize,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGlobalAvgPool {
            channels,
            spatial_h,
            spatial_w,
        } => {
            let (actual_c, actual_h, actual_w) = shape_resolve::resolve_global_avg_pool_dims(
                *channels,
                *spatial_h,
                *spatial_w,
                input_metas.first().and_then(|m| m.as_ref()),
            );
            let result = float_dispatch::pool::dispatch_global_avg_pool_direct(
                inputs, actual_c, actual_h, actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineResize { mode } => {
            let result = float_dispatch::spatial::dispatch_resize(inputs, *mode)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlinePad { mode } => {
            let result = float_dispatch::spatial::dispatch_pad(inputs, *mode)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineInstanceNorm { size, epsilon } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            let result = float_dispatch::norm::dispatch_instance_norm(
                inputs,
                actual,
                f32::from_bits(*epsilon),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGroupNorm {
            num_groups,
            epsilon,
        } => {
            let result = float_dispatch::norm::dispatch_group_norm(
                inputs,
                *num_groups as usize,
                f32::from_bits(*epsilon),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }

        // ── Fused norm + activation (epilogue fusion) ────────────────
        TapeKernel::InlineRmsNormActivation {
            size,
            epsilon,
            activation,
        } => {
            let actual = shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineLayerNormActivation {
            size,
            epsilon,
            activation,
        } => {
            let actual = shape_resolve::resolve_last_dim_with_weight(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
                inputs.get(1).map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_layer_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineGroupNormActivation {
            num_groups,
            epsilon,
            activation,
        } => {
            let result = float_dispatch::norm::dispatch_group_norm(
                inputs,
                *num_groups as usize,
                f32::from_bits(*epsilon),
            )?;
            out_buf.extend_from_slice(&result);
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchResult::InOutBuf)
        }

        TapeKernel::InlineLRN {
            size,
            alpha,
            beta,
            bias,
        } => {
            let result = float_dispatch::norm::dispatch_lrn(
                inputs,
                *size as usize,
                hologram_core::op::bits_to_f32(*alpha),
                hologram_core::op::bits_to_f32(*beta),
                hologram_core::op::bits_to_f32(*bias),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineTopK { axis, largest } => {
            let result = float_dispatch::misc::dispatch_top_k(inputs, *axis as usize, *largest)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineScatterND => {
            let result = float_dispatch::misc::dispatch_scatter_nd(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineCumSum { axis } => {
            let result = float_dispatch::misc::dispatch_cumsum(inputs, *axis as usize)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineNonZero => {
            let result = float_dispatch::misc::dispatch_nonzero(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineCompress { axis } => {
            let result = float_dispatch::misc::dispatch_compress(inputs, *axis as usize)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineReverseSequence {
            batch_axis,
            time_axis,
        } => {
            let result = float_dispatch::misc::dispatch_reverse_sequence(
                inputs,
                *batch_axis as usize,
                *time_axis as usize,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }

        // ── Inline custom ops (Phase 9a.3–9a.4) ─────────────────────────
        // Try backend (GPU) first, then direct CPU kernel call.
        TapeKernel::InlineMatMul { m, k, n } => {
            // Try N-D metadata first, then fall back to buffer-length heuristic.
            let meta_dims = shape_resolve::resolve_matmul_dims(
                *m,
                *k,
                *n,
                input_metas.first().and_then(|m| m.as_ref()),
                input_metas.get(1).and_then(|m| m.as_ref()),
                inputs[0].len(),
                inputs[1].len(),
            );
            // Validate: k must divide both buffers cleanly.
            let a_floats = inputs[0].len() / 4;
            let b_floats = inputs[1].len() / 4;
            let (actual_m, actual_k, actual_n) = if meta_dims.1 > 0
                && a_floats > 0
                && b_floats > 0
                && a_floats.is_multiple_of(meta_dims.1)
                && b_floats.is_multiple_of(meta_dims.1)
            {
                meta_dims
            } else {
                // Fall back to buffer-length inference.
                crate::float_dispatch::matmul::infer_matmul_dims(
                    *m as usize,
                    *k as usize,
                    *n as usize,
                    a_floats,
                    b_floats,
                )
            };
            // Skip backend dispatch — use CPU for now to validate correctness.
            // TODO: re-enable backend.dispatch_matmul with adapted dims once validated.
            crate::float_dispatch::matmul::dispatch_matmul_into(
                inputs, actual_m, actual_k, actual_n, out_buf,
            )?;
            // Compute output meta: [batch, M, N] from resolved dims.
            let batch = if actual_m > 0 && actual_k > 0 {
                a_floats / (actual_m * actual_k)
            } else {
                1
            };
            let meta = if batch > 1 {
                hologram_core::op::TensorMeta::new(
                    hologram_core::op::FloatDType::F32,
                    &[batch, actual_m, actual_n],
                )
            } else {
                hologram_core::op::TensorMeta::new(
                    hologram_core::op::FloatDType::F32,
                    &[actual_m, actual_n],
                )
            };
            Ok(DispatchResult::InOutBufWithMeta(meta))
        }
        TapeKernel::InlineMatMulBiasActivation {
            m,
            k,
            n,
            activation,
        } => {
            // inputs: [activation_tensor, weight, bias] — all zero-copy from arena.
            let bias: &[f32] = bytemuck::try_cast_slice(inputs[2]).map_err(|_| {
                crate::error::ExecError::UnsupportedOp("bias not f32-aligned".into())
            })?;
            // Resolve runtime dimensions from N-D input metas (same as InlineMatMul).
            let meta_dims = shape_resolve::resolve_matmul_dims(
                *m,
                *k,
                *n,
                input_metas.first().and_then(|m| m.as_ref()),
                input_metas.get(1).and_then(|m| m.as_ref()),
                inputs[0].len(),
                inputs[1].len(),
            );
            let a_floats = inputs[0].len() / 4;
            let b_floats = inputs[1].len() / 4;
            let (actual_m, actual_k, actual_n) = if meta_dims.1 > 0
                && a_floats > 0
                && b_floats > 0
                && a_floats.is_multiple_of(meta_dims.1)
                && b_floats.is_multiple_of(meta_dims.1)
            {
                meta_dims
            } else {
                crate::float_dispatch::matmul::infer_matmul_dims(
                    *m as usize,
                    *k as usize,
                    *n as usize,
                    a_floats,
                    b_floats,
                )
            };
            crate::float_dispatch::matmul::dispatch_matmul_bias_activation_into(
                &inputs[..2],
                actual_m,
                actual_k,
                actual_n,
                bias,
                activation,
                out_buf,
            )?;
            let batch = if actual_m > 0 && actual_k > 0 {
                a_floats / (actual_m * actual_k)
            } else {
                1
            };
            let meta = if batch > 1 {
                hologram_core::op::TensorMeta::new(
                    hologram_core::op::FloatDType::F32,
                    &[batch, actual_m, actual_n],
                )
            } else {
                hologram_core::op::TensorMeta::new(
                    hologram_core::op::FloatDType::F32,
                    &[actual_m, actual_n],
                )
            };
            Ok(DispatchResult::InOutBufWithMeta(meta))
        }
        TapeKernel::InlineMatMulActivation {
            m,
            k,
            n,
            activation,
        } => {
            let meta_dims = shape_resolve::resolve_matmul_dims(
                *m,
                *k,
                *n,
                input_metas.first().and_then(|m| m.as_ref()),
                input_metas.get(1).and_then(|m| m.as_ref()),
                inputs[0].len(),
                inputs[1].len(),
            );
            let a_floats = inputs[0].len() / 4;
            let b_floats = inputs[1].len() / 4;
            let (actual_m, actual_k, actual_n) = if meta_dims.1 > 0
                && a_floats > 0
                && b_floats > 0
                && a_floats.is_multiple_of(meta_dims.1)
                && b_floats.is_multiple_of(meta_dims.1)
            {
                meta_dims
            } else {
                crate::float_dispatch::matmul::infer_matmul_dims(
                    *m as usize,
                    *k as usize,
                    *n as usize,
                    a_floats,
                    b_floats,
                )
            };
            crate::float_dispatch::matmul::dispatch_matmul_activation_into(
                inputs, actual_m, actual_k, actual_n, activation, out_buf,
            )?;
            let batch = if actual_m > 0 && actual_k > 0 {
                a_floats / (actual_m * actual_k)
            } else {
                1
            };
            let meta = if batch > 1 {
                hologram_core::op::TensorMeta::new(
                    hologram_core::op::FloatDType::F32,
                    &[batch, actual_m, actual_n],
                )
            } else {
                hologram_core::op::TensorMeta::new(
                    hologram_core::op::FloatDType::F32,
                    &[actual_m, actual_n],
                )
            };
            Ok(DispatchResult::InOutBufWithMeta(meta))
        }
        TapeKernel::InlineSoftmax { size } => {
            match backend.dispatch_float(&FloatOp::Softmax { size: *size }, inputs, out_buf)? {
                KernelOutput::Bytes => return Ok(DispatchResult::InOutBuf),
                #[cfg(has_metal)]
                KernelOutput::MetalBuffer(buf) => {
                    return Ok(DispatchResult::MetalBuffer(buf));
                }
                #[cfg(has_webgpu)]
                KernelOutput::WgpuDeferred => return Ok(DispatchResult::WgpuDeferred),
                KernelOutput::Skipped => {}
            }
            let actual = crate::shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_softmax_into(inputs, actual, out_buf)?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::InlineRmsNorm { size, epsilon } => {
            match backend.dispatch_float(
                &FloatOp::RmsNorm {
                    size: *size,
                    epsilon: *epsilon,
                },
                inputs,
                out_buf,
            )? {
                KernelOutput::Bytes => return Ok(DispatchResult::InOutBuf),
                #[cfg(has_metal)]
                KernelOutput::MetalBuffer(buf) => {
                    return Ok(DispatchResult::MetalBuffer(buf));
                }
                #[cfg(has_webgpu)]
                KernelOutput::WgpuDeferred => return Ok(DispatchResult::WgpuDeferred),
                KernelOutput::Skipped => {}
            }
            let actual = crate::shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::Custom(handler) => {
            let result = handler(inputs, tape_ctx.constants)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchResult::InOutBuf)
        }
    }
}

/// Binary elementwise with broadcasting. Fast paths avoid per-element modulo.
#[inline(always)]
fn binary_broadcast(a: &[f32], b: &[f32], dst: &mut [f32], f: impl Fn(f32, f32) -> f32) {
    if a.len() == b.len() {
        for (d, (&x, &y)) in dst.iter_mut().zip(a.iter().zip(b.iter())) {
            *d = f(x, y);
        }
    } else if b.len() == 1 {
        let bv = b[0];
        for (d, &x) in dst.iter_mut().zip(a.iter()) {
            *d = f(x, bv);
        }
    } else if a.len() == 1 {
        let av = a[0];
        for (d, &y) in dst.iter_mut().zip(b.iter()) {
            *d = f(av, y);
        }
    } else {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = f(a[i % a.len()], b[i % b.len()]);
        }
    }
}

/// Inline unary kernel — writes directly to out_buf as f32 via bytemuck cast.
/// No dispatch overhead, no intermediate allocation.
#[inline(always)]
fn inline_unary(input: &[u8], out_buf: &mut Vec<u8>, f: impl Fn(f32) -> f32) {
    let x: &[f32] = bytemuck::cast_slice(input);
    let byte_len = x.len() * 4;
    let base = out_buf.len();
    out_buf.resize(base + byte_len, 0);
    let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut out_buf[base..]);
    for (d, &s) in dst.iter_mut().zip(x.iter()) {
        *d = f(s);
    }
}

/// Inline binary kernel — writes directly to out_buf as f32 via bytemuck cast.
#[inline(always)]
fn inline_binary(a: &[u8], b: &[u8], out_buf: &mut Vec<u8>, f: impl Fn(f32, f32) -> f32) {
    let a: &[f32] = bytemuck::cast_slice(a);
    let b: &[f32] = bytemuck::cast_slice(b);
    let out_len = a.len().max(b.len());
    let byte_len = out_len * 4;
    let base = out_buf.len();
    out_buf.resize(base + byte_len, 0);
    let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut out_buf[base..]);
    binary_broadcast(a, b, dst, f);
}

/// Typed unary kernel — input already cast to `&[f32]` by caller.
/// Eliminates input-side bytemuck cast per kernel call.
#[inline(always)]
fn inline_unary_f32(input: &[f32], out_buf: &mut Vec<u8>, f: impl Fn(f32) -> f32) {
    let byte_len = input.len() * 4;
    let base = out_buf.len();
    out_buf.resize(base + byte_len, 0);
    let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut out_buf[base..]);
    for (d, &s) in dst.iter_mut().zip(input.iter()) {
        *d = f(s);
    }
}

/// Typed binary kernel — inputs already cast to `&[f32]` by caller.
#[inline(always)]
fn inline_binary_f32(a: &[f32], b: &[f32], out_buf: &mut Vec<u8>, f: impl Fn(f32, f32) -> f32) {
    let out_len = a.len().max(b.len());
    let byte_len = out_len * 4;
    let base = out_buf.len();
    out_buf.resize(base + byte_len, 0);
    let dst: &mut [f32] = bytemuck::cast_slice_mut(&mut out_buf[base..]);
    binary_broadcast(a, b, dst, f);
}

/// Apply a unary inline op in-place on an owned f32 buffer.
/// Avoids allocation — the kernel overwrites the input data directly.
#[inline(always)]
fn inline_unary_inplace(buf: &mut [f32], f: impl Fn(f32) -> f32) {
    for v in buf.iter_mut() {
        *v = f(*v);
    }
}

/// Dispatch an inline unary op with typed `&[f32]` input (Phase 9d).
/// Caller casts once via `arena.get_f32()`, kernel works with native types.
#[inline]
fn dispatch_inline_unary(kernel: &TapeKernel, input: &[f32], out_buf: &mut Vec<u8>) {
    match kernel {
        TapeKernel::InlineRelu => inline_unary_f32(input, out_buf, |v| v.max(0.0)),
        TapeKernel::InlineNeg => inline_unary_f32(input, out_buf, |v| -v),
        TapeKernel::InlineAbs => inline_unary_f32(input, out_buf, |v| v.abs()),
        TapeKernel::InlineSigmoid => {
            inline_unary_f32(input, out_buf, |v| 1.0 / (1.0 + (-v).exp()));
        }
        TapeKernel::InlineSilu => {
            inline_unary_f32(input, out_buf, |v| v * (1.0 / (1.0 + (-v).exp())));
        }
        TapeKernel::InlineTanh => inline_unary_f32(input, out_buf, |v| v.tanh()),
        TapeKernel::InlineGelu => {
            inline_unary_f32(input, out_buf, |v| {
                0.5 * v
                    * (1.0
                        + (std::f32::consts::FRAC_2_SQRT_PI
                            * std::f32::consts::FRAC_1_SQRT_2
                            * (v + 0.044715 * v * v * v))
                            .tanh())
            });
        }
        TapeKernel::InlineExp => inline_unary_f32(input, out_buf, |v| v.exp()),
        TapeKernel::InlineLog => inline_unary_f32(input, out_buf, |v| v.ln()),
        TapeKernel::InlineSqrt => inline_unary_f32(input, out_buf, |v| v.sqrt()),
        TapeKernel::InlineCos => inline_unary_f32(input, out_buf, |v| v.cos()),
        TapeKernel::InlineSin => inline_unary_f32(input, out_buf, |v| v.sin()),
        TapeKernel::InlineSign => inline_unary_f32(input, out_buf, |v| v.signum()),
        TapeKernel::InlineFloor => inline_unary_f32(input, out_buf, |v| v.floor()),
        TapeKernel::InlineCeil => inline_unary_f32(input, out_buf, |v| v.ceil()),
        TapeKernel::InlineRound => inline_unary_f32(input, out_buf, |v| v.round()),
        TapeKernel::InlineErf => {
            // Abramowitz & Stegun approximation (same as dispatch_kernel path).
            inline_unary_f32(input, out_buf, |v| {
                #[allow(clippy::excessive_precision)]
                const A1: f32 = 0.254_829_592;
                #[allow(clippy::excessive_precision)]
                const A2: f32 = -0.284_496_736;
                #[allow(clippy::excessive_precision)]
                const A3: f32 = 1.421_413_741;
                #[allow(clippy::excessive_precision)]
                const A4: f32 = -1.453_152_027;
                #[allow(clippy::excessive_precision)]
                const A5: f32 = 1.061_405_429;
                #[allow(clippy::excessive_precision)]
                const P: f32 = 0.327_591_1;
                let sign = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
                let x = v.abs();
                let t = 1.0 / (1.0 + P * x);
                let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
                sign * y
            });
        }
        TapeKernel::InlineReciprocal => inline_unary_f32(input, out_buf, |v| 1.0 / v),
        TapeKernel::InlineClip { min, max } => {
            let min_f = hologram_core::op::bits_to_f32(*min);
            let max_f = hologram_core::op::bits_to_f32(*max);
            inline_unary_f32(input, out_buf, |v| v.max(min_f).min(max_f));
        }
        TapeKernel::InlineNot => {
            inline_unary_f32(input, out_buf, |v| if v == 0.0 { 1.0 } else { 0.0 })
        }
        TapeKernel::InlineIsNaN => {
            // IsNaN outputs f32 1.0/0.0 in the inline path (consistent with f32 arena).
            inline_unary_f32(input, out_buf, |v| if v.is_nan() { 1.0 } else { 0.0 });
        }
        _ => unreachable!("dispatch_inline_unary called for non-unary kernel"),
    }
}

/// Dispatch an inline binary op with typed `&[f32]` inputs (Phase 9d).
#[inline]
fn dispatch_inline_binary(kernel: &TapeKernel, a: &[f32], b: &[f32], out_buf: &mut Vec<u8>) {
    match kernel {
        TapeKernel::InlineAdd => inline_binary_f32(a, b, out_buf, |x, y| x + y),
        TapeKernel::InlineMul => inline_binary_f32(a, b, out_buf, |x, y| x * y),
        TapeKernel::InlineSub => inline_binary_f32(a, b, out_buf, |x, y| x - y),
        TapeKernel::InlineDiv => inline_binary_f32(a, b, out_buf, |x, y| x / y),
        TapeKernel::InlineMin => inline_binary_f32(a, b, out_buf, |x, y| x.min(y)),
        TapeKernel::InlineMax => inline_binary_f32(a, b, out_buf, |x, y| x.max(y)),
        TapeKernel::InlinePow => inline_binary_f32(a, b, out_buf, |x, y| x.powf(y)),
        TapeKernel::InlineMod => inline_binary_f32(a, b, out_buf, |x, y| x % y),
        TapeKernel::InlineEqual => {
            inline_binary_f32(a, b, out_buf, |x, y| if x == y { 1.0 } else { 0.0 });
        }
        TapeKernel::InlineLess => {
            inline_binary_f32(a, b, out_buf, |x, y| if x < y { 1.0 } else { 0.0 });
        }
        TapeKernel::InlineLessOrEqual => {
            inline_binary_f32(a, b, out_buf, |x, y| if x <= y { 1.0 } else { 0.0 });
        }
        TapeKernel::InlineGreater => {
            inline_binary_f32(a, b, out_buf, |x, y| if x > y { 1.0 } else { 0.0 });
        }
        TapeKernel::InlineGreaterOrEqual => {
            inline_binary_f32(a, b, out_buf, |x, y| if x >= y { 1.0 } else { 0.0 });
        }
        TapeKernel::InlineAnd => {
            inline_binary_f32(
                a,
                b,
                out_buf,
                |x, y| {
                    if x != 0.0 && y != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                },
            );
        }
        TapeKernel::InlineOr => {
            inline_binary_f32(
                a,
                b,
                out_buf,
                |x, y| {
                    if x != 0.0 || y != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                },
            );
        }
        TapeKernel::InlineXor => {
            inline_binary_f32(a, b, out_buf, |x, y| {
                if (x != 0.0) ^ (y != 0.0) {
                    1.0
                } else {
                    0.0
                }
            });
        }
        TapeKernel::InlineFusedSwiGLU => {
            // silu(gate) * up = gate * sigmoid(gate) * up
            inline_binary_f32(a, b, out_buf, |g, u| g * (1.0 / (1.0 + (-g).exp())) * u);
        }
        _ => unreachable!("dispatch_inline_binary called for non-binary kernel"),
    }
}

/// Try to dispatch a unary inline op in-place on typed f32 data.
/// Returns `true` if handled.
#[inline]
fn dispatch_inplace(kernel: &TapeKernel, buf: &mut [f32]) -> bool {
    match kernel {
        TapeKernel::InlineRelu => {
            inline_unary_inplace(buf, |v| v.max(0.0));
            true
        }
        TapeKernel::InlineNeg => {
            inline_unary_inplace(buf, |v| -v);
            true
        }
        TapeKernel::InlineAbs => {
            inline_unary_inplace(buf, |v| v.abs());
            true
        }
        TapeKernel::InlineSigmoid => {
            inline_unary_inplace(buf, |v| 1.0 / (1.0 + (-v).exp()));
            true
        }
        TapeKernel::InlineSilu => {
            inline_unary_inplace(buf, |v| v * (1.0 / (1.0 + (-v).exp())));
            true
        }
        TapeKernel::InlineTanh => {
            inline_unary_inplace(buf, |v| v.tanh());
            true
        }
        TapeKernel::InlineGelu => {
            inline_unary_inplace(buf, |v| {
                0.5 * v
                    * (1.0
                        + (std::f32::consts::FRAC_2_SQRT_PI
                            * std::f32::consts::FRAC_1_SQRT_2
                            * (v + 0.044715 * v * v * v))
                            .tanh())
            });
            true
        }
        TapeKernel::InlineExp => {
            inline_unary_inplace(buf, |v| v.exp());
            true
        }
        TapeKernel::InlineReciprocal => {
            inline_unary_inplace(buf, |v| 1.0 / v);
            true
        }
        TapeKernel::InlineLog => {
            inline_unary_inplace(buf, |v| v.ln());
            true
        }
        TapeKernel::InlineSqrt => {
            inline_unary_inplace(buf, |v| v.sqrt());
            true
        }
        TapeKernel::InlineCos => {
            inline_unary_inplace(buf, |v| v.cos());
            true
        }
        TapeKernel::InlineSin => {
            inline_unary_inplace(buf, |v| v.sin());
            true
        }
        TapeKernel::InlineSign => {
            inline_unary_inplace(buf, |v| v.signum());
            true
        }
        TapeKernel::InlineFloor => {
            inline_unary_inplace(buf, |v| v.floor());
            true
        }
        TapeKernel::InlineCeil => {
            inline_unary_inplace(buf, |v| v.ceil());
            true
        }
        TapeKernel::InlineRound => {
            inline_unary_inplace(buf, |v| v.round());
            true
        }
        TapeKernel::InlineErf => {
            #[allow(clippy::excessive_precision)]
            inline_unary_inplace(buf, |v| {
                const A1: f32 = 0.254_829_592;
                const A2: f32 = -0.284_496_736;
                const A3: f32 = 1.421_413_741;
                const A4: f32 = -1.453_152_027;
                const A5: f32 = 1.061_405_429;
                const P: f32 = 0.327_591_1;
                let sign = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
                let x = v.abs();
                let t = 1.0 / (1.0 + P * x);
                sign * (1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp())
            });
            true
        }
        TapeKernel::InlineClip { min, max } => {
            let min_f = hologram_core::op::bits_to_f32(*min);
            let max_f = hologram_core::op::bits_to_f32(*max);
            inline_unary_inplace(buf, |v| v.max(min_f).min(max_f));
            true
        }
        TapeKernel::InlineNot => {
            inline_unary_inplace(buf, |v| if v == 0.0 { 1.0 } else { 0.0 });
            true
        }
        TapeKernel::InlineIsNaN => {
            inline_unary_inplace(buf, |v| if v.is_nan() { 1.0 } else { 0.0 });
            true
        }
        _ => false,
    }
}

/// Sync-safe dispatch for parallelizable ops (no RefCell access).
///
/// Only handles Float, FusedChain, Output, LutView, PrimUnary, PrimBinary.
/// LUT-GEMM and KvCache ops are excluded from parallel levels.
#[cfg(feature = "parallel")]
#[inline]
fn dispatch_kernel_par(
    kernel: &TapeKernel,
    inputs: &[&[u8]],
    input_metas: &crate::shape_resolve::InputMetas,
    _ctx: Option<&ExecutionContext>,
    constants: &ConstantStore,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    use crate::float_dispatch;
    use crate::kv::KvStore;

    match kernel {
        TapeKernel::FusedFloatChain(chain) => {
            float_dispatch::dispatch_fused_chain_into(chain, inputs, out_buf)
        }
        TapeKernel::Output => {
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(())
        }
        TapeKernel::LutView(view) | TapeKernel::PrimUnary(view) => {
            out_buf.extend_from_slice(&KvStore::apply_unary(view, inputs[0]));
            Ok(())
        }
        TapeKernel::PrimBinary(p) => {
            let r = KvStore::apply_binary(*p, inputs[0], inputs[1])?;
            out_buf.extend_from_slice(&r);
            Ok(())
        }
        // Inline hot ops — fully parallelizable.
        TapeKernel::InlineRelu => {
            inline_unary(inputs[0], out_buf, |v| v.max(0.0));
            Ok(())
        }
        TapeKernel::InlineNeg => {
            inline_unary(inputs[0], out_buf, |v| -v);
            Ok(())
        }
        TapeKernel::InlineSigmoid => {
            inline_unary(inputs[0], out_buf, |v| 1.0 / (1.0 + (-v).exp()));
            Ok(())
        }
        TapeKernel::InlineSilu => {
            inline_unary(inputs[0], out_buf, |v| v * (1.0 / (1.0 + (-v).exp())));
            Ok(())
        }
        TapeKernel::InlineTanh => {
            inline_unary(inputs[0], out_buf, |v| v.tanh());
            Ok(())
        }
        TapeKernel::InlineGelu => {
            inline_unary(inputs[0], out_buf, |v| {
                0.5 * v
                    * (1.0
                        + (std::f32::consts::FRAC_2_SQRT_PI
                            * std::f32::consts::FRAC_1_SQRT_2
                            * (v + 0.044715 * v * v * v))
                            .tanh())
            });
            Ok(())
        }
        TapeKernel::InlineExp => {
            inline_unary(inputs[0], out_buf, |v| v.exp());
            Ok(())
        }
        TapeKernel::InlineAdd => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a + b);
            Ok(())
        }
        TapeKernel::InlineMul => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a * b);
            Ok(())
        }
        TapeKernel::InlineSub => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a - b);
            Ok(())
        }
        TapeKernel::InlineDiv => {
            inline_binary(inputs[0], inputs[1], out_buf, |a, b| a / b);
            Ok(())
        }
        TapeKernel::InlineAbs => {
            inline_unary(inputs[0], out_buf, |v| v.abs());
            Ok(())
        }
        TapeKernel::InlineReciprocal => {
            inline_unary(inputs[0], out_buf, |v| 1.0 / v);
            Ok(())
        }
        // Inline custom ops — CPU-only in parallel context (no backend).
        TapeKernel::InlineMatMul { m, k, n } => {
            crate::float_dispatch::matmul::dispatch_matmul_into(
                inputs,
                *m as usize,
                *k as usize,
                *n as usize,
                out_buf,
            )
        }
        TapeKernel::InlineMatMulActivation {
            m,
            k,
            n,
            activation,
        } => crate::float_dispatch::matmul::dispatch_matmul_activation_into(
            inputs,
            *m as usize,
            *k as usize,
            *n as usize,
            activation,
            out_buf,
        ),
        TapeKernel::InlineSoftmax { size } => {
            let actual = crate::shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_softmax_into(inputs, actual, out_buf)
        }
        TapeKernel::InlineRmsNorm { size, epsilon } => {
            let actual = crate::shape_resolve::resolve_last_dim(
                *size,
                input_metas.first().and_then(|m| m.as_ref()),
                inputs.first().map(|b| b.len()).unwrap_or(0),
            );
            crate::float_dispatch::norm::dispatch_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )
        }
        TapeKernel::Custom(handler) => {
            let result = handler(inputs, constants)?;
            out_buf.extend_from_slice(&result);
            Ok(())
        }
        // These should never appear in parallel levels (filtered by needs_shared_state).
        _ => Err(crate::error::ExecError::UnsupportedOp(
            "non-parallelizable op in parallel level".into(),
        )),
    }
}

/// Apply activation element-wise to an out_buf that contains f32 data.
/// Used for epilogue fusion on LUT-GEMM paths where the kernel writes
/// to out_buf first and we apply activation as an immediate post-pass.
fn apply_activation_to_out_buf(out_buf: &mut [u8], activation: &FloatOp) {
    if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>(out_buf) {
        for v in floats.iter_mut() {
            *v = activation.apply_unary(*v);
        }
    }
}

/// LUT-GEMM Q4 dispatch for tape kernels.
fn dispatch_lut_gemm_4(
    inputs: &[&[u8]],
    cid: ConstantId,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let mut cache = tape_ctx.weight_cache.borrow_mut();
    let qw = cache.get_q4(cid, tape_ctx.constants, tape_ctx.weights)?;
    let activations: &[f32] = bytemuck::try_cast_slice(inputs[0]).map_err(|_| {
        crate::error::ExecError::UnsupportedOp("Q4: activation not f32-aligned".into())
    })?;
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let m = if k > 0 { activations.len() / k } else { 0 };
    let mut output = vec![0.0f32; m * n];
    crate::lut_gemm::lut_gemm_4bit(activations, qw, &mut output);
    out_buf.extend_from_slice(bytemuck::cast_slice(&output));
    Ok(())
}

/// LUT-GEMM Q8 dispatch for tape kernels.
fn dispatch_lut_gemm_8(
    inputs: &[&[u8]],
    cid: ConstantId,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let mut cache = tape_ctx.weight_cache.borrow_mut();
    let qw = cache.get_q8(cid, tape_ctx.constants, tape_ctx.weights)?;
    let activations: &[f32] = bytemuck::try_cast_slice(inputs[0]).map_err(|_| {
        crate::error::ExecError::UnsupportedOp("Q8: activation not f32-aligned".into())
    })?;
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let m = if k > 0 { activations.len() / k } else { 0 };
    let mut output = vec![0.0f32; m * n];
    crate::lut_gemm::lut_gemm_8bit(activations, qw, &mut output);
    out_buf.extend_from_slice(bytemuck::cast_slice(&output));
    Ok(())
}

/// KvWrite dispatch: store K/V to cache, output for downstream attention.
///
/// `heads_first` determines the input layout and output format:
/// - `true`: input is `[heads, seq, dim]`, transpose to seq-first for storage,
///   and transpose back to heads-first on output during decode.
/// - `false`: input is `[seq, heads, dim]`, store directly, output seq-first.
///
/// Returns the actual output TensorMeta (runtime shape, not compiled).
#[allow(clippy::too_many_arguments)]
fn dispatch_kv_write(
    inputs: &[&[u8]],
    layer: u32,
    n_kv_heads: u32,
    head_dim: u32,
    is_key: bool,
    heads_first: bool,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<hologram_core::op::TensorMeta> {
    let Some(kv_cell) = &tape_ctx.kv_state else {
        return Err(crate::error::ExecError::UnsupportedOp(
            "KvWrite requires TapeContext with kv_state".into(),
        ));
    };
    let input = inputs.first().copied().unwrap_or(&[]);
    if input.is_empty() || input.len() % 4 != 0 {
        out_buf.extend_from_slice(input);
        return Ok(hologram_core::op::TensorMeta::infer_1d(input.len(), 4));
    }
    let floats: &[f32] = bytemuck::cast_slice(input);
    let nkv = n_kv_heads as usize;
    let hd = head_dim as usize;
    let stride = nkv * hd;
    let seq = if stride > 0 { floats.len() / stride } else { 1 };

    // Convert to seq-first for cache storage if input is heads-first.
    let seq_first_data: Vec<f32>;
    let cache_data: &[f32] = if heads_first {
        seq_first_data = transpose_heads_to_seq_first(floats, nkv, seq, hd);
        &seq_first_data
    } else {
        floats
    };

    let mut kv = kv_cell.borrow_mut();
    if is_key {
        kv.write_layer(layer, cache_data, &[]);
    } else {
        kv.write_layer(layer, &[], cache_data);
    }

    let out_meta = if kv.write_pos() == 0 {
        // Prefill: pass through original data in its original layout.
        out_buf.extend_from_slice(input);
        // Output shape matches input shape.
        if heads_first {
            hologram_core::op::TensorMeta::new(hologram_core::op::FloatDType::F32, &[nkv, seq, hd])
        } else {
            hologram_core::op::TensorMeta::new(hologram_core::op::FloatDType::F32, &[seq, nkv, hd])
        }
    } else {
        // Decode: read full cache (seq-first) and convert to output layout.
        let total_seq = kv.write_pos() + seq;
        let full = if is_key {
            kv.read_k_through(layer, seq)
        } else {
            kv.read_v_through(layer, seq)
        };
        if heads_first {
            let heads = transpose_seq_first_to_heads(full, nkv, total_seq, hd);
            out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&heads));
            hologram_core::op::TensorMeta::new(
                hologram_core::op::FloatDType::F32,
                &[nkv, total_seq, hd],
            )
        } else {
            out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(full));
            hologram_core::op::TensorMeta::new(
                hologram_core::op::FloatDType::F32,
                &[total_seq, nkv, hd],
            )
        }
    };
    Ok(out_meta)
}

/// KvRead dispatch: read full cached K/V from the KV cache.
///
/// `heads_first` determines output layout:
/// - `true`: transpose from seq-first cache to `[heads, seq, dim]`
/// - `false`: return seq-first `[seq, heads, dim]` directly
fn dispatch_kv_read(
    layer: u32,
    n_kv_heads: u32,
    head_dim: u32,
    heads_first: bool,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let Some(kv_cell) = &tape_ctx.kv_state else {
        return Err(crate::error::ExecError::UnsupportedOp(
            "KvRead requires TapeContext with kv_state".into(),
        ));
    };
    let kv = kv_cell.borrow();
    let nkv = n_kv_heads as usize;
    let hd = head_dim as usize;
    let total_seq = kv.write_pos();
    let k = kv.read_k(layer);
    let v = kv.read_v(layer);
    if heads_first {
        let k_heads = transpose_seq_first_to_heads(k, nkv, total_seq, hd);
        let v_heads = transpose_seq_first_to_heads(v, nkv, total_seq, hd);
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&k_heads));
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&v_heads));
    } else {
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(k));
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(v));
    }
    Ok(())
}

/// Transpose KV data from heads-first `[heads, seq, dim]` to seq-first `[seq, heads, dim]`.
fn transpose_heads_to_seq_first(
    data: &[f32],
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = n_heads * seq * head_dim;
    if data.len() < total || seq == 0 || n_heads == 0 || head_dim == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; total];
    for h in 0..n_heads {
        for s in 0..seq {
            let src = (h * seq + s) * head_dim;
            let dst = (s * n_heads + h) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

/// Transpose KV data from seq-first `[seq, heads, dim]` to heads-first `[heads, seq, dim]`.
fn transpose_seq_first_to_heads(
    data: &[f32],
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = n_heads * seq * head_dim;
    if data.len() < total || seq == 0 || n_heads == 0 || head_dim == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; total];
    for s in 0..seq {
        for h in 0..n_heads {
            let src = (s * n_heads + h) * head_dim;
            let dst = (h * seq + s) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

/// A single instruction in the enum-dispatch tape.
pub struct TapeInstruction {
    /// The kernel to execute (enum variant, no heap allocation).
    pub kernel: TapeKernel,
    /// Output node index (where to store the result in the arena).
    pub output_idx: u32,
    /// Input node indices (where to gather inputs from the arena).
    pub input_indices: Vec<u32>,
    /// Element size of the output (for arena metadata).
    pub output_elem_size: u8,
    /// Pre-computed output byte size hint (0 = unknown/dynamic).
    pub output_byte_hint: u32,
    /// Byte offset into the weight archive for LUT-GEMM constants.
    /// 0 = no weight prefetch needed (non-LUT-GEMM ops).
    /// When non-zero, the executor prefetches this address in the weight
    /// archive while the previous instruction executes.
    pub weight_offset_hint: u32,
    /// If true, this Output instruction can move the input buffer directly
    /// instead of copying through `out_buf`. Set when the input has exactly
    /// one consumer (this instruction).
    pub passthrough: bool,
    /// If true, a unary inline op can overwrite its input buffer in place.
    /// Set when the input has exactly one consumer and the op preserves size.
    pub can_reuse_input: bool,
    /// Pre-computed output tensor metadata (shape + dtype) from compiled graph.
    /// `None` = not available (infer from buffer size at runtime).
    pub output_meta: Option<hologram_core::op::TensorMeta>,
}

/// Pre-compiled execution tape using enum dispatch.
///
/// Each instruction carries a [`TapeKernel`] enum variant instead of a
/// boxed closure. This eliminates vtable indirection, enables inlining
/// of small kernels, and removes per-kernel heap allocation.
pub struct EnumTape {
    /// Flat instruction array in execution order.
    pub instructions: Vec<TapeInstruction>,
    /// Level boundaries: `level_offsets[i]..level_offsets[i+1]`.
    pub level_offsets: Vec<usize>,
}

impl EnumTape {
    /// Create an empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            level_offsets: vec![0],
        }
    }

    /// Create a tape with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(n_instructions: usize, n_levels: usize) -> Self {
        let mut level_offsets = Vec::with_capacity(n_levels + 1);
        level_offsets.push(0);
        Self {
            instructions: Vec::with_capacity(n_instructions),
            level_offsets,
        }
    }

    /// Add an instruction and return its index.
    pub fn push(&mut self, instr: TapeInstruction) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx
    }

    /// Mark the end of the current level.
    pub fn end_level(&mut self) {
        self.level_offsets.push(self.instructions.len());
    }

    /// Number of levels in the tape.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.level_offsets.len().saturating_sub(1)
    }

    /// Pre-allocate output slots in the arena so `swap_insert` has buffers
    /// to recycle from the very first instruction (eliminates first-inference
    /// allocation overhead).
    pub fn prewarm_arena(&self, arena: &mut BufferArena<'_>) {
        for instr in &self.instructions {
            if instr.output_byte_hint > 0 && !instr.passthrough {
                let id = NodeId::new(instr.output_idx, 0);
                if !arena.contains(id) {
                    let buf = Vec::with_capacity(instr.output_byte_hint as usize);
                    arena.insert_with_elem_size(id, buf, instr.output_elem_size as usize);
                }
            }
        }
    }

    /// Execute the tape against the given arena and context.
    ///
    /// Uses swap-insert for zero-allocation buffer recycling after warmup.
    /// Enum dispatch replaces vtable indirection with a direct match.
    /// Processes instructions level-by-level, flushing GPU work at level
    /// boundaries (Phase 8.2: command buffer batching).
    pub fn execute(
        &self,
        arena: &mut BufferArena<'_>,
        tape_ctx: &TapeContext<'_>,
    ) -> ExecResult<()> {
        // Resolve backend once (not per-instruction).
        let backend = tape_ctx.backend.resolve();
        let mut out_buf: Vec<u8> = Vec::with_capacity(4096);
        let mut deferred_slots: Vec<(u32, u8)> = Vec::new();

        for level_idx in 0..self.n_levels() {
            let start = self.level_offsets[level_idx];
            let end = self.level_offsets[level_idx + 1];
            let level_instrs = &self.instructions[start..end];

            for (i, instr) in level_instrs.iter().enumerate() {
                let global_i = start + i;
                // Prefetch next instruction's input data and weight pages.
                if global_i + 1 < self.instructions.len() {
                    let next = &self.instructions[global_i + 1];
                    for &idx in &next.input_indices {
                        let id = NodeId::new(idx, 0);
                        if let Ok(data) = arena.get(id) {
                            prefetch_read(data.as_ptr());
                        }
                    }
                    // Prefetch weight pages for LUT-GEMM ops.
                    if next.weight_offset_hint > 0 {
                        let offset = next.weight_offset_hint as usize;
                        if offset < tape_ctx.weights.len() {
                            prefetch_read(tape_ctx.weights[offset..].as_ptr());
                        }
                    }
                }

                // ── Fast path: Output passthrough (zero-copy move) ──
                if instr.passthrough {
                    if let Some(&src_idx) = instr.input_indices.first() {
                        let src_id = NodeId::new(src_idx, 0);
                        let dst_id = NodeId::new(instr.output_idx, 0);
                        // Preserve input's runtime meta through passthrough.
                        // Only use compiled meta when it intentionally changes
                        // rank (squeeze/unsqueeze) or when input has no meta.
                        let input_meta = arena.get_meta(src_id).copied();
                        arena.move_slot(src_id, dst_id);
                        match (input_meta, instr.output_meta) {
                            (Some(im), Some(cm))
                                if im.ndim != cm.ndim && cm.n_elems() == im.n_elems() =>
                            {
                                // Intentional rank change (squeeze/unsqueeze) — use compiled.
                                arena.set_meta(dst_id, cm);
                            }
                            (Some(im), _) => {
                                // Preserve input meta (data unchanged).
                                arena.set_meta(dst_id, im);
                            }
                            (None, Some(cm)) => {
                                arena.set_meta(dst_id, cm);
                            }
                            _ => {}
                        }
                        continue;
                    }
                }

                // ── Fast path: In-place unary op (typed f32, reuse input buffer) ──
                if instr.can_reuse_input {
                    let src_id = NodeId::new(instr.input_indices[0], 0);
                    let out_id = NodeId::new(instr.output_idx, 0);
                    if let Ok(floats) = arena.get_mut_f32(src_id) {
                        dispatch_inplace(&instr.kernel, floats);
                        arena.move_slot(src_id, out_id);
                        continue;
                    }
                }

                // ── Fast path: Inline unary (direct f32 arena access, no SmallVec) ──
                if let Some(1) = instr.kernel.inline_arity() {
                    // SAFETY (release): tape builder guarantees input_indices[0] exists
                    // and the arena slot is populated by a prior instruction or seed.
                    #[cfg(debug_assertions)]
                    let input = arena.get_f32(NodeId::new(instr.input_indices[0], 0))?;
                    #[cfg(not(debug_assertions))]
                    let input = unsafe {
                        arena.get_f32_unchecked(NodeId::new(
                            *instr.input_indices.get_unchecked(0),
                            0,
                        ))
                    };
                    out_buf.clear();
                    dispatch_inline_unary(&instr.kernel, input, &mut out_buf);
                    let out_id = NodeId::new(instr.output_idx, 0);
                    arena.swap_insert_with_elem_size(
                        out_id,
                        &mut out_buf,
                        instr.output_elem_size as usize,
                    );
                    // Unary: output shape = input shape.
                    let src_id = NodeId::new(instr.input_indices[0], 0);
                    if let Some(meta) = arena.get_meta(src_id).copied() {
                        arena.set_meta(out_id, meta);
                    } else if let Some(meta) = instr.output_meta {
                        arena.set_meta(out_id, meta);
                    }
                    continue;
                }

                // ── Fast path: Inline binary (direct f32 arena access, no SmallVec) ──
                if let Some(2) = instr.kernel.inline_arity() {
                    #[cfg(debug_assertions)]
                    let (a, b) = {
                        let a = arena.get_f32(NodeId::new(instr.input_indices[0], 0))?;
                        let b = arena.get_f32(NodeId::new(instr.input_indices[1], 0))?;
                        (a, b)
                    };
                    #[cfg(not(debug_assertions))]
                    let (a, b) = unsafe {
                        let a = arena.get_f32_unchecked(NodeId::new(
                            *instr.input_indices.get_unchecked(0),
                            0,
                        ));
                        let b = arena.get_f32_unchecked(NodeId::new(
                            *instr.input_indices.get_unchecked(1),
                            0,
                        ));
                        (a, b)
                    };
                    out_buf.clear();
                    dispatch_inline_binary(&instr.kernel, a, b, &mut out_buf);
                    let out_len = out_buf.len();
                    let out_id = NodeId::new(instr.output_idx, 0);
                    arena.swap_insert_with_elem_size(
                        out_id,
                        &mut out_buf,
                        instr.output_elem_size as usize,
                    );
                    // Binary: use the input meta that matches output element count.
                    let a_id = NodeId::new(instr.input_indices[0], 0);
                    let b_id = NodeId::new(instr.input_indices[1], 0);
                    let a_meta = arena.get_meta(a_id).copied();
                    let b_meta = arena.get_meta(b_id).copied();
                    let out_elems = out_len / 4;
                    if let Some(meta) = a_meta
                        .filter(|m| m.n_elems() == out_elems)
                        .or_else(|| b_meta.filter(|m| m.n_elems() == out_elems))
                    {
                        arena.set_meta(out_id, meta);
                    } else if let Some(meta) = instr.output_meta {
                        arena.set_meta(out_id, meta);
                    }
                    continue;
                }

                // ── Fast path: Reshape/Passthrough with shape-aware TensorMeta ──
                // When input has fewer elements than the compiled output shape
                // (variable-length execution), adjust the output meta to match
                // the actual element count. This is how ONNX Reshape(-1) works.
                // ── Fast path: Reshape — zero-copy data, adjust shape metadata ──
                if matches!(
                    instr.kernel,
                    TapeKernel::InlineReshape | TapeKernel::Passthrough
                ) {
                    if let Some(&src_idx) = instr.input_indices.first() {
                        let src_id = NodeId::new(src_idx, 0);
                        let out_id = NodeId::new(instr.output_idx, 0);

                        // Compute adjusted output meta from input meta + compiled target shape.
                        // When actual element count differs from compiled (variable-length
                        // execution), find the dim that changed and scale it.
                        let adjusted_meta =
                            match (arena.get_meta(src_id).copied(), instr.output_meta) {
                                (Some(input_meta), Some(compiled_meta)) => {
                                    let actual_elems = input_meta.n_elems();
                                    let compiled_elems = compiled_meta.n_elems();
                                    if actual_elems != compiled_elems
                                        && actual_elems > 0
                                        && compiled_elems > 0
                                    {
                                        let mut adjusted = compiled_meta;
                                        let ratio = actual_elems as f64 / compiled_elems as f64;
                                        for i in 0..adjusted.ndim as usize {
                                            let scaled =
                                                (adjusted.dims[i] as f64 * ratio).round() as u32;
                                            let mut check = adjusted;
                                            check.dims[i] = scaled;
                                            if check.n_elems() == actual_elems {
                                                adjusted.dims[i] = scaled;
                                                break;
                                            }
                                        }
                                        Some(adjusted)
                                    } else {
                                        Some(compiled_meta)
                                    }
                                }
                                (_, Some(compiled_meta)) => Some(compiled_meta),
                                (Some(input_meta), None) => Some(input_meta),
                                _ => None,
                            };

                        let data = arena.get(src_id)?;
                        out_buf.clear();
                        out_buf.extend_from_slice(data);
                        arena.swap_insert_with_elem_size(
                            out_id,
                            &mut out_buf,
                            instr.output_elem_size as usize,
                        );
                        if let Some(meta) = adjusted_meta {
                            arena.set_meta(out_id, meta);
                        }
                        continue;
                    }
                }

                // ── General path: SmallVec collection + dispatch_kernel ──
                let input_metas: crate::shape_resolve::InputMetas = instr
                    .input_indices
                    .iter()
                    .map(|&idx| arena.get_meta(NodeId::new(idx, 0)).copied())
                    .collect();
                let dispatch_result = {
                    let input_refs: SmallVec<[&[u8]; 4]> = instr
                        .input_indices
                        .iter()
                        .map(|&idx| arena.get(NodeId::new(idx, 0)))
                        .collect::<ExecResult<SmallVec<_>>>()?;
                    out_buf.clear();
                    if instr.output_byte_hint > 0 {
                        out_buf.reserve(instr.output_byte_hint as usize);
                    }
                    dispatch_kernel(
                        &instr.kernel,
                        &input_refs,
                        &input_metas,
                        tape_ctx,
                        &*backend,
                        &mut out_buf,
                    )?
                };

                // Store output based on dispatch result.
                let out_id = NodeId::new(instr.output_idx, 0);

                match dispatch_result {
                    DispatchResult::InOutBuf => {
                        let out_len = out_buf.len();
                        arena.swap_insert_with_elem_size(
                            out_id,
                            &mut out_buf,
                            instr.output_elem_size as usize,
                        );
                        // Compute runtime meta from actual output + input metas.
                        // This ensures downstream ops get correct N-D shapes even
                        // when compiled shapes don't match runtime sizes.
                        if let Some(meta) = crate::shape_resolve::compute_output_meta(
                            &input_metas,
                            instr.output_meta,
                            out_len,
                            instr.output_elem_size as usize,
                        ) {
                            arena.set_meta(out_id, meta);
                        }
                    }
                    DispatchResult::InOutBufWithMeta(runtime_meta) => {
                        arena.swap_insert_with_elem_size(
                            out_id,
                            &mut out_buf,
                            instr.output_elem_size as usize,
                        );
                        arena.set_meta(out_id, runtime_meta);
                    }
                    #[cfg(has_metal)]
                    DispatchResult::MetalBuffer(metal_buf) => {
                        arena.insert_metal(out_id, metal_buf, instr.output_elem_size as usize);
                    }
                    #[cfg(has_webgpu)]
                    DispatchResult::WgpuDeferred => {
                        deferred_slots.push((instr.output_idx, instr.output_elem_size));
                    }
                }
                // Shape context override: if the compiler's ShapeContextGraph
                // resolved this node's output shape from actual input dimensions,
                // set it as the definitive TensorMeta. This overrides both
                // heuristic inference and dispatch-computed meta, ensuring all
                // downstream ops see the correct shape.
                if !tape_ctx.shape_overrides.is_empty() {
                    if let Some(shape) = tape_ctx.shape_overrides.get(&instr.output_idx) {
                        let dtype = arena
                            .get_meta(out_id)
                            .map(|m| m.dtype)
                            .unwrap_or(hologram_core::op::FloatDType::F32);
                        arena.set_meta(out_id, hologram_core::op::TensorMeta::new(dtype, shape));
                    }
                }
            } // end inner instruction loop

            // Flush deferred GPU work at level boundary (Phase 8.2 + 8.3d).
            // Metal: commits batched command buffer, waits for completion.
            // WebGPU: submits encoder, polls device, maps+reads all staging buffers.
            let deferred_data = backend.flush_deferred()?;
            for (data, &(out_idx, elem_size)) in
                deferred_data.into_iter().zip(deferred_slots.iter())
            {
                arena.insert_with_elem_size(NodeId::new(out_idx, 0), data, elem_size as usize);
            }
            deferred_slots.clear();
        } // end level loop

        Ok(())
    }

    /// Execute the tape with adaptive parallelism within levels.
    ///
    /// Levels with ≥4 instructions are dispatched in parallel via rayon.
    /// Smaller levels use sequential execution to avoid thread-pool overhead.
    /// Falls back to sequential on all levels when the `parallel` feature
    /// is disabled.
    #[cfg(feature = "parallel")]
    pub fn execute_parallel(
        &self,
        arena: &mut BufferArena<'_>,
        tape_ctx: &TapeContext<'_>,
    ) -> ExecResult<()> {
        use rayon::prelude::*;

        const PAR_THRESHOLD: usize = 4;
        let backend = tape_ctx.backend.resolve();
        let mut par_deferred_slots: Vec<(u32, u8)> = Vec::new();

        for level_idx in 0..self.n_levels() {
            let start = self.level_offsets[level_idx];
            let end = self.level_offsets[level_idx + 1];
            let level_instrs = &self.instructions[start..end];

            // Check if any instruction needs shared mutable state (RefCell).
            // LUT-GEMM and KvCache ops cannot be parallelized.
            let needs_shared_state = level_instrs.iter().any(|instr| {
                matches!(
                    instr.kernel,
                    TapeKernel::MatMulLut4(_)
                        | TapeKernel::MatMulLut8(_)
                        | TapeKernel::MatMulLut4Activation(..)
                        | TapeKernel::MatMulLut8Activation(..)
                        | TapeKernel::InlineMatMulBiasActivation { .. }
                        | TapeKernel::KvWrite { .. }
                        | TapeKernel::KvRead { .. }
                )
            });

            if level_instrs.len() >= PAR_THRESHOLD && !needs_shared_state {
                // Parallel: each instruction independently gathers inputs and dispatches.
                // For parallel dispatch, we pass only the execution context ref (Sync-safe)
                // since parallel levels never contain LUT-GEMM or KvCache ops.
                let exec_ctx = tape_ctx.ctx.as_ref();
                let results: ExecResult<Vec<(u32, Vec<u8>, u8)>> = level_instrs
                    .par_iter()
                    .map(|instr| {
                        let input_refs: SmallVec<[&[u8]; 4]> = instr
                            .input_indices
                            .iter()
                            .map(|&idx| arena.get(NodeId::new(idx, 0)))
                            .collect::<ExecResult<SmallVec<_>>>()?;
                        let input_metas: crate::shape_resolve::InputMetas = instr
                            .input_indices
                            .iter()
                            .map(|&idx| arena.get_meta(NodeId::new(idx, 0)).copied())
                            .collect();
                        let mut out_buf = Vec::with_capacity(if instr.output_byte_hint > 0 {
                            instr.output_byte_hint as usize
                        } else {
                            256
                        });
                        dispatch_kernel_par(
                            &instr.kernel,
                            &input_refs,
                            &input_metas,
                            exec_ctx,
                            tape_ctx.constants,
                            &mut out_buf,
                        )?;
                        Ok((instr.output_idx, out_buf, instr.output_elem_size))
                    })
                    .collect();

                for (output_idx, data, elem_size) in results? {
                    let out_id = NodeId::new(output_idx, 0);
                    arena.insert_with_elem_size(out_id, data, elem_size as usize);
                }
            } else {
                // Sequential: reuse single output buffer with swap-insert.
                let mut out_buf: Vec<u8> = Vec::with_capacity(4096);
                for (i, instr) in level_instrs.iter().enumerate() {
                    // Prefetch next instruction in this level.
                    if i + 1 < level_instrs.len() {
                        let next = &level_instrs[i + 1];
                        for &idx in &next.input_indices {
                            let id = NodeId::new(idx, 0);
                            if let Ok(data) = arena.get(id) {
                                prefetch_read(data.as_ptr());
                            }
                        }
                        if next.weight_offset_hint > 0 {
                            let offset = next.weight_offset_hint as usize;
                            if offset < tape_ctx.weights.len() {
                                prefetch_read(tape_ctx.weights[offset..].as_ptr());
                            }
                        }
                    }

                    // Fast path: Output passthrough.
                    if instr.passthrough {
                        if let Some(&src_idx) = instr.input_indices.first() {
                            arena.move_slot(
                                NodeId::new(src_idx, 0),
                                NodeId::new(instr.output_idx, 0),
                            );
                            continue;
                        }
                    }

                    // Fast path: In-place unary op (typed f32).
                    if instr.can_reuse_input {
                        let src_id = NodeId::new(instr.input_indices[0], 0);
                        let out_id = NodeId::new(instr.output_idx, 0);
                        if let Ok(floats) = arena.get_mut_f32(src_id) {
                            dispatch_inplace(&instr.kernel, floats);
                            arena.move_slot(src_id, out_id);
                            continue;
                        }
                    }

                    // Fast path: Inline unary (direct f32 access).
                    if let Some(1) = instr.kernel.inline_arity() {
                        let input = arena.get_f32(NodeId::new(instr.input_indices[0], 0))?;
                        out_buf.clear();
                        if instr.output_byte_hint > 0 {
                            out_buf.reserve(instr.output_byte_hint as usize);
                        }
                        dispatch_inline_unary(&instr.kernel, input, &mut out_buf);
                        let out_id = NodeId::new(instr.output_idx, 0);
                        arena.swap_insert_with_elem_size(
                            out_id,
                            &mut out_buf,
                            instr.output_elem_size as usize,
                        );
                        if let Some(meta) = instr.output_meta {
                            arena.set_meta(out_id, meta);
                        }
                        continue;
                    }

                    // Fast path: Inline binary (direct f32 access).
                    if let Some(2) = instr.kernel.inline_arity() {
                        let a = arena.get_f32(NodeId::new(instr.input_indices[0], 0))?;
                        let b = arena.get_f32(NodeId::new(instr.input_indices[1], 0))?;
                        out_buf.clear();
                        if instr.output_byte_hint > 0 {
                            out_buf.reserve(instr.output_byte_hint as usize);
                        }
                        dispatch_inline_binary(&instr.kernel, a, b, &mut out_buf);
                        let out_id = NodeId::new(instr.output_idx, 0);
                        arena.swap_insert_with_elem_size(
                            out_id,
                            &mut out_buf,
                            instr.output_elem_size as usize,
                        );
                        if let Some(meta) = instr.output_meta {
                            arena.set_meta(out_id, meta);
                        }
                        continue;
                    }

                    // General path: SmallVec + dispatch_kernel.
                    let dispatch_result = {
                        let input_refs: SmallVec<[&[u8]; 4]> = instr
                            .input_indices
                            .iter()
                            .map(|&idx| arena.get(NodeId::new(idx, 0)))
                            .collect::<ExecResult<SmallVec<_>>>()?;
                        let input_metas: crate::shape_resolve::InputMetas = instr
                            .input_indices
                            .iter()
                            .map(|&idx| arena.get_meta(NodeId::new(idx, 0)).copied())
                            .collect();
                        out_buf.clear();
                        if instr.output_byte_hint > 0 {
                            out_buf.reserve(instr.output_byte_hint as usize);
                        }
                        dispatch_kernel(
                            &instr.kernel,
                            &input_refs,
                            &input_metas,
                            tape_ctx,
                            &*backend,
                            &mut out_buf,
                        )?
                    };

                    let out_id = NodeId::new(instr.output_idx, 0);
                    // Re-collect input metas for output meta computation.
                    let input_metas: crate::shape_resolve::InputMetas = instr
                        .input_indices
                        .iter()
                        .map(|&idx| arena.get_meta(NodeId::new(idx, 0)).copied())
                        .collect();
                    match dispatch_result {
                        DispatchResult::InOutBuf => {
                            let out_len = out_buf.len();
                            arena.swap_insert_with_elem_size(
                                out_id,
                                &mut out_buf,
                                instr.output_elem_size as usize,
                            );
                            if let Some(meta) = crate::shape_resolve::compute_output_meta(
                                &input_metas,
                                instr.output_meta,
                                out_len,
                                instr.output_elem_size as usize,
                            ) {
                                arena.set_meta(out_id, meta);
                            }
                        }
                        DispatchResult::InOutBufWithMeta(runtime_meta) => {
                            arena.swap_insert_with_elem_size(
                                out_id,
                                &mut out_buf,
                                instr.output_elem_size as usize,
                            );
                            arena.set_meta(out_id, runtime_meta);
                        }
                        #[cfg(has_metal)]
                        DispatchResult::MetalBuffer(metal_buf) => {
                            arena.insert_metal(out_id, metal_buf, instr.output_elem_size as usize);
                        }
                        #[cfg(has_webgpu)]
                        DispatchResult::WgpuDeferred => {
                            par_deferred_slots.push((instr.output_idx, instr.output_elem_size));
                        }
                    }
                }
            }

            // Flush deferred GPU work at level boundary.
            let deferred_data = backend.flush_deferred()?;
            for (data, &(out_idx, elem_size)) in
                deferred_data.into_iter().zip(par_deferred_slots.iter())
            {
                arena.insert_with_elem_size(NodeId::new(out_idx, 0), data, elem_size as usize);
            }
            par_deferred_slots.clear();
        }

        Ok(())
    }
}

impl Default for EnumTape {
    fn default() -> Self {
        Self::new()
    }
}

// ── Backward-compat aliases ──────────────────────────────────────────────────

/// Backward-compatible alias for [`TapeInstruction`].
pub type BoxedInstruction = TapeInstruction;

/// Backward-compatible alias for [`EnumTape`].
pub type BoxedTape = EnumTape;

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_constants() -> ConstantStore {
        ConstantStore::new()
    }

    #[test]
    fn enum_tape_empty_executes() {
        let tape = EnumTape::new();
        let mut arena = BufferArena::new();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        assert!(tape.execute(&mut arena, &ctx).is_ok());
    }

    #[test]
    fn enum_tape_output_passthrough() {
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::Output,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 1,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![10, 20, 30]);

        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &ctx).unwrap();

        assert_eq!(arena.get(NodeId::new(1, 0)).unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn enum_tape_float_relu() {
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 8, // 2 floats × 4 bytes
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.push(TapeInstruction {
            kernel: TapeKernel::Output,
            output_idx: 2,
            input_indices: vec![1],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();

        // Input: two f32 values [-1.0, 2.0]
        let input_bytes: Vec<u8> = [(-1.0f32).to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input_bytes);

        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[0.0, 2.0]); // relu(-1)=0, relu(2)=2
    }

    #[test]
    fn enum_tape_lut_view() {
        use hologram_core::op::LutOp;
        let view = hologram_core::view::ElementWiseView::from_table(*LutOp::Relu.table());

        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::LutView(view),
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 1,
            output_byte_hint: 3,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![0, 128, 255]);

        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(1, 0)).unwrap();
        assert_eq!(out[0], LutOp::Relu.apply(0));
        assert_eq!(out[1], LutOp::Relu.apply(128));
        assert_eq!(out[2], LutOp::Relu.apply(255));
    }

    #[test]
    fn enum_tape_two_level_chain() {
        // Input(0) → Relu(1) → Output(2)
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        tape.push(TapeInstruction {
            kernel: TapeKernel::Output,
            output_idx: 2,
            input_indices: vec![1],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();

        assert_eq!(tape.n_levels(), 2);

        let input: Vec<u8> = [(-3.0f32).to_le_bytes(), 5.0f32.to_le_bytes()].concat();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input);

        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[0.0, 5.0]);
    }

    #[test]
    fn enum_tape_swap_insert_recycles_buffers() {
        // Run the same tape twice — second run should reuse allocations.
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();

        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);

        // Run 1
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), 1.0f32.to_le_bytes().to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out1 = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Run 2 (reuse arena)
        arena.insert(NodeId::new(0, 0), 2.0f32.to_le_bytes().to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out2 = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        let f1: f32 = f32::from_le_bytes(out1[..4].try_into().unwrap());
        let f2: f32 = f32::from_le_bytes(out2[..4].try_into().unwrap());
        assert_eq!(f1, 1.0);
        assert_eq!(f2, 2.0);
    }

    // ── Inline hot op tests (Phase 9a) ────────────────────────────

    #[test]
    fn inline_relu_matches_generic() {
        let input: Vec<u8> = [(-2.0f32).to_le_bytes(), 3.0f32.to_le_bytes()].concat();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);

        // Inline path
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input.clone());
        tape.execute(&mut arena, &ctx).unwrap();
        let inline_out = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Generic Float path
        let mut tape2 = EnumTape::new();
        tape2.push(TapeInstruction {
            kernel: TapeKernel::InlineRelu,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape2.end_level();
        let mut arena2 = BufferArena::new();
        arena2.insert(NodeId::new(0, 0), input);
        tape2.execute(&mut arena2, &ctx).unwrap();
        let generic_out = arena2.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Byte-for-byte match.
        assert_eq!(inline_out, generic_out, "InlineRelu must match Float(Relu)");
        let floats: &[f32] = bytemuck::cast_slice(&inline_out);
        assert_eq!(floats, &[0.0, 3.0]);
    }

    #[test]
    fn inline_add_matches_generic() {
        let a: Vec<u8> = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        let b: Vec<u8> = [10.0f32.to_le_bytes(), 20.0f32.to_le_bytes()].concat();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);

        // Inline path
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineAdd,
            output_idx: 2,
            input_indices: vec![0, 1],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), a.clone());
        arena.insert(NodeId::new(1, 0), b.clone());
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[11.0, 22.0]);
    }

    #[test]
    fn inline_mul_sigmoid_chain() {
        // Test chaining inline ops: Input → InlineSigmoid → InlineMul → Output
        let input: Vec<u8> = [0.0f32.to_le_bytes()].concat(); // sigmoid(0) = 0.5
        let two: Vec<u8> = [2.0f32.to_le_bytes()].concat();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);

        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineSigmoid,
            output_idx: 2,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineMul,
            output_idx: 3,
            input_indices: vec![2, 1],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input);
        arena.insert(NodeId::new(1, 0), two);
        tape.execute(&mut arena, &ctx).unwrap();

        let out = arena.get(NodeId::new(3, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        // sigmoid(0) * 2 = 0.5 * 2 = 1.0
        assert!((floats[0] - 1.0).abs() < 1e-5, "got {}", floats[0]);
    }

    // ── binary_broadcast tests ──────────────────────────────────────

    #[test]
    fn broadcast_same_size() {
        let mut dst = vec![0.0f32; 2];
        binary_broadcast(&[1.0, 2.0], &[3.0, 4.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![4.0, 6.0]);
    }

    #[test]
    fn broadcast_scalar_b() {
        let mut dst = vec![0.0f32; 3];
        binary_broadcast(&[1.0, 2.0, 3.0], &[10.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn broadcast_scalar_a() {
        let mut dst = vec![0.0f32; 2];
        binary_broadcast(&[10.0], &[1.0, 2.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![11.0, 12.0]);
    }

    #[test]
    fn broadcast_general() {
        let mut dst = vec![0.0f32; 3];
        binary_broadcast(&[1.0, 2.0], &[10.0, 20.0, 30.0], &mut dst, |a, b| a + b);
        assert_eq!(dst, vec![11.0, 22.0, 31.0]);
    }

    // ── Sprint 21 tests: Passthrough, new inline variants, norm direct-write ──

    /// Helper: build and execute a single-instruction tape, return output f32s.
    fn run_unary_tape(kernel: TapeKernel, input: &[f32]) -> Vec<f32> {
        let input_bytes: Vec<u8> = bytemuck::cast_slice(input).to_vec();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: (input.len() * 4) as u32,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input_bytes);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        bytemuck::cast_slice(out).to_vec()
    }

    fn run_binary_tape(kernel: TapeKernel, a: &[f32], b: &[f32]) -> Vec<f32> {
        let a_bytes: Vec<u8> = bytemuck::cast_slice(a).to_vec();
        let b_bytes: Vec<u8> = bytemuck::cast_slice(b).to_vec();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel,
            output_idx: 2,
            input_indices: vec![0, 1],
            output_elem_size: 4,
            output_byte_hint: (a.len() * 4) as u32,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), a_bytes);
        arena.insert(NodeId::new(1, 0), b_bytes);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        bytemuck::cast_slice(out).to_vec()
    }

    #[test]
    fn passthrough_identity_cast() {
        // Passthrough kernel should forward input bytes unchanged.
        let input = [1.0f32, 2.0, 3.0];
        let out = run_unary_tape(TapeKernel::Passthrough, &input);
        assert_eq!(out, input);
    }

    #[test]
    fn passthrough_empty_input() {
        let out = run_unary_tape(TapeKernel::Passthrough, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn inline_log() {
        let out = run_unary_tape(TapeKernel::InlineLog, &[1.0, std::f32::consts::E]);
        assert!((out[0] - 0.0).abs() < 1e-6, "ln(1) = 0");
        assert!((out[1] - 1.0).abs() < 1e-5, "ln(e) = 1");
    }

    #[test]
    fn inline_sqrt() {
        let out = run_unary_tape(TapeKernel::InlineSqrt, &[4.0, 9.0, 0.0]);
        assert_eq!(out, [2.0, 3.0, 0.0]);
    }

    #[test]
    fn inline_cos_sin() {
        let out_cos = run_unary_tape(TapeKernel::InlineCos, &[0.0]);
        let out_sin = run_unary_tape(TapeKernel::InlineSin, &[0.0]);
        assert!((out_cos[0] - 1.0).abs() < 1e-6, "cos(0) = 1");
        assert!(out_sin[0].abs() < 1e-6, "sin(0) = 0");
    }

    #[test]
    fn inline_sign() {
        let out = run_unary_tape(TapeKernel::InlineSign, &[-5.0, 0.0, 3.0]);
        // f32::signum() returns 1.0 for +0.0 (IEEE 754 behavior).
        assert_eq!(out, [-1.0, 1.0, 1.0]);
    }

    #[test]
    fn inline_floor_ceil_round() {
        let out_floor = run_unary_tape(TapeKernel::InlineFloor, &[1.7, -1.3]);
        let out_ceil = run_unary_tape(TapeKernel::InlineCeil, &[1.1, -1.9]);
        let out_round = run_unary_tape(TapeKernel::InlineRound, &[1.5, 2.3]);
        assert_eq!(out_floor, [1.0, -2.0]);
        assert_eq!(out_ceil, [2.0, -1.0]);
        assert_eq!(out_round, [2.0, 2.0]);
    }

    #[test]
    fn inline_erf() {
        let out = run_unary_tape(TapeKernel::InlineErf, &[0.0, 1.0]);
        assert!(out[0].abs() < 1e-5, "erf(0) = 0");
        assert!((out[1] - 0.8427).abs() < 0.01, "erf(1) ≈ 0.8427");
    }

    #[test]
    fn inline_min_max() {
        let a = [1.0f32, 5.0, 3.0];
        let b = [2.0f32, 4.0, 3.0];
        let mins = run_binary_tape(TapeKernel::InlineMin, &a, &b);
        let maxs = run_binary_tape(TapeKernel::InlineMax, &a, &b);
        assert_eq!(mins, [1.0, 4.0, 3.0]);
        assert_eq!(maxs, [2.0, 5.0, 3.0]);
    }

    #[test]
    fn inline_layer_norm() {
        // LayerNorm: normalize [1, 2, 3] with weight=[1,1,1] bias=[0,0,0]
        // mean=2, var=2/3, inv_std≈1.2247
        let x: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec();
        let w: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 1.0, 1.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[0.0f32, 0.0, 0.0]).to_vec();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineLayerNorm {
                size: 3,
                epsilon: f32::to_bits(1e-5),
            },
            output_idx: 3,
            input_indices: vec![0, 1, 2],
            output_elem_size: 4,
            output_byte_hint: 12,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), x);
        arena.insert(NodeId::new(1, 0), w);
        arena.insert(NodeId::new(2, 0), b);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(3, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats.len(), 3);
        // Normalized: (x - mean) * inv_std ≈ [-1.2247, 0, 1.2247]
        assert!((floats[0] + 1.2247).abs() < 0.01);
        assert!(floats[1].abs() < 0.01);
        assert!((floats[2] - 1.2247).abs() < 0.01);
    }

    #[test]
    fn inline_log_softmax() {
        // LogSoftmax of [0, 0, 0] → [-ln(3), -ln(3), -ln(3)]
        let x = [0.0f32, 0.0, 0.0];
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineLogSoftmax { size: 3 },
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 12,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), bytemuck::cast_slice(&x).to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        let expected = -(3.0f32.ln());
        for &v in floats {
            assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn inline_softmax_into_direct_write() {
        // Verify InlineSoftmax writes correct values and sums to 1.
        let x = [1.0f32, 2.0, 3.0];
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineSoftmax { size: 3 },
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 12,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), bytemuck::cast_slice(&x).to_vec());
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(1, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats.len(), 3);
        let sum: f32 = floats.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
        // Values should be monotonically increasing (input was sorted).
        assert!(floats[0] < floats[1]);
        assert!(floats[1] < floats[2]);
    }

    #[test]
    fn inline_gather_dispatch() {
        use hologram_core::op::FloatDType;
        // Gather: inputs[0]=indices (i64), inputs[1]=table (f32)
        // dim=1 means each entry is 1 float. indices=[2,0] → table[2]=30, table[0]=10
        let table: Vec<u8> = bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]).to_vec();
        let idx_vals: [i64; 2] = [2, 0];
        let indices: Vec<u8> = bytemuck::cast_slice(&idx_vals).to_vec();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineGather {
                dim: 1,
                dtype: FloatDType::F32,
            },
            output_idx: 2,
            // inputs[0] = indices, inputs[1] = table
            input_indices: vec![1, 0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        // node 0 = table, node 1 = indices
        arena.insert(NodeId::new(0, 0), table);
        arena.insert(NodeId::new(1, 0), indices);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[30.0, 10.0]);
    }

    #[test]
    fn inline_concat_dispatch() {
        use hologram_core::op::FloatDType;
        // Concat: [1,2] + [3,4,5] → [1,2,3,4,5]
        let a: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[3.0f32, 4.0, 5.0]).to_vec();
        let constants = empty_constants();
        let ctx = TapeContext::new(&constants, &[]);
        let mut tape = EnumTape::new();
        tape.push(TapeInstruction {
            kernel: TapeKernel::InlineConcat {
                size_a: 2,
                size_b: 3,
                dtype: FloatDType::F32,
            },
            output_idx: 2,
            input_indices: vec![0, 1],
            output_elem_size: 4,
            output_byte_hint: 20,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), a);
        arena.insert(NodeId::new(1, 0), b);
        tape.execute(&mut arena, &ctx).unwrap();
        let out = arena.get(NodeId::new(2, 0)).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(out);
        assert_eq!(floats, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
