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
    /// Float op dispatched via `dispatch_float_into`.
    Float(FloatOp),
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
    /// KV cache write (autoregressive generation).
    KvWrite {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        is_key: bool,
    },
    /// KV cache read (autoregressive generation).
    KvRead {
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
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
    /// Inline Softmax with baked row size.
    InlineSoftmax { size: u32 },
    /// Inline RmsNorm with baked row size and epsilon (as f32::to_bits()).
    InlineRmsNorm { size: u32, epsilon: u32 },

    /// Custom op — handler baked at tape build time from registry.
    Custom(crate::kv::CustomHandler),
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
            | TapeKernel::InlineReciprocal => Some(1),
            TapeKernel::InlineAdd
            | TapeKernel::InlineMul
            | TapeKernel::InlineSub
            | TapeKernel::InlineDiv => Some(2),
            _ => None,
        }
    }
}

/// Result of kernel dispatch — tells the execute loop how to store the output.
enum DispatchResult {
    /// Output written to `out_buf`. Store via swap_insert.
    InOutBuf,
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
    tape_ctx: &TapeContext<'_>,
    backend: &dyn crate::backend::ComputeBackend,
    out_buf: &mut Vec<u8>,
) -> ExecResult<DispatchResult> {
    use crate::backend::KernelOutput;
    use crate::float_dispatch;
    use crate::kv::KvStore;

    match kernel {
        TapeKernel::Float(op) => {
            // Try selected backend (GPU/accelerator) first.
            match backend.dispatch_float(op, inputs, out_buf)? {
                KernelOutput::Bytes => return Ok(DispatchResult::InOutBuf),
                #[cfg(has_metal)]
                KernelOutput::MetalBuffer(buf) => {
                    return Ok(DispatchResult::MetalBuffer(buf));
                }
                #[cfg(has_webgpu)]
                KernelOutput::WgpuDeferred => return Ok(DispatchResult::WgpuDeferred),
                KernelOutput::Skipped => {}
            }
            // Fallback to CPU dispatch.
            float_dispatch::dispatch_float_into(op, inputs, tape_ctx.ctx.as_ref(), out_buf)?;
            Ok(DispatchResult::InOutBuf)
        }
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
        TapeKernel::KvWrite {
            layer,
            n_kv_heads,
            head_dim,
            is_key,
        } => {
            dispatch_kv_write(
                inputs,
                *layer,
                *n_kv_heads,
                *head_dim,
                *is_key,
                tape_ctx,
                out_buf,
            )?;
            Ok(DispatchResult::InOutBuf)
        }
        TapeKernel::KvRead {
            layer,
            n_kv_heads,
            head_dim,
        } => {
            dispatch_kv_read(*layer, *n_kv_heads, *head_dim, tape_ctx, out_buf)?;
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

        // ── Inline custom ops (Phase 9a.3–9a.4) ─────────────────────────
        // Try backend (GPU) first, then direct CPU kernel call.
        TapeKernel::InlineMatMul { m, k, n } => {
            match backend.dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize, out_buf)? {
                KernelOutput::Bytes => return Ok(DispatchResult::InOutBuf),
                #[cfg(has_metal)]
                KernelOutput::MetalBuffer(buf) => {
                    return Ok(DispatchResult::MetalBuffer(buf));
                }
                #[cfg(has_webgpu)]
                KernelOutput::WgpuDeferred => return Ok(DispatchResult::WgpuDeferred),
                KernelOutput::Skipped => {}
            }
            crate::float_dispatch::matmul::dispatch_matmul_into(
                inputs,
                *m as usize,
                *k as usize,
                *n as usize,
                out_buf,
            )?;
            Ok(DispatchResult::InOutBuf)
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
            let actual = crate::float_dispatch::resolve_size(*size, inputs);
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
            let actual = crate::float_dispatch::resolve_size(*size, inputs);
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
        TapeKernel::InlineReciprocal => inline_unary_f32(input, out_buf, |v| 1.0 / v),
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
    ctx: Option<&ExecutionContext>,
    constants: &ConstantStore,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    use crate::float_dispatch;
    use crate::kv::KvStore;

    match kernel {
        TapeKernel::Float(op) => float_dispatch::dispatch_float_into(op, inputs, ctx, out_buf),
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
        TapeKernel::InlineSoftmax { size } => {
            let actual = crate::float_dispatch::resolve_size(*size, inputs);
            crate::float_dispatch::norm::dispatch_softmax_into(inputs, actual, out_buf)
        }
        TapeKernel::InlineRmsNorm { size, epsilon } => {
            let actual = crate::float_dispatch::resolve_size(*size, inputs);
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

/// KvWrite dispatch: transpose heads→seq, write to cache, output for downstream attention.
#[allow(clippy::too_many_arguments)]
fn dispatch_kv_write(
    inputs: &[&[u8]],
    layer: u32,
    n_kv_heads: u32,
    head_dim: u32,
    is_key: bool,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let Some(kv_cell) = &tape_ctx.kv_state else {
        return Err(crate::error::ExecError::UnsupportedOp(
            "KvWrite requires TapeContext with kv_state".into(),
        ));
    };
    let input = inputs.first().copied().unwrap_or(&[]);
    if input.is_empty() || input.len() % 4 != 0 {
        out_buf.extend_from_slice(input);
        return Ok(());
    }
    let floats: &[f32] = bytemuck::cast_slice(input);
    let nkv = n_kv_heads as usize;
    let hd = head_dim as usize;
    let stride = nkv * hd;
    let seq = if stride > 0 { floats.len() / stride } else { 1 };

    // Transpose from heads-first [heads, seq, dim] to seq-first [seq, heads, dim].
    let seq_first = transpose_heads_to_seq_first(floats, nkv, seq, hd);

    let mut kv = kv_cell.borrow_mut();
    if is_key {
        kv.write_layer(layer, &seq_first, &[]);
    } else {
        kv.write_layer(layer, &[], &seq_first);
    }

    if kv.write_pos() == 0 {
        // Prefill: pass through original heads-first data.
        out_buf.extend_from_slice(input);
    } else {
        // Decode: read full cache and transpose back to heads-first.
        let total_seq = kv.write_pos() + seq;
        let full = if is_key {
            kv.read_k_through(layer, seq)
        } else {
            kv.read_v_through(layer, seq)
        };
        let heads_first = transpose_seq_first_to_heads(full, nkv, total_seq, hd);
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&heads_first));
    }
    Ok(())
}

/// KvRead dispatch: read full cached K/V from the KV cache.
fn dispatch_kv_read(
    layer: u32,
    n_kv_heads: u32,
    head_dim: u32,
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
    // Transpose to heads-first for attention kernel.
    let k_heads = transpose_seq_first_to_heads(k, nkv, total_seq, hd);
    let v_heads = transpose_seq_first_to_heads(v, nkv, total_seq, hd);
    // Concatenate K and V as output (downstream Attention op expects both).
    out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&k_heads));
    out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&v_heads));
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
                        arena.move_slot(NodeId::new(src_idx, 0), NodeId::new(instr.output_idx, 0));
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
                    let out_id = NodeId::new(instr.output_idx, 0);
                    arena.swap_insert_with_elem_size(
                        out_id,
                        &mut out_buf,
                        instr.output_elem_size as usize,
                    );
                    continue;
                }

                // ── General path: SmallVec collection + dispatch_kernel ──
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
                        tape_ctx,
                        &*backend,
                        &mut out_buf,
                    )?
                };

                // Store output based on dispatch result.
                let out_id = NodeId::new(instr.output_idx, 0);
                match dispatch_result {
                    DispatchResult::InOutBuf => {
                        arena.swap_insert_with_elem_size(
                            out_id,
                            &mut out_buf,
                            instr.output_elem_size as usize,
                        );
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
                        let mut out_buf = Vec::with_capacity(if instr.output_byte_hint > 0 {
                            instr.output_byte_hint as usize
                        } else {
                            256
                        });
                        dispatch_kernel_par(
                            &instr.kernel,
                            &input_refs,
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
                        continue;
                    }

                    // General path: SmallVec + dispatch_kernel.
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
                            tape_ctx,
                            &*backend,
                            &mut out_buf,
                        )?
                    };

                    let out_id = NodeId::new(instr.output_idx, 0);
                    match dispatch_result {
                        DispatchResult::InOutBuf => {
                            arena.swap_insert_with_elem_size(
                                out_id,
                                &mut out_buf,
                                instr.output_elem_size as usize,
                            );
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
            kernel: TapeKernel::Float(FloatOp::Relu),
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 8, // 2 floats × 4 bytes
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
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
            kernel: TapeKernel::Float(FloatOp::Relu),
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 0,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
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
            kernel: TapeKernel::Float(FloatOp::Relu),
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
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
        });
        tape.end_level();
        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), input.clone());
        tape.execute(&mut arena, &ctx).unwrap();
        let inline_out = arena.get(NodeId::new(1, 0)).unwrap().to_vec();

        // Generic Float path
        let mut tape2 = EnumTape::new();
        tape2.push(TapeInstruction {
            kernel: TapeKernel::Float(FloatOp::Relu),
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            output_byte_hint: 8,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
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
}
