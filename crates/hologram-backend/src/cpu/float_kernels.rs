//! Native IEEE-754 CPU kernels (f32 / bf16 / f16).
//!
//! Selected when the KernelCall's `dtype` tag indicates a float dtype.
//! Mirrors the byte-domain kernels in semantics but at native precision.

use alloc::vec::Vec;

use crate::cpu::dtype::*;
#[cfg(not(feature = "std"))]
use crate::cpu::mathf::FloatExt;
use crate::error::BackendError;
use crate::kernel_call::*;
use crate::workspace::Workspace;

/// Byte width of a fixed-width dtype. The float kernels only ever see IEEE /
/// bfloat tags; a sub-byte or unrecognized tag is rejected rather than silently
/// assigned a plausible default width.
#[inline]
fn elem_size(dtype: u8) -> Result<usize, BackendError> {
    bytes_per_element(dtype).ok_or(BackendError::UnsupportedOp(
        "float kernel: dtype has no fixed element width (sub-byte or unknown)",
    ))
}

/// Resolve an **optional affine operand** (`gamma` / `beta`) of a normalization.
///
/// An *empty* operand means "no affine" — the identity, which is correct. A
/// *present but too short* operand is a different thing entirely: silently
/// falling back to `gamma = 1` / `beta = 0` drops the model's learned
/// per-feature scale and bias and returns a plausible, wrong tensor. Absent is
/// not the same as present-but-wrong.
///
/// `x` and `output` have always been validated this way (`ok_or(SlotOutOfRange)`);
/// `gamma`/`beta` were the lone operands with a soft `unwrap_or(&[])`. A
/// scalar/broadcast gamma (shape `[1]`, legal in ONNX-style graphs) compiled,
/// loaded, and executed with its scale silently ignored. Witness:
/// `rms_norm_with_a_short_gamma_fails_loud_instead_of_dropping_the_scale`.
#[inline]
pub(crate) fn affine_operand(bytes: &[u8], want: usize, slot: u32) -> Result<&[u8], BackendError> {
    if bytes.is_empty() {
        return Ok(&[]); // absent: identity affine
    }
    bytes.get(..want).ok_or(BackendError::SlotOutOfRange(slot))
}

#[inline]
fn elem_count_to_bytes(n: usize, dtype: u8) -> Result<usize, BackendError> {
    Ok(n * elem_size(dtype)?)
}

// Scratch for matmul's pre-transposed B. Under `std` it is a thread-local
// `RefCell<Vec<f32>>`, amortizing the `vec![0f32; k * n]` allocation across
// kernel invocations on the same thread — for trillion-parameter inference
// loops the difference between O(calls) and O(1) allocations. On `no_std`
// targets (wasm / embedded) there is no thread-local, so each call gets a
// fresh scratch buffer; the result is identical, only the amortization is
// lost.
#[cfg(feature = "std")]
fn with_matmul_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    std::thread_local! {
        static MATMUL_BT_SCRATCH: core::cell::RefCell<Vec<f32>> =
            const { core::cell::RefCell::new(Vec::new()) };
    }
    // Nested use on one thread — e.g. a pooled task helping-drained onto a
    // publisher that already holds this scratch — falls back to a fresh
    // buffer: identical semantics, one extra allocation, never a panic.
    MATMUL_BT_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut s) => f(&mut s),
        Err(_) => f(&mut Vec::new()),
    })
}

#[cfg(not(feature = "std"))]
fn with_matmul_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    let mut scratch = Vec::new();
    f(&mut scratch)
}

// Scratch for conv2d's im2col matrix. Distinct thread-local from the matmul
// `bt` scratch so conv can hold the lowered `[K, N]` patch matrix while the
// matmul engine uses its own buffer — the two never alias. Same amortization
// story: O(1) allocations across an inference loop under `std`, fresh per call
// on `no_std`.
#[cfg(feature = "std")]
fn with_conv_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    std::thread_local! {
        static CONV_IM2COL_SCRATCH: core::cell::RefCell<Vec<f32>> =
            const { core::cell::RefCell::new(Vec::new()) };
    }
    // Nested use on one thread — e.g. a pooled task helping-drained onto a
    // publisher that already holds this scratch — falls back to a fresh
    // buffer: identical semantics, one extra allocation, never a panic.
    CONV_IM2COL_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut s) => f(&mut s),
        Err(_) => f(&mut Vec::new()),
    })
}

#[cfg(not(feature = "std"))]
fn with_conv_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    let mut scratch = Vec::new();
    f(&mut scratch)
}

// Three reusable f32 buffers (A, B, output) for widening a low-precision
// matmul (bf16 / f16) into the f32 engine and narrowing the result back.
// Widening is O(mk + kn + mn) — amortized against the O(mkn) GEMM — and lets
// bf16/f16 matmul share the cache-oblivious / packed kernel instead of a
// strided scalar triple-loop. Thread-local so the loop pays no per-call alloc.
#[cfg(feature = "std")]
fn with_widen_scratch<R>(f: impl FnOnce(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>) -> R) -> R {
    std::thread_local! {
        static WIDEN: core::cell::RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            const { core::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    }
    // Nested use on one thread — e.g. a pooled task helping-drained onto a
    // publisher that already holds this scratch — falls back to a fresh
    // buffer: identical semantics, one extra allocation, never a panic.
    WIDEN.with(|cell| match cell.try_borrow_mut() {
        Ok(mut g) => {
            let (a, b, o) = &mut *g;
            f(a, b, o)
        }
        Err(_) => f(&mut Vec::new(), &mut Vec::new(), &mut Vec::new()),
    })
}

#[cfg(not(feature = "std"))]
fn with_widen_scratch<R>(f: impl FnOnce(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>) -> R) -> R {
    let (mut a, mut b, mut o) = (Vec::new(), Vec::new(), Vec::new());
    f(&mut a, &mut b, &mut o)
}

// Four reusable f32 buffers for widening a 3-input op (attention: Q, K, V, out)
// into the f32 engine and narrowing back. Same amortization as `with_widen`.
#[cfg(feature = "std")]
fn with_widen4_scratch<R>(
    f: impl FnOnce(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>) -> R,
) -> R {
    type Widen4Buf = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    std::thread_local! {
        static WIDEN4: core::cell::RefCell<Widen4Buf> =
            const { core::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
    }
    // Nested use on one thread — e.g. a pooled task helping-drained onto a
    // publisher that already holds this scratch — falls back to a fresh
    // buffer: identical semantics, one extra allocation, never a panic.
    WIDEN4.with(|cell| match cell.try_borrow_mut() {
        Ok(mut g) => {
            let (a, b, c, dd) = &mut *g;
            f(a, b, c, dd)
        }
        Err(_) => f(
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        ),
    })
}

#[cfg(not(feature = "std"))]
fn with_widen4_scratch<R>(
    f: impl FnOnce(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>) -> R,
) -> R {
    let (mut a, mut b, mut c, mut dd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    f(&mut a, &mut b, &mut c, &mut dd)
}

pub fn unary_float<W: Workspace>(
    c: &UnaryCall,
    ws: &mut W,
    f: fn(f32) -> f32,
    dtype: u8,
) -> Result<(), BackendError> {
    unary_float_acc(c, ws, f, None, dtype)
}

/// Whole-slice f32 elementwise unary SIMD primitive: `out = act(inp)`.
pub type SimdUnaryF32 = fn(&[f32], &mut [f32]);

/// Elementwise unary float op, optionally accelerated by a whole-slice SIMD
/// primitive on the contiguous f32 path. Transcendental activations
/// (sigmoid/silu/tanh/gelu) pass their vectorized `*_slice` form — bit-identical
/// to the scalar `f`, so the LUT/dequant SSOT contract is preserved — because a
/// scalar libm call per element cannot vectorize. Every other unary op passes
/// `None` and uses the scalar closure `f` (which the compiler autovectorizes for
/// the simple arithmetic ops anyway).
pub fn unary_float_acc<W: Workspace>(
    c: &UnaryCall,
    ws: &mut W,
    f: fn(f32) -> f32,
    simd: Option<SimdUnaryF32>,
    dtype: u8,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, dtype)?;
    // Zero-copy split-borrow + bytemuck cast (no fallback). Every
    // `Workspace` consumed by hologram's CPU compute must supply
    // `split_borrow`; the test `Ws` impls above and `BufferArena`
    // both do. Eliminates the `.to_vec()` clones the previous design
    // used to dodge the borrow checker.
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    let inp = reads[0]
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < bytes {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    if dtype == DTYPE_F32 {
        if let (Ok(i32s), Ok(o32s)) = (
            bytemuck::try_cast_slice::<u8, f32>(&inp[..bytes]),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..bytes]),
        ) {
            match simd {
                Some(simd_fn) => simd_fn(&i32s[..n], &mut o32s[..n]),
                None => {
                    for i in 0..n {
                        o32s[i] = f(i32s[i]);
                    }
                }
            }
            return Ok(());
        }
    }
    for i in 0..n {
        let v = read_float(inp, i, dtype);
        write_float(out, i, f(v), dtype);
    }
    Ok(())
}

/// Whole-slice f32 elementwise SIMD primitive: `out[i] = op(a[i], b[i])`.
type SimdBinaryF32 = fn(&[f32], &[f32], &mut [f32]);

pub fn binary_float<W: Workspace>(
    c: &BinaryCall,
    ws: &mut W,
    f: fn(f32, f32) -> f32,
    dtype: u8,
) -> Result<(), BackendError> {
    binary_float_acc(c, ws, f, None, dtype)
}

/// Elementwise binary float op, optionally accelerated by a whole-slice SIMD
/// primitive on the contiguous f32 path. `add` / `mul` pass the corresponding
/// `simd::simd_f32_*` (so those vectorized primitives are on the production
/// path, not just tests); every other op passes `None` and uses the scalar
/// closure `f` (which the compiler autovectorizes anyway).
pub fn binary_float_acc<W: Workspace>(
    c: &BinaryCall,
    ws: &mut W,
    f: fn(f32, f32) -> f32,
    simd: Option<SimdBinaryF32>,
    dtype: u8,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, dtype)?;
    let (reads, out) = ws
        .split_borrow(&[c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let a = reads[0]
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b = reads[1]
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < bytes {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    if dtype == DTYPE_F32 {
        if let (Ok(a32), Ok(b32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(&a[..bytes]),
            bytemuck::try_cast_slice::<u8, f32>(&b[..bytes]),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..bytes]),
        ) {
            match simd {
                Some(s) => s(a32, b32, o32),
                None => {
                    for i in 0..n {
                        o32[i] = f(a32[i], b32[i]);
                    }
                }
            }
            return Ok(());
        }
    }
    for i in 0..n {
        let va = read_float(a, i, dtype);
        let vb = read_float(b, i, dtype);
        write_float(out, i, f(va, vb), dtype);
    }
    Ok(())
}

pub fn matmul_float<W: Workspace>(c: &MatMulCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;

    // Zero-copy split-borrow + bytemuck f32 view + blocked-tile +
    // runtime-SIMD path. The transposed-B scratch is thread-local so
    // back-to-back matmul calls on the same thread don't re-allocate.
    let (reads, out) = ws
        .split_borrow(&[c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let a = reads[0]
        .get(..m * k * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    // A packed-B weight occupies `layout::packed_len(k,n)` elements (≥ k·n);
    // a plain weight is row-major `k×n`. View the matching extent (the packed
    // length comes from the single source of truth for the layout).
    let b_len = if c.b_packed {
        crate::layout::packed_len(k, n) * es
    } else {
        k * n * es
    };
    let b = reads[1]
        .get(..b_len)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < m * n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }

    if dt == DTYPE_F32 {
        // Zero-copy: the workspace buffers are already contiguous, 64-byte
        // aligned f32, so view them directly and call the shared kernel. (The
        // prism-canonical `TensorAxis` surface — `HologramTensorMatmulF32` —
        // wraps the same kernel for external consumers per ADR-031; the
        // runtime hot path must not pay its marshalling copy of A and B.)
        if let (Ok(a32), Ok(b32), Ok(out32)) = (
            bytemuck::try_cast_slice::<u8, f32>(a),
            bytemuck::try_cast_slice::<u8, f32>(b),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..m * n * 4]),
        ) {
            if c.b_packed {
                // B is the compile-time panel-packed weight: the leaf streams
                // it contiguously (no strided gather). Zero runtime copy.
                crate::cpu::simd::matmul_f32_packed(a32, b32, out32, m, k, n);
            } else {
                with_matmul_scratch(|bt| {
                    crate::cpu::simd::matmul_f32_blocked(a32, b32, out32, m, k, n, bt);
                });
            }
            return Ok(());
        }
    }

    // bf16 / f16: accumulate in f32 (identical to the old scalar path's
    // `acc: f32`), then narrow the result. `A` (m×k) is always widened — it is
    // tiny at decode. `B` (the k×n weight) is the large operand and was the
    // problem: the old path materialized the WHOLE weight to an f32 scratch on
    // every call via a scalar per-element `read_float`, so an M=1 decode
    // re-widened the entire constant weight each token (its dominant cost).
    //   • small M (decode / short prefill): stream B directly through
    //     `matmul_lowp_gemv` — the weight is widened in-register, never
    //     materialized (the bf16/f16 analog of the int8 decode kernel).
    //   • large M (prefill): materialize B once with a **vectorized** widen
    //     (the cost amortizes over the M output rows), then the shared f32
    //     kernel. Both are bit-identical (same `a·widen(b)` sum order).
    if dt == DTYPE_BF16 || dt == DTYPE_F16 {
        const LOWP_STREAM_M_MAX: usize = 8;
        let is_f16 = dt == DTYPE_F16;
        with_widen_scratch(|a32, b32, o32| {
            a32.clear();
            a32.extend((0..m * k).map(|i| read_float(a, i, dt)));
            // `resize` zero-fills only the growth. A preceding `clear()` would
            // zero all `m·n` (and `k·n`) elements on every call — stores that
            // both kernels below immediately overwrite in full.
            o32.resize(m * n, 0.0);
            if m <= LOWP_STREAM_M_MAX {
                crate::cpu::simd::matmul_lowp_gemv(a32, b, o32, m, k, n, is_f16);
            } else {
                b32.resize(k * n, 0.0);
                crate::cpu::simd::widen_lowp_to_f32(b, b32, is_f16);
                with_matmul_scratch(|bt| {
                    crate::cpu::simd::matmul_f32_blocked(a32, b32, o32, m, k, n, bt);
                });
            }
            for (i, &v) in o32.iter().enumerate() {
                write_float(out, i, v, dt);
            }
        });
        return Ok(());
    }

    // f32 / f16 / bf16 are handled above; every other float dtype (f64) is
    // rejected at dispatch. No scalar compute fallback exists.
    Err(BackendError::UnsupportedOp(
        "matmul: only f32/f16/bf16 are supported compute dtypes",
    ))
}

/// Fused dequantize → matmul: `out = A · dequant(Bq)`. `Bq` (i8/i4, per-tensor
/// or per-channel) is dequantized into a **transient** f32 panel — the dense
/// weight never occupies a pool slot — then the tuned blocked f32 kernel runs
/// unchanged (no change to the FMA inner loop, so the perf floors hold).
/// Decode the `i`-th quantized weight of a `[k,n]` panel to its integer level.
/// `quant_dtype` is loop-invariant at every call site, so inlining lets LLVM
/// unswitch the dispatch out of the dequant loop.
/// The quantized weight encodings the **scalar** (W8A32) dequant loop can read
/// directly, validated once before the loop. Tiers that need extra operands to
/// decode — `e8cb`, whose weights are codebook indices — are not members, so
/// they are rejected up front instead of silently decoding to zero (the old
/// `_ => 0` arm turned an unhandled tier into a zero-filled result).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarQuant {
    I8,
    U8,
    I4,
}

impl ScalarQuant {
    fn from_tag(tag: u8) -> Result<Self, BackendError> {
        use crate::cpu::dtype::*;
        match tag {
            DTYPE_I8 => Ok(Self::I8),
            DTYPE_U8 => Ok(Self::U8),
            DTYPE_I4 => Ok(Self::I4),
            _ => Err(BackendError::UnsupportedOp(
                "matmul_dequant: quant_dtype is not scalar-decodable \
                 (e8cb needs its codebook operand and the fused omajor path)",
            )),
        }
    }

    /// Total: `self` is a validated variant, so there is no fallback arm. The
    /// dispatch is loop-invariant and unswitches out of the dequant loop.
    #[inline(always)]
    fn read(self, bq: &[u8], i: usize) -> i32 {
        match self {
            Self::I8 => (bq[i] as i8) as i32,
            Self::U8 => bq[i] as i32,
            Self::I4 => {
                let byte = bq[i / 2];
                let nib = if i.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                let v = nib as i32;
                if v >= 8 {
                    v - 16
                } else {
                    v
                }
            }
        }
    }
}

pub fn matmul_dequant_float<W: Workspace>(
    c: &MatMulDequantCall,
    ws: &mut W,
) -> Result<(), BackendError> {
    use crate::cpu::dtype::*;
    let (m, k, n) = (c.m as usize, c.k as usize, c.n as usize);
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    if c.dtype != DTYPE_F32 {
        return Err(BackendError::UnsupportedOp(
            "matmul_dequant: only f32 output is supported",
        ));
    }
    let kn = k * n;
    // The weight's stored size is tier data (i4 packs two per byte; e8cb stores
    // one index per 8-element group). An unregistered tag is an error, never a
    // guessed length.
    let tier = crate::quant_tier::quant_tier(DTypeId(c.quant_dtype)).ok_or(
        BackendError::UnsupportedOp("matmul_dequant: unknown quantized weight tier"),
    )?;
    let in_bytes = match tier.weight_bytes(k, n) {
        Some(b) => b,
        None => {
            return Err(BackendError::UnsupportedOp(
                "matmul_dequant: weight shape is not representable in this tier",
            ))
        }
    };
    // A vector-quantized tier must carry its codebook operand, and only such a
    // tier may carry one — mismatch means the call was not built by a compiler
    // that understands this tier, so reject rather than decode garbage.
    if tier.needs_codebook != c.has_codebook() {
        return Err(BackendError::UnsupportedOp(
            "matmul_dequant: codebook operand missing for a VQ tier (or present without one)",
        ));
    }
    let per_ch = c.per_channel();
    // Operand order mirrors `buffers()`: `a, bq[, scales, zero_points][, codebook]`.
    let mut spec = [c.a; 5];
    let mut n_spec = 2;
    spec[1] = c.bq;
    if per_ch {
        spec[2] = c.scales;
        spec[3] = c.zero_points;
        n_spec = 4;
    }
    let cb_idx = n_spec;
    if c.has_codebook() {
        spec[n_spec] = c.codebook;
        n_spec += 1;
    }
    let (reads, out) = ws
        .split_borrow(&spec[..n_spec], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let a = reads[0]
        .get(..m * k * 4)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let bq = reads[1]
        .get(..in_bytes)
        .ok_or(BackendError::SlotOutOfRange(c.bq.slot))?;
    let (scales, zps): (&[u8], &[u8]) = if per_ch {
        (reads[2], reads[3])
    } else {
        (&[], &[])
    };
    if out.len() < m * n * 4 {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let channels = c.channels as usize;
    let inner = (c.inner as usize).max(1);
    let scale = f32::from_bits(c.scale_bits);
    let zp = c.zero_point;
    let quant_dtype = c.quant_dtype;

    // Output-major W8A8 decode GEMV — compile-time-fused constant weights
    // (`fuse_const_i8_decode`). The omajor layout pairs exclusively with
    // W8A8 (a single emitter), and `matmul_i8_pc_omajor` runs the identical
    // exact-integer function on every target, so this arm has no per-arch
    // gate, no W8A32 downgrade, and no fallthrough: an output-major weight
    // must never be read with the `[k,n]` interpretation below — any
    // contract violation fails loud.
    {
        use crate::kernel_call::mm_act_quant;
        let omajor = c.bq_omajor;
        let w8a8 = c.act_quant == mm_act_quant::W8A8_TOKEN_SYM;
        if omajor || w8a8 {
            if !(omajor && w8a8) {
                return Err(BackendError::UnsupportedOp(
                    "matmul_dequant: bq_omajor and W8A8 must be paired",
                ));
            }
            // A fused omajor GEMV exists for this tier, and `k` is a whole
            // number of its groups — both read from the tier registry.
            let dtype_ok = tier.omajor_fusable && tier.omajor_k_ok(k);
            let symmetric = per_ch
                && dtype_ok
                && channels == n
                && inner == 1
                && zps
                    .chunks_exact(4)
                    .all(|z| i32::from_le_bytes([z[0], z[1], z[2], z[3]]) == 0);
            if !symmetric || k > mm_act_quant::K_MAX {
                return Err(BackendError::UnsupportedOp(
                    "matmul_dequant: W8A8 requires symmetric per-channel i8/i4/e8cb within the k bound",
                ));
            }
            let a32 = bytemuck::try_cast_slice::<u8, f32>(a)
                .map_err(|_| BackendError::SlotOutOfRange(c.a.slot))?;
            let scale32 = bytemuck::try_cast_slice::<u8, f32>(scales)
                .map_err(|_| BackendError::SlotOutOfRange(c.scales.slot))?;
            let out32 = bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..m * n * 4])
                .map_err(|_| BackendError::SlotOutOfRange(c.output.slot))?;
            if tier.needs_codebook {
                // VQ tier: `u8` codebook indices, 8× fewer streamed bytes. The
                // codebook is the **model's** — a constant operand, not an
                // engine table — so two models with different codebooks coexist.
                // Its length is fixed at the full index space so any `u8` index
                // dereferences in range without a per-call bounds scan.
                let cb_bytes = reads[cb_idx];
                let want = DTypeId::E8CB_MAX_ENTRIES * tier.group_dim as usize;
                if cb_bytes.len() < want {
                    return Err(BackendError::SlotOutOfRange(c.codebook.slot));
                }
                let codebook = bytemuck::cast_slice::<u8, i8>(&cb_bytes[..want]);
                crate::cpu::simd::matmul_e8cb_omajor(a32, bq, codebook, scale32, out32, m, k, n);
            } else if quant_dtype == DTYPE_I4 {
                // LUT tier: packed nibbles, half the streamed bytes.
                crate::cpu::simd::matmul_i4_pc_omajor(a32, bq, scale32, out32, m, k, n);
            } else {
                let bq_i8 = bytemuck::cast_slice::<u8, i8>(bq);
                crate::cpu::simd::matmul_i8_pc_omajor(a32, bq_i8, scale32, out32, m, k, n);
            }
            return Ok(());
        }
    }

    // Fast path — fused per-channel symmetric int8 → f32 at small M (decode).
    // Reads the i8 weight directly (no f32 materialization), factoring the
    // per-column scale to the writeback. On the decode (M ≤ 3) shape the f32
    // register tile (MR = 4) has not engaged, so this beats the dequant-then-
    // matmul below — which must first materialize a `k·n` f32 panel.
    //
    // Dispatched on **every** SIMD target. x86-64 was excluded on the theory
    // that "the scalar fused kernel would lose to the tuned f32 path", but
    // `matmul_i8_per_channel` has an AVX2 inner (`matmul_i8_pc_avx2`) and the
    // exclusion left x86 materializing the whole weight: measured 18.8 ms vs
    // 0.56 ms at `1×896×4864`, a 34× loss, masked until W8A8 stopped being
    // applied implicitly. It also meant x86 and aarch64/wasm computed *different
    // bytes* for the same W8A32 matmul at `m ≤ 3`, since the two paths factor
    // the per-column scale differently. They now agree.
    //
    // Requires per-output-column scales (channels == n, inner == 1) and
    // all-zero zero-points (symmetric).
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    {
        if per_ch
            && quant_dtype == DTYPE_I8
            && m <= crate::kernel_call::decode_gate::FUSED_W8A32_MAX_M
            && channels == n
            && inner == 1
            && zps
                .chunks_exact(4)
                .all(|z| i32::from_le_bytes([z[0], z[1], z[2], z[3]]) == 0)
        {
            if let (Ok(a32), Ok(scale32), Ok(out32)) = (
                bytemuck::try_cast_slice::<u8, f32>(a),
                bytemuck::try_cast_slice::<u8, f32>(scales),
                bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..m * n * 4]),
            ) {
                let bq_i8 = bytemuck::cast_slice::<u8, i8>(bq);
                crate::cpu::simd::matmul_i8_per_channel(a32, bq_i8, scale32, out32, m, k, n);
                return Ok(());
            }
        }
    }

    // `bdq` is a reused thread-local (zero alloc per call after warm-up) holding
    // the dequantized B panel. A/out are workspace slots — 64-byte aligned by
    // construction — so the f32 views always succeed; an unaligned operand is a
    // contract violation and fails loud (no scalar/copy fallback), matching
    // `matmul_float`.
    with_widen_scratch(|_a_unused, bdq, _o_unused| {
        // Size to `kn` without a redundant zero-fill: the dequant loop below
        // overwrites every element, so `resize` (not `clear` + `resize`) is
        // enough — after warmup `len == kn`, so this zeros nothing. The old
        // clear+resize re-zeroed the whole panel each call (a wasted `kn`-float
        // write that made the fused path lose to the unfused one).
        if bdq.len() > kn {
            bdq.truncate(kn);
        } else {
            bdq.resize(kn, 0.0);
        }
        // Dequantize the B panel into `bdq`. The weight encoding is validated
        // **once**, here: a tier the scalar loop cannot decode (e8cb, whose
        // weights are codebook indices) is rejected rather than read as zeros.
        // Pre-cast the per-channel scale/zp tables to typed slices once
        // (aligned workspace slots always cast) rather than reconstructing each
        // scalar with `from_le_bytes` per element — that per-element
        // reconstruction is what made the fused dequant slower than the
        // standalone Dequantize kernel. `sq` is loop-invariant, so the inlined
        // `read` match unswitches and the contiguous inner map autovectorizes.
        let sq = ScalarQuant::from_tag(quant_dtype)?;
        let sca = if per_ch {
            bytemuck::try_cast_slice::<u8, f32>(scales).ok()
        } else {
            None
        };
        let zpa = if per_ch {
            bytemuck::try_cast_slice::<u8, i32>(zps).ok()
        } else {
            None
        };
        match (per_ch, sca, zpa) {
            // Per-channel with aligned typed tables (the production path).
            (true, Some(sa), Some(za)) => {
                for (i, slot) in bdq.iter_mut().enumerate() {
                    let ch = (i / inner) % channels;
                    *slot = (sq.read(bq, i) - za[ch]) as f32 * sa[ch];
                }
            }
            // Per-tensor: scalar scale/zp — the inner map is a straight f32 line.
            (false, _, _) => {
                for (i, slot) in bdq.iter_mut().enumerate() {
                    *slot = (sq.read(bq, i) - zp) as f32 * scale;
                }
            }
            // Misaligned per-channel tables (non-`BufferArena` workspace only):
            // reconstruct per element.
            _ => {
                for (i, slot) in bdq.iter_mut().enumerate() {
                    let ch = (i / inner) % channels;
                    let s = f32::from_le_bytes([
                        scales[ch * 4],
                        scales[ch * 4 + 1],
                        scales[ch * 4 + 2],
                        scales[ch * 4 + 3],
                    ]);
                    let z = i32::from_le_bytes([
                        zps[ch * 4],
                        zps[ch * 4 + 1],
                        zps[ch * 4 + 2],
                        zps[ch * 4 + 3],
                    ]);
                    *slot = (sq.read(bq, i) - z) as f32 * s;
                }
            }
        }
        let a32 = bytemuck::try_cast_slice::<u8, f32>(a)
            .map_err(|_| BackendError::SlotOutOfRange(c.a.slot))?;
        let out32 = bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..m * n * 4])
            .map_err(|_| BackendError::SlotOutOfRange(c.output.slot))?;
        with_matmul_scratch(|bt| {
            crate::cpu::simd::matmul_f32_blocked(a32, bdq, out32, m, k, n, bt);
        });
        Ok(())
    })
}

/// Fused epilogue for `MatMulDequant`: `out = act(out [+ residual])`, applied
/// in place over the `m·n` results while they are still hot in cache — the
/// same functions the f32 matmul epilogues use, so the decode projection's
/// `act(A·dequant(Bq) + bias)` is one call with no separately materialized
/// or addressed intermediate. A no-op for calls without an epilogue.
pub fn matmul_dequant_epilogue<W: Workspace>(
    c: &MatMulDequantCall,
    ws: &mut W,
) -> Result<(), BackendError> {
    if c.act == 0 && !c.has_residual() {
        return Ok(());
    }
    let count = (c.m as usize) * (c.n as usize);
    if count == 0 {
        return Ok(());
    }
    let reads_spec: &[crate::workspace::BufferRef] = if c.has_residual() {
        core::slice::from_ref(&c.residual)
    } else {
        &[]
    };
    let (reads, out) = ws
        .split_borrow(reads_spec, c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let out_bytes = out
        .get_mut(..count * 4)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let out32 = bytemuck::try_cast_slice_mut::<u8, f32>(out_bytes)
        .map_err(|_| BackendError::SlotOutOfRange(c.output.slot))?;
    if c.has_residual() {
        let res = reads[0]
            .get(..count * 4)
            .ok_or(BackendError::SlotOutOfRange(c.residual.slot))?;
        let res32 = bytemuck::try_cast_slice::<u8, f32>(res)
            .map_err(|_| BackendError::SlotOutOfRange(c.residual.slot))?;
        for (o, &r) in out32.iter_mut().zip(res32) {
            *o += r;
        }
    }
    if c.act != 0 {
        apply_fused_act_f32(c.act, out32);
    }
    Ok(())
}

/// Selector → activation function for a fused matmul epilogue.
fn fused_act_fn(act: u8) -> fn(f32) -> f32 {
    use crate::kernel_call::fused_activation as fa;
    match act {
        fa::RELU => relu_f,
        fa::GELU => gelu_f,
        fa::SILU => silu_f,
        fa::SIGMOID => sigmoid_f,
        fa::TANH => tanh_f,
        fa::ELU => elu_f,
        fa::SELU => selu_f,
        fa::EXP => exp_f,
        _ => |x| x,
    }
}

/// Apply a fused activation over an **f32** output slice in place. The
/// transcendental activations (sigmoid/silu/tanh/gelu) route through their
/// vectorized `*_slice` form — the scalar `fused_act_fn` (a deterministic-exp
/// closure applied per element through a fn pointer) is markedly slower than
/// one vectorized pass, and applying it in the epilogue regressed fused
/// `matmul → activation`. The `*_slice` forms need a distinct input (silu/gelu
/// re-read `x` in their final multiply), so they read from the reused matmul
/// scratch copied from the output; the cheap/rare acts (relu/elu/selu/exp) keep
/// the scalar closure. Bit-identical to the scalar path (the slice twins are
/// bit-identical to their scalar `*_f` reference by construction).
fn apply_fused_act_f32(act: u8, o32: &mut [f32]) {
    use crate::kernel_call::fused_activation as fa;
    let slice: Option<SimdUnaryF32> = match act {
        fa::SIGMOID => Some(sigmoid_slice),
        fa::TANH => Some(tanh_slice),
        fa::SILU => Some(silu_slice),
        fa::GELU => Some(gelu_slice),
        _ => None,
    };
    match slice {
        Some(sfn) => with_matmul_scratch(|scratch| {
            scratch.clear();
            scratch.extend_from_slice(o32);
            sfn(scratch, o32);
        }),
        None => {
            let f = fused_act_fn(act);
            for v in o32.iter_mut() {
                *v = f(*v);
            }
        }
    }
}

/// **Fused matmul + activation (content-addressed fusion).** Computes the
/// matmul into the output slot, then applies the activation *in place* over
/// the `m·n` results while they are still hot in cache — so the activation
/// has no separate input/output buffer and no second dispatch. Equivalent
/// to `activation(matmul(a, b))`, verified against the f64 reference.
pub fn matmul_activation_float<W: Workspace>(
    c: &MatMulActivationCall,
    ws: &mut W,
) -> Result<(), BackendError> {
    matmul_float(&c.mm, ws)?;
    let count = (c.mm.m as usize) * (c.mm.n as usize);
    if count == 0 {
        return Ok(());
    }
    let dt = c.mm.dtype;
    let es = elem_size(dt)?;
    let f = fused_act_fn(c.act);
    let out = ws.write(c.mm.output);
    if out.len() < count * es {
        return Err(BackendError::SlotOutOfRange(c.mm.output.slot));
    }
    if dt == DTYPE_F32 {
        if let Ok(o32s) = bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..count * 4]) {
            apply_fused_act_f32(c.act, o32s);
            return Ok(());
        }
    }
    for i in 0..count {
        let v = read_float(out, i, dt);
        write_float(out, i, f(v), dt);
    }
    Ok(())
}

/// **Fused matmul + residual-add (content-addressed fusion).** Computes the
/// matmul into the output slot, then adds the residual tensor *in place* over
/// the `m·n` results while they are hot in cache — the transformer skip
/// connection `y = A·B + residual` as one op, eliding both the matmul's
/// intermediate and the separate (bandwidth-bound) add pass. Equivalent to
/// `add(matmul(a, b), residual)`, verified against the f64 reference.
pub fn matmul_add_float<W: Workspace>(c: &MatMulAddCall, ws: &mut W) -> Result<(), BackendError> {
    matmul_float(&c.mm, ws)?;
    let count = (c.mm.m as usize) * (c.mm.n as usize);
    if count == 0 {
        return Ok(());
    }
    let dt = c.mm.dtype;
    let es = elem_size(dt)?;
    // Disjoint borrow: residual (read) + matmul output (write).
    let (reads, out) = ws
        .split_borrow(&[c.residual], c.mm.output)
        .ok_or(BackendError::SlotOutOfRange(c.mm.output.slot))?;
    let res = reads[0]
        .get(..count * es)
        .ok_or(BackendError::SlotOutOfRange(c.residual.slot))?;
    if out.len() < count * es {
        return Err(BackendError::SlotOutOfRange(c.mm.output.slot));
    }
    if dt == DTYPE_F32 {
        if let (Ok(o32s), Ok(r32s)) = (
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..count * 4]),
            bytemuck::try_cast_slice::<u8, f32>(res),
        ) {
            for (o, &r) in o32s.iter_mut().zip(r32s.iter()) {
                *o += r;
            }
            return Ok(());
        }
    }
    for i in 0..count {
        let v = read_float(out, i, dt) + read_float(res, i, dt);
        write_float(out, i, v, dt);
    }
    Ok(())
}

/// **Fused matmul + residual-add + activation (content-addressed fusion).**
/// Computes the matmul into the output slot, adds the residual, then applies
/// the activation — all over the `m·n` results while hot in cache, eliding the
/// matmul product, the post-add sum, *and* the activation intermediate as
/// distinct addressed values. Equivalent to `act(add(matmul(a, b), residual))`,
/// so it inherits the V&V of its three component kernels.
pub fn matmul_add_activation_float<W: Workspace>(
    c: &MatMulAddActivationCall,
    ws: &mut W,
) -> Result<(), BackendError> {
    matmul_float(&c.mm, ws)?;
    let count = (c.mm.m as usize) * (c.mm.n as usize);
    if count == 0 {
        return Ok(());
    }
    let dt = c.mm.dtype;
    let es = elem_size(dt)?;
    let f = fused_act_fn(c.act);
    let (reads, out) = ws
        .split_borrow(&[c.residual], c.mm.output)
        .ok_or(BackendError::SlotOutOfRange(c.mm.output.slot))?;
    let res = reads[0]
        .get(..count * es)
        .ok_or(BackendError::SlotOutOfRange(c.residual.slot))?;
    if out.len() < count * es {
        return Err(BackendError::SlotOutOfRange(c.mm.output.slot));
    }
    if dt == DTYPE_F32 {
        if let (Ok(o32s), Ok(r32s)) = (
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..count * 4]),
            bytemuck::try_cast_slice::<u8, f32>(res),
        ) {
            // Residual add (autovectorizes) then the vectorized activation pass.
            for (o, &r) in o32s.iter_mut().zip(r32s.iter()) {
                *o += r;
            }
            apply_fused_act_f32(c.act, o32s);
            return Ok(());
        }
    }
    for i in 0..count {
        let v = read_float(out, i, dt) + read_float(res, i, dt);
        write_float(out, i, f(v), dt);
    }
    Ok(())
}

/// **im2col** (valid conv receptive-field gather). Reads `input [Cin,Hin,Win]`
/// and writes `output [Cin·kh·kw, Hout·Wout]`, where row `(ci·kh+kh')·kw+kw'`
/// column `oh·Wout+ow` is `input[ci, oh·sh+kh', ow·sw+kw']`. Mirrors the
/// receptive-field map of `conv2d_f32_engine`, so `W·im2col(x)` reproduces the
/// convolution. Pure gather — every supported float dtype via read/write_float.
pub fn im2col_float<W: Workspace>(c: &Im2ColCall, ws: &mut W) -> Result<(), BackendError> {
    let (cin, hin, win) = (c.channels as usize, c.h_in as usize, c.w_in as usize);
    let (hout, wout) = (c.h_out as usize, c.w_out as usize);
    let (kh, kw) = (c.k_h as usize, c.k_w as usize);
    let (sh, sw) = ((c.stride_h as usize).max(1), (c.stride_w as usize).max(1));
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let nn = hout * wout;
    let kk = cin * kh * kw;
    if kk == 0 || nn == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..cin * hin * win * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < kk * nn * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path: pure gather over `&[f32]` — the contiguous `ow` store
    // autovectorizes and each read is a direct load (no per-element `read_float`
    // dtype match). Bit-identical (pure data movement).
    if dt == DTYPE_F32 {
        if let (Ok(i32s), Ok(o32s)) = (
            bytemuck::try_cast_slice::<u8, f32>(inp),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..kk * nn * es]),
        ) {
            for ci in 0..cin {
                for kh_ in 0..kh {
                    for kw_ in 0..kw {
                        let krow = (ci * kh + kh_) * kw + kw_;
                        for oh in 0..hout {
                            let ih = oh * sh + kh_;
                            let orow =
                                &mut o32s[krow * nn + oh * wout..krow * nn + oh * wout + wout];
                            if ih < hin {
                                let ibase = (ci * hin + ih) * win;
                                for (ow, o) in orow.iter_mut().enumerate() {
                                    let iw = ow * sw + kw_;
                                    *o = if iw < win { i32s[ibase + iw] } else { 0.0 };
                                }
                            } else {
                                orow.fill(0.0);
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for ci in 0..cin {
        for kh_ in 0..kh {
            for kw_ in 0..kw {
                let krow = (ci * kh + kh_) * kw + kw_;
                for oh in 0..hout {
                    let ih = oh * sh + kh_;
                    for ow in 0..wout {
                        let iw = ow * sw + kw_;
                        let v = if ih < hin && iw < win {
                            read_float(inp, (ci * hin + ih) * win + iw, dt)
                        } else {
                            0.0
                        };
                        write_float(out, krow * nn + oh * wout + ow, v, dt);
                    }
                }
            }
        }
    }
    Ok(())
}

/// **col2im** — the adjoint of [`im2col_float`]. Reads a patch matrix
/// `input [Cin·kh·kw, Hout·Wout]` and scatter-**adds** each entry back to its
/// source pixel in `output [Cin,Hin,Win]`; overlapping receptive fields
/// accumulate. This is exactly the input-gradient of a convolution given the
/// upstream patch-space gradient, so it closes conv's VJP composition.
pub fn col2im_float<W: Workspace>(c: &Im2ColCall, ws: &mut W) -> Result<(), BackendError> {
    let (cin, hin, win) = (c.channels as usize, c.h_in as usize, c.w_in as usize);
    let (hout, wout) = (c.h_out as usize, c.w_out as usize);
    let (kh, kw) = (c.k_h as usize, c.k_w as usize);
    let (sh, sw) = ((c.stride_h as usize).max(1), (c.stride_w as usize).max(1));
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let nn = hout * wout;
    let kk = cin * kh * kw;
    if kk == 0 || nn == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..kk * nn * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < cin * hin * win * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path: scatter-add over `&[f32]` (direct loads/stores, no
    // per-element `read_float`). Bit-identical for f32 (read/write_float are the
    // identity there); the scatter order is unchanged.
    if dt == DTYPE_F32 {
        if let (Ok(i32s), Ok(o32s)) = (
            bytemuck::try_cast_slice::<u8, f32>(inp),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..cin * hin * win * es]),
        ) {
            for o in o32s[..cin * hin * win].iter_mut() {
                *o = 0.0;
            }
            for ci in 0..cin {
                for kh_ in 0..kh {
                    for kw_ in 0..kw {
                        let krow = (ci * kh + kh_) * kw + kw_;
                        for oh in 0..hout {
                            let ih = oh * sh + kh_;
                            if ih >= hin {
                                continue;
                            }
                            let prow = krow * nn + oh * wout;
                            let ibase = (ci * hin + ih) * win;
                            for ow in 0..wout {
                                let iw = ow * sw + kw_;
                                if iw < win {
                                    o32s[ibase + iw] += i32s[prow + ow];
                                }
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    // Zero the image, then accumulate overlapping patches.
    for i in 0..cin * hin * win {
        write_float(out, i, 0.0, dt);
    }
    for ci in 0..cin {
        for kh_ in 0..kh {
            for kw_ in 0..kw {
                let krow = (ci * kh + kh_) * kw + kw_;
                for oh in 0..hout {
                    let ih = oh * sh + kh_;
                    if ih >= hin {
                        continue;
                    }
                    for ow in 0..wout {
                        let iw = ow * sw + kw_;
                        if iw >= win {
                            continue;
                        }
                        let v = read_float(inp, krow * nn + oh * wout + ow, dt);
                        let idx = (ci * hin + ih) * win + iw;
                        let acc = read_float(out, idx, dt) + v;
                        write_float(out, idx, acc, dt);
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn gemm_float<W: Workspace>(c: &GemmCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let alpha = f32::from_bits(c.alpha_bits as u32);
    let beta = f32::from_bits(c.beta_bits as u32);

    let (reads, out) = ws
        .split_borrow(&[c.a, c.b, c.c], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let a = reads[0]
        .get(..m * k * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b = reads[1]
        .get(..k * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    let cc = reads[2]
        .get(..m * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.c.slot))?;
    if out.len() < m * n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }

    // α·(A·B) + β·C through the one engine for every supported float dtype:
    // f32 zero-copy, f16/bf16 widened. No scalar fallback.
    if dt == DTYPE_F32 {
        let (a32, b32, c32, out32) = (
            bytemuck::cast_slice::<u8, f32>(a),
            bytemuck::cast_slice::<u8, f32>(b),
            bytemuck::cast_slice::<u8, f32>(cc),
            bytemuck::cast_slice_mut::<u8, f32>(&mut out[..m * n * 4]),
        );
        with_matmul_scratch(|bt| {
            crate::cpu::simd::matmul_f32_blocked(a32, b32, out32, m, k, n, bt);
        });
        for i in 0..m * n {
            out32[i] = alpha * out32[i] + beta * c32[i];
        }
        return Ok(());
    }
    if dt == DTYPE_BF16 || dt == DTYPE_F16 {
        with_widen_scratch(|a32, b32, o32| {
            a32.clear();
            a32.extend((0..m * k).map(|i| read_float(a, i, dt)));
            b32.clear();
            b32.extend((0..k * n).map(|i| read_float(b, i, dt)));
            o32.clear();
            o32.resize(m * n, 0.0);
            with_matmul_scratch(|bt| {
                crate::cpu::simd::matmul_f32_blocked(a32, b32, o32, m, k, n, bt);
            });
            for (i, &v) in o32.iter().enumerate() {
                let bias = read_float(cc, i, dt) * beta;
                write_float(out, i, alpha * v + bias, dt);
            }
        });
        return Ok(());
    }
    Err(BackendError::UnsupportedOp(
        "gemm: only f32/f16/bf16 are supported compute dtypes",
    ))
}

pub fn conv2d_float<W: Workspace>(c: &Conv2dCall, ws: &mut W) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let cin = c.channels_in as usize;
    let cout = c.channels_out as usize;
    let h_in = c.h_in as usize;
    let w_in = c.w_in as usize;
    let h_out = c.h_out as usize;
    let w_out = c.w_out as usize;
    let k_h = c.k_h as usize;
    let k_w = c.k_w as usize;
    let s_h = (c.stride_h as usize).max(1);
    let s_w = (c.stride_w as usize).max(1);
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total_in = b * cin * h_in * w_in * es;
    let total_w = cout * cin * k_h * k_w * es;
    let total_out = b * cout * h_out * w_out * es;
    let (reads, out) = ws
        .split_borrow(&[c.x, c.w], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    if total_in == 0 || total_w == 0 || total_out == 0 {
        for o in out.iter_mut() {
            *o = 0;
        }
        return Ok(());
    }
    let xs = reads[0]
        .get(..total_in)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let ws_w = reads[1]
        .get(..total_w)
        .ok_or(BackendError::SlotOutOfRange(c.w.slot))?;
    if out.len() < total_out {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }

    // Convolution *is* a per-batch GEMM: the weight already lives in `[cout, K]`
    // row-major layout (K = cin·kh·kw), so lowering the input patches to an
    // im2col matrix `Xcol[K, ho·wo]` makes each batch a dense
    // `out_b[cout, ho·wo] = W[cout, K] · Xcol[K, ho·wo]`, routed through the one
    // cache-oblivious / parallel matmul engine. Every supported float dtype goes
    // through that engine — f32 zero-copy, f16/bf16 widened (they are sub-f32
    // storage with no native compute, so f32 accumulation *is* their semantics).
    // There is no scalar fallback.
    let dims = ConvDims {
        b,
        cin,
        cout,
        h_in,
        w_in,
        h_out,
        w_out,
        k_h,
        k_w,
        s_h,
        s_w,
    };
    if dt == DTYPE_F32 {
        let (xs32, w32, out32) = (
            bytemuck::cast_slice::<u8, f32>(xs),
            bytemuck::cast_slice::<u8, f32>(ws_w),
            bytemuck::cast_slice_mut::<u8, f32>(out),
        );
        conv2d_f32_engine(xs32, w32, out32, &dims);
        return Ok(());
    }
    if dt != DTYPE_BF16 && dt != DTYPE_F16 {
        return Err(BackendError::UnsupportedOp(
            "conv2d: only f32/f16/bf16 are supported compute dtypes",
        ));
    }
    // f16 / bf16: widen inputs to f32, run the engine, narrow the result.
    with_widen_scratch(|x32, w32, o32| {
        x32.clear();
        x32.extend((0..b * cin * h_in * w_in).map(|i| read_float(xs, i, dt)));
        w32.clear();
        w32.extend((0..cout * cin * k_h * k_w).map(|i| read_float(ws_w, i, dt)));
        o32.clear();
        o32.resize(b * cout * h_out * w_out, 0.0);
        conv2d_f32_engine(x32, w32, o32, &dims);
        for (i, &v) in o32.iter().enumerate() {
            write_float(out, i, v, dt);
        }
    });
    Ok(())
}

struct ConvDims {
    b: usize,
    cin: usize,
    cout: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    k_h: usize,
    k_w: usize,
    s_h: usize,
    s_w: usize,
}

/// im2col + per-batch GEMM on f32 slices (the shared engine core for every conv
/// dtype). `col` (im2col patches) and `bt` (matmul transpose scratch) are reused
/// thread-locals; the receptive-field zero-fill matches ONNX valid convolution.
fn conv2d_f32_engine(xs32: &[f32], w32: &[f32], out32: &mut [f32], d: &ConvDims) {
    let kk = d.cin * d.k_h * d.k_w;
    let nn = d.h_out * d.w_out;
    with_conv_scratch(|col| {
        col.clear();
        col.resize(kk * nn, 0.0);
        with_matmul_scratch(|bt| {
            for bi in 0..d.b {
                for ci in 0..d.cin {
                    for kh in 0..d.k_h {
                        for kw in 0..d.k_w {
                            let krow = (ci * d.k_h + kh) * d.k_w + kw;
                            let xbase = (bi * d.cin + ci) * d.h_in;
                            for oh in 0..d.h_out {
                                let ih = oh * d.s_h + kh;
                                let crow = krow * nn + oh * d.w_out;
                                for ow in 0..d.w_out {
                                    let iw = ow * d.s_w + kw;
                                    col[crow + ow] = if ih < d.h_in && iw < d.w_in {
                                        xs32[(xbase + ih) * d.w_in + iw]
                                    } else {
                                        0.0
                                    };
                                }
                            }
                        }
                    }
                }
                let ob = &mut out32[bi * d.cout * nn..(bi + 1) * d.cout * nn];
                crate::cpu::simd::matmul_f32_blocked(w32, col, ob, d.cout, kk, nn, bt);
            }
        });
    });
}

/// Resolve a norm's declared epsilon. `0` selects the pinned default `1e-9`
/// (the historical behavior for an absent declaration); any **positive
/// finite** declared value is honored exactly — the declared structure is
/// authoritative, and the kernel never second-guesses it with a floor. A
/// negative, NaN, or infinite declaration is refused loud
/// (refuse-not-fabricate) instead of being silently rewritten.
fn norm_epsilon(epsilon_bits: u64) -> Result<f32, BackendError> {
    if epsilon_bits == 0 {
        return Ok(1e-9);
    }
    let e = f32::from_bits(epsilon_bits as u32);
    if e.is_finite() && e > 0.0 {
        Ok(e)
    } else {
        Err(BackendError::UnsupportedOp(
            "norm epsilon must be a positive finite f32 (0 selects the default)",
        ))
    }
}

pub fn layer_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    if c.num_groups > 0 {
        return group_norm_float(c, ws);
    }
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total = bsz * f * es;
    let eps = norm_epsilon(c.epsilon_bits)?;

    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma, c.beta], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = affine_operand(reads[1], f * es, c.gamma.slot)?;
    let beta = affine_operand(reads[2], f * es, c.beta.slot)?;
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path (mirrors `rms_norm_float`): view the aligned slot bytes as
    // `&[f32]`, take the mean with the vectorized `simd_f32_sum`, center into
    // the output buffer as scratch, take the variance as `simd_f32_dot` of the
    // centered row against itself (the numerically-stable two-pass form the
    // scalar loop uses), then scale in place. The generic `read_float` loop
    // below is kept as the bf16/f16/misaligned fallback.
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..total]),
        ) {
            let g32 = bytemuck::try_cast_slice::<u8, f32>(gamma).ok();
            let b32 = bytemuck::try_cast_slice::<u8, f32>(beta).ok();
            let inv_f = 1.0 / f as f32;
            for bi in 0..bsz {
                let row = &x32[bi * f..bi * f + f];
                let mean = crate::cpu::simd::simd_f32_sum(row) * inv_f;
                let orow = &mut o32[bi * f..bi * f + f];
                for (o, &v) in orow.iter_mut().zip(row) {
                    *o = v - mean;
                }
                let var = crate::cpu::simd::simd_f32_dot(orow, orow) * inv_f;
                let inv_std = 1.0 / libm::sqrtf(var + eps);
                match (g32, b32) {
                    (Some(g), Some(bta)) if g.len() >= f && bta.len() >= f => {
                        for ((o, &gj), &bj) in orow.iter_mut().zip(&g[..f]).zip(&bta[..f]) {
                            *o = *o * inv_std * gj + bj;
                        }
                    }
                    (Some(g), _) if g.len() >= f => {
                        for (o, &gj) in orow.iter_mut().zip(&g[..f]) {
                            *o = *o * inv_std * gj;
                        }
                    }
                    (_, Some(bta)) if bta.len() >= f => {
                        for (o, &bj) in orow.iter_mut().zip(&bta[..f]) {
                            *o = *o * inv_std + bj;
                        }
                    }
                    _ => {
                        for o in orow.iter_mut() {
                            *o *= inv_std;
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for bi in 0..bsz {
        let row_off = bi * f;
        let mut mean = 0f32;
        for j in 0..f {
            mean += read_float(xs, row_off + j, dt);
        }
        mean /= f as f32;
        let mut var = 0f32;
        for j in 0..f {
            let d = read_float(xs, row_off + j, dt) - mean;
            var += d * d;
        }
        var /= f as f32;
        let inv_std = 1.0 / libm::sqrtf(var + eps);
        for j in 0..f {
            let g = if !gamma.is_empty() {
                read_float(gamma, j, dt)
            } else {
                1.0
            };
            let bv = if !beta.is_empty() {
                read_float(beta, j, dt)
            } else {
                0.0
            };
            let v = (read_float(xs, row_off + j, dt) - mean) * inv_std * g + bv;
            write_float(out, row_off + j, v, dt);
        }
    }
    Ok(())
}

/// GroupNorm / InstanceNorm (ONNX): each of `batch` samples carries `feature`
/// (= `channels` × spatial) elements split into `num_groups` contiguous groups;
/// each group is mean/variance-normalized independently, then scaled per channel
/// by `gamma`/`beta` (length `channels`). InstanceNorm is the `num_groups ==
/// channels` case. Routed here from `layer_norm_float` when `num_groups > 0`.
pub fn group_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.batch as usize;
    let f = c.feature as usize;
    let ch = c.channels as usize;
    let g = c.num_groups as usize;
    if n == 0 || f == 0 {
        return Ok(());
    }
    // Divisibility is a shape invariant; reject violations rather than compute a
    // silently-wrong result over ragged groups.
    if ch == 0 || g == 0 || !f.is_multiple_of(ch) || !f.is_multiple_of(g) || !ch.is_multiple_of(g) {
        return Err(BackendError::UnsupportedOp(
            "group_norm: require channels|feature, num_groups|feature, num_groups|channels",
        ));
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total = n * f * es;
    let eps = norm_epsilon(c.epsilon_bits)?;
    let spatial = f / ch; // elements per channel
    let group_size = f / g; // elements per normalization group

    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma, c.beta], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = affine_operand(reads[1], ch * es, c.gamma.slot)?;
    let beta = affine_operand(reads[2], ch * es, c.beta.slot)?;
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path (the LayerNorm pattern, generalized to per-group + per-
    // channel γ/β): `simd_f32_sum` mean, stable two-pass variance via
    // `simd_f32_dot` of the centered group, then scale in contiguous
    // `spatial`-sized runs that share one channel's γ/β. The scalar loop below
    // stays the bf16/f16/misaligned fallback.
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..total]),
        ) {
            let g32 = bytemuck::try_cast_slice::<u8, f32>(gamma).ok();
            let b32 = bytemuck::try_cast_slice::<u8, f32>(beta).ok();
            let inv_gs = 1.0 / group_size as f32;
            let chans_per_group = ch / g;
            for ni in 0..n {
                let sample = ni * f;
                for gi in 0..g {
                    let gbase = sample + gi * group_size;
                    let row = &x32[gbase..gbase + group_size];
                    let mean = crate::cpu::simd::simd_f32_sum(row) * inv_gs;
                    let orow = &mut o32[gbase..gbase + group_size];
                    for (o, &v) in orow.iter_mut().zip(row) {
                        *o = v - mean;
                    }
                    let var = crate::cpu::simd::simd_f32_dot(orow, orow) * inv_gs;
                    let inv_std = 1.0 / libm::sqrtf(var + eps);
                    let chan_base = gi * chans_per_group;
                    for local_ch in 0..chans_per_group {
                        let ci = chan_base + local_ch;
                        let gv = match g32 {
                            Some(gg) if gg.len() >= ch => gg[ci],
                            _ => 1.0,
                        };
                        let bv = match b32 {
                            Some(bb) if bb.len() >= ch => bb[ci],
                            _ => 0.0,
                        };
                        let run = &mut orow[local_ch * spatial..local_ch * spatial + spatial];
                        for o in run.iter_mut() {
                            *o = *o * inv_std * gv + bv;
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for ni in 0..n {
        let sample = ni * f;
        for gi in 0..g {
            let gbase = sample + gi * group_size;
            let mut mean = 0f32;
            for i in 0..group_size {
                mean += read_float(xs, gbase + i, dt);
            }
            mean /= group_size as f32;
            let mut var = 0f32;
            for i in 0..group_size {
                let d = read_float(xs, gbase + i, dt) - mean;
                var += d * d;
            }
            var /= group_size as f32;
            let inv_std = 1.0 / libm::sqrtf(var + eps);
            for i in 0..group_size {
                // Channel of this element within the sample: contiguous layout is
                // [channel][spatial], so channel = (group offset) / spatial.
                let ci = (gi * group_size + i) / spatial;
                let gv = if !gamma.is_empty() {
                    read_float(gamma, ci, dt)
                } else {
                    1.0
                };
                let bv = if !beta.is_empty() {
                    read_float(beta, ci, dt)
                } else {
                    0.0
                };
                let v = (read_float(xs, gbase + i, dt) - mean) * inv_std * gv + bv;
                write_float(out, gbase + i, v, dt);
            }
        }
    }
    Ok(())
}

pub fn add_rms_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total = bsz * f * es;
    let eps = norm_epsilon(c.epsilon_bits)?;
    let has_residual = c.has_residual();

    let (reads, out) = if has_residual {
        ws.split_borrow(&[c.x, c.residual, c.gamma], c.output)
    } else {
        ws.split_borrow(&[c.x, c.gamma], c.output)
    }
    .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let residual = if has_residual {
        Some(
            reads[1]
                .get(..total)
                .ok_or(BackendError::SlotOutOfRange(c.residual.slot))?,
        )
    } else {
        None
    };
    let gamma_idx = if has_residual { 2 } else { 1 };
    let gamma = affine_operand(reads[gamma_idx], f * es, c.gamma.slot)?;
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path (see `rms_norm_float`): materialize `x + residual` into the
    // output view, take the sum-of-squares with `simd_f32_dot`, and scale in
    // place — all vectorized. Runs every decode layer per token.
    const DTYPE_F32: u8 = 8;
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..total]),
        ) {
            let r32 = residual.and_then(|r| bytemuck::try_cast_slice::<u8, f32>(r).ok());
            let g32 = bytemuck::try_cast_slice::<u8, f32>(gamma).ok();
            let has_g = matches!(g32, Some(g) if g.len() >= f);
            for bi in 0..bsz {
                let row = &x32[bi * f..bi * f + f];
                let rrow = r32.map(|r| &r[bi * f..bi * f + f]);
                let orow = &mut o32[bi * f..bi * f + f];
                let sumsq = match rrow {
                    Some(rr) => {
                        for ((o, &v), &rv) in orow.iter_mut().zip(row).zip(rr) {
                            *o = v + rv;
                        }
                        crate::cpu::simd::simd_f32_dot(orow, orow)
                    }
                    None => crate::cpu::simd::simd_f32_dot(row, row),
                };
                let inv = 1.0 / libm::sqrtf(sumsq / f as f32 + eps);
                match (rrow.is_some(), has_g) {
                    (true, true) => {
                        let g = &g32.unwrap()[..f];
                        for (o, &gj) in orow.iter_mut().zip(g) {
                            *o *= inv * gj;
                        }
                    }
                    (true, false) => {
                        for o in orow.iter_mut() {
                            *o *= inv;
                        }
                    }
                    (false, true) => {
                        let g = &g32.unwrap()[..f];
                        for ((o, &v), &gj) in orow.iter_mut().zip(row).zip(g) {
                            *o = v * inv * gj;
                        }
                    }
                    (false, false) => {
                        for (o, &v) in orow.iter_mut().zip(row) {
                            *o = v * inv;
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for bi in 0..bsz {
        let row_off = bi * f;
        let mut sumsq = 0f32;
        for j in 0..f {
            let v = read_float(xs, row_off + j, dt)
                + residual
                    .map(|r| read_float(r, row_off + j, dt))
                    .unwrap_or(0.0);
            sumsq += v * v;
        }
        let inv_rms = 1.0 / libm::sqrtf(sumsq / f as f32 + eps);
        for j in 0..f {
            let v = read_float(xs, row_off + j, dt)
                + residual
                    .map(|r| read_float(r, row_off + j, dt))
                    .unwrap_or(0.0);
            let g = if !gamma.is_empty() {
                read_float(gamma, j, dt)
            } else {
                1.0
            };
            write_float(out, row_off + j, v * inv_rms * g, dt);
        }
    }
    Ok(())
}

pub fn rms_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total = bsz * f * es;
    let eps = norm_epsilon(c.epsilon_bits)?;
    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = affine_operand(reads[1], f * es, c.gamma.slot)?;
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path: view the aligned slot bytes as `&[f32]` (the arena pads
    // slots to 64 bytes, so the cast succeeds), take the sum-of-squares with
    // the vectorized `simd_f32_dot`, and normalize over the contiguous view
    // (an elementwise map — the compiler vectorizes it). Every decode layer
    // runs this per token; the generic `read_float` loop below (kept as the
    // bf16/f16/misaligned fallback) blocked both the reduction and the map.
    const DTYPE_F32: u8 = 8;
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..total]),
        ) {
            let g32 = bytemuck::try_cast_slice::<u8, f32>(gamma).ok();
            for bi in 0..bsz {
                let row = &x32[bi * f..bi * f + f];
                let sumsq = crate::cpu::simd::simd_f32_dot(row, row);
                let inv_rms = 1.0 / libm::sqrtf(sumsq / f as f32 + eps);
                let orow = &mut o32[bi * f..bi * f + f];
                match g32 {
                    Some(g) if g.len() >= f => {
                        for ((o, &v), &gj) in orow.iter_mut().zip(row).zip(&g[..f]) {
                            *o = v * inv_rms * gj;
                        }
                    }
                    _ => {
                        for (o, &v) in orow.iter_mut().zip(row) {
                            *o = v * inv_rms;
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for bi in 0..bsz {
        let row_off = bi * f;
        let mut sumsq = 0f32;
        for j in 0..f {
            let v = read_float(xs, row_off + j, dt);
            sumsq += v * v;
        }
        let inv_rms = 1.0 / libm::sqrtf(sumsq / f as f32 + eps);
        for j in 0..f {
            let g = if !gamma.is_empty() {
                read_float(gamma, j, dt)
            } else {
                1.0
            };
            let v = read_float(xs, row_off + j, dt) * inv_rms * g;
            write_float(out, row_off + j, v, dt);
        }
    }
    Ok(())
}

pub fn softmax_float<W: Workspace>(
    c: &SoftmaxCall,
    ws: &mut W,
    log_form: bool,
) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let f = c.feature as usize;
    if b == 0 || f == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total = b * f * es;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }

    // f32 fast path: view the aligned slot bytes as `&[f32]` so the max pass is
    // the vectorized `simd_f32_max` and the framing (subtract-max, normalize)
    // walks contiguous `&[f32]` runs instead of per-element `read_float` /
    // `write_float`. The exp pass and the sequential denominator sum are
    // unchanged. The generic loop below stays as the bf16/f16/misaligned path.
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..total]),
        ) {
            with_matmul_scratch(|exps| {
                for bi in 0..b {
                    let row = &x32[bi * f..bi * f + f];
                    let max_v = crate::cpu::simd::simd_f32_max(row);
                    // All-(−∞) row: pinned total semantics — softmax of an
                    // impossible distribution is exactly zero everywhere;
                    // log-softmax is exactly −∞ (log 0). The shift below
                    // would otherwise make the row NaN.
                    if max_v == f32::NEG_INFINITY {
                        let orow = &mut o32[bi * f..bi * f + f];
                        orow.fill(if log_form { f32::NEG_INFINITY } else { 0.0 });
                        continue;
                    }
                    exps.clear();
                    exps.extend(row.iter().map(|&v| v - max_v));
                    crate::cpu::simd::simd_f32_exp_inplace(exps);
                    let mut sum = 0f32;
                    for &e in exps.iter() {
                        sum += e;
                    }
                    let orow = &mut o32[bi * f..bi * f + f];
                    if log_form {
                        let log_sum = libm::logf(sum.max(1e-30)) + max_v;
                        for (o, &v) in orow.iter_mut().zip(row) {
                            *o = v - log_sum;
                        }
                    } else {
                        let denom = sum.max(1e-30);
                        for (o, &e) in orow.iter_mut().zip(exps.iter()) {
                            *o = e / denom;
                        }
                    }
                }
            });
            return Ok(());
        }
    }

    // Reuse the thread-local matmul scratch as a per-row exp buffer.
    // Reset between rows; never reallocates after the first call of
    // matching feature size.
    with_matmul_scratch(|exps| {
        for bi in 0..b {
            let row_off = bi * f;
            let mut max_v = f32::NEG_INFINITY;
            for j in 0..f {
                max_v = max_v.max(read_float(xs, row_off + j, dt));
            }
            // All-(−∞) row: same pinned semantics as the f32 fast path.
            if max_v == f32::NEG_INFINITY {
                for j in 0..f {
                    write_float(
                        out,
                        row_off + j,
                        if log_form { f32::NEG_INFINITY } else { 0.0 },
                        dt,
                    );
                }
                continue;
            }
            exps.clear();
            exps.reserve(f);
            for j in 0..f {
                exps.push(read_float(xs, row_off + j, dt) - max_v);
            }
            // Vectorized deterministic exp (bit-identical across targets);
            // the sum keeps its original sequential reduction order.
            crate::cpu::simd::simd_f32_exp_inplace(exps);
            let mut sum = 0f32;
            for &e in exps.iter() {
                sum += e;
            }
            let log_sum = libm::logf(sum.max(1e-30)) + max_v;
            for (j, &e) in exps.iter().enumerate() {
                let v = if log_form {
                    read_float(xs, row_off + j, dt) - log_sum
                } else {
                    e / sum.max(1e-30)
                };
                write_float(out, row_off + j, v, dt);
            }
        }
    });
    Ok(())
}

/// The associative fold a [`reduce_float`] performs. Carries enough structure
/// (unlike a bare `fn(f32,f32)->f32`) to let the **full-reduction** f32 path
/// dispatch to the multi-accumulator SIMD primitives — an opaque fn pointer in
/// the innermost fold is a serial dependent chain the compiler cannot vectorize.
#[derive(Clone, Copy)]
pub enum ReduceKind {
    Sum,
    Prod,
    Min,
    Max,
}

impl ReduceKind {
    #[inline]
    fn fold(self) -> (fn(f32, f32) -> f32, f32) {
        match self {
            ReduceKind::Sum => (|a, b| a + b, 0.0),
            ReduceKind::Prod => (|a, b| a * b, 1.0),
            ReduceKind::Min => (|a, b| a.min(b), f32::INFINITY),
            ReduceKind::Max => (|a, b| a.max(b), f32::NEG_INFINITY),
        }
    }
}

pub fn reduce_float<W: Workspace>(
    c: &ReduceCall,
    ws: &mut W,
    kind: ReduceKind,
    mean: bool,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let (f, init) = kind.fold();
    let plan = ReducePlan::new(c, n)?;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < plan.out_count * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path: view both slots as `&[f32]` and replace the per-element
    // `out_offset` (a full rank-length div/mod decode of `i` on every element)
    // with an incremental odometer that carries the output offset alongside the
    // input walk — a reduced axis contributes 0 to the offset, a kept axis its
    // output stride. Identical fold order (linear `i`) and identical cell
    // mapping to the scalar path, so the reduction result is unchanged.
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(out),
        ) {
            // Full reduction (single output cell): the odometer degenerates to a
            // serial fold into `o32[0]`. Use the multi-accumulator SIMD
            // primitives instead. Min/Max are order-independent (bit-identical);
            // Sum's chunked order differs in the last ULP exactly as the f32
            // norm/dot reductions already do; Prod has no SIMD primitive.
            if plan.out_count == 1 {
                let acc = match kind {
                    ReduceKind::Sum => crate::cpu::simd::simd_f32_sum(&x32[..n]),
                    ReduceKind::Max => crate::cpu::simd::simd_f32_max(&x32[..n]),
                    ReduceKind::Min => crate::cpu::simd::simd_f32_min(&x32[..n]),
                    ReduceKind::Prod => x32[..n].iter().fold(1.0f32, |p, &v| p * v),
                };
                o32[0] = if mean {
                    acc / plan.reduced_count as f32
                } else {
                    acc
                };
                for o in o32[1..].iter_mut() {
                    *o = 0.0;
                }
                return Ok(());
            }
            for o in o32[..plan.out_count].iter_mut() {
                *o = init;
            }
            let rank = plan.rank;
            let mut coord = [0usize; MAX_RANK];
            let mut oo = 0usize;
            for &xv in x32[..n].iter() {
                o32[oo] = f(o32[oo], xv);
                // Advance the odometer over the input dims (rightmost fastest),
                // tracking `oo` incrementally.
                let mut ax = rank;
                while ax > 0 {
                    ax -= 1;
                    coord[ax] += 1;
                    if !plan.reduced[ax] {
                        oo += plan.out_stride[ax];
                    }
                    if coord[ax] < plan.in_dims[ax] {
                        break;
                    }
                    coord[ax] = 0;
                    if !plan.reduced[ax] {
                        oo -= plan.out_stride[ax] * plan.in_dims[ax];
                    }
                }
            }
            if mean {
                let inv = 1.0 / plan.reduced_count as f32;
                for o in o32[..plan.out_count].iter_mut() {
                    *o *= inv;
                }
            }
            // Zero any trailing cells beyond the reduced output (slot may be wider).
            for o in o32[plan.out_count..].iter_mut() {
                *o = 0.0;
            }
            return Ok(());
        }
    }
    // Initialize each output cell to the fold identity, then fold every input
    // element into the cell its non-reduced coordinates address.
    for o in 0..plan.out_count {
        write_float(out, o, init, dt);
    }
    let mut coord = [0usize; MAX_RANK];
    for i in 0..n {
        let oo = plan.out_offset(i, &mut coord);
        let acc = f(read_float(out, oo, dt), read_float(xs, i, dt));
        write_float(out, oo, acc, dt);
    }
    if mean {
        let inv = 1.0 / plan.reduced_count as f32;
        for o in 0..plan.out_count {
            write_float(out, o, read_float(out, o, dt) * inv, dt);
        }
    }
    // Zero any trailing bytes beyond the reduced output (the slot may be wider).
    for b in out.iter_mut().skip(plan.out_count * es) {
        *b = 0;
    }
    Ok(())
}

/// Resolved layout for an axis reduction: which axes collapse, the row-major
/// output strides over the keepdims (`reduced → 1`) shape, and the per-element
/// map from an input linear index to its output cell. Shared by the float and
/// byte reduce kernels so both fold over identical geometry.
pub(crate) struct ReducePlan {
    rank: usize,
    in_dims: [usize; MAX_RANK],
    /// Row-major stride into the output for each axis; a reduced axis has its
    /// coordinate forced to 0, so its stride is never actually consumed.
    out_stride: [usize; MAX_RANK],
    reduced: [bool; MAX_RANK],
    pub out_count: usize,
    pub reduced_count: usize,
}

impl ReducePlan {
    pub(crate) fn new(c: &ReduceCall, n: usize) -> Result<Self, BackendError> {
        let rank = c.rank as usize;
        if rank > MAX_RANK {
            return Err(BackendError::UnsupportedOp("reduce: rank must be ≤ 8"));
        }
        // rank 0 (or no dims) ⇒ a single-element full reduction.
        let mut in_dims = [1usize; MAX_RANK];
        let mut reduced = [false; MAX_RANK];
        let mut out_dims = [1usize; MAX_RANK];
        let mut out_count = 1usize;
        let mut reduced_count = 1usize;
        let full = rank == 0 || c.axes_mask == 0;
        for i in 0..rank {
            in_dims[i] = c.dims[i] as usize;
            let is_red = full || (c.axes_mask >> i) & 1 == 1;
            reduced[i] = is_red;
            if is_red {
                reduced_count *= in_dims[i];
            } else {
                out_dims[i] = in_dims[i];
                out_count *= in_dims[i];
            }
        }
        // Row-major strides over the keepdims output shape.
        let mut out_stride = [0usize; MAX_RANK];
        let mut s = 1usize;
        for i in (0..rank).rev() {
            out_stride[i] = s;
            s *= out_dims[i];
        }
        // Guard the declared input count against the shape product.
        let prod: usize = (0..rank).map(|i| in_dims[i]).product::<usize>().max(1);
        if rank > 0 && prod != n {
            return Err(BackendError::UnsupportedOp(
                "reduce: dims product ≠ element_count",
            ));
        }
        Ok(Self {
            rank,
            in_dims,
            out_stride,
            reduced,
            out_count,
            reduced_count,
        })
    }

    /// Output cell for input linear index `i` (decodes `i` into `coord`).
    #[inline]
    #[allow(clippy::needless_range_loop)] // index addresses coord + per-axis dims/strides
    pub(crate) fn out_offset(&self, i: usize, coord: &mut [usize; MAX_RANK]) -> usize {
        let mut rem = i;
        for a in (0..self.rank).rev() {
            coord[a] = rem % self.in_dims[a];
            rem /= self.in_dims[a];
        }
        let mut oo = 0usize;
        for a in 0..self.rank {
            if !self.reduced[a] {
                oo += coord[a] * self.out_stride[a];
            }
        }
        oo
    }
}

pub fn cumsum_float<W: Workspace>(c: &ReduceCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let mut acc = 0f32;
    for i in 0..n {
        acc += read_float(xs, i, dt);
        write_float(out, i, acc, dt);
    }
    Ok(())
}

pub fn pool_float<W: Workspace>(
    c: &PoolCall,
    ws: &mut W,
    take_max: bool,
) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let ch = c.channels as usize;
    let h_in = c.h_in as usize;
    let w_in = c.w_in as usize;
    let h_out = (c.h_out as usize).max(1);
    let w_out = (c.w_out as usize).max(1);
    let k_h = (c.k_h as usize).max(1);
    let k_w = (c.k_w as usize).max(1);
    let s_h = (c.stride_h as usize).max(1);
    let s_w = (c.stride_w as usize).max(1);
    if b * ch * h_in * w_in == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let total_in = b * ch * h_in * w_in * es;
    let total_out = b * ch * h_out * w_out * es;
    let (reads, out) = ws
        .split_borrow(&[c.x], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total_in)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    if out.len() < total_out {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path: direct `&[f32]` loads in the pooling window instead of a
    // per-element `read_float` dtype match. Bit-identical (same accumulation
    // order; f32 read/write_float are the identity).
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(xs),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..total_out]),
        ) {
            for bi in 0..b {
                for ci in 0..ch {
                    let plane = (bi * ch + ci) * h_in;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut acc = if take_max { f32::NEG_INFINITY } else { 0f32 };
                            let mut count = 0u32;
                            for kh in 0..k_h {
                                let ih = oh * s_h + kh;
                                if ih >= h_in {
                                    continue;
                                }
                                let rbase = (plane + ih) * w_in;
                                for kw in 0..k_w {
                                    let iw = ow * s_w + kw;
                                    if iw < w_in {
                                        let v = x32[rbase + iw];
                                        if take_max {
                                            acc = acc.max(v);
                                        } else {
                                            acc += v;
                                        }
                                        count += 1;
                                    }
                                }
                            }
                            let result = if take_max {
                                acc
                            } else if count > 0 {
                                acc / count as f32
                            } else {
                                0.0
                            };
                            o32[((bi * ch + ci) * h_out + oh) * w_out + ow] = result;
                        }
                    }
                }
            }
            return Ok(());
        }
    }
    for bi in 0..b {
        for ci in 0..ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = if take_max { f32::NEG_INFINITY } else { 0f32 };
                    let mut count = 0u32;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = oh * s_h + kh;
                            let iw = ow * s_w + kw;
                            if ih < h_in && iw < w_in {
                                let v =
                                    read_float(xs, ((bi * ch + ci) * h_in + ih) * w_in + iw, dt);
                                if take_max {
                                    acc = acc.max(v);
                                } else {
                                    acc += v;
                                }
                                count += 1;
                            }
                        }
                    }
                    let result = if take_max {
                        acc
                    } else if count > 0 {
                        acc / count as f32
                    } else {
                        0.0
                    };
                    let oi = ((bi * ch + ci) * h_out + oh) * w_out + ow;
                    write_float(out, oi, result, dt);
                }
            }
        }
    }
    Ok(())
}

pub fn attention_float<W: Workspace>(c: &AttentionCall, ws: &mut W) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let h = c.heads as usize;
    let s = c.seq as usize;
    let d = c.head_dim as usize;
    if b == 0 || h == 0 || s == 0 || d == 0 {
        return Ok(());
    }
    // Grouped-query attention: K/V carry `kv_heads` heads (0 ⇒ multi-head ==
    // heads). Each query head maps to kv head `hi / (h / hkv)`; require an even
    // grouping so the mapping is exact (no silent-wrong).
    let hkv = if c.kv_heads == 0 {
        h
    } else {
        c.kv_heads as usize
    };
    if hkv == 0 || !h.is_multiple_of(hkv) {
        return Err(BackendError::UnsupportedOp(
            "attention: heads must be a multiple of kv_heads (grouped-query)",
        ));
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let q_total = b * h * s * d;
    let kv_total = b * hkv * s * d;
    let (reads, out) = ws
        .split_borrow(&[c.q, c.k, c.v], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let q = reads[0]
        .get(..q_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.q.slot))?;
    let kk = reads[1]
        .get(..kv_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.k.slot))?;
    let v = reads[2]
        .get(..kv_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.v.slot))?;
    if out.len() < q_total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // Softmax score divisor: explicit multiplier (its reciprocal) when given,
    // else the standard `1/√head_dim`.
    let scale = match c.scale_bits {
        0 => libm::sqrtf(d as f32).max(1.0),
        bits => {
            let m = f32::from_bits(bits);
            if m > 0.0 {
                1.0 / m
            } else {
                libm::sqrtf(d as f32).max(1.0)
            }
        }
    };
    let causal = c.causal;

    // Scaled dot-product attention is two matmuls per (batch, head): QKᵀ scores
    // and the P·V context. Every supported float dtype runs through the one f32
    // engine core — f32 zero-copy, f16/bf16 widened (sub-f32 storage). No scalar
    // fallback. Q/K/V rows are contiguous over the head dimension, so the score
    // reduction is a contiguous SIMD dot and the context is a contiguous AXPY.
    if dt == DTYPE_F32 {
        let (q32, k32, v32, out32) = (
            bytemuck::cast_slice::<u8, f32>(q),
            bytemuck::cast_slice::<u8, f32>(kk),
            bytemuck::cast_slice::<u8, f32>(v),
            bytemuck::cast_slice_mut::<u8, f32>(out),
        );
        attention_f32_engine(q32, k32, v32, out32, b, h, hkv, s, d, scale, causal);
        return Ok(());
    }
    if dt != DTYPE_BF16 && dt != DTYPE_F16 {
        return Err(BackendError::UnsupportedOp(
            "attention: only f32/f16/bf16 are supported compute dtypes",
        ));
    }
    // f16 / bf16: widen Q/K/V to f32, run the engine, narrow the result.
    with_widen4_scratch(|q32, k32, v32, o32| {
        q32.clear();
        q32.extend((0..q_total).map(|i| read_float(q, i, dt)));
        k32.clear();
        k32.extend((0..kv_total).map(|i| read_float(kk, i, dt)));
        v32.clear();
        v32.extend((0..kv_total).map(|i| read_float(v, i, dt)));
        o32.clear();
        o32.resize(q_total, 0.0);
        attention_f32_engine(q32, k32, v32, o32, b, h, hkv, s, d, scale, causal);
        for (i, &val) in o32.iter().enumerate() {
            write_float(out, i, val, dt);
        }
    });
    Ok(())
}

/// Scaled dot-product attention on f32 slices (the shared engine core for every
/// attention dtype). Scores via contiguous `simd_f32_dot`; context via
/// contiguous AXPY. The per-row score buffer is the reused matmul scratch.
#[allow(clippy::too_many_arguments)]
fn attention_f32_engine(
    q32: &[f32],
    k32: &[f32],
    v32: &[f32],
    out32: &mut [f32],
    b: usize,
    h: usize,
    hkv: usize,
    s: usize,
    d: usize,
    scale: f32,
    causal: bool,
) {
    // Grouped-query mapping: consecutive `group` query heads share one kv head.
    let group = h / hkv;
    with_matmul_scratch(|scores| {
        for bi in 0..b {
            for hi in 0..h {
                let q_off = (bi * h + hi) * s * d;
                // K/V are indexed by the kv head this query head reads.
                let kv_off = (bi * hkv + hi / group) * s * d;
                for qi in 0..s {
                    let qrow = &q32[q_off + qi * d..q_off + qi * d + d];
                    scores.clear();
                    scores.resize(s, 0.0);
                    for (kj, score) in scores.iter_mut().enumerate() {
                        // Causal mask: query qi attends only to keys kj ≤ qi.
                        if causal && kj > qi {
                            *score = f32::NEG_INFINITY;
                            continue;
                        }
                        let krow = &k32[kv_off + kj * d..kv_off + kj * d + d];
                        *score = crate::cpu::simd::simd_f32_dot(qrow, krow) / scale;
                    }
                    let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for sc in scores.iter_mut() {
                        *sc -= max_s;
                    }
                    // Vectorized deterministic exp; masked −∞ scores stay
                    // exactly 0. Sequential sum order unchanged.
                    crate::cpu::simd::simd_f32_exp_inplace(scores);
                    let mut sum = 0f32;
                    for &sc in scores.iter() {
                        sum += sc;
                    }
                    let denom = sum.max(1e-30);
                    let orow = &mut out32[q_off + qi * d..q_off + qi * d + d];
                    orow.fill(0.0);
                    for (kj, &sc) in scores.iter().enumerate() {
                        let p = sc / denom;
                        let vrow = &v32[kv_off + kj * d..kv_off + kj * d + d];
                        // orow += p · vrow (broadcast-scalar AXPY, bit-identical
                        // to the scalar accumulation it replaces).
                        crate::cpu::simd::simd_f32_axpy(orow, p, vrow);
                    }
                }
            }
        }
    });
}

/// One `(batch, head, query-row)` of the fused decode attention: scores over
/// `past ∥ new` keys (read where they lie — the concatenation is never
/// materialized), plus the additive mask, then max-shift → deterministic exp →
/// sum → probability-weighted context over both V segments.
///
/// This is the unit both the serial loop and every pool participant run, so a
/// partitioned call is bit-identical to serial by construction: each output
/// row is computed whole, by one participant, in this exact order. The key
/// iteration order is `past` then `new`, which makes the split form
/// bit-identical to attention over the precatenated buffer.
///
/// # Safety
/// `q_row` addresses `d` f32; `k_past`/`v_past` address `past·d` f32 from the
/// row's kv-head base (null iff `past == 0`); `k_new`/`v_new` address `new·d`
/// (null iff `new == 0`); `mask_row` addresses `past + new` f32; `out_row`
/// addresses `d` f32; `scores` addresses `past + new` f32 scratch, exclusive
/// to the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_attention_row(
    q_row: *const f32,
    k_past: *const f32,
    v_past: *const f32,
    k_new: *const f32,
    v_new: *const f32,
    mask_row: *const f32,
    out_row: *mut f32,
    past: usize,
    new: usize,
    d: usize,
    scale: f32,
    scores: *mut f32,
) {
    let l = past + new;
    let q = core::slice::from_raw_parts(q_row, d);
    let sc = core::slice::from_raw_parts_mut(scores, l);
    for (kj, slot) in sc.iter_mut().enumerate() {
        let krow = if kj < past {
            core::slice::from_raw_parts(k_past.add(kj * d), d)
        } else {
            core::slice::from_raw_parts(k_new.add((kj - past) * d), d)
        };
        *slot = crate::cpu::simd::simd_f32_dot(q, krow) / scale + *mask_row.add(kj);
    }
    let max_s = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // A row whose mask erases *every* key has no visible context; its
    // semantics is pinned to the exact zero vector. Without this guard the
    // max-shift below would compute `−∞ − (−∞) = NaN` and the row would be
    // garbage — a threshold accident, not a semantics. The mask is the single
    // visibility authority, so "no key visible" is a legal, total input.
    if max_s == f32::NEG_INFINITY {
        core::slice::from_raw_parts_mut(out_row, d).fill(0.0);
        return;
    }
    for v in sc.iter_mut() {
        *v -= max_s;
    }
    // Deterministic vectorized exp: a fully-masked (−∞) key becomes exactly
    // 0.0, so padded bucket rows contribute nothing — bit for bit.
    crate::cpu::simd::simd_f32_exp_inplace(sc);
    let mut sum = 0f32;
    for &v in sc.iter() {
        sum += v;
    }
    let denom = sum.max(1e-30);
    let out = core::slice::from_raw_parts_mut(out_row, d);
    out.fill(0.0);
    for (kj, &v) in sc.iter().enumerate() {
        let p = v / denom;
        let vrow = if kj < past {
            core::slice::from_raw_parts(v_past.add(kj * d), d)
        } else {
            core::slice::from_raw_parts(v_new.add((kj - past) * d), d)
        };
        crate::cpu::simd::simd_f32_axpy(out, p, vrow);
    }
}

/// Fused decode attention (see [`DecodeAttentionCall`]): split KV read in
/// place, additive mask, `q_rows` decoupled from key length, GQA. f32
/// compute; f16/bf16 storage widens exactly as the legacy attention does.
pub fn decode_attention_float<W: Workspace>(
    c: &DecodeAttentionCall,
    ws: &mut W,
) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let h = c.heads as usize;
    let m = c.q_rows as usize;
    let past = c.past_len as usize;
    let new = c.new_len as usize;
    let d = c.head_dim as usize;
    let l = past + new;
    if b == 0 || h == 0 || m == 0 || d == 0 {
        return Ok(());
    }
    if l == 0 {
        return Err(BackendError::UnsupportedOp(
            "decode_attention: past_len + new_len must be at least 1",
        ));
    }
    let hkv = if c.kv_heads == 0 {
        h
    } else {
        c.kv_heads as usize
    };
    if hkv == 0 || !h.is_multiple_of(hkv) {
        return Err(BackendError::UnsupportedOp(
            "decode_attention: heads must be a multiple of kv_heads (grouped-query)",
        ));
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let q_total = b * h * m * d;
    let past_total = b * hkv * past * d;
    let new_total = b * hkv * new * d;
    let mask_total = m * l;
    let (reads, out) = ws
        .split_borrow(
            &[c.q, c.k_past, c.v_past, c.k_new, c.v_new, c.mask],
            c.output,
        )
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let q = reads[0]
        .get(..q_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.q.slot))?;
    let kp = reads[1]
        .get(..past_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.k_past.slot))?;
    let vp = reads[2]
        .get(..past_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.v_past.slot))?;
    let kn = reads[3]
        .get(..new_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.k_new.slot))?;
    let vn = reads[4]
        .get(..new_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.v_new.slot))?;
    // The mask is always f32, whatever the compute dtype.
    let mask = reads[5]
        .get(..mask_total * 4)
        .ok_or(BackendError::SlotOutOfRange(c.mask.slot))?;
    if out.len() < q_total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let scale = match c.scale_bits {
        0 => libm::sqrtf(d as f32).max(1.0),
        bits => {
            let sm = f32::from_bits(bits);
            if sm > 0.0 {
                1.0 / sm
            } else {
                libm::sqrtf(d as f32).max(1.0)
            }
        }
    };
    let mask32 = bytemuck::cast_slice::<u8, f32>(mask);

    if dt == DTYPE_F32 {
        let (q32, kp32, vp32, kn32, vn32, out32) = (
            bytemuck::cast_slice::<u8, f32>(q),
            bytemuck::cast_slice::<u8, f32>(kp),
            bytemuck::cast_slice::<u8, f32>(vp),
            bytemuck::cast_slice::<u8, f32>(kn),
            bytemuck::cast_slice::<u8, f32>(vn),
            bytemuck::cast_slice_mut::<u8, f32>(out),
        );
        decode_attention_f32_engine(
            q32, kp32, vp32, kn32, vn32, mask32, out32, b, h, hkv, m, past, new, d, scale,
        );
        return Ok(());
    }
    if dt != DTYPE_BF16 && dt != DTYPE_F16 {
        return Err(BackendError::UnsupportedOp(
            "decode_attention: only f32/f16/bf16 are supported compute dtypes",
        ));
    }
    with_widen4_scratch(|qb, kb, vb, ob| {
        // Widen Q and both KV segments; `kb`/`vb` hold past ∥ new back to back
        // so the engine's split pointers land inside one scratch each.
        qb.clear();
        qb.extend((0..q_total).map(|i| read_float(q, i, dt)));
        kb.clear();
        kb.extend((0..past_total).map(|i| read_float(kp, i, dt)));
        kb.extend((0..new_total).map(|i| read_float(kn, i, dt)));
        vb.clear();
        vb.extend((0..past_total).map(|i| read_float(vp, i, dt)));
        vb.extend((0..new_total).map(|i| read_float(vn, i, dt)));
        ob.clear();
        ob.resize(q_total, 0.0);
        let (kp32, kn32) = kb.split_at(past_total);
        let (vp32, vn32) = vb.split_at(past_total);
        decode_attention_f32_engine(
            qb, kp32, vp32, kn32, vn32, mask32, ob, b, h, hkv, m, past, new, d, scale,
        );
        for (i, &val) in ob.iter().enumerate() {
            write_float(out, i, val, dt);
        }
    });
    Ok(())
}

/// Serial engine: every `(batch, head, query-row)` through
/// [`decode_attention_row`] in order. The pooled paths run the same row fn
/// over partitioned row ranges, so serial and pooled are bit-identical.
#[allow(clippy::too_many_arguments)]
fn decode_attention_f32_engine(
    q32: &[f32],
    kp32: &[f32],
    vp32: &[f32],
    kn32: &[f32],
    vn32: &[f32],
    mask32: &[f32],
    out32: &mut [f32],
    b: usize,
    h: usize,
    hkv: usize,
    m: usize,
    past: usize,
    new: usize,
    d: usize,
    scale: f32,
) {
    let rows = b * h * m;
    let l = past + new;
    // wasm multi-core: fork-join the rows across the embedder pool. The
    // publisher (this thread) carries per-participant score scratch — workers
    // never allocate, per the pool's embedder contract.
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        feature = "wasm-threads"
    ))]
    {
        let parts = crate::cpu::wasm_pool::participants();
        if parts > 1 {
            let pooled = with_matmul_scratch(|scores| {
                scores.clear();
                scores.resize(parts * l, 0.0);
                crate::cpu::wasm_pool::fork_join_attn(crate::cpu::wasm_pool::AttnJob {
                    q: q32.as_ptr(),
                    k_past: kp32.as_ptr(),
                    v_past: vp32.as_ptr(),
                    k_new: kn32.as_ptr(),
                    v_new: vn32.as_ptr(),
                    mask: mask32.as_ptr(),
                    out: out32.as_mut_ptr(),
                    scores: scores.as_mut_ptr(),
                    b,
                    h,
                    hkv,
                    m,
                    past,
                    new,
                    d,
                    scale_bits: scale.to_bits(),
                })
            });
            if pooled {
                return;
            }
        }
    }
    // Native multi-core: disjoint row ranges across the pool, each task
    // running the same walker with its own thread-local scratch. Work-based
    // admission, as everywhere.
    #[cfg(all(
        feature = "parallel",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        use crate::cpu::parallel::{self, SendConst, SendMut};
        let w = parallel::pool().width();
        if w > 1
            && rows >= 2
            && (rows as u64) * (l as u64) * (d as u64) >= crate::cpu::simd::GEMV_PAR_THRESHOLD
        {
            let tiles = parallel::output_tiles(1, rows, w, 1);
            if tiles.len() > 1 {
                let (qp, kpp, vpp, knp, vnp, mp, op) = (
                    SendConst(q32.as_ptr()),
                    SendConst(kp32.as_ptr()),
                    SendConst(vp32.as_ptr()),
                    SendConst(kn32.as_ptr()),
                    SendConst(vn32.as_ptr()),
                    SendConst(mask32.as_ptr()),
                    SendMut(out32.as_mut_ptr()),
                );
                let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                    .into_iter()
                    .map(|(_, _, r0, nrows)| {
                        Box::new(move || {
                            let (qp, kpp, vpp, knp, vnp, mp, op) = (qp, kpp, vpp, knp, vnp, mp, op);
                            with_matmul_scratch(|scores| {
                                scores.clear();
                                scores.resize(l, 0.0);
                                // SAFETY: disjoint row ranges; scratch is this
                                // task's thread-local.
                                unsafe {
                                    decode_attention_tile_rows(
                                        qp.0,
                                        kpp.0,
                                        vpp.0,
                                        knp.0,
                                        vnp.0,
                                        mp.0,
                                        op.0,
                                        b,
                                        h,
                                        hkv,
                                        m,
                                        past,
                                        new,
                                        d,
                                        scale,
                                        r0,
                                        nrows,
                                        scores.as_mut_ptr(),
                                    );
                                }
                            });
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                parallel::pool().run(tasks);
                return;
            }
        }
    }
    with_matmul_scratch(|scores| {
        scores.clear();
        scores.resize(l, 0.0);
        // SAFETY: slice lengths validated by the caller; the full row range is
        // one exclusive borrow; scratch is this thread's.
        unsafe {
            decode_attention_tile_rows(
                q32.as_ptr(),
                kp32.as_ptr(),
                vp32.as_ptr(),
                kn32.as_ptr(),
                vn32.as_ptr(),
                mask32.as_ptr(),
                out32.as_mut_ptr(),
                b,
                h,
                hkv,
                m,
                past,
                new,
                d,
                scale,
                0,
                rows,
                scores.as_mut_ptr(),
            );
        }
    });
}

/// Test-only re-entry into the decode-attention engine with raw f32 slices —
/// lets the pool witness drive serial-vs-pooled through the exact production
/// dispatch (wasm fork-join / native tiles / serial) without a workspace.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_engine_for_tests(
    q: &[f32],
    kp: &[f32],
    vp: &[f32],
    kn: &[f32],
    vn: &[f32],
    mask: &[f32],
    out: &mut [f32],
    b: usize,
    h: usize,
    hkv: usize,
    m: usize,
    past: usize,
    new: usize,
    d: usize,
) {
    let scale = libm::sqrtf(d as f32).max(1.0);
    decode_attention_f32_engine(
        q, kp, vp, kn, vn, mask, out, b, h, hkv, m, past, new, d, scale,
    );
}

/// Pool executor: one participant's contiguous row range of the decode
/// attention — the same [`decode_attention_tile_rows`] walker the serial
/// engine runs, over a disjoint slice of rows, with this participant's own
/// stripe of the publisher-allocated score scratch. Bit-identical to serial
/// by construction.
///
/// # Safety
/// Called only from the `wasm_pool` fork-join: the job's buffers outlive the
/// join, participant row ranges are disjoint, and `scores` has
/// `participants · (past + new)` f32 capacity.
#[cfg(all(
    target_arch = "wasm32",
    feature = "wasm-threads",
    target_feature = "simd128"
))]
pub(crate) unsafe fn pool_exec_attn(
    job: &crate::cpu::wasm_pool::AttnJob,
    part: usize,
    parts: usize,
) {
    let rows = job.b * job.h * job.m;
    let start = part * rows / parts;
    let end = (part + 1) * rows / parts;
    if start >= end {
        return;
    }
    let l = job.past + job.new;
    decode_attention_tile_rows(
        job.q,
        job.k_past,
        job.v_past,
        job.k_new,
        job.v_new,
        job.mask,
        job.out,
        job.b,
        job.h,
        job.hkv,
        job.m,
        job.past,
        job.new,
        job.d,
        f32::from_bits(job.scale_bits),
        start,
        end - start,
        job.scores.add(part * l),
    );
}

/// A contiguous **global-row range** of the decode attention, through
/// [`decode_attention_row`] — the unit the serial engine, each native pool
/// task, and each wasm pool participant all run, which is what makes any
/// partition bit-identical to serial. Global row `r` decomposes as
/// `(batch, head, query-row) = (r / (h·m), (r / m) % h, r % m)`.
///
/// # Safety
/// Pointers address the full tensors (`q`/`out`: `b·h·m·d` f32; `k/v_past`:
/// `b·hkv·past·d`; `k/v_new`: `b·hkv·new·d`; `mask`: `m·(past+new)`), callers'
/// row ranges are disjoint, and `scores` addresses `past + new` f32 exclusive
/// to this caller.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_attention_tile_rows(
    q: *const f32,
    k_past: *const f32,
    v_past: *const f32,
    k_new: *const f32,
    v_new: *const f32,
    mask: *const f32,
    out: *mut f32,
    b: usize,
    h: usize,
    hkv: usize,
    m: usize,
    past: usize,
    new: usize,
    d: usize,
    scale: f32,
    row0: usize,
    rows: usize,
    scores: *mut f32,
) {
    let _ = b;
    let group = h / hkv;
    let l = past + new;
    for r in row0..row0 + rows {
        let bi = r / (h * m);
        let hi = (r / m) % h;
        let qi = r % m;
        let kvh = bi * hkv + hi / group;
        decode_attention_row(
            q.add(r * d),
            if past == 0 {
                k_past
            } else {
                k_past.add(kvh * past * d)
            },
            if past == 0 {
                v_past
            } else {
                v_past.add(kvh * past * d)
            },
            if new == 0 {
                k_new
            } else {
                k_new.add(kvh * new * d)
            },
            if new == 0 {
                v_new
            } else {
                v_new.add(kvh * new * d)
            },
            mask.add(qi * l),
            out.add(r * d),
            past,
            new,
            d,
            scale,
            scores,
        );
    }
}

pub fn where_float<W: Workspace>(c: &WhereCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let (reads, out) = ws
        .split_borrow(&[c.cond, c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let cond = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.cond.slot))?;
    let a = reads[1]
        .get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b = reads[2]
        .get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..n {
        let pick_a = cond.get(i).copied().unwrap_or(0) != 0;
        let v = if pick_a {
            read_float(a, i, dt)
        } else {
            read_float(b, i, dt)
        };
        write_float(out, i, v, dt);
    }
    Ok(())
}

pub fn layout_float<W: Workspace>(c: &LayoutCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, c.dtype)?;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < bytes {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // Zero-copy byte copy — `copy_from_slice` is one `memcpy`.
    out[..bytes].copy_from_slice(inp);
    Ok(())
}

/// Concat — the closed `PrimitiveOp::Concat` constructor (ADR-053): place
/// `a` then `b` into the output (`out = a ∥ b`). Concatenation is intrinsically
/// a constructor (it produces new content), so unlike the addressing-class
/// relabels it is a real placement; it is dtype-agnostic byte movement, so one
/// kernel serves both the float and byte domains. The output buffer is sized to
/// `a ∥ b` by the compiler from the concatenated output shape. n-ary concat is
/// expressed as a left-associated chain of this binary primitive.
pub fn concat_float<W: Workspace>(c: &BinaryCall, ws: &mut W) -> Result<(), BackendError> {
    let (reads, out) = ws
        .split_borrow(&[c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let (alen, blen) = (reads[0].len(), reads[1].len());
    if out.len() < alen + blen {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    out[..alen].copy_from_slice(reads[0]);
    out[alen..alen + blen].copy_from_slice(reads[1]);
    Ok(())
}

/// Transpose (axis permutation) — the irreducible re-indexing op. Output axis
/// `i` is input axis `perm[i]`; each output element is gathered from its
/// permuted input position. Dtype-agnostic element-wise byte copy (one kernel
/// for every dtype), up to rank 8.
pub fn transpose_float<W: Workspace>(c: &TransposeCall, ws: &mut W) -> Result<(), BackendError> {
    let rank = c.rank as usize;
    if rank == 0 || rank > MAX_RANK {
        return Err(BackendError::UnsupportedOp("transpose: rank must be 1..=8"));
    }
    let es = elem_size(c.dtype)?;
    let in_dims = &c.dims[..rank];
    let perm = &c.perm[..rank];
    let total: usize = in_dims.iter().map(|&d| d as usize).product();

    // Row-major input strides and permuted output extents.
    let mut in_strides = [0usize; MAX_RANK];
    let mut stride = 1usize;
    for i in (0..rank).rev() {
        in_strides[i] = stride;
        stride *= in_dims[i] as usize;
    }
    let mut out_dims = [1usize; MAX_RANK];
    for i in 0..rank {
        out_dims[i] = in_dims[perm[i] as usize] as usize;
    }

    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0];
    if inp.len() < total * es || out.len() < total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let mut coord = [0usize; MAX_RANK];
    for o in 0..total {
        // Decompose the output linear index into per-axis output coordinates.
        let mut rem = o;
        for i in (0..rank).rev() {
            coord[i] = rem % out_dims[i];
            rem /= out_dims[i];
        }
        // out coord i lives on input axis perm[i]; gather that input element.
        let mut in_idx = 0usize;
        for i in 0..rank {
            in_idx += coord[i] * in_strides[perm[i] as usize];
        }
        out[o * es..o * es + es].copy_from_slice(&inp[in_idx * es..in_idx * es + es]);
    }
    Ok(())
}

/// Resize (nearest-neighbor) — reuses `ExpandCall`'s in/out dims. Each output
/// coordinate maps to the nearest input coordinate `floor(o · in/out)` per
/// axis (ONNX `nearest`, `asymmetric`-style). Dtype-agnostic gather, rank ≤ 8.
pub fn resize_float<W: Workspace>(c: &ExpandCall, ws: &mut W) -> Result<(), BackendError> {
    let rank = c.rank as usize;
    if rank == 0 || rank > MAX_RANK {
        return Err(BackendError::UnsupportedOp("resize: rank must be 1..=8"));
    }
    let es = elem_size(c.dtype)?;
    let in_dims = &c.in_dims[..rank];
    let out_dims = &c.out_dims[..rank];
    let out_total: usize = out_dims.iter().map(|&d| d as usize).product();

    let mut in_strides = [0usize; MAX_RANK];
    let mut stride = 1usize;
    for i in (0..rank).rev() {
        in_strides[i] = stride;
        stride *= in_dims[i] as usize;
    }
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0];
    if out.len() < out_total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let mut coord = [0usize; MAX_RANK];
    for o in 0..out_total {
        let mut rem = o;
        for i in (0..rank).rev() {
            coord[i] = rem % out_dims[i] as usize;
            rem /= out_dims[i] as usize;
        }
        let mut in_idx = 0usize;
        for i in 0..rank {
            // Nearest source index: floor(out_coord · in_dim / out_dim).
            let src = coord[i] * in_dims[i] as usize / out_dims[i] as usize;
            in_idx += src.min(in_dims[i] as usize - 1) * in_strides[i];
        }
        out[o * es..o * es + es].copy_from_slice(&inp[in_idx * es..in_idx * es + es]);
    }
    Ok(())
}

/// LRN (local response normalization) over the channel axis of an
/// `[batch, channels, inner]` tensor (inner = H·W). For each element,
/// `out = x / (bias + (α/size)·Σ_{j∈window} x[j]²)^β`, the window spanning the
/// `size` channels centred on the element's channel (ONNX LRN).
pub fn lrn_float<W: Workspace>(c: &LrnCall, ws: &mut W) -> Result<(), BackendError> {
    let (b, ch, inner) = (c.batch as usize, c.channels as usize, c.inner as usize);
    if ch == 0 || c.size == 0 {
        return Err(BackendError::UnsupportedOp(
            "lrn: channels/size must be > 0",
        ));
    }
    let dt = c.dtype;
    let size = c.size as usize;
    let alpha = f32::from_bits(c.alpha_bits);
    let beta = f32::from_bits(c.beta_bits);
    let bias = f32::from_bits(c.bias_bits);
    let total = b * ch * inner;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let x = reads[0];
    if out.len() < total * elem_size(dt)? {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // Window [c - (size-1)/2, c + size/2] (ONNX), clamped to [0, ch).
    let lo_off = (size - 1) / 2;
    for n in 0..b {
        for cc in 0..ch {
            let c0 = cc.saturating_sub(lo_off);
            let c1 = (cc + size / 2 + 1).min(ch); // exclusive end
            for i in 0..inner {
                let mut sumsq = 0f32;
                for j in c0..c1 {
                    let v = read_float(x, (n * ch + j) * inner + i, dt);
                    sumsq += v * v;
                }
                // `libm::powf` (not `f32::powf`) so the kernel builds on the
                // no_std / bare-metal target, matching the rest of this file.
                let denom = libm::powf(bias + (alpha / size as f32) * sumsq, beta);
                let idx = (n * ch + cc) * inner + i;
                write_float(out, idx, read_float(x, idx, dt) / denom, dt);
            }
        }
    }
    Ok(())
}

/// RoPE (rotary positional embedding), rotate-half form. `cos`/`sin` are full
/// per-element tables (same layout as `x`). Within each head of width
/// `head_dim` (split into halves at `half = head_dim/2`): the first half maps
/// to `x·cos − x₂·sin`, the second to `x·cos + x₁·sin`, where `x₂`/`x₁` are the
/// paired elements across the half boundary. f32/bf16/f16 supported.
pub fn rope_float<W: Workspace>(c: &RoPECall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let d = c.head_dim as usize;
    if d == 0 || !d.is_multiple_of(2) || !n.is_multiple_of(d) {
        return Err(BackendError::UnsupportedOp(
            "rope: head_dim must be even and divide the element count",
        ));
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let half = d / 2;
    let (reads, out) = ws
        .split_borrow(&[c.x, c.cos, c.sin], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let (x, cos, sin) = (reads[0], reads[1], reads[2]);
    if out.len() < n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // f32 fast path: view the slots as `&[f32]` and walk head-by-head, splitting
    // the per-element `pos = e % d` / partner addressing into two contiguous
    // half-loops. Each half is a straight-line f32 map that autovectorizes; the
    // scalar `read_float` loop below (with its per-element `%`/`-`) stays the
    // bf16/f16/misaligned fallback. RoPE runs on Q and K every layer, every
    // decode token — this is decode-hot.
    if dt == DTYPE_F32 {
        if let (Ok(x32), Ok(c32), Ok(s32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(x),
            bytemuck::try_cast_slice::<u8, f32>(cos),
            bytemuck::try_cast_slice::<u8, f32>(sin),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..n * es]),
        ) {
            if c32.len() >= n && s32.len() >= n && x32.len() >= n {
                let heads = n / d;
                for h in 0..heads {
                    let base = h * d;
                    let xh = &x32[base..base + d];
                    let ch = &c32[base..base + d];
                    let sh = &s32[base..base + d];
                    let oh = &mut o32[base..base + d];
                    // Lower half: partner is the matching element in the upper half.
                    for j in 0..half {
                        oh[j] = xh[j] * ch[j] - xh[j + half] * sh[j];
                    }
                    // Upper half: partner is in the lower half.
                    for j in half..d {
                        oh[j] = xh[j] * ch[j] + xh[j - half] * sh[j];
                    }
                }
                return Ok(());
            }
        }
    }
    for e in 0..n {
        let pos = e % d;
        let base = e - pos;
        let xe = read_float(x, e, dt);
        let ce = read_float(cos, e, dt);
        let se = read_float(sin, e, dt);
        let v = if pos < half {
            // pair partner is the matching element in the upper half.
            xe * ce - read_float(x, base + pos + half, dt) * se
        } else {
            xe * ce + read_float(x, base + pos - half, dt) * se
        };
        write_float(out, e, v, dt);
    }
    Ok(())
}

/// Contiguous-inner-run kernel for the f32 [`broadcast_binary_float`] fast path.
/// Generic over the elementwise op `F` so each call site monomorphizes to a
/// concrete inner loop LLVM can vectorize (`op` always receives `(small, other)`
/// in that order; the caller bakes any operand swap into the closure). The
/// odometer over the outer axes is identical to the scalar path's.
#[inline]
#[allow(clippy::too_many_arguments)]
fn broadcast_binary_f32_run<F: Fn(f32, f32) -> f32>(
    s32: &[f32],
    ot32: &[f32],
    o32: &mut [f32],
    in_strides: &[usize; MAX_RANK],
    out_dims: &[u32],
    inner_len: usize,
    inner_stride: usize,
    num_rows: usize,
    op: F,
) {
    let rank = out_dims.len();
    let mut coord = [0usize; MAX_RANK];
    let mut sbase = 0usize;
    let mut o = 0usize;
    for _ in 0..num_rows {
        let orow = &mut o32[o..o + inner_len];
        let otrow = &ot32[o..o + inner_len];
        if inner_stride == 0 {
            // Last axis broadcasts: one small value across the whole run.
            let sv = s32[sbase];
            for (od, &ov) in orow.iter_mut().zip(otrow) {
                *od = op(sv, ov);
            }
        } else {
            // Last axis maps 1:1 (stride 1): small run is contiguous.
            let srow = &s32[sbase..sbase + inner_len];
            for ((od, &sv), &ov) in orow.iter_mut().zip(srow).zip(otrow) {
                *od = op(sv, ov);
            }
        }
        o += inner_len;
        let mut ax = rank - 1;
        while ax > 0 {
            ax -= 1;
            coord[ax] += 1;
            sbase += in_strides[ax];
            if coord[ax] < out_dims[ax] as usize {
                break;
            }
            sbase -= in_strides[ax] * out_dims[ax] as usize;
            coord[ax] = 0;
        }
    }
}

/// Fused `Expand → elementwise-binary`: `out[o] = op(small[bcast(o)], other[o])`
/// (operands swapped when `!small_is_lhs`). The `small` (pre-Expand) operand is
/// read with stride-0 broadcast indexing in place, so the broadcasted tensor is
/// **never materialized** — the zero-movement realization of Expand for its
/// dominant consumer (bias/scale broadcast, the norm-VJP `Expand → Mul`).
pub fn broadcast_binary_float<W: Workspace>(
    c: &BroadcastBinaryCall,
    ws: &mut W,
) -> Result<(), BackendError> {
    use crate::kernel_call::broadcast_op;
    let rank = c.rank as usize;
    if rank == 0 || rank > MAX_RANK {
        return Err(BackendError::UnsupportedOp(
            "broadcast_binary: rank must be 1..=8",
        ));
    }
    let dt = c.dtype;
    let es = elem_size(dt)?;
    let in_dims = &c.in_dims[..rank];
    let out_dims = &c.out_dims[..rank];
    let out_total: usize = out_dims.iter().map(|&d| d as usize).product();
    let small_total: usize = in_dims.iter().map(|&d| d as usize).product();
    // Row-major strides over the small operand; a broadcast axis (in_dim == 1)
    // contributes stride 0, so it re-reads index 0 along that axis.
    let mut in_strides = [0usize; MAX_RANK];
    let mut stride = 1usize;
    for i in (0..rank).rev() {
        in_strides[i] = if in_dims[i] == 1 { 0 } else { stride };
        stride *= in_dims[i] as usize;
    }
    let (reads, out) = ws
        .split_borrow(&[c.small, c.other], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let small = reads[0]
        .get(..small_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.small.slot))?;
    let other = reads[1]
        .get(..out_total * es)
        .ok_or(BackendError::SlotOutOfRange(c.other.slot))?;
    if out.len() < out_total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let op: fn(f32, f32) -> f32 = match c.op {
        broadcast_op::ADD => |a, b| a + b,
        broadcast_op::SUB => |a, b| a - b,
        broadcast_op::MUL => |a, b| a * b,
        _ => return Err(BackendError::UnsupportedOp("broadcast_binary: bad op")),
    };
    // Walk the output in contiguous runs along the last axis, advancing the
    // small-operand base index with an incremental odometer over the outer
    // axes — no per-element div/mod. The inner run's small stride is 0 (the
    // last axis broadcasts: `small[sbase]` constant) or 1 (it maps 1:1:
    // `small[sbase..]` contiguous), so the inner loop autovectorizes.
    let inner_len = out_dims[rank - 1] as usize;
    let inner_stride = in_strides[rank - 1];
    if inner_len == 0 {
        return Ok(());
    }
    let num_rows = out_total / inner_len;
    // f32 fast path: view the three slots as `&[f32]` and run the contiguous
    // inner op with a **concrete** monomorphized closure (an indirect `fn`
    // pointer defeats the inner-loop vectorization the odometer sets up). The
    // odometer geometry is identical to the scalar path below; only the leaf
    // element access changes from `read_float`/`write_float` to a direct load.
    if dt == DTYPE_F32 {
        if let (Ok(s32), Ok(ot32), Ok(o32)) = (
            bytemuck::try_cast_slice::<u8, f32>(small),
            bytemuck::try_cast_slice::<u8, f32>(other),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..out_total * es]),
        ) {
            match (c.op, c.small_is_lhs) {
                (broadcast_op::ADD, _) => broadcast_binary_f32_run(
                    s32,
                    ot32,
                    o32,
                    &in_strides,
                    out_dims,
                    inner_len,
                    inner_stride,
                    num_rows,
                    |s, o| s + o,
                ),
                (broadcast_op::MUL, _) => broadcast_binary_f32_run(
                    s32,
                    ot32,
                    o32,
                    &in_strides,
                    out_dims,
                    inner_len,
                    inner_stride,
                    num_rows,
                    |s, o| s * o,
                ),
                (broadcast_op::SUB, true) => broadcast_binary_f32_run(
                    s32,
                    ot32,
                    o32,
                    &in_strides,
                    out_dims,
                    inner_len,
                    inner_stride,
                    num_rows,
                    |s, o| s - o,
                ),
                (broadcast_op::SUB, false) => broadcast_binary_f32_run(
                    s32,
                    ot32,
                    o32,
                    &in_strides,
                    out_dims,
                    inner_len,
                    inner_stride,
                    num_rows,
                    |s, o| o - s,
                ),
                _ => return Err(BackendError::UnsupportedOp("broadcast_binary: bad op")),
            }
            return Ok(());
        }
    }
    let mut coord = [0usize; MAX_RANK];
    let mut sbase = 0usize;
    let mut o = 0usize;
    for _ in 0..num_rows {
        if c.small_is_lhs {
            for j in 0..inner_len {
                let sv = read_float(small, sbase + j * inner_stride, dt);
                let ov = read_float(other, o + j, dt);
                write_float(out, o + j, op(sv, ov), dt);
            }
        } else {
            for j in 0..inner_len {
                let sv = read_float(small, sbase + j * inner_stride, dt);
                let ov = read_float(other, o + j, dt);
                write_float(out, o + j, op(ov, sv), dt);
            }
        }
        o += inner_len;
        // Advance the odometer over the outer axes (last outer axis fastest).
        let mut ax = rank - 1;
        while ax > 0 {
            ax -= 1;
            coord[ax] += 1;
            sbase += in_strides[ax];
            if coord[ax] < out_dims[ax] as usize {
                break;
            }
            sbase -= in_strides[ax] * out_dims[ax] as usize;
            coord[ax] = 0;
        }
    }
    Ok(())
}

/// Expand (broadcast): replicate `input` to `out_dims`. An axis with
/// `in_dims[i] == 1` reads input index 0 (broadcast); every other axis maps
/// 1:1. Dtype-agnostic gather. When the sole consumer is an elementwise
/// `{Add,Sub,Mul}`, the runtime fuses this into [`broadcast_binary_float`] so
/// the broadcast is never materialized; this materializing path covers the
/// remaining cases (e.g. Expand feeding a matmul/concat). Rank ≤ 8.
pub fn expand_float<W: Workspace>(c: &ExpandCall, ws: &mut W) -> Result<(), BackendError> {
    let rank = c.rank as usize;
    if rank == 0 || rank > MAX_RANK {
        return Err(BackendError::UnsupportedOp("expand: rank must be 1..=8"));
    }
    let es = elem_size(c.dtype)?;
    let in_dims = &c.in_dims[..rank];
    let out_dims = &c.out_dims[..rank];
    let out_total: usize = out_dims.iter().map(|&d| d as usize).product();

    // Row-major input strides (a broadcast axis contributes stride 0).
    let mut in_strides = [0usize; MAX_RANK];
    let mut stride = 1usize;
    for i in (0..rank).rev() {
        in_strides[i] = if in_dims[i] == 1 { 0 } else { stride };
        stride *= in_dims[i] as usize;
    }

    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0];
    if out.len() < out_total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let mut coord = [0usize; MAX_RANK];
    for o in 0..out_total {
        let mut rem = o;
        for i in (0..rank).rev() {
            coord[i] = rem % out_dims[i] as usize;
            rem /= out_dims[i] as usize;
        }
        let mut in_idx = 0usize;
        for i in 0..rank {
            in_idx += coord[i] * in_strides[i];
        }
        out[o * es..o * es + es].copy_from_slice(&inp[in_idx * es..in_idx * es + es]);
    }
    Ok(())
}

// Elementwise float impls.
#[inline]
pub fn relu_f(x: f32) -> f32 {
    x.max(0.0)
}
#[inline]
pub fn neg_f(x: f32) -> f32 {
    -x
}
#[inline]
pub fn abs_f(x: f32) -> f32 {
    x.abs()
}
#[inline]
pub fn sign_f(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
#[inline]
pub fn is_nan_f(x: f32) -> f32 {
    if x.is_nan() {
        1.0
    } else {
        0.0
    }
}
#[inline]
pub fn ceil_f(x: f32) -> f32 {
    libm::ceilf(x)
}
#[inline]
pub fn floor_f(x: f32) -> f32 {
    libm::floorf(x)
}
#[inline]
pub fn round_f(x: f32) -> f32 {
    libm::roundf(x)
}
#[inline]
pub fn sqrt_f(x: f32) -> f32 {
    libm::sqrtf(x)
}
#[inline]
pub fn recip_f(x: f32) -> f32 {
    1.0 / x
}
#[inline]
pub fn exp_f(x: f32) -> f32 {
    libm::expf(x)
}
#[inline]
pub fn log_f(x: f32) -> f32 {
    libm::logf(x.max(1e-30))
}
#[inline]
pub fn log1p_f(x: f32) -> f32 {
    libm::log1pf(x)
}
#[inline]
pub fn sin_f(x: f32) -> f32 {
    libm::sinf(x)
}
#[inline]
pub fn cos_f(x: f32) -> f32 {
    libm::cosf(x)
}
#[inline]
pub fn tan_f(x: f32) -> f32 {
    libm::tanf(x)
}
#[inline]
pub fn asin_f(x: f32) -> f32 {
    libm::asinf(x.clamp(-1.0, 1.0))
}
#[inline]
pub fn acos_f(x: f32) -> f32 {
    libm::acosf(x.clamp(-1.0, 1.0))
}
#[inline]
pub fn atan_f(x: f32) -> f32 {
    libm::atanf(x)
}
#[inline]
pub fn erf_f(x: f32) -> f32 {
    libm::erff(x)
}
// The four transcendental activations below are expressed through the
// deterministic `exp_f32_det` rather than libm. Two reasons: (1) libm's
// `expf`/`tanhf` are platform-divergent, so a content-addressed system saw
// different activation bits per target — routing through `exp_f32_det` (≤2e-6
// rel err vs libm, and bit-identical across scalar/AVX2/AVX-512/NEON/wasm)
// removes that latent divergence; (2) it lets the f32 whole-slice path
// (`*_slice` below) be **bit-identical** to this scalar reference, so the SSOT
// contract with the f16/bf16 LUT and the `DequantActivation` densified table
// (both built from `lut_act_ref`) is preserved while the f32 path vectorizes.
// Keep each scalar body and its `*_slice` twin in lock-step: identical op
// order, no FMA contraction.
#[inline]
pub fn sigmoid_f(x: f32) -> f32 {
    1.0 / (1.0 + crate::cpu::simd::exp_f32_det(-x))
}
#[inline]
pub fn tanh_f(x: f32) -> f32 {
    // tanh(x) = (e^{2x} − 1)/(e^{2x} + 1); the clamped `exp_f32_det` saturates
    // to ±1 at the tails exactly as libm tanh does.
    let e = crate::cpu::simd::exp_f32_det(2.0 * x);
    (e - 1.0) / (e + 1.0)
}
#[inline]
pub fn gelu_f(x: f32) -> f32 {
    0.5 * x * (1.0 + tanh_f(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}
#[inline]
pub fn silu_f(x: f32) -> f32 {
    x * sigmoid_f(x)
}

/// Whole-slice `sigmoid` (`out[i] = 1/(1+e^{-x})`), bit-identical to
/// per-element [`sigmoid_f`]: write `−x`, run the vectorized deterministic exp,
/// then the (autovectorized) `1/(1+e)` map.
pub fn sigmoid_slice(inp: &[f32], out: &mut [f32]) {
    let n = inp.len().min(out.len());
    for (o, &x) in out[..n].iter_mut().zip(&inp[..n]) {
        *o = -x;
    }
    crate::cpu::simd::simd_f32_exp_inplace(&mut out[..n]);
    for o in out[..n].iter_mut() {
        *o = 1.0 / (1.0 + *o);
    }
}

/// Whole-slice `silu` (`x·sigmoid(x)`), bit-identical to [`silu_f`].
pub fn silu_slice(inp: &[f32], out: &mut [f32]) {
    let n = inp.len().min(out.len());
    for (o, &x) in out[..n].iter_mut().zip(&inp[..n]) {
        *o = -x;
    }
    crate::cpu::simd::simd_f32_exp_inplace(&mut out[..n]);
    for (o, &x) in out[..n].iter_mut().zip(&inp[..n]) {
        *o = x * (1.0 / (1.0 + *o));
    }
}

/// Whole-slice `tanh`, bit-identical to [`tanh_f`].
pub fn tanh_slice(inp: &[f32], out: &mut [f32]) {
    let n = inp.len().min(out.len());
    for (o, &x) in out[..n].iter_mut().zip(&inp[..n]) {
        *o = 2.0 * x;
    }
    crate::cpu::simd::simd_f32_exp_inplace(&mut out[..n]);
    for o in out[..n].iter_mut() {
        *o = (*o - 1.0) / (*o + 1.0);
    }
}

/// Whole-slice `gelu` (tanh approximation), bit-identical to [`gelu_f`]. The
/// exp argument is `2·(0.7978846·(x + 0.044715·x³))` — the exact expansion of
/// `tanh_f`'s `2.0 * u` on `u =` gelu's inner term.
pub fn gelu_slice(inp: &[f32], out: &mut [f32]) {
    let n = inp.len().min(out.len());
    for (o, &x) in out[..n].iter_mut().zip(&inp[..n]) {
        let u = 0.797_884_6 * (x + 0.044_715 * x * x * x);
        *o = 2.0 * u;
    }
    crate::cpu::simd::simd_f32_exp_inplace(&mut out[..n]);
    for (o, &x) in out[..n].iter_mut().zip(&inp[..n]) {
        let tanh_u = (*o - 1.0) / (*o + 1.0);
        *o = 0.5 * x * (1.0 + tanh_u);
    }
}

/// The reference `f32 → f32` activation for a `lut_act::*` id. The single source
/// of truth for the transcendental activations that have a dense finite-domain
/// table form — both the f16/bf16 LUT (`cpu::lut`) and the quantized-domain
/// densification (`DequantActivation`) build their tables from this, so all
/// three paths are bit-identical by construction.
#[inline]
pub fn lut_act_ref(act: u8) -> fn(f32) -> f32 {
    use crate::kernel_call::lut_act;
    match act {
        lut_act::SIGMOID => sigmoid_f,
        lut_act::TANH => tanh_f,
        lut_act::GELU => gelu_f,
        lut_act::SILU => silu_f,
        lut_act::EXP => exp_f,
        _ => erf_f,
    }
}

#[inline]
pub fn elu_f(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        libm::expf(x) - 1.0
    }
}
#[inline]
pub fn selu_f(x: f32) -> f32 {
    let alpha = 1.673_263_2_f32;
    let scale = 1.050_701_f32;
    if x >= 0.0 {
        scale * x
    } else {
        scale * alpha * (libm::expf(x) - 1.0)
    }
}

#[inline]
pub fn add_f(a: f32, b: f32) -> f32 {
    a + b
}
#[inline]
pub fn sub_f(a: f32, b: f32) -> f32 {
    a - b
}
#[inline]
pub fn mul_f(a: f32, b: f32) -> f32 {
    a * b
}
#[inline]
pub fn div_f(a: f32, b: f32) -> f32 {
    // IEEE-754 native: x/0 → ±∞, 0/0 → NaN. No silent 0.0 substitution.
    a / b
}
#[inline]
pub fn pow_f(a: f32, b: f32) -> f32 {
    libm::powf(a, b)
}
#[inline]
pub fn mod_f(a: f32, b: f32) -> f32 {
    // Floored modulo (sign follows divisor, ONNX Mod fmod=0). b==0 → NaN
    // naturally (a/0=±∞, floor(±∞)·0=NaN), per IEEE — no silent 0.0.
    a - (a / b).floor() * b
}
#[inline]
pub fn min_f(a: f32, b: f32) -> f32 {
    a.min(b)
}
#[inline]
pub fn max_f(a: f32, b: f32) -> f32 {
    a.max(b)
}
#[inline]
pub fn equal_f(a: f32, b: f32) -> f32 {
    if a == b {
        1.0
    } else {
        0.0
    }
}
#[inline]
pub fn less_f(a: f32, b: f32) -> f32 {
    if a < b {
        1.0
    } else {
        0.0
    }
}
#[inline]
pub fn less_or_equal_f(a: f32, b: f32) -> f32 {
    if a <= b {
        1.0
    } else {
        0.0
    }
}
#[inline]
pub fn greater_f(a: f32, b: f32) -> f32 {
    if a > b {
        1.0
    } else {
        0.0
    }
}
#[inline]
pub fn greater_or_equal_f(a: f32, b: f32) -> f32 {
    if a >= b {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod scalar_quant_tests {
    use super::*;

    /// The scalar (W8A32) dequant loop must **reject** a tier it cannot decode
    /// rather than read it as zeros. `e8cb` weights are codebook indices: read
    /// as raw bytes they are meaningless, and the previous `_ => 0` arm turned
    /// an unhandled tier into a silently zero-filled result.
    #[test]
    fn scalar_quant_rejects_tiers_it_cannot_decode() {
        assert!(ScalarQuant::from_tag(DTYPE_I8).is_ok());
        assert!(ScalarQuant::from_tag(DTYPE_U8).is_ok());
        assert!(ScalarQuant::from_tag(DTYPE_I4).is_ok());
        // Needs a codebook operand + the fused omajor path.
        assert!(ScalarQuant::from_tag(DTYPE_E8CB).is_err());
        // Float and unknown tags are not weight encodings at all.
        assert!(ScalarQuant::from_tag(DTYPE_F32).is_err());
        assert!(ScalarQuant::from_tag(200).is_err());
    }

    /// The validated reader is total and matches the encodings bit-for-bit.
    #[test]
    fn scalar_quant_read_matches_the_encodings() {
        let bytes: alloc::vec::Vec<u8> = alloc::vec![0, 1, 127, 128, 255];
        let sq = ScalarQuant::from_tag(DTYPE_I8).unwrap();
        for (i, w) in [0i32, 1, 127, -128, -1].iter().enumerate() {
            assert_eq!(sq.read(&bytes, i), *w, "i8 elem {i}");
        }
        let squ = ScalarQuant::from_tag(DTYPE_U8).unwrap();
        for (i, b) in bytes.iter().enumerate() {
            assert_eq!(squ.read(&bytes, i), *b as i32, "u8 elem {i}");
        }
        // i4: low nibble is element 2k, high nibble 2k+1; each sign-extended.
        let packed: alloc::vec::Vec<u8> = alloc::vec![0xE1, 0x0F];
        let sq4 = ScalarQuant::from_tag(DTYPE_I4).unwrap();
        assert_eq!(sq4.read(&packed, 0), 1); // 0x1
        assert_eq!(sq4.read(&packed, 1), -2); // 0xE = 14 - 16
        assert_eq!(sq4.read(&packed, 2), -1); // 0xF = 15 - 16
        assert_eq!(sq4.read(&packed, 3), 0); // 0x0
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    /// The whole-slice activation twins MUST be bit-identical to the scalar
    /// SSOT (`sigmoid_f`/`silu_f`/`tanh_f`/`gelu_f`) — the f16/bf16 LUT and the
    /// `DequantActivation` densified table are built from the scalar form, so
    /// any divergence would silently break the "bit-identical by construction"
    /// contract those paths rely on. Non-fused, no reassociation, so this is
    /// exact equality, not an epsilon check.
    #[test]
    fn slice_activations_bit_identical_to_scalar() {
        // Span the full working range incl. the exp clamp tails and 0.
        let xs: Vec<f32> = (-1000..=1000).map(|i| i as f32 * 0.1).collect();
        type Case = (fn(&[f32], &mut [f32]), fn(f32) -> f32, &'static str);
        let cases: &[Case] = &[
            (sigmoid_slice, sigmoid_f, "sigmoid"),
            (silu_slice, silu_f, "silu"),
            (tanh_slice, tanh_f, "tanh"),
            (gelu_slice, gelu_f, "gelu"),
        ];
        for (slice_fn, scalar_fn, name) in cases {
            let mut got = vec![0f32; xs.len()];
            slice_fn(&xs, &mut got);
            for (i, (&g, &x)) in got.iter().zip(&xs).enumerate() {
                let want = scalar_fn(x);
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "{name} mismatch at x={x} (idx {i}): slice {g} vs scalar {want}"
                );
            }
        }
    }
}
