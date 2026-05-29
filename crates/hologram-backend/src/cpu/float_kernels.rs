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

#[inline]
fn elem_size(dtype: u8) -> usize {
    bytes_per_element(dtype)
}

#[inline]
fn elem_count_to_bytes(n: usize, dtype: u8) -> usize {
    n * elem_size(dtype)
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
    MATMUL_BT_SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
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
    CONV_IM2COL_SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
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
    WIDEN.with(|cell| {
        let mut g = cell.borrow_mut();
        let (a, b, o) = &mut *g;
        f(a, b, o)
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
    WIDEN4.with(|cell| {
        let mut g = cell.borrow_mut();
        let (a, b, c, dd) = &mut *g;
        f(a, b, c, dd)
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
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, dtype);
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
            for i in 0..n {
                o32s[i] = f(i32s[i]);
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
    let bytes = elem_count_to_bytes(n, dtype);
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
    let es = elem_size(dt);

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

    // bf16 / f16: widen operands to f32, run the shared cache-oblivious engine
    // (f32 accumulation — identical to the old scalar path's `acc: f32`), then
    // narrow the result. Shares the optimized kernel instead of a strided
    // scalar triple-loop.
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
    let in_bytes = match c.quant_dtype {
        DTYPE_I4 => kn.div_ceil(2),
        DTYPE_I8 | DTYPE_U8 => kn,
        _ => {
            return Err(BackendError::UnsupportedOp(
                "matmul_dequant: quant_dtype must be i8/u8/i4",
            ))
        }
    };
    let per_ch = c.per_channel();
    let reads_spec: &[crate::workspace::BufferRef] = if per_ch {
        &[c.a, c.bq, c.scales, c.zero_points]
    } else {
        &[c.a, c.bq]
    };
    let (reads, out) = ws
        .split_borrow(reads_spec, c.output)
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
    // `bdq` is a reused thread-local (zero alloc per call after warm-up) holding
    // the dequantized B panel. A/out are workspace slots — 64-byte aligned by
    // construction — so the f32 views always succeed; an unaligned operand is a
    // contract violation and fails loud (no scalar/copy fallback), matching
    // `matmul_float`.
    with_widen_scratch(|_a_unused, bdq, _o_unused| {
        bdq.clear();
        bdq.resize(kn, 0.0);
        for (i, slot) in bdq.iter_mut().enumerate() {
            let q: i32 = match quant_dtype {
                DTYPE_I8 => (bq[i] as i8) as i32,
                DTYPE_U8 => bq[i] as i32,
                DTYPE_I4 => {
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
                _ => 0,
            };
            let (s, z) = if per_ch {
                let ch = (i / inner) % channels;
                (
                    f32::from_le_bytes([
                        scales[ch * 4],
                        scales[ch * 4 + 1],
                        scales[ch * 4 + 2],
                        scales[ch * 4 + 3],
                    ]),
                    i32::from_le_bytes([
                        zps[ch * 4],
                        zps[ch * 4 + 1],
                        zps[ch * 4 + 2],
                        zps[ch * 4 + 3],
                    ]),
                )
            } else {
                (scale, zp)
            };
            *slot = (q - z) as f32 * s;
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
    let es = elem_size(dt);
    let f = fused_act_fn(c.act);
    let out = ws.write(c.mm.output);
    if out.len() < count * es {
        return Err(BackendError::SlotOutOfRange(c.mm.output.slot));
    }
    if dt == DTYPE_F32 {
        if let Ok(o32s) = bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..count * 4]) {
            for v in o32s.iter_mut() {
                *v = f(*v);
            }
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
            for (o, &r) in o32s.iter_mut().zip(r32s.iter()) {
                *o = f(*o + r);
            }
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
    let total = bsz * f * es;
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);

    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma, c.beta], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = reads[1].get(..f * es).unwrap_or(&[]);
    let beta = reads[2].get(..f * es).unwrap_or(&[]);
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
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
    let es = elem_size(dt);
    let total = n * f * es;
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);
    let spatial = f / ch; // elements per channel
    let group_size = f / g; // elements per normalization group

    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma, c.beta], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = reads[1].get(..ch * es).unwrap_or(&[]);
    let beta = reads[2].get(..ch * es).unwrap_or(&[]);
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
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
    let es = elem_size(dt);
    let total = bsz * f * es;
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);
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
    let gamma = reads[gamma_idx].get(..f * es).unwrap_or(&[]);
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
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
    let es = elem_size(dt);
    let total = bsz * f * es;
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);
    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = reads[1].get(..f * es).unwrap_or(&[]);
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
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
    let es = elem_size(dt);
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
            exps.clear();
            exps.reserve(f);
            let mut sum = 0f32;
            for j in 0..f {
                let e = libm::expf(read_float(xs, row_off + j, dt) - max_v);
                sum += e;
                exps.push(e);
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

pub fn reduce_float<W: Workspace>(
    c: &ReduceCall,
    ws: &mut W,
    f: fn(f32, f32) -> f32,
    init: f32,
    mean: bool,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 {
        return Ok(());
    }
    let dt = c.dtype;
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
    let es = elem_size(dt);
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
                    let mut sum = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = libm::expf(*sc - max_s);
                        sum += *sc;
                    }
                    let denom = sum.max(1e-30);
                    let orow = &mut out32[q_off + qi * d..q_off + qi * d + d];
                    orow.fill(0.0);
                    for (kj, &sc) in scores.iter().enumerate() {
                        let p = sc / denom;
                        let vrow = &v32[kv_off + kj * d..kv_off + kj * d + d];
                        for (o, &vv) in orow.iter_mut().zip(vrow) {
                            *o += p * vv;
                        }
                    }
                }
            }
        }
    });
}

pub fn where_float<W: Workspace>(c: &WhereCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let dt = c.dtype;
    let es = elem_size(dt);
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
    let bytes = elem_count_to_bytes(n, c.dtype);
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
    let es = elem_size(c.dtype);
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
    let es = elem_size(c.dtype);
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
    if out.len() < total * elem_size(dt) {
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
    let es = elem_size(dt);
    let half = d / 2;
    let (reads, out) = ws
        .split_borrow(&[c.x, c.cos, c.sin], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let (x, cos, sin) = (reads[0], reads[1], reads[2]);
    if out.len() < n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
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
    let es = elem_size(dt);
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
    let es = elem_size(c.dtype);
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
#[inline]
pub fn sigmoid_f(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}
#[inline]
pub fn tanh_f(x: f32) -> f32 {
    libm::tanhf(x)
}
#[inline]
pub fn gelu_f(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::tanhf(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}
#[inline]
pub fn silu_f(x: f32) -> f32 {
    x * sigmoid_f(x)
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
