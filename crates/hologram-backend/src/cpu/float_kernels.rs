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

// Reused byte buffer for marshalling matmul operands into the prism
// `TensorAxis::matmul` single-input contract (`[m,k,n] || A || B`). Same
// thread-local-vs-fresh story as `with_matmul_scratch`, so routing matmul
// through the prism axis surface costs no per-call allocation under `std`.
#[cfg(feature = "std")]
fn with_matmul_input_scratch<R>(f: impl FnOnce(&mut Vec<u8>) -> R) -> R {
    std::thread_local! {
        static MATMUL_IN_SCRATCH: core::cell::RefCell<Vec<u8>> =
            const { core::cell::RefCell::new(Vec::new()) };
    }
    MATMUL_IN_SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(not(feature = "std"))]
fn with_matmul_input_scratch<R>(f: impl FnOnce(&mut Vec<u8>) -> R) -> R {
    let mut scratch = Vec::new();
    f(&mut scratch)
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

pub fn binary_float<W: Workspace>(
    c: &BinaryCall,
    ws: &mut W,
    f: fn(f32, f32) -> f32,
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
            for i in 0..n {
                o32[i] = f(a32[i], b32[i]);
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
    let b = reads[1]
        .get(..k * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < m * n * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }

    if dt == DTYPE_F32 {
        // Dispatch through the prism `TensorAxis` surface (wiki ADR-031):
        // marshal `[m,k,n] || A || B` into the reused scratch and invoke the
        // axis kernel. The marshalling is Θ(m·k + k·n) against the kernel's
        // Θ(m·k·n) compute — negligible at model scale.
        use prism::tensor::TensorAxis;
        let mn = m * n * 4;
        let res = with_matmul_input_scratch(|inp| {
            inp.clear();
            inp.reserve(crate::prism_axes::HOLOGRAM_MATMUL_HEADER_BYTES + a.len() + b.len());
            inp.extend_from_slice(&(m as u32).to_le_bytes());
            inp.extend_from_slice(&(k as u32).to_le_bytes());
            inp.extend_from_slice(&(n as u32).to_le_bytes());
            inp.extend_from_slice(a);
            inp.extend_from_slice(b);
            crate::prism_axes::HologramTensorMatmulF32::matmul(inp, &mut out[..mn])
        });
        return res
            .map(|_| ())
            .map_err(|_| BackendError::Dispatch("matmul axis"));
    }

    // Non-f32 dtypes (bf16, f16, f64): per-element codec. The split-
    // borrow still gives zero-copy `&[u8]` views.
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                let av = read_float(a, i * k + kk, dt);
                let bv = read_float(b, kk * n + j, dt);
                acc += av * bv;
            }
            write_float(out, i * n + j, acc, dt);
        }
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

    if dt == DTYPE_F32 {
        if let (Ok(a32), Ok(b32), Ok(c32), Ok(out32)) = (
            bytemuck::try_cast_slice::<u8, f32>(a),
            bytemuck::try_cast_slice::<u8, f32>(b),
            bytemuck::try_cast_slice::<u8, f32>(cc),
            bytemuck::try_cast_slice_mut::<u8, f32>(&mut out[..m * n * 4]),
        ) {
            with_matmul_scratch(|bt| {
                crate::cpu::simd::matmul_f32_blocked(a32, b32, out32, m, k, n, bt);
            });
            for i in 0..m * n {
                out32[i] = alpha * out32[i] + beta * c32[i];
            }
            return Ok(());
        }
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += read_float(a, i * k + kk, dt) * read_float(b, kk * n + j, dt);
            }
            let bias = read_float(cc, i * n + j, dt) * beta;
            write_float(out, i * n + j, alpha * acc + bias, dt);
        }
    }
    Ok(())
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
    for bi in 0..b {
        for co in 0..cout {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0f32;
                    for ci in 0..cin {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * s_h + kh;
                                let iw = ow * s_w + kw;
                                if ih < h_in && iw < w_in {
                                    let xi = ((bi * cin + ci) * h_in + ih) * w_in + iw;
                                    let wi = ((co * cin + ci) * k_h + kh) * k_w + kw;
                                    acc += read_float(xs, xi, dt) * read_float(ws_w, wi, dt);
                                }
                            }
                        }
                    }
                    let oi = ((bi * cout + co) * h_out + oh) * w_out + ow;
                    write_float(out, oi, acc, dt);
                }
            }
        }
    }
    Ok(())
}

pub fn layer_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
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
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let mut acc = init;
    for i in 0..n {
        acc = f(acc, read_float(xs, i, dt));
    }
    if mean {
        acc /= n as f32;
    }
    write_float(out, 0, acc, dt);
    for o in out.iter_mut().skip(es) {
        *o = 0;
    }
    Ok(())
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
    let dt = c.dtype;
    let es = elem_size(dt);
    let total = b * h * s * d;
    let (reads, out) = ws
        .split_borrow(&[c.q, c.k, c.v], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let q = reads[0]
        .get(..total * es)
        .ok_or(BackendError::SlotOutOfRange(c.q.slot))?;
    let kk = reads[1]
        .get(..total * es)
        .ok_or(BackendError::SlotOutOfRange(c.k.slot))?;
    let v = reads[2]
        .get(..total * es)
        .ok_or(BackendError::SlotOutOfRange(c.v.slot))?;
    if out.len() < total * es {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let scale = libm::sqrtf(d as f32).max(1.0);

    // Per-row score buffer reused across all (b, h, q) iterations.
    with_matmul_scratch(|scores| {
        for bi in 0..b {
            for hi in 0..h {
                let head_off = (bi * h + hi) * s * d;
                for qi in 0..s {
                    scores.clear();
                    scores.resize(s, 0.0);
                    for (kj, score) in scores.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for di in 0..d {
                            acc += read_float(q, head_off + qi * d + di, dt)
                                * read_float(kk, head_off + kj * d + di, dt);
                        }
                        *score = acc / scale;
                    }
                    let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = libm::expf(*sc - max_s);
                        sum += *sc;
                    }
                    let denom = sum.max(1e-30);
                    for di in 0..d {
                        let mut acc = 0f32;
                        for (kj, &sc) in scores.iter().enumerate() {
                            acc += (sc / denom) * read_float(v, head_off + kj * d + di, dt);
                        }
                        write_float(out, head_off + qi * d + di, acc, dt);
                    }
                }
            }
        }
    });
    Ok(())
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
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}
#[inline]
pub fn pow_f(a: f32, b: f32) -> f32 {
    libm::powf(a, b)
}
#[inline]
pub fn mod_f(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        0.0
    } else {
        a - (a / b).floor() * b
    }
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
