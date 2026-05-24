//! Native IEEE-754 CPU kernels (f32 / bf16 / f16).
//!
//! Selected when the KernelCall's `dtype` tag indicates a float dtype.
//! Mirrors the byte-domain kernels in semantics but at native precision.

use crate::kernel_call::*;
use crate::workspace::Workspace;
use crate::error::BackendError;
use crate::cpu::dtype::*;

#[inline]
fn elem_size(dtype: u8) -> usize { bytes_per_element(dtype) }

#[inline]
fn elem_count_to_bytes(n: usize, dtype: u8) -> usize { n * elem_size(dtype) }

pub fn unary_float<W: Workspace>(
    c: &UnaryCall, ws: &mut W,
    f: fn(f32) -> f32, dtype: u8,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, dtype);
    let inp = ws.read(c.input).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < bytes { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for i in 0..n {
        let v = read_float(&inp, i, dtype);
        write_float(out, i, f(v), dtype);
    }
    Ok(())
}

pub fn binary_float<W: Workspace>(
    c: &BinaryCall, ws: &mut W,
    f: fn(f32, f32) -> f32, dtype: u8,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, dtype);
    let a = ws.read(c.a).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws.read(c.b).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < bytes { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for i in 0..n {
        let va = read_float(&a, i, dtype);
        let vb = read_float(&b, i, dtype);
        write_float(out, i, f(va, vb), dtype);
    }
    Ok(())
}

pub fn matmul_float<W: Workspace>(c: &MatMulCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let a = ws.read(c.a).get(..m * k * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws.read(c.b).get(..k * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < m * n * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }

    // Fast path: f32 dtype → SIMD dot via simd_f32_dot.
    if dt == DTYPE_F32 {
        let a32: Vec<f32> = (0..m * k).map(|i| read_f32(&a, i)).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| read_f32(&b, i)).collect();
        let mut bt = vec![0f32; k * n];
        for kk in 0..k {
            for j in 0..n {
                bt[j * k + kk] = b32[kk * n + j];
            }
        }
        for i in 0..m {
            let row = &a32[i * k..i * k + k];
            for j in 0..n {
                let col = &bt[j * k..j * k + k];
                let acc = crate::cpu::simd::simd_f32_dot(row, col);
                write_f32(out, i * n + j, acc);
            }
        }
        return Ok(());
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                let av = read_float(&a, i * k + kk, dt);
                let bv = read_float(&b, kk * n + j, dt);
                acc += av * bv;
            }
            write_float(out, i * n + j, acc, dt);
        }
    }
    Ok(())
}

pub fn fused_matmul_activation_float<W: Workspace>(c: &FusedMatMulActivationCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let a = ws.read(c.a).get(..m * k * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws.read(c.b).get(..k * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < m * n * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }

    let act_fn = resolve_activation_f32(c.activation);

    if dt == DTYPE_F32 {
        let a32: Vec<f32> = (0..m * k).map(|i| read_f32(&a, i)).collect();
        let b32: Vec<f32> = (0..k * n).map(|i| read_f32(&b, i)).collect();
        let mut bt = vec![0f32; k * n];
        for kk in 0..k {
            for j in 0..n {
                bt[j * k + kk] = b32[kk * n + j];
            }
        }
        for i in 0..m {
            let row = &a32[i * k..i * k + k];
            for j in 0..n {
                let col = &bt[j * k..j * k + k];
                let acc = crate::cpu::simd::simd_f32_dot(row, col);
                write_f32(out, i * n + j, act_fn(acc));
            }
        }
        return Ok(());
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                let av = read_float(&a, i * k + kk, dt);
                let bv = read_float(&b, kk * n + j, dt);
                acc += av * bv;
            }
            write_float(out, i * n + j, act_fn(acc), dt);
        }
    }
    Ok(())
}

/// Resolve an activation discriminant to an f32 function pointer.
fn resolve_activation_f32(activation: u16) -> fn(f32) -> f32 {
    use hologram_ops::OpKind;
    match activation {
        a if a == OpKind::Relu as u16 => relu_f,
        a if a == OpKind::Sigmoid as u16 => sigmoid_f,
        a if a == OpKind::Tanh as u16 => tanh_f,
        a if a == OpKind::Gelu as u16 => gelu_f,
        a if a == OpKind::Silu as u16 => silu_f,
        a if a == OpKind::Elu as u16 => elu_f,
        a if a == OpKind::Selu as u16 => selu_f,
        a if a == OpKind::Exp as u16 => exp_f,
        a if a == OpKind::Log as u16 => log_f,
        a if a == OpKind::Abs as u16 => abs_f,
        _ => |x| x, // identity
    }
}

pub fn fused_unary_chain_float<W: Workspace>(c: &FusedUnaryChainCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let dt = c.dtype;
    let es = elem_size(dt);
    let input = ws.read(c.input).get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < n * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    // Resolve chain function pointers once.
    let chain_len = c.chain_len as usize;
    let mut fns: [fn(f32) -> f32; 8] = [|x| x; 8];
    for (j, f) in fns.iter_mut().enumerate().take(chain_len) {
        *f = resolve_activation_f32(c.chain[j]);
    }
    for i in 0..n {
        let mut x = read_float(&input, i, dt);
        for f in &fns[..chain_len] {
            x = f(x);
        }
        write_float(out, i, x, dt);
    }
    Ok(())
}

pub fn fused_conv2d_activation_float<W: Workspace>(c: &FusedConv2dActivationCall, ws: &mut W) -> Result<(), BackendError> {
    // Delegate to unfused conv2d, then apply activation in-place.
    let conv = Conv2dCall {
        x: c.x, w: c.w, output: c.output,
        batch: c.batch, channels_in: c.channels_in, channels_out: c.channels_out,
        h_in: c.h_in, w_in: c.w_in, h_out: c.h_out, w_out: c.w_out,
        k_h: c.k_h, k_w: c.k_w,
        stride_h: c.stride_h, stride_w: c.stride_w,
        pad_h: c.pad_h, pad_w: c.pad_w, dtype: c.dtype,
    };
    conv2d_float(&conv, ws)?;
    let n = (c.batch * c.channels_out * c.h_out * c.w_out) as usize;
    let dt = c.dtype;
    let es = elem_size(dt);
    let act_fn = resolve_activation_f32(c.activation);
    let out = ws.write(c.output);
    for i in 0..n.min(out.len() / es) {
        let v = read_float(out, i, dt);
        write_float(out, i, act_fn(v), dt);
    }
    Ok(())
}

pub fn fused_norm_activation_float<W: Workspace>(c: &FusedNormActivationCall, ws: &mut W) -> Result<(), BackendError> {
    // Delegate to unfused layer_norm, then apply activation in-place.
    let norm = NormCall {
        x: c.x, gamma: c.gamma, beta: c.beta,
        residual: c.residual, output: c.output,
        batch: c.batch, feature: c.feature,
        epsilon_bits: c.epsilon_bits, dtype: c.dtype,
    };
    layer_norm_float(&norm, ws)?;
    let n = (c.batch * c.feature) as usize;
    let dt = c.dtype;
    let es = elem_size(dt);
    let act_fn = resolve_activation_f32(c.activation);
    let out = ws.write(c.output);
    for i in 0..n.min(out.len() / es) {
        let v = read_float(out, i, dt);
        write_float(out, i, act_fn(v), dt);
    }
    Ok(())
}

pub fn gemm_float<W: Workspace>(c: &GemmCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let a = ws.read(c.a).get(..m * k * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws.read(c.b).get(..k * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let cc = ws.read(c.c).get(..m * n * es)
        .ok_or(BackendError::SlotOutOfRange(c.c.slot))?
        .to_vec();
    let alpha = f32::from_bits(c.alpha_bits as u32);
    let beta = f32::from_bits(c.beta_bits as u32);
    let out = ws.write(c.output);
    if out.len() < m * n * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += read_float(&a, i * k + kk, dt) * read_float(&b, kk * n + j, dt);
            }
            let bias = read_float(&cc, i * n + j, dt) * beta;
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
    if total_in == 0 || total_w == 0 || total_out == 0 {
        let out = ws.write(c.output);
        for o in out.iter_mut() { *o = 0; }
        return Ok(());
    }
    let xs = ws.read(c.x).get(..total_in)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?
        .to_vec();
    let ws_w = ws.read(c.w).get(..total_w)
        .ok_or(BackendError::SlotOutOfRange(c.w.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < total_out { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
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
                                    acc += read_float(&xs, xi, dt) * read_float(&ws_w, wi, dt);
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
    if bsz == 0 || f == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let total = bsz * f * es;
    let xs = ws.read(c.x).get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?
        .to_vec();
    let gamma = ws.read(c.gamma).get(..f * es).map(|s| s.to_vec()).unwrap_or_default();
    let beta = ws.read(c.beta).get(..f * es).map(|s| s.to_vec()).unwrap_or_default();
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);
    let out = ws.write(c.output);
    if out.len() < total { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for bi in 0..bsz {
        let row_off = bi * f;
        let mut mean = 0f32;
        for j in 0..f { mean += read_float(&xs, row_off + j, dt); }
        mean /= f as f32;
        let mut var = 0f32;
        for j in 0..f {
            let d = read_float(&xs, row_off + j, dt) - mean;
            var += d * d;
        }
        var /= f as f32;
        let inv_std = 1.0 / libm::sqrtf(var + eps);
        for j in 0..f {
            let g = if !gamma.is_empty() { read_float(&gamma, j, dt) } else { 1.0 };
            let bv = if !beta.is_empty() { read_float(&beta, j, dt) } else { 0.0 };
            let v = (read_float(&xs, row_off + j, dt) - mean) * inv_std * g + bv;
            write_float(out, row_off + j, v, dt);
        }
    }
    Ok(())
}

pub fn add_rms_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let total = bsz * f * es;
    let xs = ws.read(c.x).get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?
        .to_vec();
    let residual: Vec<u8> = if c.has_residual() {
        ws.read(c.residual).get(..total)
            .ok_or(BackendError::SlotOutOfRange(c.residual.slot))?
            .to_vec()
    } else {
        vec![0u8; total]
    };
    let gamma = ws.read(c.gamma).get(..f * es).map(|s| s.to_vec()).unwrap_or_default();
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);
    let out = ws.write(c.output);
    if out.len() < total { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for bi in 0..bsz {
        let row_off = bi * f;
        let mut added: Vec<f32> = Vec::with_capacity(f);
        let mut sumsq = 0f32;
        for j in 0..f {
            let v = read_float(&xs, row_off + j, dt) + read_float(&residual, row_off + j, dt);
            added.push(v);
            sumsq += v * v;
        }
        let inv_rms = 1.0 / libm::sqrtf(sumsq / f as f32 + eps);
        for (j, &v) in added.iter().enumerate() {
            let g = if !gamma.is_empty() { read_float(&gamma, j, dt) } else { 1.0 };
            write_float(out, row_off + j, v * inv_rms * g, dt);
        }
    }
    Ok(())
}

pub fn rms_norm_float<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let total = bsz * f * es;
    let xs = ws.read(c.x).get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?
        .to_vec();
    let gamma = ws.read(c.gamma).get(..f * es).map(|s| s.to_vec()).unwrap_or_default();
    let eps = f32::from_bits(c.epsilon_bits as u32).abs().max(1e-9);
    let out = ws.write(c.output);
    if out.len() < total { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for bi in 0..bsz {
        let row_off = bi * f;
        let mut sumsq = 0f32;
        for j in 0..f {
            let v = read_float(&xs, row_off + j, dt);
            sumsq += v * v;
        }
        let inv_rms = 1.0 / libm::sqrtf(sumsq / f as f32 + eps);
        for j in 0..f {
            let g = if !gamma.is_empty() { read_float(&gamma, j, dt) } else { 1.0 };
            let v = read_float(&xs, row_off + j, dt) * inv_rms * g;
            write_float(out, row_off + j, v, dt);
        }
    }
    Ok(())
}

pub fn softmax_float<W: Workspace>(c: &SoftmaxCall, ws: &mut W, log_form: bool) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let f = c.feature as usize;
    if b == 0 || f == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let total = b * f * es;
    let xs = ws.read(c.input).get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < total { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for bi in 0..b {
        let row_off = bi * f;
        let mut max_v = f32::NEG_INFINITY;
        for j in 0..f { max_v = max_v.max(read_float(&xs, row_off + j, dt)); }
        let mut exps: Vec<f32> = Vec::with_capacity(f);
        let mut sum = 0f32;
        for j in 0..f {
            let e = libm::expf(read_float(&xs, row_off + j, dt) - max_v);
            sum += e;
            exps.push(e);
        }
        let log_sum = libm::logf(sum.max(1e-30)) + max_v;
        for (j, &e) in exps.iter().enumerate() {
            let v = if log_form {
                read_float(&xs, row_off + j, dt) - log_sum
            } else {
                e / sum.max(1e-30)
            };
            write_float(out, row_off + j, v, dt);
        }
    }
    Ok(())
}

pub fn reduce_float<W: Workspace>(
    c: &ReduceCall, ws: &mut W,
    f: fn(f32, f32) -> f32, init: f32, mean: bool,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let xs = ws.read(c.input).get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let mut acc = init;
    for i in 0..n {
        acc = f(acc, read_float(&xs, i, dt));
    }
    if mean { acc /= n as f32; }
    let out = ws.write(c.output);
    if out.len() < es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    write_float(out, 0, acc, dt);
    for o in out.iter_mut().skip(es) { *o = 0; }
    Ok(())
}

pub fn cumsum_float<W: Workspace>(c: &ReduceCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let xs = ws.read(c.input).get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < n * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    let mut acc = 0f32;
    for i in 0..n {
        acc += read_float(&xs, i, dt);
        write_float(out, i, acc, dt);
    }
    Ok(())
}

pub fn pool_float<W: Workspace>(c: &PoolCall, ws: &mut W, take_max: bool) -> Result<(), BackendError> {
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
    if b * ch * h_in * w_in == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let total_in = b * ch * h_in * w_in * es;
    let total_out = b * ch * h_out * w_out * es;
    let xs = ws.read(c.x).get(..total_in)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?
        .to_vec();
    let out = ws.write(c.output);
    if out.len() < total_out { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
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
                                let v = read_float(&xs, ((bi * ch + ci) * h_in + ih) * w_in + iw, dt);
                                if take_max { acc = acc.max(v); } else { acc += v; }
                                count += 1;
                            }
                        }
                    }
                    let result = if take_max { acc } else if count > 0 { acc / count as f32 } else { 0.0 };
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
    if b == 0 || h == 0 || s == 0 || d == 0 { return Ok(()); }
    let dt = c.dtype;
    let es = elem_size(dt);
    let total = b * h * s * d;
    let q = ws.read(c.q).get(..total * es)
        .ok_or(BackendError::SlotOutOfRange(c.q.slot))?.to_vec();
    let kk = ws.read(c.k).get(..total * es)
        .ok_or(BackendError::SlotOutOfRange(c.k.slot))?.to_vec();
    let v = ws.read(c.v).get(..total * es)
        .ok_or(BackendError::SlotOutOfRange(c.v.slot))?.to_vec();
    let out = ws.write(c.output);
    if out.len() < total * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    let scale = libm::sqrtf(d as f32).max(1.0);
    for bi in 0..b {
        for hi in 0..h {
            let head_off = (bi * h + hi) * s * d;
            for qi in 0..s {
                let mut scores = vec![0f32; s];
                for (kj, score) in scores.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for di in 0..d {
                        acc += read_float(&q, head_off + qi * d + di, dt)
                             * read_float(&kk, head_off + kj * d + di, dt);
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
                        acc += (sc / denom)
                             * read_float(&v, head_off + kj * d + di, dt);
                    }
                    write_float(out, head_off + qi * d + di, acc, dt);
                }
            }
        }
    }
    Ok(())
}

pub fn where_float<W: Workspace>(c: &WhereCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let dt = c.dtype;
    let es = elem_size(dt);
    let cond = ws.read(c.cond).get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.cond.slot))?.to_vec();
    let a = ws.read(c.a).get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?.to_vec();
    let b = ws.read(c.b).get(..n * es)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?.to_vec();
    let out = ws.write(c.output);
    if out.len() < n * es { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    for i in 0..n {
        let pick_a = cond.get(i).copied().unwrap_or(0) != 0;
        let v = if pick_a { read_float(&a, i, dt) } else { read_float(&b, i, dt) };
        write_float(out, i, v, dt);
    }
    Ok(())
}

pub fn layout_float<W: Workspace>(c: &LayoutCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = elem_count_to_bytes(n, c.dtype);
    let inp = ws.read(c.input).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?.to_vec();
    let out = ws.write(c.output);
    if out.len() < bytes { return Err(BackendError::SlotOutOfRange(c.output.slot)); }
    out[..bytes].copy_from_slice(&inp);
    Ok(())
}

// Elementwise float impls.
#[inline] pub fn relu_f(x: f32) -> f32 { x.max(0.0) }
#[inline] pub fn neg_f(x: f32) -> f32 { -x }
#[inline] pub fn abs_f(x: f32) -> f32 { x.abs() }
#[inline] pub fn sign_f(x: f32) -> f32 { if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 } }
#[inline] pub fn is_nan_f(x: f32) -> f32 { if x.is_nan() { 1.0 } else { 0.0 } }
#[inline] pub fn ceil_f(x: f32) -> f32 { libm::ceilf(x) }
#[inline] pub fn floor_f(x: f32) -> f32 { libm::floorf(x) }
#[inline] pub fn round_f(x: f32) -> f32 { libm::roundf(x) }
#[inline] pub fn sqrt_f(x: f32) -> f32 { libm::sqrtf(x) }
#[inline] pub fn recip_f(x: f32) -> f32 { 1.0 / x }
#[inline] pub fn exp_f(x: f32) -> f32 { libm::expf(x) }
#[inline] pub fn log_f(x: f32) -> f32 { libm::logf(x.max(1e-30)) }
#[inline] pub fn log1p_f(x: f32) -> f32 { libm::log1pf(x) }
#[inline] pub fn sin_f(x: f32) -> f32 { libm::sinf(x) }
#[inline] pub fn cos_f(x: f32) -> f32 { libm::cosf(x) }
#[inline] pub fn tan_f(x: f32) -> f32 { libm::tanf(x) }
#[inline] pub fn asin_f(x: f32) -> f32 { libm::asinf(x.clamp(-1.0, 1.0)) }
#[inline] pub fn acos_f(x: f32) -> f32 { libm::acosf(x.clamp(-1.0, 1.0)) }
#[inline] pub fn atan_f(x: f32) -> f32 { libm::atanf(x) }
#[inline] pub fn erf_f(x: f32) -> f32 { libm::erff(x) }
#[inline] pub fn sigmoid_f(x: f32) -> f32 { 1.0 / (1.0 + libm::expf(-x)) }
#[inline] pub fn tanh_f(x: f32) -> f32 { libm::tanhf(x) }
#[inline] pub fn gelu_f(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::tanhf(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}
#[inline] pub fn silu_f(x: f32) -> f32 { x * sigmoid_f(x) }
#[inline] pub fn elu_f(x: f32) -> f32 { if x >= 0.0 { x } else { libm::expf(x) - 1.0 } }
#[inline] pub fn selu_f(x: f32) -> f32 {
    let alpha = 1.673_263_2_f32;
    let scale = 1.050_701_f32;
    if x >= 0.0 { scale * x } else { scale * alpha * (libm::expf(x) - 1.0) }
}

#[inline] pub fn add_f(a: f32, b: f32) -> f32 { a + b }
#[inline] pub fn sub_f(a: f32, b: f32) -> f32 { a - b }
#[inline] pub fn mul_f(a: f32, b: f32) -> f32 { a * b }
#[inline] pub fn div_f(a: f32, b: f32) -> f32 { if b == 0.0 { 0.0 } else { a / b } }
#[inline] pub fn pow_f(a: f32, b: f32) -> f32 { libm::powf(a, b) }
#[inline] pub fn mod_f(a: f32, b: f32) -> f32 { if b == 0.0 { 0.0 } else { a - (a / b).floor() * b } }
#[inline] pub fn min_f(a: f32, b: f32) -> f32 { a.min(b) }
#[inline] pub fn max_f(a: f32, b: f32) -> f32 { a.max(b) }
#[inline] pub fn equal_f(a: f32, b: f32) -> f32 { if a == b { 1.0 } else { 0.0 } }
#[inline] pub fn less_f(a: f32, b: f32) -> f32 { if a < b { 1.0 } else { 0.0 } }
#[inline] pub fn less_or_equal_f(a: f32, b: f32) -> f32 { if a <= b { 1.0 } else { 0.0 } }
#[inline] pub fn greater_f(a: f32, b: f32) -> f32 { if a > b { 1.0 } else { 0.0 } }
#[inline] pub fn greater_or_equal_f(a: f32, b: f32) -> f32 { if a >= b { 1.0 } else { 0.0 } }
