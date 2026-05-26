//! CPU kernel dispatch (spec IX.2).
//!
//! Each arm of `dispatch` routes to a function implementing the op's
//! semantics. Per spec C-1 / O-2: kernels are the execution form;
//! the Term tree (in `hologram-ops`) is the formal spec; equivalence
//! is verified by per-op tests.

use alloc::vec::Vec;

use crate::cpu::dtype::is_float;
use crate::cpu::float_kernels as ff;
use crate::error::BackendError;
use crate::kernel_call::*;
use crate::workspace::Workspace;

pub fn dispatch<W: Workspace>(call: &KernelCall, ws: &mut W) -> Result<(), BackendError> {
    if let Some(rv) = try_dispatch_float(call, ws) {
        return rv;
    }
    match call {
        // Direct PrimitiveOp wrappers.
        KernelCall::Neg(c) => unary_w8(c, ws, neg_byte),
        KernelCall::Bnot(c) => unary_w8(c, ws, bnot_byte),
        KernelCall::Succ(c) => unary_w8(c, ws, succ_byte),
        KernelCall::Pred(c) => unary_w8(c, ws, pred_byte),
        KernelCall::Add(c) => binary_w8(c, ws, add_byte),
        KernelCall::Sub(c) => binary_w8(c, ws, sub_byte),
        KernelCall::Mul(c) => binary_w8(c, ws, mul_byte),
        KernelCall::Xor(c) => binary_w8(c, ws, xor_byte),
        KernelCall::And(c) => binary_w8(c, ws, and_byte),
        KernelCall::Or(c) => binary_w8(c, ws, or_byte),

        // Elementwise unary (W8 byte-domain reference).
        KernelCall::Relu(c) => unary_w8(c, ws, relu_byte),
        KernelCall::Abs(c) => unary_w8(c, ws, abs_byte),
        KernelCall::Sign(c) => unary_w8(c, ws, sign_byte),
        KernelCall::IsNaN(c) => unary_w8(c, ws, is_nan_byte),
        KernelCall::Ceil(c) => unary_w8(c, ws, identity_byte),
        KernelCall::Floor(c) => unary_w8(c, ws, identity_byte),
        KernelCall::Round(c) => unary_w8(c, ws, identity_byte),
        KernelCall::Sigmoid(c) => unary_w8(c, ws, sigmoid_byte),
        KernelCall::Tanh(c) => unary_w8(c, ws, tanh_byte),
        KernelCall::Gelu(c) => unary_w8(c, ws, gelu_byte),
        KernelCall::Silu(c) => unary_w8(c, ws, silu_byte),
        KernelCall::Elu(c) => unary_w8(c, ws, elu_byte),
        KernelCall::Selu(c) => unary_w8(c, ws, selu_byte),
        KernelCall::Exp(c) => unary_w8(c, ws, exp_byte),
        KernelCall::Log(c) => unary_w8(c, ws, log_byte),
        KernelCall::Log1p(c) => unary_w8(c, ws, log_byte),
        KernelCall::Sqrt(c) => unary_w8(c, ws, sqrt_byte),
        KernelCall::Reciprocal(c) => unary_w8(c, ws, recip_byte),
        KernelCall::Sin(c) => unary_w8(c, ws, sin_byte),
        KernelCall::Cos(c) => unary_w8(c, ws, cos_byte),
        KernelCall::Tan(c) => unary_w8(c, ws, tan_byte),
        KernelCall::Asin(c) => unary_w8(c, ws, asin_byte),
        KernelCall::Acos(c) => unary_w8(c, ws, acos_byte),
        KernelCall::Atan(c) => unary_w8(c, ws, atan_byte),
        KernelCall::Erf(c) => unary_w8(c, ws, erf_byte),

        // Elementwise binary.
        KernelCall::Div(c) => binary_w8(c, ws, div_byte),
        KernelCall::Pow(c) => binary_w8(c, ws, pow_byte),
        KernelCall::Mod(c) => binary_w8(c, ws, mod_byte),
        KernelCall::Min(c) => binary_w8(c, ws, min_byte),
        KernelCall::Max(c) => binary_w8(c, ws, max_byte),
        KernelCall::Equal(c) => binary_w8(c, ws, equal_byte),
        KernelCall::Less(c) => binary_w8(c, ws, less_byte),
        KernelCall::LessOrEqual(c) => binary_w8(c, ws, less_or_equal_byte),
        KernelCall::Greater(c) => binary_w8(c, ws, greater_byte),
        KernelCall::GreaterOrEqual(c) => binary_w8(c, ws, greater_or_equal_byte),

        // Reshape is a true relabel — row-major bytes unchanged — so a byte
        // copy is correct. The rest move/transform data and carry no params in
        // LayoutCall; they fail loud rather than silently copy (see the float
        // path for the full rationale).
        // Reshape relabel + Slice (ProjectField, input = sub-region) + Pad
        // (output = interior region of a zeroed buffer) are byte copies through
        // the offset-honoring read/write.
        KernelCall::Reshape(c) | KernelCall::Slice(c) | KernelCall::Pad(c) => layout_copy(c, ws),
        // Concat is byte placement (dtype-agnostic) — the same kernel serves
        // both domains.
        KernelCall::Concat(c) => ff::concat_float(c, ws),
        // Transpose / Expand = dtype-agnostic gather (one kernel both domains).
        KernelCall::Transpose(c) => ff::transpose_float(c, ws),
        KernelCall::Expand(c) => ff::expand_float(c, ws),
        KernelCall::Resize(c) => ff::resize_float(c, ws),
        // MatMul (byte ring).
        KernelCall::MatMul(c) | KernelCall::FusedSwiGlu(c) => matmul_w8(c, ws),

        // Where: if cond != 0 select a else b.
        KernelCall::Where(c) => where_w8(c, ws),

        // Gemm: α·A·B + β·C  (W8 byte-domain).
        KernelCall::Gemm(c) => gemm_w8(c, ws),

        // Conv2d / transpose.
        KernelCall::Conv2d(c) | KernelCall::ConvTranspose2d(c) => conv2d_w8(c, ws),

        // im2col / col2im: float-only (they appear in the autodiff backward
        // graph of float convs). The byte ring has no meaning here, so fail
        // loud rather than behave as identity.
        KernelCall::Im2Col(_) | KernelCall::Col2Im(_) => Err(BackendError::UnsupportedOp(
            "im2col/col2im: float-only (byte-ring patch gather is not defined)",
        )),

        // Normalizations.
        KernelCall::LayerNorm(c) | KernelCall::GroupNorm(c) | KernelCall::InstanceNorm(c) => {
            layer_norm_w8(c, ws)
        }
        KernelCall::RmsNorm(c) => rms_norm_w8(c, ws),
        KernelCall::AddRmsNorm(c) => add_rms_norm_w8(c, ws),

        // Reductions: per-batch fold over feature axis.
        KernelCall::ReduceSum(c) => reduce_w8(c, ws, |a, b| a.wrapping_add(b), 0, false),
        KernelCall::ReduceMean(c) => reduce_w8(c, ws, |a, b| a.wrapping_add(b), 0, true),
        KernelCall::ReduceProd(c) => reduce_w8(c, ws, |a, b| a.wrapping_mul(b), 1, false),
        KernelCall::ReduceMin(c) => reduce_w8(c, ws, |a, b| a.min(b), 255, false),
        KernelCall::ReduceMax(c) => reduce_w8(c, ws, |a, b| a.max(b), 0, false),
        KernelCall::CumSum(c) => cumsum_w8(c, ws),

        // Softmax.
        KernelCall::Softmax(c) => softmax_w8(c, ws, false),
        KernelCall::LogSoftmax(c) => softmax_w8(c, ws, true),

        // Pooling.
        KernelCall::MaxPool2d(c) => pool_w8(c, ws, true),
        KernelCall::AvgPool2d(c) | KernelCall::GlobalAvgPool(c) => pool_w8(c, ws, false),

        // Attention.
        KernelCall::Attention(c) => attention_w8(c, ws),

        // RoPE / Clip / Lrn: float-only or parameter-carrying; the byte ring is
        // not meaningful, so fail loud rather than behave as identity. (Clip
        // normally desugars to Min∘Max before lowering; this guards a directly
        // constructed under-specified call.)
        KernelCall::RotaryEmbedding(_) => Err(BackendError::UnsupportedOp(
            "RotaryEmbedding: float-only (byte-ring rotation is not defined)",
        )),
        KernelCall::Clip(_) => Err(BackendError::UnsupportedOp(
            "Clip: (min, max) bounds not represented in UnaryCall",
        )),
        KernelCall::Lrn(_) => Err(BackendError::UnsupportedOp(
            "Lrn: float-only (byte-ring LRN is not defined)",
        )),

        // Quantization (spec X-5): dequantize INT8 / packed-INT4 → float.
        KernelCall::Dequantize(c) => dequantize(c, ws),

        // Content-addressed fusion: matmul + activation epilogue. Fusion
        // only fires for float matmuls, so this is handled by the float
        // fast path above; this arm keeps the match exhaustive.
        KernelCall::MatMulActivation(c) => ff::matmul_activation_float(c, ws),
        KernelCall::MatMulAdd(c) => ff::matmul_add_float(c, ws),
        KernelCall::MatMulAddActivation(c) => ff::matmul_add_activation_float(c, ws),
    }
}

/// Dequantize a packed-integer buffer (INT8 or INT4) into a dense float
/// buffer using `output = (q − zero_point) · scale`.
fn dequantize<W: Workspace>(c: &DequantizeCall, ws: &mut W) -> Result<(), BackendError> {
    use crate::cpu::dtype::*;
    let n = c.element_count as usize;
    let in_bytes_needed = match c.quant_dtype {
        DTYPE_I4 => n.div_ceil(2),
        DTYPE_I8 => n,
        _ => return Err(BackendError::SlotOutOfRange(c.input.slot)),
    };
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..in_bytes_needed)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    let scale = f32::from_bits(c.scale_bits);
    let zp = c.zero_point;

    // Compute the dequantized f32 value for element index `i`.
    let dequant_at = |i: usize| -> f32 {
        let q: i32 = match c.quant_dtype {
            DTYPE_I8 => (inp[i] as i8) as i32,
            DTYPE_I4 => {
                // Two nibbles per byte; low nibble is element 2k, high
                // nibble is element 2k+1. Sign-extend each 4-bit value.
                let byte = inp[i / 2];
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
        (q - zp) as f32 * scale
    };

    let bytes_per_out = match c.dtype {
        DTYPE_F32 => 4,
        DTYPE_BF16 | DTYPE_F16 => 2,
        DTYPE_F64 => 8,
        _ => return Err(BackendError::SlotOutOfRange(c.output.slot)),
    };
    if out.len() < n * bytes_per_out {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..n {
        let v = dequant_at(i);
        match c.dtype {
            DTYPE_F32 => write_f32(out, i, v),
            DTYPE_BF16 => write_bf16(out, i, v),
            DTYPE_F16 => write_f16(out, i, v),
            DTYPE_F64 => {
                let bytes = (v as f64).to_le_bytes();
                out[i * 8..i * 8 + 8].copy_from_slice(&bytes);
            }
            _ => {}
        }
    }
    Ok(())
}

#[inline]
fn unary_w8<W: Workspace>(c: &UnaryCall, ws: &mut W, f: fn(u8) -> u8) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..n {
        out[i] = f(inp[i]);
    }
    Ok(())
}

#[inline]
fn binary_w8<W: Workspace>(
    c: &BinaryCall,
    ws: &mut W,
    f: fn(u8, u8) -> u8,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let (reads, out) = ws
        .split_borrow(&[c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let a = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b = reads[1]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..n {
        out[i] = f(a[i], b[i]);
    }
    Ok(())
}

#[inline]
fn layout_copy<W: Workspace>(c: &LayoutCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    out[..n].copy_from_slice(inp);
    Ok(())
}

/// Gemm: out = α·A·B + β·C, byte-domain.
fn gemm_w8<W: Workspace>(c: &GemmCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let alpha = (c.alpha_bits & 0xFF) as u8;
    let beta = (c.beta_bits & 0xFF) as u8;
    let (reads, out) = ws
        .split_borrow(&[c.a, c.b, c.c], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let a = reads[0]
        .get(..m * k)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b = reads[1]
        .get(..k * n)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    let cc = reads[2]
        .get(..m * n)
        .ok_or(BackendError::SlotOutOfRange(c.c.slot))?;
    if out.len() < m * n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc: u8 = 0;
            for kk in 0..k {
                let p = a[i * k + kk].wrapping_mul(b[kk * n + j]);
                acc = acc.wrapping_add(p);
            }
            let scaled = acc.wrapping_mul(alpha);
            let bias = cc[i * n + j].wrapping_mul(beta);
            out[i * n + j] = scaled.wrapping_add(bias);
        }
    }
    Ok(())
}

/// Conv2d (no padding for the byte-domain reference). Iterates output
/// (h_out, w_out) windows over the input × kernel.
fn conv2d_w8<W: Workspace>(c: &Conv2dCall, ws: &mut W) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let cin = c.channels_in as usize;
    let cout = c.channels_out as usize;
    let h_in = c.h_in as usize;
    let w_in = c.w_in as usize;
    let h_out = c.h_out as usize;
    let w_out = c.w_out as usize;
    let k_h = c.k_h as usize;
    let k_w = c.k_w as usize;
    let s_h = c.stride_h.max(1) as usize;
    let s_w = c.stride_w.max(1) as usize;
    let total_in = b * cin * h_in * w_in;
    let total_w = cout * cin * k_h * k_w;
    let total_out = b * cout * h_out * w_out;
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
                    let mut acc: u8 = 0;
                    for ci in 0..cin {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * s_h + kh;
                                let iw = ow * s_w + kw;
                                if ih < h_in && iw < w_in {
                                    let xi = ((bi * cin + ci) * h_in + ih) * w_in + iw;
                                    let wi = ((co * cin + ci) * k_h + kh) * k_w + kw;
                                    let p = xs[xi].wrapping_mul(ws_w[wi]);
                                    acc = acc.wrapping_add(p);
                                }
                            }
                        }
                    }
                    out[((bi * cout + co) * h_out + oh) * w_out + ow] = acc;
                }
            }
        }
    }
    Ok(())
}

/// LayerNorm (byte-domain reference): subtract mean, scale by gamma, add beta.
fn layer_norm_w8<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    if c.num_groups > 0 {
        return group_norm_w8(c, ws);
    }
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma, c.beta], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..bsz * f)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = reads[1].get(..f).unwrap_or(&[]);
    let beta = reads[2].get(..f).unwrap_or(&[]);
    if out.len() < bsz * f {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for bi in 0..bsz {
        let row = &xs[bi * f..bi * f + f];
        let mean = row.iter().fold(0u32, |a, b| a.wrapping_add(*b as u32)) / f.max(1) as u32;
        let mean = (mean & 0xFF) as u8;
        for j in 0..f {
            let centered = row[j].wrapping_sub(mean);
            let g = *gamma.get(j).unwrap_or(&1);
            let bv = *beta.get(j).unwrap_or(&0);
            out[bi * f + j] = centered.wrapping_mul(g).wrapping_add(bv);
        }
    }
    Ok(())
}

/// GroupNorm / InstanceNorm (byte-domain reference): per-sample, per-group
/// mean subtraction with per-channel `gamma`/`beta`. Mirrors `group_norm_float`
/// over the wrapping byte ring.
fn group_norm_w8<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.batch as usize;
    let f = c.feature as usize;
    let ch = c.channels as usize;
    let g = c.num_groups as usize;
    if n == 0 || f == 0 {
        return Ok(());
    }
    if ch == 0 || g == 0 || !f.is_multiple_of(ch) || !f.is_multiple_of(g) || !ch.is_multiple_of(g) {
        return Err(BackendError::UnsupportedOp(
            "group_norm: require channels|feature, num_groups|feature, num_groups|channels",
        ));
    }
    let spatial = f / ch;
    let group_size = f / g;
    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma, c.beta], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..n * f)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = reads[1].get(..ch).unwrap_or(&[]);
    let beta = reads[2].get(..ch).unwrap_or(&[]);
    if out.len() < n * f {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for ni in 0..n {
        let sample = ni * f;
        for gi in 0..g {
            let gbase = sample + gi * group_size;
            let grp = &xs[gbase..gbase + group_size];
            let mean =
                grp.iter().fold(0u32, |a, b| a.wrapping_add(*b as u32)) / group_size.max(1) as u32;
            let mean = (mean & 0xFF) as u8;
            for i in 0..group_size {
                let ci = (gi * group_size + i) / spatial;
                let centered = grp[i].wrapping_sub(mean);
                let gv = *gamma.get(ci).unwrap_or(&1);
                let bv = *beta.get(ci).unwrap_or(&0);
                out[gbase + i] = centered.wrapping_mul(gv).wrapping_add(bv);
            }
        }
    }
    Ok(())
}

/// RmsNorm (byte-domain): scale by inverse RMS, multiply by gamma.
fn rms_norm_w8<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.x, c.gamma], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..bsz * f)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let gamma = reads[1].get(..f).unwrap_or(&[]);
    if out.len() < bsz * f {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for bi in 0..bsz {
        let row = &xs[bi * f..bi * f + f];
        let sumsq: u32 = row
            .iter()
            .map(|&v| (v as u32).wrapping_mul(v as u32))
            .fold(0u32, |a, b| a.wrapping_add(b));
        let mean_sq = sumsq / f.max(1) as u32;
        let rms = libm::sqrtf((mean_sq as f32).max(1.0));
        let inv_rms = if rms > 0.0 {
            (255.0 / rms).clamp(0.0, 255.0) as u8
        } else {
            0
        };
        for j in 0..f {
            let g = *gamma.get(j).unwrap_or(&1);
            out[bi * f + j] = row[j].wrapping_mul(inv_rms).wrapping_mul(g);
        }
    }
    Ok(())
}

/// Fused Add+RmsNorm: out = rms_norm(x + residual).
fn add_rms_norm_w8<W: Workspace>(c: &NormCall, ws: &mut W) -> Result<(), BackendError> {
    let bsz = c.batch as usize;
    let f = c.feature as usize;
    if bsz == 0 || f == 0 {
        return Ok(());
    }
    let has_residual = c.has_residual();
    let (reads, out) = if has_residual {
        ws.split_borrow(&[c.x, c.residual, c.gamma], c.output)
    } else {
        ws.split_borrow(&[c.x, c.gamma], c.output)
    }
    .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..bsz * f)
        .ok_or(BackendError::SlotOutOfRange(c.x.slot))?;
    let residual: Option<&[u8]> = if has_residual {
        Some(
            reads[1]
                .get(..bsz * f)
                .ok_or(BackendError::SlotOutOfRange(c.residual.slot))?,
        )
    } else {
        None
    };
    let gamma_idx = if has_residual { 2 } else { 1 };
    let gamma = reads[gamma_idx].get(..f).unwrap_or(&[]);
    if out.len() < bsz * f {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for bi in 0..bsz {
        let row_off = bi * f;
        let added_j = |j: usize| -> u8 {
            xs[row_off + j].wrapping_add(residual.map(|r| r[row_off + j]).unwrap_or(0))
        };
        let mut sumsq: u32 = 0;
        for j in 0..f {
            let v = added_j(j) as u32;
            sumsq = sumsq.wrapping_add(v.wrapping_mul(v));
        }
        let mean_sq = sumsq / f.max(1) as u32;
        let rms = libm::sqrtf((mean_sq as f32).max(1.0));
        let inv_rms = if rms > 0.0 {
            (255.0 / rms).clamp(0.0, 255.0) as u8
        } else {
            0
        };
        for j in 0..f {
            let g = *gamma.get(j).unwrap_or(&1);
            out[row_off + j] = added_j(j).wrapping_mul(inv_rms).wrapping_mul(g);
        }
    }
    Ok(())
}

/// Reduction: fold input chunk-by-chunk through `f`, optionally averaging.
fn reduce_w8<W: Workspace>(
    c: &ReduceCall,
    ws: &mut W,
    f: fn(u8, u8) -> u8,
    init: u8,
    mean: bool,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    if n == 0 {
        for o in out.iter_mut() {
            *o = 0;
        }
        return Ok(());
    }
    let xs = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    let acc = xs.iter().copied().fold(init, f);
    let final_value = if mean {
        let total: u32 = xs.iter().map(|&v| v as u32).sum();
        ((total / n.max(1) as u32) & 0xFF) as u8
    } else {
        acc
    };
    if out.is_empty() {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    out[0] = final_value;
    for o in out.iter_mut().skip(1) {
        *o = 0;
    }
    Ok(())
}

/// Cumulative sum along the (single-axis) input.
fn cumsum_w8<W: Workspace>(c: &ReduceCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let mut acc: u8 = 0;
    for i in 0..n {
        acc = acc.wrapping_add(xs[i]);
        out[i] = acc;
    }
    Ok(())
}

/// Softmax (byte-domain reference): subtract row max, exponentiate via
/// the byte-domain `exp_byte` LUT, normalize. `log_form` returns log-softmax.
fn softmax_w8<W: Workspace>(
    c: &SoftmaxCall,
    ws: &mut W,
    log_form: bool,
) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let f = c.feature as usize;
    if b == 0 || f == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let xs = reads[0]
        .get(..b * f)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < b * f {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for bi in 0..b {
        let row = &xs[bi * f..bi * f + f];
        let max_v = *row.iter().max().unwrap_or(&0);
        let exps: Vec<u32> = row
            .iter()
            .map(|&v| {
                let centered = (v as i32 - max_v as i32) as f32 / 32.0;
                (libm::expf(centered) * 255.0) as u32
            })
            .collect();
        let sum: u32 = exps.iter().sum();
        let denom = sum.max(1) as f32;
        for j in 0..f {
            let p = exps[j] as f32 / denom;
            let v = if log_form {
                (libm::logf(p.max(1e-9)) * 32.0 + 128.0).clamp(0.0, 255.0) as u8
            } else {
                (p * 255.0).clamp(0.0, 255.0) as u8
            };
            out[bi * f + j] = v;
        }
    }
    Ok(())
}

/// Pooling: per-window max/mean fold.
fn pool_w8<W: Workspace>(c: &PoolCall, ws: &mut W, take_max: bool) -> Result<(), BackendError> {
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
    let total_in = b * ch * h_in * w_in;
    let total_out = b * ch * h_out * w_out;
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
                    let mut acc: u32 = 0;
                    let mut count: u32 = 0;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = oh * s_h + kh;
                            let iw = ow * s_w + kw;
                            if ih < h_in && iw < w_in {
                                let v = xs[((bi * ch + ci) * h_in + ih) * w_in + iw];
                                if take_max {
                                    acc = acc.max(v as u32);
                                } else {
                                    acc = acc.wrapping_add(v as u32);
                                }
                                count += 1;
                            }
                        }
                    }
                    let result = if take_max {
                        acc as u8
                    } else {
                        acc.checked_div(count)
                            .map(|v| (v & 0xFF) as u8)
                            .unwrap_or(0)
                    };
                    out[((bi * ch + ci) * h_out + oh) * w_out + ow] = result;
                }
            }
        }
    }
    Ok(())
}

/// Attention: out = softmax(Q · K^T / √d) · V — byte-domain reference.
fn attention_w8<W: Workspace>(c: &AttentionCall, ws: &mut W) -> Result<(), BackendError> {
    let b = c.batch as usize;
    let h = c.heads as usize;
    let s = c.seq as usize;
    let d = c.head_dim as usize;
    if b == 0 || h == 0 || s == 0 || d == 0 {
        return Ok(());
    }
    let total = b * h * s * d;
    let (reads, out) = ws
        .split_borrow(&[c.q, c.k, c.v], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let q = reads[0]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.q.slot))?;
    let kk = reads[1]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.k.slot))?;
    let v = reads[2]
        .get(..total)
        .ok_or(BackendError::SlotOutOfRange(c.v.slot))?;
    if out.len() < total {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    let scale = (libm::sqrtf(d as f32)).max(1.0);
    let mut scores: Vec<f32> = Vec::with_capacity(s);
    for bi in 0..b {
        for hi in 0..h {
            let head_off = (bi * h + hi) * s * d;
            for qi in 0..s {
                let q_row = &q[head_off + qi * d..head_off + qi * d + d];
                scores.clear();
                let mut max_score = f32::NEG_INFINITY;
                for kj in 0..s {
                    let k_row = &kk[head_off + kj * d..head_off + kj * d + d];
                    let mut acc: u32 = 0;
                    for di in 0..d {
                        acc = acc.wrapping_add((q_row[di] as u32).wrapping_mul(k_row[di] as u32));
                    }
                    let sc = acc as f32 / scale;
                    if sc > max_score {
                        max_score = sc;
                    }
                    scores.push(sc);
                }
                let mut sum = 0f32;
                for sc in scores.iter_mut() {
                    *sc = libm::expf(*sc - max_score);
                    sum += *sc;
                }
                let denom = sum.max(1e-9);
                for di in 0..d {
                    let mut acc: f32 = 0.0;
                    for (kj, &e) in scores.iter().enumerate() {
                        let v_row = &v[head_off + kj * d..head_off + kj * d + d];
                        acc += (e / denom) * v_row[di] as f32;
                    }
                    out[head_off + qi * d + di] = acc.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(())
}

/// Naive byte-domain matmul kernel (no SIMD). Only correct when the
/// dtype encoding maps the algebraic ring Z/(2^8)Z onto byte storage —
/// which is the W8 contract from spec III.
fn matmul_w8<W: Workspace>(c: &MatMulCall, ws: &mut W) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let (reads, out) = ws
        .split_borrow(&[c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let a_bytes = reads[0]
        .get(..m * k)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b_bytes = reads[1]
        .get(..k * n)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < m * n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    // `ikj` loop order: the inner loop over `j` walks `b_bytes[kk*n + j]`
    // and `out[i*n + j]` contiguously (the `ijk` form strided B down its
    // columns, defeating the cache and auto-vectorization). Addition in
    // Z/(2^8)Z is associative + commutative, so reordering the accumulation
    // preserves the exact W8-ring result. Zero rows of A are skipped.
    for v in out[..m * n].iter_mut() {
        *v = 0;
    }
    for i in 0..m {
        let orow = &mut out[i * n..i * n + n];
        for kk in 0..k {
            let a_ik = a_bytes[i * k + kk];
            if a_ik == 0 {
                continue;
            }
            let brow = &b_bytes[kk * n..kk * n + n];
            for j in 0..n {
                orow[j] = orow[j].wrapping_add(a_ik.wrapping_mul(brow[j]));
            }
        }
    }
    Ok(())
}

fn where_w8<W: Workspace>(c: &WhereCall, ws: &mut W) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let (reads, out) = ws
        .split_borrow(&[c.cond, c.a, c.b], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let cond = reads[0]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.cond.slot))?;
    let a = reads[1]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?;
    let b = reads[2]
        .get(..n)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?;
    if out.len() < n {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..n {
        out[i] = if cond[i] != 0 { a[i] } else { b[i] };
    }
    Ok(())
}

// W8 PrimitiveOp implementations (spec V.7: bytes are bit patterns; the
// algebraic ring Z/(2^8)Z is the carrier).
#[inline]
fn neg_byte(x: u8) -> u8 {
    x.wrapping_neg()
}
#[inline]
fn bnot_byte(x: u8) -> u8 {
    !x
}
#[inline]
fn succ_byte(x: u8) -> u8 {
    x.wrapping_add(1)
}
#[inline]
fn pred_byte(x: u8) -> u8 {
    x.wrapping_sub(1)
}
#[inline]
fn add_byte(a: u8, b: u8) -> u8 {
    a.wrapping_add(b)
}
#[inline]
fn sub_byte(a: u8, b: u8) -> u8 {
    a.wrapping_sub(b)
}
#[inline]
fn mul_byte(a: u8, b: u8) -> u8 {
    a.wrapping_mul(b)
}
#[inline]
fn xor_byte(a: u8, b: u8) -> u8 {
    a ^ b
}
#[inline]
fn and_byte(a: u8, b: u8) -> u8 {
    a & b
}
#[inline]
fn or_byte(a: u8, b: u8) -> u8 {
    a | b
}

// Activation byte kernels (W8 reference; LUT specialization plugs in
// for production hot paths).
#[inline]
fn relu_byte(x: u8) -> u8 {
    // u8 has no negative; ReLU collapses to identity in the unsigned ring.
    // The kernel still serves as the reference for sign-aware dtypes.
    x
}
#[inline]
fn abs_byte(x: u8) -> u8 {
    x
}
#[inline]
fn sign_byte(x: u8) -> u8 {
    if x == 0 {
        0
    } else {
        1
    }
}
#[inline]
fn is_nan_byte(_x: u8) -> u8 {
    0
}
#[inline]
fn identity_byte(x: u8) -> u8 {
    x
}
#[inline]
fn sigmoid_byte(x: u8) -> u8 {
    // Byte-domain logistic: map the u8 onto [-4, 4], apply the exact
    // sigmoid, then re-quantize to [0, 255].
    let f = (x as f32 / 255.0 - 0.5) * 8.0;
    let s = 1.0 / (1.0 + libm::expf(-f));
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn tanh_byte(x: u8) -> u8 {
    let f = (x as f32 / 255.0 - 0.5) * 4.0;
    let s = (libm::tanhf(f) + 1.0) / 2.0;
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn gelu_byte(x: u8) -> u8 {
    let f = (x as f32 / 255.0 - 0.5) * 8.0;
    let g = 0.5 * f * (1.0 + libm::tanhf(0.797_884_6 * (f + 0.044_715 * f * f * f)));
    let s = ((g + 4.0) / 8.0).clamp(0.0, 1.0);
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn silu_byte(x: u8) -> u8 {
    let f = (x as f32 / 255.0 - 0.5) * 8.0;
    let g = f / (1.0 + libm::expf(-f));
    let s = ((g + 4.0) / 8.0).clamp(0.0, 1.0);
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn elu_byte(x: u8) -> u8 {
    // Byte-domain ELU (α = 1): decode onto [-4, 4], apply `x if x≥0 else
    // exp(x)−1`, re-encode like the rest of the activation family. Replaces the
    // prior `relu_byte` alias, which silently dropped the negative branch.
    let f = (x as f32 / 255.0 - 0.5) * 8.0;
    let g = if f >= 0.0 { f } else { libm::expf(f) - 1.0 };
    let s = ((g + 4.0) / 8.0).clamp(0.0, 1.0);
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn selu_byte(x: u8) -> u8 {
    // Byte-domain SELU with the canonical constants (λ ≈ 1.0507, α ≈ 1.6733),
    // same decode/encode convention as the activation family.
    const LAMBDA: f32 = 1.050_700_9;
    const ALPHA: f32 = 1.673_263_2;
    let f = (x as f32 / 255.0 - 0.5) * 8.0;
    let g = if f >= 0.0 {
        LAMBDA * f
    } else {
        LAMBDA * ALPHA * (libm::expf(f) - 1.0)
    };
    let s = ((g + 4.0) / 8.0).clamp(0.0, 1.0);
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn pow_byte(a: u8, b: u8) -> u8 {
    // Power in the wrapping byte ring is iterated multiplication — consistent
    // with `mul_byte` being `wrapping_mul`. Replaces the prior `mul_byte` alias
    // (which computed a·b, not aᵇ).
    a.wrapping_pow(b as u32)
}
#[inline]
fn exp_byte(x: u8) -> u8 {
    let f = x as f32 / 255.0 * 5.0;
    let e = libm::expf(f) / libm::expf(5.0);
    (e * 255.0 + 0.5) as u8
}
#[inline]
fn log_byte(x: u8) -> u8 {
    if x == 0 {
        return 0;
    }
    let f = x as f32 / 255.0;
    let l = (libm::logf(f) + 6.0) / 6.0;
    (l.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
#[inline]
fn sqrt_byte(x: u8) -> u8 {
    let f = x as f32 / 255.0;
    (libm::sqrtf(f) * 255.0 + 0.5) as u8
}
#[inline]
fn recip_byte(x: u8) -> u8 {
    if x == 0 {
        return 255;
    }
    let f = x as f32 / 255.0;
    let r = (1.0 / f / 255.0).clamp(0.0, 1.0);
    (r * 255.0 + 0.5) as u8
}
#[inline]
fn sin_byte(x: u8) -> u8 {
    // [0, 255] → [0, τ] → sin → [-1, 1] → [0, 1] → [0, 255].
    let f = x as f32 / 255.0 * core::f32::consts::TAU;
    let s = (libm::sinf(f) + 1.0) / 2.0;
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn cos_byte(x: u8) -> u8 {
    let f = x as f32 / 255.0 * core::f32::consts::TAU;
    let s = (libm::cosf(f) + 1.0) / 2.0;
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn tan_byte(x: u8) -> u8 {
    // tan has poles at ±π/2; clamp the saturating result to keep the
    // byte encoding well-defined across the full input domain.
    let f = x as f32 / 255.0 * core::f32::consts::TAU;
    let t = libm::tanf(f).clamp(-8.0, 8.0);
    let s = (t / 16.0 + 0.5).clamp(0.0, 1.0);
    (s * 255.0 + 0.5) as u8
}
#[inline]
fn asin_byte(x: u8) -> u8 {
    // [0, 255] → [-1, 1] → asin → [-π/2, π/2] → [0, 1] → [0, 255].
    let f = (x as f32 / 127.5 - 1.0).clamp(-1.0, 1.0);
    let s = libm::asinf(f) / core::f32::consts::PI + 0.5;
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
#[inline]
fn acos_byte(x: u8) -> u8 {
    // [0, 255] → [-1, 1] → acos → [0, π] → [0, 1] → [0, 255].
    let f = (x as f32 / 127.5 - 1.0).clamp(-1.0, 1.0);
    let s = libm::acosf(f) / core::f32::consts::PI;
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
#[inline]
fn atan_byte(x: u8) -> u8 {
    // [0, 255] → [-4, 4] → atan → [-π/2, π/2] → [0, 1] → [0, 255].
    let f = x as f32 / 31.875 - 4.0;
    let s = libm::atanf(f) / core::f32::consts::PI + 0.5;
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
#[inline]
fn erf_byte(x: u8) -> u8 {
    // [0, 255] → [-3, 3] → erf → [-1, 1] → [0, 1] → [0, 255].
    let f = x as f32 / 42.5 - 3.0;
    let s = (libm::erff(f) + 1.0) / 2.0;
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// Binary helpers.
#[inline]
fn div_byte(a: u8, b: u8) -> u8 {
    a.checked_div(b).unwrap_or(0)
}
#[inline]
fn mod_byte(a: u8, b: u8) -> u8 {
    a.checked_rem(b).unwrap_or(0)
}
#[inline]
fn min_byte(a: u8, b: u8) -> u8 {
    a.min(b)
}
#[inline]
fn max_byte(a: u8, b: u8) -> u8 {
    a.max(b)
}
#[inline]
fn equal_byte(a: u8, b: u8) -> u8 {
    if a == b {
        1
    } else {
        0
    }
}
#[inline]
fn less_byte(a: u8, b: u8) -> u8 {
    if a < b {
        1
    } else {
        0
    }
}
#[inline]
fn less_or_equal_byte(a: u8, b: u8) -> u8 {
    if a <= b {
        1
    } else {
        0
    }
}
#[inline]
fn greater_byte(a: u8, b: u8) -> u8 {
    if a > b {
        1
    } else {
        0
    }
}
#[inline]
fn greater_or_equal_byte(a: u8, b: u8) -> u8 {
    if a >= b {
        1
    } else {
        0
    }
}

/// Float-typed dispatch. Returns `Some(result)` if the call's dtype is a
/// float dtype and the corresponding float kernel handled it; `None`
/// otherwise (caller falls through to byte-domain dispatch).
fn try_dispatch_float<W: Workspace>(
    call: &KernelCall,
    ws: &mut W,
) -> Option<Result<(), BackendError>> {
    use crate::cpu::dtype::DTYPE_F64;
    use KernelCall as K;
    // Single dtype-support policy: hologram computes in f16/bf16/f32. f64 is a
    // super-f32 storage format with no native engine — computing it at f32
    // precision would be a silent downgrade, so it is rejected outright rather
    // than producing reduced-precision or zero output. (No model frontend emits
    // f64 today; this guards against a hand-built or future call.)
    if call_dtype(call) == DTYPE_F64 {
        return Some(Err(BackendError::UnsupportedOp(
            "f64 is not a supported compute dtype (hologram computes in f16/bf16/f32)",
        )));
    }
    match call {
        // Direct PrimitiveOp wrappers — float forms.
        K::Neg(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::neg_f, dtype_of_unary(c)))
        }
        K::Add(c) if is_float_binary(c) => Some(ff::binary_float_acc(
            c,
            ws,
            ff::add_f,
            Some(crate::cpu::simd::simd_f32_add),
            dtype_of_binary(c),
        )),
        K::Sub(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::sub_f, dtype_of_binary(c)))
        }
        K::Mul(c) if is_float_binary(c) => Some(ff::binary_float_acc(
            c,
            ws,
            ff::mul_f,
            Some(crate::cpu::simd::simd_f32_mul),
            dtype_of_binary(c),
        )),

        // Elementwise unary float forms.
        K::Relu(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::relu_f, dtype_of_unary(c)))
        }
        K::Sigmoid(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::sigmoid_f, dtype_of_unary(c)))
        }
        K::Tanh(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::tanh_f, dtype_of_unary(c)))
        }
        K::Gelu(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::gelu_f, dtype_of_unary(c)))
        }
        K::Silu(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::silu_f, dtype_of_unary(c)))
        }
        K::Elu(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::elu_f, dtype_of_unary(c)))
        }
        K::Selu(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::selu_f, dtype_of_unary(c)))
        }
        K::Exp(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::exp_f, dtype_of_unary(c)))
        }
        K::Log(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::log_f, dtype_of_unary(c)))
        }
        K::Log1p(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::log1p_f, dtype_of_unary(c)))
        }
        K::Sqrt(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::sqrt_f, dtype_of_unary(c)))
        }
        K::Reciprocal(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::recip_f, dtype_of_unary(c)))
        }
        K::Sin(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::sin_f, dtype_of_unary(c)))
        }
        K::Cos(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::cos_f, dtype_of_unary(c)))
        }
        K::Tan(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::tan_f, dtype_of_unary(c)))
        }
        K::Asin(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::asin_f, dtype_of_unary(c)))
        }
        K::Acos(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::acos_f, dtype_of_unary(c)))
        }
        K::Atan(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::atan_f, dtype_of_unary(c)))
        }
        K::Ceil(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::ceil_f, dtype_of_unary(c)))
        }
        K::Floor(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::floor_f, dtype_of_unary(c)))
        }
        K::Round(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::round_f, dtype_of_unary(c)))
        }
        K::Erf(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::erf_f, dtype_of_unary(c)))
        }
        K::IsNaN(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::is_nan_f, dtype_of_unary(c)))
        }
        K::Sign(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::sign_f, dtype_of_unary(c)))
        }
        K::Abs(c) if is_float_unary(c) => {
            Some(ff::unary_float(c, ws, ff::abs_f, dtype_of_unary(c)))
        }

        // Elementwise binary float forms.
        K::Div(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::div_f, dtype_of_binary(c)))
        }
        K::Pow(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::pow_f, dtype_of_binary(c)))
        }
        K::Mod(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::mod_f, dtype_of_binary(c)))
        }
        K::Min(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::min_f, dtype_of_binary(c)))
        }
        K::Max(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::max_f, dtype_of_binary(c)))
        }
        K::Equal(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::equal_f, dtype_of_binary(c)))
        }
        K::Less(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::less_f, dtype_of_binary(c)))
        }
        K::LessOrEqual(c) if is_float_binary(c) => Some(ff::binary_float(
            c,
            ws,
            ff::less_or_equal_f,
            dtype_of_binary(c),
        )),
        K::Greater(c) if is_float_binary(c) => {
            Some(ff::binary_float(c, ws, ff::greater_f, dtype_of_binary(c)))
        }
        K::GreaterOrEqual(c) if is_float_binary(c) => Some(ff::binary_float(
            c,
            ws,
            ff::greater_or_equal_f,
            dtype_of_binary(c),
        )),

        // Linear algebra / convolution.
        K::MatMulActivation(c) => Some(ff::matmul_activation_float(c, ws)),
        K::MatMulAdd(c) => Some(ff::matmul_add_float(c, ws)),
        K::MatMulAddActivation(c) => Some(ff::matmul_add_activation_float(c, ws)),
        K::MatMul(c) if is_float(c.dtype) => Some(ff::matmul_float(c, ws)),
        // FusedSwiGlu is `silu(x·W_gate) · (x·W_up)` — it needs **two** weight
        // operands, but `MatMulCall` carries one (`b`). It cannot be computed
        // faithfully in the current representation; the old code silently ran a
        // plain matmul and dropped the gate. Fail loud rather than return a
        // wrong tensor. (Completing it requires a two-weight call form + the
        // compiler lowering to populate it — see the representational-gap note.)
        K::FusedSwiGlu(c) if is_float(c.dtype) => Some(Err(BackendError::UnsupportedOp(
            "FusedSwiGlu: gate+up weights not representable in MatMulCall (one operand)",
        ))),
        K::Gemm(c) if is_float(c.dtype) => Some(ff::gemm_float(c, ws)),
        K::Conv2d(c) | K::ConvTranspose2d(c) if is_float(c.dtype) => Some(ff::conv2d_float(c, ws)),
        K::Im2Col(c) if is_float(c.dtype) => Some(ff::im2col_float(c, ws)),
        K::Col2Im(c) if is_float(c.dtype) => Some(ff::col2im_float(c, ws)),

        // Normalizations.
        K::LayerNorm(c) | K::GroupNorm(c) | K::InstanceNorm(c) if is_float(c.dtype) => {
            Some(ff::layer_norm_float(c, ws))
        }
        K::RmsNorm(c) if is_float(c.dtype) => Some(ff::rms_norm_float(c, ws)),
        K::AddRmsNorm(c) if is_float(c.dtype) => Some(ff::add_rms_norm_float(c, ws)),

        // Reductions.
        K::ReduceSum(c) if is_float(c.dtype) => {
            Some(ff::reduce_float(c, ws, |a, b| a + b, 0.0, false))
        }
        K::ReduceMean(c) if is_float(c.dtype) => {
            Some(ff::reduce_float(c, ws, |a, b| a + b, 0.0, true))
        }
        K::ReduceProd(c) if is_float(c.dtype) => {
            Some(ff::reduce_float(c, ws, |a, b| a * b, 1.0, false))
        }
        K::ReduceMin(c) if is_float(c.dtype) => Some(ff::reduce_float(
            c,
            ws,
            |a, b| a.min(b),
            f32::INFINITY,
            false,
        )),
        K::ReduceMax(c) if is_float(c.dtype) => Some(ff::reduce_float(
            c,
            ws,
            |a, b| a.max(b),
            f32::NEG_INFINITY,
            false,
        )),
        K::CumSum(c) if is_float(c.dtype) => Some(ff::cumsum_float(c, ws)),

        // Softmax. (Backward variants are rejected by the byte-path grad arm.)
        K::Softmax(c) if is_float(c.dtype) => Some(ff::softmax_float(c, ws, false)),
        K::LogSoftmax(c) if is_float(c.dtype) => Some(ff::softmax_float(c, ws, true)),

        // Pooling.
        K::MaxPool2d(c) if is_float(c.dtype) => Some(ff::pool_float(c, ws, true)),
        K::AvgPool2d(c) | K::GlobalAvgPool(c) if is_float(c.dtype) => {
            Some(ff::pool_float(c, ws, false))
        }

        // Attention. (AttentionGrad is rejected by the byte-path grad arm.)
        K::Attention(c) if is_float(c.dtype) => Some(ff::attention_float(c, ws)),

        // Where.
        K::Where(c) if is_float(c.dtype) => Some(ff::where_float(c, ws)),

        // Reshape is a *true relabel*: a row-major buffer's bytes are unchanged
        // by a logical shape change, so a dtype-aware byte copy is exactly
        // correct.
        K::Reshape(c) if is_float(c.dtype) => Some(ff::layout_float(c, ws)),
        // Concat is the closed `PrimitiveOp::Concat` constructor: place a ∥ b.
        K::Concat(c) if is_float(c.dtype) => Some(ff::concat_float(c, ws)),
        // Slice is `ProjectField`: the compiler sets the input BufferRef to the
        // sub-region [byte_offset, byte_offset+byte_len), so the dtype-aware
        // copy reads exactly that field. (The content-addressed executor turns
        // this into a zero-movement view; this kernel is the direct-dispatch
        // path.)
        K::Slice(c) if is_float(c.dtype) => Some(ff::layout_float(c, ws)),
        // Pad = placement into a zeroed buffer: the compiler sets the output
        // BufferRef to the interior region [lo, lo+data) so the copy writes the
        // data there and the freshly-zeroed pad regions remain zero — the
        // degenerate Concat(zeros, x, zeros) realized by offset placement.
        K::Pad(c) if is_float(c.dtype) => Some(ff::layout_float(c, ws)),
        // Transpose is the irreducible re-indexing kernel (gather by perm).
        K::Transpose(c) if is_float(c.dtype) => Some(ff::transpose_float(c, ws)),
        // Expand is the broadcast gather (stride-0 on size-1 axes).
        K::Expand(c) if is_float(c.dtype) => Some(ff::expand_float(c, ws)),
        // Resize is the nearest-neighbor gather (reuses ExpandCall's dims).
        K::Resize(c) if is_float(c.dtype) => Some(ff::resize_float(c, ws)),

        // Parameterized ops whose parameters are *not carried* by the kernel-
        // call representation: Clip needs (min, max), RotaryEmbedding needs the
        // rotation table / θ / positions, Lrn needs (size, α, β, bias). The
        // graph node has no attribute slot for them (only Quant/Conv attrs
        // exist), so a float kernel here would silently behave as identity —
        // corrupting any model that uses them. Fail loud until the parameters
        // are plumbed through (graph attrs → call fields → codec). The byte
        // ring keeps its documented reference approximation; this guard only
        // covers the numeric float path that real models execute.
        K::Clip(c) if is_float(c.dtype) => Some(Err(BackendError::UnsupportedOp(
            "Clip: (min, max) bounds not carried by UnaryCall — parameters dropped at lowering",
        ))),
        K::RotaryEmbedding(c) if is_float(c.dtype) => Some(ff::rope_float(c, ws)),
        K::Lrn(c) if is_float(c.dtype) => Some(ff::lrn_float(c, ws)),

        _ => None,
    }
}

#[inline]
fn is_float_unary(c: &UnaryCall) -> bool {
    is_float(c.dtype)
}
#[inline]
fn dtype_of_unary(c: &UnaryCall) -> u8 {
    c.dtype
}
#[inline]
fn is_float_binary(c: &BinaryCall) -> bool {
    is_float(c.dtype)
}
#[inline]
fn dtype_of_binary(c: &BinaryCall) -> u8 {
    c.dtype
}
