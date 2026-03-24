use super::helpers::*;
use crate::error::{ExecError, ExecResult};

/// Fast approximate inverse square root (Quake III-style with two Newton-Raphson steps).
///
/// Two NR iterations give ~0.001% max relative error — sufficient for
/// normalization layers and matching standard `1.0 / sqrt(x)` to `< 1e-4`.
#[inline]
fn fast_rsqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = f32::to_bits(x);
    let i = 0x5f37_59df - (i >> 1);
    let mut y = f32::from_bits(i);
    y = y * (1.5 - half * y * y); // First Newton-Raphson iteration
    y = y * (1.5 - half * y * y); // Second iteration for higher precision
    y
}

pub(super) fn dispatch_softmax(inputs: &[&[u8]], size: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }

    let mut out = x.into_owned();
    softmax_in_place(&mut out, size);
    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_log_softmax(inputs: &[&[u8]], size: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.into_owned();
    for row in out.chunks_mut(size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
        for v in row.iter_mut() {
            *v -= log_sum_exp;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_rms_norm(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    if weight.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight: [{size}]"),
            actual: format!("{} floats", weight.len()),
        });
    }
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.into_owned();
    rms_norm_in_place(&mut out, &weight, size, epsilon);
    Ok(f32_vec_to_bytes(out))
}

/// Fused Add + RMS normalization: rmsnorm(x + residual, weight, epsilon).
/// Inputs: [x (f32), residual (f32), weight (f32)].
/// Avoids materializing the intermediate x + residual buffer.
pub(super) fn dispatch_add_rms_norm(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let residual = cast_f32(inputs[1])?;
    let weight = cast_f32(inputs[2])?;
    if weight.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight: [{size}]"),
            actual: format!("{} floats", weight.len()),
        });
    }
    if x.len() != residual.len() {
        return Err(ExecError::ShapeMismatch {
            expected: "x and residual same length".to_string(),
            actual: format!("x={}, residual={}", x.len(), residual.len()),
        });
    }
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    // Compute x + residual in-place, then normalize
    let mut out: Vec<f32> = x
        .iter()
        .zip(residual.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    rms_norm_in_place(&mut out, &weight, size, epsilon);
    Ok(f32_vec_to_bytes(out))
}

/// Fused Add + RMS normalization writing directly into out_buf (zero intermediate Vec).
pub(crate) fn dispatch_add_rms_norm_into(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let x = cast_f32(inputs[0])?;
    let residual = cast_f32(inputs[1])?;
    let weight = cast_f32(inputs[2])?;
    if weight.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight: [{size}]"),
            actual: format!("{} floats", weight.len()),
        });
    }
    if x.len() != residual.len() {
        return Err(ExecError::ShapeMismatch {
            expected: "x and residual same length".to_string(),
            actual: format!("x={}, residual={}", x.len(), residual.len()),
        });
    }
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let out = alloc_f32_in(out_buf, x.len());
    for (o, (&a, &b)) in out.iter_mut().zip(x.iter().zip(residual.iter())) {
        *o = a + b;
    }
    rms_norm_in_place(out, &weight, size, epsilon);
    Ok(())
}

/// Softmax writing directly into out_buf (zero intermediate Vec).
pub(crate) fn dispatch_softmax_into(
    inputs: &[&[u8]],
    size: usize,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }

    let out = alloc_f32_in(out_buf, x.len());
    out.copy_from_slice(&x);
    softmax_in_place(out, size);
    Ok(())
}

/// RmsNorm writing directly into out_buf (zero intermediate Vec).
pub(crate) fn dispatch_rms_norm_into(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    if weight.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight: [{size}]"),
            actual: format!("{} floats", weight.len()),
        });
    }
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let out = alloc_f32_in(out_buf, x.len());
    out.copy_from_slice(&x);
    rms_norm_in_place(out, &weight, size, epsilon);
    Ok(())
}

pub(super) fn dispatch_layer_norm(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias = cast_f32(inputs[2])?;
    if weight.len() != size || bias.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight/bias: [{size}]"),
            actual: format!("weight={}, bias={}", weight.len(), bias.len()),
        });
    }
    let mut out = x.into_owned();
    layer_norm_in_place(&mut out, &weight, &bias, size, epsilon);
    Ok(f32_vec_to_bytes(out))
}

/// LogSoftmax writing directly into out_buf (zero intermediate Vec).
pub(crate) fn dispatch_log_softmax_into(
    inputs: &[&[u8]],
    size: usize,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let x = cast_f32(inputs[0])?;
    let actual_size = if size == 0 { x.len() } else { size };
    if actual_size > 0 && x.len() % actual_size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {actual_size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let out = alloc_f32_in(out_buf, x.len());
    out.copy_from_slice(&x);
    for row in out.chunks_mut(actual_size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
        for v in row.iter_mut() {
            *v -= log_sum_exp;
        }
    }
    Ok(())
}

/// LayerNorm writing directly into out_buf (zero intermediate Vec).
pub(crate) fn dispatch_layer_norm_into(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias = cast_f32(inputs[2])?;
    let actual_size = if size == 0 { x.len() } else { size };
    if weight.len() != actual_size || bias.len() != actual_size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight/bias: [{actual_size}]"),
            actual: format!("weight={}, bias={}", weight.len(), bias.len()),
        });
    }
    let out = alloc_f32_in(out_buf, x.len());
    out.copy_from_slice(&x);
    layer_norm_in_place(out, &weight, &bias, actual_size, epsilon);
    Ok(())
}

pub(super) fn dispatch_instance_norm(
    inputs: &[&[u8]],
    size: usize,
    epsilon: f32,
) -> ExecResult<Vec<u8>> {
    // inputs: [data, scale, bias]
    // InstanceNorm: normalize each (N,C) spatial slice independently
    // size = number of spatial elements per channel (H*W)
    let data = cast_f32(inputs[0])?;
    let scale = cast_f32(inputs[1])?;
    let bias = cast_f32(inputs[2])?;

    let n_channels = scale.len();
    let spatial = if n_channels > 0 {
        data.len() / n_channels
    } else {
        data.len()
    };
    let actual_size = if size > 0 { size } else { spatial };

    let mut out = data.into_owned();

    for c in 0..n_channels {
        let start = c * actual_size;
        let end = (start + actual_size).min(out.len());
        if start >= out.len() {
            break;
        }
        let slice = &out[start..end];

        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        let var: f32 =
            slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / slice.len() as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();

        let s = if c < scale.len() { scale[c] } else { 1.0 };
        let b = if c < bias.len() { bias[c] } else { 0.0 };

        for v in out[start..end].iter_mut() {
            *v = (*v - mean) * inv_std * s + b;
        }
    }

    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_lrn(
    inputs: &[&[u8]],
    size: usize,
    alpha: f32,
    beta: f32,
    bias: f32,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // LRN: across channels. data=[N,C,H,W]
    // out[n,c,h,w] = data[n,c,h,w] / (bias + alpha/size * sum(data[n,c',h,w]^2))^beta
    // where c' ranges over [max(0,c-floor(size/2)), min(C-1,c+floor(size/2))]

    // Simplified: treat as 1-D across the entire buffer
    let n = data.len();
    let half = size / 2;
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let sum_sq: f32 = data[lo..hi].iter().map(|v| v * v).sum();
        let denom = (bias + alpha / size as f32 * sum_sq).powf(beta);
        out[i] = data[i] / denom;
    }

    Ok(f32_vec_to_bytes(out))
}

// ── Shared in-place kernels ─────────────────────────────────────────────

/// Softmax in-place on a mutable f32 slice.
#[inline]
fn softmax_in_place(out: &mut [f32], size: usize) {
    let uniform = 1.0f32 / size as f32;
    for row in out.chunks_mut(size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::INFINITY {
            let inf_count = row.iter().filter(|&&v| v == f32::INFINITY).count();
            let w = if inf_count > 0 {
                1.0f32 / inf_count as f32
            } else {
                uniform
            };
            for v in row.iter_mut() {
                *v = if *v == f32::INFINITY { w } else { 0.0 };
            }
            continue;
        }
        if !max.is_finite() {
            for v in row.iter_mut() {
                *v = uniform;
            }
            continue;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        } else {
            for v in row.iter_mut() {
                *v = uniform;
            }
        }
    }
}

/// RmsNorm in-place on a mutable f32 slice.
#[inline]
fn rms_norm_in_place(out: &mut [f32], weight: &[f32], size: usize, epsilon: f32) {
    for row in out.chunks_mut(size) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
        let inv_rms = fast_rsqrt(ms + epsilon);
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * inv_rms * w;
        }
    }
}

/// LayerNorm in-place on a mutable f32 slice.
#[inline]
fn layer_norm_in_place(out: &mut [f32], weight: &[f32], bias: &[f32], size: usize, epsilon: f32) {
    for row in out.chunks_mut(size) {
        let mean: f32 = row.iter().sum::<f32>() / size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / size as f32;
        let inv_std = fast_rsqrt(var + epsilon);
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * weight[i] + bias[i];
        }
    }
}
