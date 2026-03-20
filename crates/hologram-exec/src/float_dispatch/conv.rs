use super::helpers::*;
use crate::error::ExecResult;

// ── im2col + GEMM conv2d core ────────────────────────────────────────────────

/// Core conv2d using im2col + matmul pattern.
///
/// Replaces 8-level nested loops with two flat phases:
/// 1. im2col: gather input patches into a column matrix
/// 2. GEMM: weight × col → output (cache-friendly, autovectorizable)
#[allow(clippy::too_many_arguments)]
fn conv2d_core(
    data: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    ic_per_group: usize,
    h_in: usize,
    w_in: usize,
    oc: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
) -> Vec<f32> {
    let oc_per_group = oc / group.max(1);
    let kernel_size = ic_per_group * kh * kw;
    let spatial_out = h_out * w_out;
    let ic = ic_per_group * group;

    let mut out = vec![0.0f32; n * oc * spatial_out];

    // Process one batch at a time to bound col buffer size.
    let mut col = vec![0.0f32; spatial_out * kernel_size];

    for batch in 0..n {
        for g in 0..group {
            // Phase 1: im2col — flat loop over output spatial × kernel elements.
            for (flat, col_val) in col.iter_mut().enumerate() {
                let out_pos = flat / kernel_size;
                let k = flat % kernel_size;
                let oh = out_pos / w_out;
                let ow = out_pos % w_out;
                let ic_idx = k / (kh * kw);
                let k_rem = k % (kh * kw);
                let fh = k_rem / kw;
                let fw = k_rem % kw;

                let ih = oh * sh + fh * dh;
                let iw = ow * sw + fw * dw;
                // Check padded bounds.
                *col_val = if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                    let abs_ic = g * ic_per_group + ic_idx;
                    let d_idx = ((batch * ic + abs_ic) * h_in + (ih - ph)) * w_in + (iw - pw);
                    if d_idx < data.len() {
                        data[d_idx]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }

            // Phase 2: GEMM — weight[oc_per_group, kernel_size] × col[kernel_size, spatial_out]
            // ikj loop order for cache-friendly access to col and output.
            let w_base = g * oc_per_group * kernel_size;
            let o_base = batch * oc * spatial_out + g * oc_per_group * spatial_out;

            for oc_idx in 0..oc_per_group {
                // Add bias once per output channel.
                let abs_oc = g * oc_per_group + oc_idx;
                let bias_val = bias.and_then(|b| b.get(abs_oc).copied()).unwrap_or(0.0);
                let o_row_start = o_base + oc_idx * spatial_out;
                if bias_val != 0.0 {
                    for v in &mut out[o_row_start..o_row_start + spatial_out] {
                        *v = bias_val;
                    }
                }

                // ikj matmul: iterate over k (inner dim) in outer loop.
                for k in 0..kernel_size {
                    let w_idx = w_base + oc_idx * kernel_size + k;
                    let w_val = if w_idx < weight.len() {
                        weight[w_idx]
                    } else {
                        continue;
                    };
                    if w_val == 0.0 {
                        continue;
                    }
                    // col is [spatial_out × kernel_size] row-major; access col[pos * kernel_size + k]
                    // to get the k-th element of each spatial patch.
                    let o_row = &mut out[o_row_start..o_row_start + spatial_out];
                    for pos in 0..spatial_out {
                        o_row[pos] += w_val * col[pos * kernel_size + k];
                    }
                }
            }
        }
    }

    out
}

/// Conv2d with explicit input shapes (avoids ambiguous shape inference).
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_conv2d_with_shapes(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let bias: Option<Vec<f32>> = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
        Some(cast_f32(bias_bytes)?.to_vec())
    } else {
        None
    };

    let ds = input_shapes.first().cloned().unwrap_or_default();
    let ws = input_shapes.get(1).cloned().unwrap_or_default();

    let (n, _ic, h_in, w_in) = if ds.len() == 4 {
        (ds[0], ds[1], ds[2], ds[3])
    } else {
        return Err(crate::error::ExecError::UnsupportedOp(format!(
            "Conv2d: expected 4D input shape, got {:?}",
            ds
        )));
    };
    let oc = ws.first().copied().unwrap_or(1);
    let ic_per_group = ws.get(1).copied().unwrap_or(1);

    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let out = conv2d_core(
        &data,
        &weight,
        bias.as_deref(),
        n,
        ic_per_group,
        h_in,
        w_in,
        oc,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        group,
    );
    Ok(f32_vec_to_bytes(out))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_conv2d(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let bias: Option<Vec<f32>> = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
        Some(cast_f32(bias_bytes)?.to_vec())
    } else {
        None
    };

    // Infer shapes: data=[N,C,H,W], weight=[OC,IC/group,KH,KW]
    let oc = weight.len() / (kh * kw * (weight.len() / (kh * kw))).max(1);
    let ic_per_group = if oc > 0 {
        weight.len() / (oc * kh * kw)
    } else {
        1
    };
    let ic = ic_per_group * group;
    let spatial = data.len() / ic.max(1);
    let h_in = (spatial as f32).sqrt() as usize;
    let w_in = if h_in > 0 { spatial / h_in } else { 1 };
    let n = data.len() / (ic * h_in * w_in).max(1);

    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let out = conv2d_core(
        &data,
        &weight,
        bias.as_deref(),
        n,
        ic_per_group,
        h_in,
        w_in,
        oc,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        group,
    );
    Ok(f32_vec_to_bytes(out))
}

// ── ConvTranspose ────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_conv_transpose(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
    output_pad_h: usize,
    output_pad_w: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    let weight_per_filter = kh * kw;
    let ic = if weight_per_filter > 0 {
        weight.len() / weight_per_filter
    } else {
        return Ok(vec![]);
    };
    let data_channels = if data.is_empty() {
        1
    } else {
        let total_spatial = ic * weight_per_filter;
        let oc_per_group = ic / group.max(1);
        let _ = total_spatial;
        let _ = oc_per_group;
        group
    };
    let _ = data_channels;

    let total = data.len();
    let ic_actual = weight.len() / (kh * kw);
    let oc_per_group = if ic_actual > 0 {
        ic_actual / group.max(1)
    } else {
        1
    };
    let in_channels = group;
    let spatial_per_channel = total / in_channels.max(1);
    let h_in = (spatial_per_channel as f32).sqrt() as usize;
    let w_in = if h_in > 0 {
        spatial_per_channel / h_in
    } else {
        1
    };

    let h_out = (h_in.saturating_sub(1)) * sh + dh * (kh - 1) + output_pad_h + 1 - 2 * ph;
    let w_out = (w_in.saturating_sub(1)) * sw + dw * (kw - 1) + output_pad_w + 1 - 2 * pw;
    let oc = oc_per_group * group;

    let mut out = vec![0.0f32; oc * h_out * w_out];

    // Add bias — flat loop over output elements
    if has_bias {
        if let Ok(b) = cast_f32(bias_bytes) {
            let hw = h_out * w_out;
            for (flat, out_val) in out.iter_mut().enumerate() {
                let c = flat / hw;
                *out_val = if c < b.len() { b[c] } else { 0.0 };
            }
        }
    }

    // Transposed convolution: scatter input values through the kernel.
    // Flat outer loop over (group × spatial), flat inner loop over (oc_per_group × kernel).
    let hw_in = h_in * w_in;
    for flat_in in 0..(group * hw_in) {
        let g = flat_in / hw_in;
        let spatial = flat_in % hw_in;
        let ih = spatial / w_in;
        let iw = spatial % w_in;
        let abs_ic = g; // simplified: 1 input channel per group
        let d_idx = (abs_ic * h_in + ih) * w_in + iw;
        let d_val = if d_idx < data.len() {
            data[d_idx]
        } else {
            continue;
        };
        for k_flat in 0..(oc_per_group * kh * kw) {
            let oc_idx = k_flat / (kh * kw);
            let k_rem = k_flat % (kh * kw);
            let ky = k_rem / kw;
            let kx = k_rem % kw;
            let abs_oc = g * oc_per_group + oc_idx;
            let oh = ih * sh + ky * dh;
            let ow = iw * sw + kx * dw;
            if oh >= ph && ow >= pw {
                let oh = oh - ph;
                let ow = ow - pw;
                if oh < h_out && ow < w_out {
                    let w_idx = ((abs_ic * oc_per_group + oc_idx) * kh + ky) * kw + kx;
                    let o_idx = (abs_oc * h_out + oh) * w_out + ow;
                    if w_idx < weight.len() && o_idx < out.len() {
                        out[o_idx] += d_val * weight[w_idx];
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}
