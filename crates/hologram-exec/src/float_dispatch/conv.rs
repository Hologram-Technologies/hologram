use super::helpers::*;
use crate::error::ExecResult;

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
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    // Extract shapes from propagated input_shapes.
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

    let mut out = vec![0.0f32; n * oc * h_out * w_out];
    let oc_per_group = oc / group.max(1);

    for batch in 0..n {
        for g in 0..group {
            for oc_idx in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc_idx;
                let bias_val = if has_bias {
                    let b = cast_f32(bias_bytes).unwrap_or_default();
                    if abs_oc < b.len() {
                        b[abs_oc]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = bias_val;
                        for ic_idx in 0..ic_per_group {
                            let abs_ic = g * ic_per_group + ic_idx;
                            for fh in 0..kh {
                                let ih = oh * sh + fh * dh;
                                if ih < ph || ih >= h_in + ph {
                                    continue;
                                }
                                let ih = ih - ph;
                                for fw in 0..kw {
                                    let iw = ow * sw + fw * dw;
                                    if iw < pw || iw >= w_in + pw {
                                        continue;
                                    }
                                    let iw = iw - pw;
                                    let d_idx = ((batch * (ic_per_group * group) + abs_ic) * h_in
                                        + ih)
                                        * w_in
                                        + iw;
                                    let w_idx =
                                        ((abs_oc * ic_per_group + ic_idx) * kh + fh) * kw + fw;
                                    if d_idx < data.len() && w_idx < weight.len() {
                                        sum += data[d_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        let o_idx = ((batch * oc + abs_oc) * h_out + oh) * w_out + ow;
                        out[o_idx] = sum;
                    }
                }
            }
        }
    }

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
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    // Infer shapes: data=[N,C,H,W], weight=[OC,IC/group,KH,KW]
    let oc = weight.len() / (kh * kw * (weight.len() / (kh * kw))).max(1);
    // More robust: total weight = OC * (IC/group) * KH * KW
    let ic_per_group = if oc > 0 {
        weight.len() / (oc * kh * kw)
    } else {
        1
    };
    let ic = ic_per_group * group;
    let spatial = data.len() / ic.max(1);
    // Infer H, W from spatial (assume square if ambiguous)
    let h_in = (spatial as f32).sqrt() as usize;
    let w_in = if h_in > 0 { spatial / h_in } else { 1 };

    // For N>1 batches, we need to figure out batch size
    let n = data.len() / (ic * h_in * w_in).max(1);

    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let mut out = vec![0.0f32; n * oc * h_out * w_out];

    let oc_per_group = oc / group.max(1);

    for batch in 0..n {
        for g in 0..group {
            for oc_idx in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc_idx;
                let bias_val = if has_bias {
                    let b = cast_f32(bias_bytes).unwrap_or_default();
                    if abs_oc < b.len() {
                        b[abs_oc]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = bias_val;
                        for ic_idx in 0..ic_per_group {
                            let abs_ic = g * ic_per_group + ic_idx;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = oh * sh + ky * dh;
                                    let ix = ow * sw + kx * dw;
                                    let iy = iy as isize - ph as isize;
                                    let ix = ix as isize - pw as isize;
                                    if iy >= 0
                                        && iy < h_in as isize
                                        && ix >= 0
                                        && ix < w_in as isize
                                    {
                                        let d_idx = ((batch * ic + abs_ic) * h_in + iy as usize)
                                            * w_in
                                            + ix as usize;
                                        let w_idx =
                                            ((abs_oc * ic_per_group + ic_idx) * kh + ky) * kw + kx;
                                        if d_idx < data.len() && w_idx < weight.len() {
                                            sum += data[d_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let o_idx = ((batch * oc + abs_oc) * h_out + oh) * w_out + ow;
                        if o_idx < out.len() {
                            out[o_idx] = sum;
                        }
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

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

    // weight=[IC, OC/group, KH, KW]
    let weight_per_filter = kh * kw;
    let ic = if weight_per_filter > 0 {
        weight.len() / weight_per_filter
    } else {
        return Ok(vec![]);
    };
    // ic here = IC * (OC/group), need to separate
    // Actually weight = [IC, OC/group, KH, KW] so total = IC * (OC/group) * KH * KW
    // We need additional info to split IC and OC/group. Use group to help.
    // Heuristic: assume IC comes from data channel count
    let data_channels = if data.is_empty() {
        1
    } else {
        // data=[N,IC,H,W], try to infer
        let total_spatial = ic * weight_per_filter; // weight.len()
        let oc_per_group = ic / group.max(1); // This isn't quite right
                                              // Simpler: just treat as single batch for now
        let _ = total_spatial;
        let _ = oc_per_group;
        group // fallback
    };
    let _ = data_channels;

    // For transposed conv: H_out = (H_in - 1) * stride - 2*pad + dilation*(kernel-1) + output_pad + 1
    // Infer input spatial dims from data
    // This is complex without shape metadata. Do a simplified version.
    let total = data.len();
    let ic_actual = weight.len() / (kh * kw);
    // weight=[IC, OC/group, KH, KW], so ic_actual = IC * OC/group
    // For group=1: IC channels in, OC channels out
    let oc_per_group = if ic_actual > 0 {
        ic_actual / group.max(1)
    } else {
        1
    };
    // Heuristic: assume square spatial, batch=1
    let in_channels = group; // minimal assumption
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
