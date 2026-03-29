use super::helpers::*;
use crate::error::ExecResult;

// ── im2col + GEMM conv2d core ────────────────────────────────────────────────

/// Core conv2d using im2col + GEMM pattern.
///
/// 1. im2col: gather input patches into a column matrix [kernel_size × spatial_out]
/// 2. GEMM: weight[oc_per_group, kernel_size] × col[kernel_size, spatial_out] → out
///
/// The GEMM phase uses BLAS sgemm when available (Accelerate on macOS),
/// falling back to the parallel tiled matmul kernel otherwise.
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

    // im2col buffer: [kernel_size × spatial_out] (column-major friendly for GEMM).
    // We store it as row-major [kernel_size, spatial_out] so weight × col works
    // directly with BLAS sgemm (row-major C = A × B).
    let mut col = vec![0.0f32; kernel_size * spatial_out];

    for batch in 0..n {
        for g in 0..group {
            // Phase 1: im2col — build col[kernel_size, spatial_out].
            // Each row k of col contains the k-th element of every spatial patch.
            for k in 0..kernel_size {
                let ic_idx = k / (kh * kw);
                let k_rem = k % (kh * kw);
                let fh = k_rem / kw;
                let fw = k_rem % kw;
                let abs_ic = g * ic_per_group + ic_idx;
                let col_row = &mut col[k * spatial_out..(k + 1) * spatial_out];

                for (out_pos, col_val) in col_row.iter_mut().enumerate() {
                    let oh = out_pos / w_out;
                    let ow = out_pos % w_out;
                    let ih = oh * sh + fh * dh;
                    let iw = ow * sw + fw * dw;

                    *col_val = if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
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
            }

            // Phase 2: GEMM — weight[oc_per_group, kernel_size] × col[kernel_size, spatial_out]
            //                → out_slice[oc_per_group, spatial_out]
            let w_slice = &weight[g * oc_per_group * kernel_size
                ..(g * oc_per_group * kernel_size + oc_per_group * kernel_size).min(weight.len())];
            let o_base = batch * oc * spatial_out + g * oc_per_group * spatial_out;
            let o_len = oc_per_group * spatial_out;
            let o_slice = &mut out[o_base..o_base + o_len];

            // Initialize output with bias.
            if let Some(b) = bias {
                for oc_idx in 0..oc_per_group {
                    let abs_oc = g * oc_per_group + oc_idx;
                    let bias_val = b.get(abs_oc).copied().unwrap_or(0.0);
                    if bias_val != 0.0 {
                        let row = &mut o_slice[oc_idx * spatial_out..(oc_idx + 1) * spatial_out];
                        for v in row.iter_mut() {
                            *v = bias_val;
                        }
                    }
                }
            }

            // GEMM: C = A × B where A=[oc_per_group, kernel_size], B=[kernel_size, spatial_out]
            // beta=1.0 to accumulate onto bias.
            let beta = if bias.is_some() { 1.0 } else { 0.0 };

            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            {
                super::matmul::blas::sgemm_full(
                    super::matmul::GemmParams {
                        m: oc_per_group,
                        n: spatial_out,
                        k: kernel_size,
                        alpha: 1.0,
                        beta,
                        trans_a: false,
                        trans_b: false,
                    },
                    w_slice,
                    &col,
                    o_slice,
                );
            }

            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            {
                // Tiled matmul kernel (parallelized via rayon when available).
                // matmul_k_outer writes into o_slice, overwriting bias.
                // So we save bias, compute matmul (which zeros first), then add bias back.
                if bias.is_some() {
                    // Bias already written to o_slice above. Save it, matmul will overwrite.
                    let saved_bias: Vec<f32> = o_slice.to_vec();
                    super::matmul::matmul_k_outer(
                        w_slice,
                        &col,
                        o_slice,
                        oc_per_group,
                        kernel_size,
                        spatial_out,
                    );
                    for (o, b) in o_slice.iter_mut().zip(saved_bias.iter()) {
                        *o += b;
                    }
                } else {
                    super::matmul::matmul_k_outer(
                        w_slice,
                        &col,
                        o_slice,
                        oc_per_group,
                        kernel_size,
                        spatial_out,
                    );
                }
            }
        }
    }

    out
}

/// Conv2d with explicit spatial dimensions from the op fields.
///
/// All dispatch paths route through this function — no shape guessing needed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv2d_direct(
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
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let bias: Option<Vec<f32>> = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
        Some(cast_f32(bias_bytes)?.to_vec())
    } else {
        None
    };

    // Derive N, OC, IC/group from buffer lengths + known spatial dims.
    let ic = if h_in > 0 && w_in > 0 {
        data.len() / (h_in * w_in)
    } else {
        1
    };
    let n = if ic > 0 && h_in > 0 && w_in > 0 {
        data.len() / (ic * h_in * w_in)
    } else {
        1
    };
    let oc = weight.len() / (kh * kw).max(1) / (ic / group.max(1)).max(1);
    let ic_per_group = (ic / group.max(1)).max(1);

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

/// Conv2d with explicit input shapes from shape vectors (used by KvStore path).
///
/// Delegates to `dispatch_conv2d_direct` after extracting H/W from shapes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv2d_with_shapes(
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
    let ds = input_shapes.first().cloned().unwrap_or_default();
    if ds.len() != 4 {
        return Err(crate::error::ExecError::UnsupportedOp(format!(
            "Conv2d: expected 4D input shape, got {:?}",
            ds
        )));
    }
    let h_in = ds[2];
    let w_in = ds[3];
    dispatch_conv2d_direct(inputs, kh, kw, sh, sw, ph, pw, dh, dw, group, h_in, w_in)
}

// ── ConvTranspose ────────────────────────────────────────────────────────────

/// Transposed 2-D convolution with explicit spatial dimensions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv_transpose(
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
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    let ic_actual = weight.len() / (kh * kw).max(1);
    let oc_per_group = if ic_actual > 0 {
        ic_actual / group.max(1)
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
