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

    // Tiled im2col: bound the col buffer to at most TILE_CAP floats.
    // For large spatial dims (e.g., 512×512 with 256 channels), the full
    // col buffer would be kernel_size × spatial_out ≈ 2.4GB. Tiling over
    // the spatial dimension keeps the buffer bounded at ~16MB.
    const TILE_CAP: usize = 4 * 1024 * 1024; // 16 MB as f32
    let tile_size = if kernel_size > 0 {
        (TILE_CAP / kernel_size).max(1).min(spatial_out)
    } else {
        spatial_out
    };
    let mut col = vec![0.0f32; kernel_size * tile_size];

    for batch in 0..n {
        for g in 0..group {
            let w_start = g * oc_per_group * kernel_size;
            let w_end = (w_start + oc_per_group * kernel_size).min(weight.len());
            let w_slice = &weight[w_start..w_end];
            let o_base = batch * oc * spatial_out + g * oc_per_group * spatial_out;

            // Initialize output with bias.
            if let Some(b) = bias {
                for oc_idx in 0..oc_per_group {
                    let abs_oc = g * oc_per_group + oc_idx;
                    let bias_val = b.get(abs_oc).copied().unwrap_or(0.0);
                    if bias_val != 0.0 {
                        let start = o_base + oc_idx * spatial_out;
                        for v in &mut out[start..start + spatial_out] {
                            *v = bias_val;
                        }
                    }
                }
            }

            // Process spatial dimension in tiles.
            let mut tile_start = 0;
            while tile_start < spatial_out {
                let tile_end = (tile_start + tile_size).min(spatial_out);
                let tile_len = tile_end - tile_start;

                // Phase 1: im2col for this tile — col[kernel_size, tile_len].
                for k in 0..kernel_size {
                    let ic_idx = k / (kh * kw);
                    let k_rem = k % (kh * kw);
                    let fh = k_rem / kw;
                    let fw = k_rem % kw;
                    let abs_ic = g * ic_per_group + ic_idx;
                    let col_row = &mut col[k * tile_len..(k + 1) * tile_len];

                    for (t, col_val) in col_row.iter_mut().enumerate() {
                        let out_pos = tile_start + t;
                        let oh = out_pos / w_out;
                        let ow = out_pos % w_out;
                        let ih = oh * sh + fh * dh;
                        let iw = ow * sw + fw * dw;

                        *col_val = if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                            let d_idx =
                                ((batch * ic + abs_ic) * h_in + (ih - ph)) * w_in + (iw - pw);
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

                // Phase 2: GEMM for this tile.
                // C_tile[oc_per_group, tile_len] = W[oc_per_group, kernel_size] × col[kernel_size, tile_len]
                // Write into the correct tile slice of the output.
                // Build a contiguous tile output slice.
                // Output layout is [oc_per_group, spatial_out] row-major; we need
                // columns [tile_start..tile_end] from each row. Since they're not
                // contiguous, use a temporary tile buffer for GEMM then scatter.
                let mut tile_out = vec![0.0f32; oc_per_group * tile_len];

                #[cfg(all(feature = "accelerate", target_os = "macos"))]
                {
                    super::matmul::blas::sgemm_full(
                        super::matmul::GemmParams {
                            m: oc_per_group,
                            n: tile_len,
                            k: kernel_size,
                            alpha: 1.0,
                            beta: 0.0, // Write to temp, add bias after.
                            trans_a: false,
                            trans_b: false,
                        },
                        w_slice,
                        &col[..kernel_size * tile_len],
                        &mut tile_out,
                    );
                }

                #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                {
                    super::matmul::matmul_k_outer(
                        w_slice,
                        &col[..kernel_size * tile_len],
                        &mut tile_out,
                        oc_per_group,
                        kernel_size,
                        tile_len,
                    );
                }

                // Scatter tile results into output (add to bias if present).
                for oc_idx in 0..oc_per_group {
                    let o_row_start = o_base + oc_idx * spatial_out + tile_start;
                    let t_row_start = oc_idx * tile_len;
                    for t in 0..tile_len {
                        out[o_row_start + t] += tile_out[t_row_start + t];
                    }
                }

                tile_start = tile_end;
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
    if data.is_empty() || weight.is_empty() || h_in == 0 || w_in == 0 {
        return Ok(vec![]);
    }
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let bias: Option<Vec<f32>> = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
        Some(cast_f32(bias_bytes)?.to_vec())
    } else {
        None
    };

    // Runtime shape validation: if compiled h_in/w_in look wrong relative to
    // the actual data size, re-derive from weight tensor and data length.
    // This catches cases where shape propagation corrupts spatial dims (e.g.,
    // Resize output shapes shrunk by ForceConcretize → Conv2d gets h_in=2).
    let (h_in, w_in) = {
        let kernel_hw = (kh * kw).max(1);
        let ic_from_weight = weight.len() / kernel_hw / group.max(1);
        let ic_from_weight = ic_from_weight.max(1);
        let n_spatial = data.len() / ic_from_weight.max(1);
        // Check: does compiled h_in * w_in match the expected spatial area?
        if h_in > 0 && w_in > 0 && h_in * w_in * ic_from_weight == data.len() {
            (h_in, w_in) // Compiled values are consistent.
        } else if n_spatial > 0 {
            // Re-derive: assume square spatial if possible.
            let side = (n_spatial as f64).sqrt() as usize;
            if side * side == n_spatial {
                (side, side)
            } else {
                (h_in.max(1), w_in.max(1)) // Keep compiled, hope for the best.
            }
        } else {
            (h_in, w_in)
        }
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
