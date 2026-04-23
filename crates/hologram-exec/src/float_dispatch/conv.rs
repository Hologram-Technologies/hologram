pub(crate) use super::conv_transpose::{dispatch_conv_transpose, ConvTransposeOutputPad};
use super::conv_winograd::{conv2d_depthwise, conv2d_winograd_f23};
use super::helpers::*;
use crate::error::ExecResult;

// ── Conv2D attribute & call builders ────────────────────────────────────────
//
// Shared parameter structs that let the dispatch + kernel entry points stay
// readable and sidestep the `clippy::too_many_arguments` lint (which the
// project rule forbids us from `#[allow]`-ing). Build with `new(..)` and
// chain `with_*` setters for any non-default knobs.

/// Kernel attributes common to every 2-D convolution entry point.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv2dAttrs {
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub ph: usize,
    pub pw: usize,
    pub dh: usize,
    pub dw: usize,
    pub group: usize,
}

impl Conv2dAttrs {
    /// New attrs with stride 1, no padding, dilation 1, and no grouping.
    #[inline]
    pub fn new(kh: usize, kw: usize) -> Self {
        Self {
            kh,
            kw,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            dh: 1,
            dw: 1,
            group: 1,
        }
    }

    #[inline]
    pub fn with_stride(mut self, sh: usize, sw: usize) -> Self {
        self.sh = sh;
        self.sw = sw;
        self
    }

    #[inline]
    pub fn with_padding(mut self, ph: usize, pw: usize) -> Self {
        self.ph = ph;
        self.pw = pw;
        self
    }

    #[inline]
    pub fn with_dilation(mut self, dh: usize, dw: usize) -> Self {
        self.dh = dh;
        self.dw = dw;
        self
    }

    #[inline]
    pub fn with_group(mut self, group: usize) -> Self {
        self.group = group;
        self
    }
}

/// Input feature-map shape `(n, channels, h_in, w_in)` — the four dimensions
/// common to both depthwise and grouped 2-D convolutions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv2dInputShape {
    pub n: usize,
    pub channels: usize,
    pub h_in: usize,
    pub w_in: usize,
}

impl Conv2dInputShape {
    #[inline]
    pub fn new(n: usize, channels: usize, h_in: usize, w_in: usize) -> Self {
        Self {
            n,
            channels,
            h_in,
            w_in,
        }
    }
}

/// Full call parameters for [`conv2d_depthwise`](super::conv_winograd::conv2d_depthwise).
#[derive(Debug, Clone, Copy)]
pub(super) struct Conv2dDepthwiseCall<'a> {
    pub(super) data: &'a [f32],
    pub(super) weight: &'a [f32],
    pub(super) bias: Option<&'a [f32]>,
    pub(super) input_shape: Conv2dInputShape,
    pub(super) h_out: usize,
    pub(super) w_out: usize,
    pub(super) attrs: Conv2dAttrs,
}

impl<'a> Conv2dDepthwiseCall<'a> {
    #[inline]
    pub(super) fn new(
        data: &'a [f32],
        weight: &'a [f32],
        input_shape: Conv2dInputShape,
        attrs: Conv2dAttrs,
    ) -> Self {
        Self {
            data,
            weight,
            bias: None,
            input_shape,
            h_out: 0,
            w_out: 0,
            attrs,
        }
    }

    #[inline]
    pub(super) fn with_output_hw(mut self, h_out: usize, w_out: usize) -> Self {
        self.h_out = h_out;
        self.w_out = w_out;
        self
    }

    #[inline]
    pub(super) fn with_bias(mut self, bias: Option<&'a [f32]>) -> Self {
        self.bias = bias;
        self
    }
}

/// Full call parameters for [`conv2d_core`] (grouped 2-D convolution).
#[derive(Debug, Clone, Copy)]
struct Conv2dCoreCall<'a> {
    data: &'a [f32],
    weight: &'a [f32],
    bias: Option<&'a [f32]>,
    /// Batch + input shape. The `channels` field is re-interpreted as
    /// `ic_per_group` (per-group input channels) by this kernel.
    input_shape: Conv2dInputShape,
    oc: usize,
    h_out: usize,
    w_out: usize,
    attrs: Conv2dAttrs,
}

impl<'a> Conv2dCoreCall<'a> {
    #[inline]
    fn new(
        data: &'a [f32],
        weight: &'a [f32],
        input_shape: Conv2dInputShape,
        attrs: Conv2dAttrs,
    ) -> Self {
        Self {
            data,
            weight,
            bias: None,
            input_shape,
            oc: 0,
            h_out: 0,
            w_out: 0,
            attrs,
        }
    }

    #[inline]
    fn with_output(mut self, oc: usize, h_out: usize, w_out: usize) -> Self {
        self.oc = oc;
        self.h_out = h_out;
        self.w_out = w_out;
        self
    }

    #[inline]
    fn with_bias(mut self, bias: Option<&'a [f32]>) -> Self {
        self.bias = bias;
        self
    }
}

// ── im2col + GEMM conv2d core ────────────────────────────────────────────────

/// Core conv2d using im2col + GEMM pattern.
///
/// 1. im2col: gather input patches into a column matrix [kernel_size × spatial_out]
/// 2. GEMM: weight[oc_per_group, kernel_size] × col[kernel_size, spatial_out] → out
///
/// The GEMM phase uses BLAS sgemm when available (Accelerate on macOS),
/// falling back to the parallel tiled matmul kernel otherwise.
#[inline(always)]
fn conv2d_core(call: Conv2dCoreCall<'_>) -> Vec<f32> {
    let Conv2dCoreCall {
        data,
        weight,
        bias,
        input_shape:
            Conv2dInputShape {
                n,
                channels: _ic_channels_slot,
                h_in,
                w_in,
            },
        oc,
        h_out,
        w_out,
        attrs:
            Conv2dAttrs {
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                group,
            },
    } = call;
    // `ic_per_group` is stored in the Conv2dInputShape.channels slot for this kernel.
    let ic_per_group = _ic_channels_slot;
    let oc_per_group = oc / group.max(1);
    let kernel_size = ic_per_group * kh * kw;
    let spatial_out = h_out * w_out;
    let ic = ic_per_group * group;

    // Depthwise fast path: group == channels, 1 input channel per group.
    // Direct loop avoids im2col overhead for single-channel inner products.
    if ic_per_group == 1 && oc_per_group == 1 {
        return conv2d_depthwise(
            Conv2dDepthwiseCall::new(
                data,
                weight,
                Conv2dInputShape::new(n, ic, h_in, w_in),
                Conv2dAttrs::new(kh, kw)
                    .with_stride(sh, sw)
                    .with_padding(ph, pw)
                    .with_dilation(dh, dw),
            )
            .with_bias(bias)
            .with_output_hw(h_out, w_out),
        );
    }

    // 1×1 pointwise fast path: when kernel is 1×1 with no padding/dilation,
    // the convolution reduces to a matrix multiply: for each batch/group,
    //   output[oc_per_group, spatial] = weight[oc_per_group, ic_per_group] × input[ic_per_group, spatial]
    // Route directly to BLAS sgemm for orders-of-magnitude speedup over
    // the general im2col path. This is the dominant case in SD UNet
    // mid-blocks (1280→1280 pointwise with [1, 1280, 64, 64] activations).
    if kh == 1 && kw == 1 && ph == 0 && pw == 0 && dh == 1 && dw == 1 && sh == 1 && sw == 1 {
        let mut out = vec![0.0f32; n * oc * spatial_out];
        for batch in 0..n {
            for g in 0..group {
                let w_start = g * oc_per_group * ic_per_group;
                let d_start = batch * ic * spatial_out + g * ic_per_group * spatial_out;
                let o_start = batch * oc * spatial_out + g * oc_per_group * spatial_out;

                let w_slice = &weight[w_start..w_start + oc_per_group * ic_per_group];
                let d_slice = &data[d_start..d_start + ic_per_group * spatial_out];
                let o_slice = &mut out[o_start..o_start + oc_per_group * spatial_out];

                // weight is [oc_per_group, ic_per_group], data is [ic_per_group, spatial]
                // output is [oc_per_group, spatial]
                #[cfg(all(feature = "accelerate", target_os = "macos"))]
                {
                    super::matmul::blas::sgemm(
                        oc_per_group,
                        spatial_out,
                        ic_per_group,
                        w_slice,
                        d_slice,
                        o_slice,
                    );
                }
                #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                {
                    super::matmul::matmul_k_outer(
                        w_slice,
                        d_slice,
                        o_slice,
                        oc_per_group,
                        ic_per_group,
                        spatial_out,
                    );
                }
            }

            // Add bias. Bias may have fewer elements than `oc` when grouped
            // conv shares a smaller bias vector — clamp to bias length.
            if let Some(b) = bias {
                for (c, &bv) in b.iter().enumerate().take(oc) {
                    let base = batch * oc * spatial_out + c * spatial_out;
                    for s in &mut out[base..base + spatial_out] {
                        *s += bv;
                    }
                }
            }
        }
        return out;
    }

    // Winograd F(2,3) fast path: 3×3 kernel, stride=1, dilation=1, sufficient channels.
    // Reduces multiplications by 2.25× — the dominant 3×3 conv case in UNet/VAE.
    if kh == 3
        && kw == 3
        && sh == 1
        && sw == 1
        && dh == 1
        && dw == 1
        && ph == 1
        && pw == 1
        && ic_per_group >= 16
    {
        return conv2d_winograd_f23(
            data, weight, bias, n, ic, h_in, w_in, oc, h_out, w_out, group,
        );
    }

    let mut out = vec![0.0f32; n * oc * spatial_out];

    // Tiled im2col: bound the col buffer to at most TILE_CAP floats.
    // Tiled im2col: bound the col buffer to at most TILE_CAP floats.
    const TILE_CAP: usize = 4 * 1024 * 1024; // 16 MB as f32
    let tile_size = (TILE_CAP.checked_div(kernel_size).unwrap_or(spatial_out))
        .max(1)
        .min(spatial_out);
    let mut col = vec![0.0f32; kernel_size * tile_size];
    // Pre-allocate tile buffers once — reused across all tiles to avoid per-tile allocation.
    let mut tile_out = vec![0.0f32; oc_per_group * tile_size];
    // For LUT-GEMM: col_t (transposed im2col) and lut_out (GEMM result).
    // These need to be separate buffers since lut_gemm writes to output while reading input.
    let mut col_t_buf = vec![0.0f32; tile_size * kernel_size];
    let mut lut_out_buf = vec![0.0f32; tile_size * oc_per_group];

    for batch in 0..n {
        for g in 0..group {
            let w_start = g * oc_per_group * kernel_size;
            let w_end = (w_start + oc_per_group * kernel_size).min(weight.len());
            let w_slice = &weight[w_start..w_end];

            // LUT-GEMM Q4 path for non-BLAS platforms (WASM, Linux without MKL).
            // On macOS with Accelerate, BLAS sgemm is faster — skip quantization.
            // Transpose W, quantize once per group, reuse across all spatial tiles.
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            let qw = if group <= 1 && oc_per_group >= 64 && kernel_size >= 16 {
                let mut w_t = vec![0.0f32; oc_per_group * kernel_size];
                for oc_idx in 0..oc_per_group {
                    for k in 0..kernel_size {
                        w_t[k * oc_per_group + oc_idx] = w_slice[oc_idx * kernel_size + k];
                    }
                }
                Some(crate::lut_gemm::quantize::quantize_4bit(
                    &w_t,
                    kernel_size as u32,
                    oc_per_group as u32,
                ))
            } else {
                None
            };
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            let qw: Option<crate::lut_gemm::quantize::QuantizedWeights4> = None;

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
                // Fast path for stride=1, dilation=1: consecutive output positions
                // within a row map to consecutive input positions, enabling memcpy.
                let use_fast_im2col = sh == 1 && sw == 1 && dh == 1 && dw == 1;

                for k in 0..kernel_size {
                    let ic_idx = k / (kh * kw);
                    let k_rem = k % (kh * kw);
                    let fh = k_rem / kw;
                    let fw = k_rem % kw;
                    let abs_ic = g * ic_per_group + ic_idx;
                    let col_row = &mut col[k * tile_len..(k + 1) * tile_len];

                    if use_fast_im2col {
                        let d_channel_base = (batch * ic + abs_ic) * h_in * w_in;
                        // For each output row in this tile, compute the contiguous
                        // interior range where both h and w are in-bounds, then memcpy.
                        let mut t = 0;
                        while t < tile_len {
                            let out_pos = tile_start + t;
                            let oh = out_pos / w_out;
                            let ow_start = out_pos % w_out;
                            // How many positions remain in this output row within the tile.
                            let row_remaining = (w_out - ow_start).min(tile_len - t);

                            let ih = oh + fh;
                            if ih < ph || ih >= h_in + ph {
                                // Entire row segment is padding — zero fill.
                                col_row[t..t + row_remaining].fill(0.0);
                                t += row_remaining;
                                continue;
                            }
                            let ih_actual = ih - ph;

                            // Width range in-bounds: iw_actual = ow + fw - pw must be in [0, w_in).
                            // → ow >= pw - fw  and  ow < w_in + pw - fw
                            let ow_valid_lo = pw.saturating_sub(fw);
                            let ow_valid_hi = (w_in + pw - fw).min(w_out);
                            let ow_end = ow_start + row_remaining;

                            // Leading zeros (left padding).
                            if ow_start < ow_valid_lo {
                                let zero_end = ow_valid_lo.min(ow_end);
                                let zlen = zero_end - ow_start;
                                col_row[t..t + zlen].fill(0.0);
                                t += zlen;
                                if t >= tile_len
                                    || ow_start + (t - (tile_start + out_pos - ow_start)) >= ow_end
                                {
                                    continue;
                                }
                            }

                            // Interior: contiguous copy from data.
                            let cur_ow = ow_start + (t - (tile_start + out_pos - ow_start));
                            if cur_ow < ow_valid_hi && cur_ow < ow_end {
                                let copy_end = ow_valid_hi.min(ow_end);
                                let copy_len = copy_end - cur_ow;
                                let iw_start = cur_ow + fw - pw;
                                let src_start = d_channel_base + ih_actual * w_in + iw_start;
                                let src_end = src_start + copy_len;
                                if src_end <= data.len() {
                                    col_row[t..t + copy_len]
                                        .copy_from_slice(&data[src_start..src_end]);
                                } else {
                                    // Fallback: element-wise with bounds check.
                                    for i in 0..copy_len {
                                        let idx = src_start + i;
                                        col_row[t + i] =
                                            if idx < data.len() { data[idx] } else { 0.0 };
                                    }
                                }
                                t += copy_len;
                            }

                            // Trailing zeros (right padding).
                            let final_ow = ow_start + (t - (tile_start + out_pos - ow_start));
                            if final_ow < ow_end {
                                let zlen = ow_end - final_ow;
                                col_row[t..t + zlen].fill(0.0);
                                t += zlen;
                            }
                        }
                    } else {
                        // General path: per-element with division and bounds checks.
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
                }

                // Phase 2: GEMM — W[oc_per_group, kernel_size] × col[kernel_size, tile_len].
                if let Some(ref qw) = qw {
                    // LUT-GEMM Q4: transpose col → col_t_buf, GEMM → lut_out_buf.
                    // Both buffers are pre-allocated, zero per-tile allocation.
                    let col_t_len = tile_len * kernel_size;
                    let lut_out_len = tile_len * oc_per_group;
                    // Transpose col[K, tile_len] → col_t_buf[tile_len, K].
                    for t in 0..tile_len {
                        for k in 0..kernel_size {
                            col_t_buf[t * kernel_size + k] = col[k * tile_len + t];
                        }
                    }
                    lut_out_buf[..lut_out_len].fill(0.0);
                    #[cfg(feature = "parallel")]
                    crate::lut_gemm::lut_gemm_4bit_par(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );
                    #[cfg(not(feature = "parallel"))]
                    crate::lut_gemm::lut_gemm_4bit(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );
                    // Scatter from [tile_len, oc] directly into output.
                    for t in 0..tile_len {
                        for oc_idx in 0..oc_per_group {
                            let o_pos = o_base + oc_idx * spatial_out + tile_start + t;
                            out[o_pos] += lut_out_buf[t * oc_per_group + oc_idx];
                        }
                    }
                    tile_start = tile_end;
                    continue; // Skip the f32 GEMM + scatter below.
                } else {
                    let to_len = oc_per_group * tile_len;
                    tile_out[..to_len].fill(0.0);
                    #[cfg(all(feature = "accelerate", target_os = "macos"))]
                    {
                        super::matmul::blas::sgemm_full(
                            super::matmul::GemmParams {
                                m: oc_per_group,
                                n: tile_len,
                                k: kernel_size,
                                alpha: 1.0,
                                beta: 0.0,
                                trans_a: false,
                                trans_b: false,
                            },
                            w_slice,
                            &col[..kernel_size * tile_len],
                            &mut tile_out[..to_len],
                        );
                    }

                    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                    {
                        super::matmul::matmul_k_outer(
                            w_slice,
                            &col[..kernel_size * tile_len],
                            &mut tile_out[..to_len],
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
pub(crate) fn dispatch_conv2d_direct(
    inputs: &[&[u8]],
    attrs: Conv2dAttrs,
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    let Conv2dAttrs {
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        group,
    } = attrs;
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    if data.is_empty() || weight.is_empty() || h_in == 0 || w_in == 0 {
        return Ok(vec![]);
    }
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let bias = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
        Some(cast_f32(bias_bytes)?)
    } else {
        None
    };

    // Trust the passed-in h_in/w_in — these come from resolve_spatial_dims()
    // which prefers runtime TensorMeta (propagated via InOutBufWithMeta) over
    // compiled values. Only fall back to heuristic derivation when both are 0.
    let (h_in, w_in) = if h_in > 0 && w_in > 0 {
        (h_in, w_in)
    } else {
        // Last resort: derive square spatial dims from total elements.
        let total = data.len();
        let side = (total as f64).sqrt() as usize;
        if side > 0 && side * side == total {
            (side, side)
        } else {
            (1, total.max(1))
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
        Conv2dCoreCall::new(
            &data,
            &weight,
            Conv2dInputShape::new(n, ic_per_group, h_in, w_in),
            attrs,
        )
        .with_bias(bias.as_deref())
        .with_output(oc, h_out, w_out),
    );
    Ok(f32_vec_to_bytes(out))
}

/// Conv2d with explicit input shapes from shape vectors (used by KvStore path).
///
/// Delegates to `dispatch_conv2d_direct` after extracting H/W from shapes.
pub(crate) fn dispatch_conv2d_with_shapes(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    attrs: Conv2dAttrs,
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
    dispatch_conv2d_direct(inputs, attrs, h_in, w_in)
}

/// Conv2d with pre-quantized 4-bit LUT-GEMM weights (compile-time quantized).
///
/// The weight quantization + transpose was done at compile time. At runtime:
/// 1. im2col: gather input patches → col[kernel_size, tile_len]
/// 2. Transpose col → col_t[tile_len, kernel_size]
/// 3. LUT-GEMM: col_t × pre_quantized_weights → [tile_len, oc_per_group]
/// 4. Scatter to output
///
/// Zero quantization/transpose overhead at runtime.
pub(crate) fn dispatch_conv2d_lut4(
    inputs: &[&[u8]],
    cid: hologram_graph::constant::ConstantId,
    tape_ctx: &crate::tape::TapeContext<'_>,
    attrs: Conv2dAttrs,
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    // On macOS with Accelerate, BLAS sgemm is faster than LUT-GEMM Q4.
    // Fall back to the f32 BLAS path — the pre-quantized weights are still in the
    // archive for non-BLAS targets (WASM, Linux without MKL).
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        let _ = (cid, tape_ctx);
        dispatch_conv2d_direct(inputs, attrs, h_in, w_in)
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        let Conv2dAttrs {
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            group,
        } = attrs;
        let data = cast_f32(inputs[0])?;
        let weight = cast_f32(inputs[1])?;
        if data.is_empty() || weight.is_empty() || h_in == 0 || w_in == 0 {
            return Ok(vec![]);
        }
        let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
        let bias = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
            Some(cast_f32(bias_bytes)?)
        } else {
            None
        };

        // Derive N, IC, OC from spatial dims and buffer lengths.
        // data is [N, IC, H, W], weight is [OC, IC/group, KH, KW].
        let h_in = h_in.max(1);
        let w_in = w_in.max(1);
        let spatial = h_in * w_in;
        let ic = data.len().checked_div(spatial).unwrap_or(1);
        let n = ic
            .checked_mul(spatial)
            .and_then(|denom| data.len().checked_div(denom))
            .unwrap_or(1);
        let ic_per_group = (ic / group.max(1)).max(1);
        let kernel_size = ic_per_group * kh * kw;
        let oc = weight.len().checked_div(kernel_size).unwrap_or(1);
        let oc_per_group = oc / group.max(1);
        let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        let spatial_out = h_out * w_out;

        // Resolve pre-quantized weights from constant store.
        let mut cache = tape_ctx.weight_cache.write();
        let qw = cache.get_q4(cid, tape_ctx.constants, tape_ctx.weights)?;

        // Validate quantized weight dimensions match runtime-derived dimensions.
        // If mismatched, fall back to the non-quantized path.
        if qw.rows as usize != kernel_size || qw.cols as usize != oc_per_group {
            tracing::warn!(
                qw_rows = qw.rows,
                qw_cols = qw.cols,
                kernel_size,
                oc_per_group,
                "Conv2dLut4 dimension mismatch — falling back to f32 path"
            );
            drop(cache);
            return dispatch_conv2d_direct(inputs, attrs, h_in, w_in);
        }

        let mut out = vec![0.0f32; n * oc * spatial_out];

        // Tiled im2col (same tile sizing as conv2d_core).
        const TILE_CAP: usize = 4 * 1024 * 1024; // 16 MB as f32
        let tile_size = (TILE_CAP.checked_div(kernel_size).unwrap_or(spatial_out))
            .max(1)
            .min(spatial_out);
        let mut col = vec![0.0f32; kernel_size * tile_size];
        // Pre-allocate transpose + output buffers — reused across all tiles.
        let mut col_t_buf = vec![0.0f32; tile_size * kernel_size];
        let mut lut_out_buf = vec![0.0f32; tile_size * oc_per_group];

        for batch in 0..n {
            for g in 0..group {
                let o_base = batch * oc * spatial_out + g * oc_per_group * spatial_out;

                // Initialize output with bias.
                if let Some(ref b) = bias {
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

                let mut tile_start = 0;
                while tile_start < spatial_out {
                    let tile_end = (tile_start + tile_size).min(spatial_out);
                    let tile_len = tile_end - tile_start;

                    // Phase 1: im2col for this tile.
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

                    // Phase 2: Transpose col → col_t_buf and LUT-GEMM → lut_out_buf.
                    let col_t_len = tile_len * kernel_size;
                    let lut_out_len = tile_len * oc_per_group;
                    for t in 0..tile_len {
                        for k in 0..kernel_size {
                            col_t_buf[t * kernel_size + k] = col[k * tile_len + t];
                        }
                    }
                    lut_out_buf[..lut_out_len].fill(0.0);
                    #[cfg(feature = "parallel")]
                    crate::lut_gemm::lut_gemm_4bit_par(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );
                    #[cfg(not(feature = "parallel"))]
                    crate::lut_gemm::lut_gemm_4bit(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );

                    // Scatter from [tile_len, oc] to output [oc, spatial_out].
                    for t in 0..tile_len {
                        for oc_idx in 0..oc_per_group {
                            let o_pos = o_base + oc_idx * spatial_out + tile_start + t;
                            out[o_pos] += lut_out_buf[t * oc_per_group + oc_idx];
                        }
                    }

                    tile_start = tile_end;
                }
            }
        }

        Ok(f32_vec_to_bytes(out))
    } // #[cfg(not(accelerate + macos))]
}
