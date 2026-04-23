use super::conv::Conv2dAttrs;
use super::helpers::*;
use crate::error::ExecResult;

// ── ConvTranspose ────────────────────────────────────────────────────────────

/// Output padding for [`dispatch_conv_transpose`]. Default is `(0, 0)`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConvTransposeOutputPad {
    pub h: usize,
    pub w: usize,
}

impl ConvTransposeOutputPad {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_hw(mut self, h: usize, w: usize) -> Self {
        self.h = h;
        self.w = w;
        self
    }
}

/// Transposed 2-D convolution with explicit spatial dimensions.
pub(crate) fn dispatch_conv_transpose(
    inputs: &[&[u8]],
    attrs: Conv2dAttrs,
    output_pad: ConvTransposeOutputPad,
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
    let ConvTransposeOutputPad {
        h: output_pad_h,
        w: output_pad_w,
    } = output_pad;
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
