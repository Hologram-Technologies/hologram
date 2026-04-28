//! Canonical `ConvTranspose` op (2-D transposed convolution) —
//! semantic identity, executable form, and CPU reference kernel.
//!
//! Reference behaviour: NCHW input, weight shape
//! `[C_in, C_out/group, kH, kW]` (note the swap from `Conv2d`'s
//! `[C_out, C_in/group, kH, kW]`), optional bias of length `C_out`.
//! Direct fold-and-scatter formulation — for each input position the
//! kernel scatters its weighted contribution into the output. The
//! planner derives shape from chain tensor dims.

use crate::attrs::ConvTransposeAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Pre-resolved arguments for `conv_transpose_2d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTransposeCall {
    /// Input data span.
    pub input: SlotSpan,
    /// Weight span (`[C_in, C_out/group, kH, kW]`).
    pub weight: SlotSpan,
    /// Bias span (zero-length if absent).
    pub bias: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Input channels.
    pub c_in: u32,
    /// Output channels.
    pub c_out: u32,
    /// Input height.
    pub h_in: u32,
    /// Input width.
    pub w_in: u32,
    /// Output height.
    pub h_out: u32,
    /// Output width.
    pub w_out: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Vertical padding.
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
    /// Vertical dilation.
    pub dilation_h: u32,
    /// Horizontal dilation.
    pub dilation_w: u32,
    /// Group count.
    pub group: u32,
}

/// Marker struct for the canonical `conv_transpose_2d` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvTranspose2d(pub ConvTransposeAttrs);

impl Op for ConvTranspose2d {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "conv_transpose_2d"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Convolution
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::ConvTranspose2dBackward)
    }
}

/// Forward: scatter-style transposed 2-D convolution.
pub fn conv_transpose_2d(storage: &mut [f32], call: &ConvTransposeCall) {
    let n = call.n as usize;
    let c_in = call.c_in as usize;
    let c_out = call.c_out as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let h_out = call.h_out as usize;
    let w_out = call.w_out as usize;
    let kh = call.kernel_h as usize;
    let kw = call.kernel_w as usize;
    let sh = call.stride_h as usize;
    let sw = call.stride_w as usize;
    let ph = call.pad_h as isize;
    let pw = call.pad_w as isize;
    let dh = call.dilation_h as usize;
    let dw = call.dilation_w as usize;
    let group = call.group.max(1) as usize;
    let c_in_per_g = c_in / group;
    let c_out_per_g = c_out / group;
    let in_chw = c_in * h_in * w_in;
    let out_chw = c_out * h_out * w_out;
    let weight_per_ic = c_out_per_g * kh * kw;

    let bias_present = call.bias.len > 0;

    // Initialise output (with bias broadcast across spatial dims if present).
    for ni in 0..n {
        for oc in 0..c_out {
            let bias = if bias_present {
                storage[call.bias.offset + oc]
            } else {
                0.0
            };
            let plane_off = call.output.offset + ni * out_chw + oc * h_out * w_out;
            for i in 0..h_out * w_out {
                storage[plane_off + i] = bias;
            }
        }
    }

    // Scatter input contributions into output.
    for ni in 0..n {
        for g in 0..group {
            for ic_local in 0..c_in_per_g {
                let ic = g * c_in_per_g + ic_local;
                for ih in 0..h_in {
                    for iw in 0..w_in {
                        let in_val = storage
                            [call.input.offset + ni * in_chw + ic * h_in * w_in + ih * w_in + iw];
                        for oc_local in 0..c_out_per_g {
                            let oc = g * c_out_per_g + oc_local;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oh = ih * sh + ky * dh;
                                    let ow = iw * sw + kx * dw;
                                    let oh_padded = oh as isize - ph;
                                    let ow_padded = ow as isize - pw;
                                    if oh_padded < 0
                                        || ow_padded < 0
                                        || oh_padded >= h_out as isize
                                        || ow_padded >= w_out as isize
                                    {
                                        continue;
                                    }
                                    let w_idx = call.weight.offset
                                        + ic * weight_per_ic
                                        + oc_local * kh * kw
                                        + ky * kw
                                        + kx;
                                    let out_idx = call.output.offset
                                        + ni * out_chw
                                        + oc * h_out * w_out
                                        + oh_padded as usize * w_out
                                        + ow_padded as usize;
                                    storage[out_idx] += in_val * storage[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Pre-resolved arguments for `ConvTranspose2d` backward.
///
/// Same geometry as `ConvTransposeCall`. Accumulates into `dx`,
/// `dw`, `db`. Empty grad slots are skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dGradCall {
    /// Forward input.
    pub input: SlotSpan,
    /// Forward weight (`[C_in, C_out/group, kH, kW]`).
    pub weight: SlotSpan,
    /// Upstream gradient `dy` (length = forward output).
    pub dy: SlotSpan,
    /// Gradient slot for input.
    pub dx: SlotSpan,
    /// Gradient slot for weight.
    pub dw: SlotSpan,
    /// Gradient slot for bias (length = `c_out` if present).
    pub db: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Input channels.
    pub c_in: u32,
    /// Output channels.
    pub c_out: u32,
    /// Input height.
    pub h_in: u32,
    /// Input width.
    pub w_in: u32,
    /// Output height.
    pub h_out: u32,
    /// Output width.
    pub w_out: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Vertical stride.
    pub stride_h: u32,
    /// Horizontal stride.
    pub stride_w: u32,
    /// Vertical padding.
    pub pad_h: u32,
    /// Horizontal padding.
    pub pad_w: u32,
    /// Vertical dilation.
    pub dilation_h: u32,
    /// Horizontal dilation.
    pub dilation_w: u32,
    /// Group count.
    pub group: u32,
}

/// Backward of `conv_transpose_2d`. Walks the same index space as the
/// forward scatter; `dx` and `dw` accumulate per-iteration. `db` is
/// computed in a separate pass over the output to avoid the
/// many-to-one bias-broadcast double-counting that the forward loop
/// would otherwise introduce.
pub fn conv_transpose_2d_grad(storage: &mut [f32], call: &ConvTranspose2dGradCall) {
    let n = call.n as usize;
    let c_in = call.c_in as usize;
    let c_out = call.c_out as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let h_out = call.h_out as usize;
    let w_out = call.w_out as usize;
    let kh = call.kernel_h as usize;
    let kw = call.kernel_w as usize;
    let sh = call.stride_h as usize;
    let sw = call.stride_w as usize;
    let ph = call.pad_h as isize;
    let pw = call.pad_w as isize;
    let dh = call.dilation_h as usize;
    let dwd = call.dilation_w as usize;
    let group = call.group.max(1) as usize;
    let c_in_per_g = c_in / group;
    let c_out_per_g = c_out / group;
    let in_chw = c_in * h_in * w_in;
    let out_chw = c_out * h_out * w_out;
    let weight_per_ic = c_out_per_g * kh * kw;
    let want_dx = call.dx.len > 0;
    let want_dw = call.dw.len > 0;
    let want_db = call.db.len > 0;

    if want_db {
        for ni in 0..n {
            for oc in 0..c_out {
                let plane = call.dy.offset + ni * out_chw + oc * h_out * w_out;
                let mut acc = 0.0_f32;
                for i in 0..h_out * w_out {
                    acc += storage[plane + i];
                }
                storage[call.db.offset + oc] += acc;
            }
        }
    }

    if !want_dx && !want_dw {
        return;
    }

    for ni in 0..n {
        for g in 0..group {
            for ic_local in 0..c_in_per_g {
                let ic = g * c_in_per_g + ic_local;
                for ih in 0..h_in {
                    for iw in 0..w_in {
                        let in_idx = ni * in_chw + ic * h_in * w_in + ih * w_in + iw;
                        let in_val = if want_dw {
                            storage[call.input.offset + in_idx]
                        } else {
                            0.0
                        };
                        let mut dx_acc = 0.0_f32;
                        for oc_local in 0..c_out_per_g {
                            let oc = g * c_out_per_g + oc_local;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oh = ih * sh + ky * dh;
                                    let ow = iw * sw + kx * dwd;
                                    let oh_padded = oh as isize - ph;
                                    let ow_padded = ow as isize - pw;
                                    if oh_padded < 0
                                        || ow_padded < 0
                                        || oh_padded >= h_out as isize
                                        || ow_padded >= w_out as isize
                                    {
                                        continue;
                                    }
                                    let w_idx_local =
                                        ic * weight_per_ic + oc_local * kh * kw + ky * kw + kx;
                                    let dy_idx = ni * out_chw
                                        + oc * h_out * w_out
                                        + oh_padded as usize * w_out
                                        + ow_padded as usize;
                                    let dy = storage[call.dy.offset + dy_idx];
                                    if want_dx {
                                        let w_val = storage[call.weight.offset + w_idx_local];
                                        dx_acc += dy * w_val;
                                    }
                                    if want_dw {
                                        storage[call.dw.offset + w_idx_local] += in_val * dy;
                                    }
                                }
                            }
                        }
                        if want_dx {
                            storage[call.dx.offset + in_idx] += dx_acc;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_transpose_identity_kernel_passes_through() {
        // 1×1×2×2 input, 1×1×1×1 weight=1, no bias → 1×1×2×2 output equal to input.
        let mut s = vec![0.0_f32; 4 + 1 + 4];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        s[4] = 1.0;
        let call = ConvTransposeCall {
            input: SlotSpan { offset: 0, len: 4 },
            weight: SlotSpan { offset: 4, len: 1 },
            bias: SlotSpan::empty(0),
            output: SlotSpan { offset: 5, len: 4 },
            n: 1,
            c_in: 1,
            c_out: 1,
            h_in: 2,
            w_in: 2,
            h_out: 2,
            w_out: 2,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        };
        conv_transpose_2d(&mut s, &call);
        assert_eq!(&s[5..9], &[1.0, 2.0, 3.0, 4.0]);
    }

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    #[derive(Clone, Copy)]
    struct Geom {
        n: u32,
        c_in: u32,
        c_out: u32,
        h_in: u32,
        w_in: u32,
        h_out: u32,
        w_out: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    }

    fn run_forward(x: &[f32], w: &[f32], b: &[f32], g: Geom) -> Vec<f32> {
        let xn = x.len();
        let wn = w.len();
        let bn = b.len();
        let yn = (g.n * g.c_out * g.h_out * g.w_out) as usize;
        let mut s = vec![0.0_f32; xn + wn + bn + yn];
        s[..xn].copy_from_slice(x);
        s[xn..xn + wn].copy_from_slice(w);
        s[xn + wn..xn + wn + bn].copy_from_slice(b);
        let call = ConvTransposeCall {
            input: span(0, xn),
            weight: span(xn, wn),
            bias: span(xn + wn, bn),
            output: span(xn + wn + bn, yn),
            n: g.n,
            c_in: g.c_in,
            c_out: g.c_out,
            h_in: g.h_in,
            w_in: g.w_in,
            h_out: g.h_out,
            w_out: g.w_out,
            kernel_h: g.kernel_h,
            kernel_w: g.kernel_w,
            stride_h: g.stride_h,
            stride_w: g.stride_w,
            pad_h: g.pad_h,
            pad_w: g.pad_w,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        };
        conv_transpose_2d(&mut s, &call);
        s[xn + wn + bn..].to_vec()
    }

    #[test]
    fn conv_transpose_2d_grad_matches_finite_difference() {
        // 1×1×2×2 input, 3×3 kernel, stride 2, no pad → 1×1×5×5 output.
        let g = Geom {
            n: 1,
            c_in: 1,
            c_out: 1,
            h_in: 2,
            w_in: 2,
            h_out: 5,
            w_out: 5,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        let xn = 4;
        let wn = 9;
        let bn = 1;
        let yn = 25;
        let x: Vec<f32> = (0..xn).map(|i| ((i as f32) * 0.31 - 0.4).sin()).collect();
        let w: Vec<f32> = (0..wn).map(|i| ((i as f32) * 0.17).cos()).collect();
        let b: Vec<f32> = (0..bn).map(|i| 0.05 + 0.1 * i as f32).collect();
        let dy: Vec<f32> = (0..yn).map(|i| ((i as f32) * 0.11).sin()).collect();

        let mut s = vec![0.0_f32; xn + wn + bn + yn + xn + wn + bn];
        s[..xn].copy_from_slice(&x);
        s[xn..xn + wn].copy_from_slice(&w);
        s[xn + wn..xn + wn + bn].copy_from_slice(&b);
        s[xn + wn + bn..xn + wn + bn + yn].copy_from_slice(&dy);
        let dx_off = xn + wn + bn + yn;
        let dw_off = dx_off + xn;
        let db_off = dw_off + wn;
        let call = ConvTranspose2dGradCall {
            input: span(0, xn),
            weight: span(xn, wn),
            dy: span(xn + wn + bn, yn),
            dx: span(dx_off, xn),
            dw: span(dw_off, wn),
            db: span(db_off, bn),
            n: g.n,
            c_in: g.c_in,
            c_out: g.c_out,
            h_in: g.h_in,
            w_in: g.w_in,
            h_out: g.h_out,
            w_out: g.w_out,
            kernel_h: g.kernel_h,
            kernel_w: g.kernel_w,
            stride_h: g.stride_h,
            stride_w: g.stride_w,
            pad_h: g.pad_h,
            pad_w: g.pad_w,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        };
        conv_transpose_2d_grad(&mut s, &call);
        let dx = s[dx_off..dx_off + xn].to_vec();
        let dw = s[dw_off..dw_off + wn].to_vec();
        let db = s[db_off..db_off + bn].to_vec();

        let dot = |y: &[f32]| -> f32 { y.iter().zip(dy.iter()).map(|(a, b)| a * b).sum() };
        let h = 1e-3_f32;
        for i in 0..xn {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let fd =
                (dot(&run_forward(&xp, &w, &b, g)) - dot(&run_forward(&xm, &w, &b, g))) / (2.0 * h);
            assert!((dx[i] - fd).abs() < 5e-2);
        }
        for i in 0..wn {
            let mut wp = w.clone();
            wp[i] += h;
            let mut wm = w.clone();
            wm[i] -= h;
            let fd =
                (dot(&run_forward(&x, &wp, &b, g)) - dot(&run_forward(&x, &wm, &b, g))) / (2.0 * h);
            assert!((dw[i] - fd).abs() < 5e-2);
        }
        for i in 0..bn {
            let mut bp = b.clone();
            bp[i] += h;
            let mut bm = b.clone();
            bm[i] -= h;
            let fd =
                (dot(&run_forward(&x, &w, &bp, g)) - dot(&run_forward(&x, &w, &bm, g))) / (2.0 * h);
            assert!((db[i] - fd).abs() < 5e-2);
        }
    }
}
