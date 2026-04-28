//! Canonical `Conv2d` op — semantic identity, executable form, and CPU
//! reference kernel.
//!
//! Direct (no im2col) reference for correctness only. NCHW layout,
//! weight shape `[C_out, C_in/group, kH, kW]`, optional bias of length
//! `C_out`. When `call.bias.len == 0` the kernel treats the bias as
//! zero.

use crate::attrs::Conv2dAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical `conv2d` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conv2d(pub Conv2dAttrs);

impl Op for Conv2d {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "conv2d"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Convolution
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::Conv2dBackward)
    }
}

/// Pre-resolved arguments for a 2-D convolution (direct reference).
///
/// Layout convention: NCHW. Input is `[N, C_in, H_in, W_in]`, weight is
/// `[C_out, C_in/group, kH, kW]`, bias is `[C_out]`, output is
/// `[N, C_out, H_out, W_out]`. The planner derives `n`, `h_out`,
/// `w_out`, `c_in`, `c_out` from the chain's tensor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dCall {
    /// Input data span.
    pub input: SlotSpan,
    /// Weight span.
    pub weight: SlotSpan,
    /// Bias span (zero-length if absent — kernel treats as zeros).
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

/// Forward: `out[n, oc, h, w] = bias[oc] + Σ input * weight`.
pub fn conv2d(storage: &mut [f32], call: &Conv2dCall) {
    let n = call.n as usize;
    let c_in = call.c_in as usize;
    let c_out = call.c_out as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let h_out = call.h_out as usize;
    let w_out = call.w_out as usize;
    let kh = call.kernel_h as usize;
    let kw = call.kernel_w as usize;
    let group = call.group.max(1) as usize;
    debug_assert_eq!(c_in % group, 0);
    debug_assert_eq!(c_out % group, 0);
    let c_in_per_g = c_in / group;
    let c_out_per_g = c_out / group;

    let bias_present = call.bias.len > 0;

    let in_chw = c_in * h_in * w_in;
    let out_chw = c_out * h_out * w_out;
    let weight_per_oc = c_in_per_g * kh * kw;

    for ni in 0..n {
        for g in 0..group {
            for oc_local in 0..c_out_per_g {
                let oc = g * c_out_per_g + oc_local;
                let bias = if bias_present {
                    storage[call.bias.offset + oc]
                } else {
                    0.0
                };
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = bias;
                        for ic_local in 0..c_in_per_g {
                            let ic = g * c_in_per_g + ic_local;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ih = (oh * call.stride_h as usize) as isize
                                        + (ky * call.dilation_h as usize) as isize
                                        - call.pad_h as isize;
                                    let iw = (ow * call.stride_w as usize) as isize
                                        + (kx * call.dilation_w as usize) as isize
                                        - call.pad_w as isize;
                                    if ih < 0
                                        || iw < 0
                                        || ih >= h_in as isize
                                        || iw >= w_in as isize
                                    {
                                        continue;
                                    }
                                    let in_idx = call.input.offset
                                        + ni * in_chw
                                        + ic * h_in * w_in
                                        + ih as usize * w_in
                                        + iw as usize;
                                    let w_idx = call.weight.offset
                                        + oc * weight_per_oc
                                        + ic_local * kh * kw
                                        + ky * kw
                                        + kx;
                                    acc += storage[in_idx] * storage[w_idx];
                                }
                            }
                        }
                        let out_idx = call.output.offset
                            + ni * out_chw
                            + oc * h_out * w_out
                            + oh * w_out
                            + ow;
                        storage[out_idx] = acc;
                    }
                }
            }
        }
    }
}

/// Pre-resolved arguments for `Conv2d` backward.
///
/// Same geometry as `Conv2dCall`. Accumulates into `dx`, `dw`, `db`.
/// Any of the gradient slots may be empty (`len == 0`) to skip its
/// accumulation — useful when only a subset of inputs requires grad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dGradCall {
    /// Forward input.
    pub input: SlotSpan,
    /// Forward weight.
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

/// Backward of `conv2d`. Accumulates into `dx`, `dw`, `db`.
pub fn conv2d_grad(storage: &mut [f32], call: &Conv2dGradCall) {
    let n = call.n as usize;
    let c_in = call.c_in as usize;
    let c_out = call.c_out as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let h_out = call.h_out as usize;
    let w_out = call.w_out as usize;
    let kh = call.kernel_h as usize;
    let kw = call.kernel_w as usize;
    let group = call.group.max(1) as usize;
    debug_assert_eq!(c_in % group, 0);
    debug_assert_eq!(c_out % group, 0);
    let c_in_per_g = c_in / group;
    let c_out_per_g = c_out / group;

    let in_chw = c_in * h_in * w_in;
    let out_chw = c_out * h_out * w_out;
    let weight_per_oc = c_in_per_g * kh * kw;
    let want_dx = call.dx.len > 0;
    let want_dw = call.dw.len > 0;
    let want_db = call.db.len > 0;

    for ni in 0..n {
        for g in 0..group {
            for oc_local in 0..c_out_per_g {
                let oc = g * c_out_per_g + oc_local;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let dy_idx =
                            call.dy.offset + ni * out_chw + oc * h_out * w_out + oh * w_out + ow;
                        let dy = storage[dy_idx];
                        if want_db {
                            storage[call.db.offset + oc] += dy;
                        }
                        for ic_local in 0..c_in_per_g {
                            let ic = g * c_in_per_g + ic_local;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ih = (oh * call.stride_h as usize) as isize
                                        + (ky * call.dilation_h as usize) as isize
                                        - call.pad_h as isize;
                                    let iw = (ow * call.stride_w as usize) as isize
                                        + (kx * call.dilation_w as usize) as isize
                                        - call.pad_w as isize;
                                    if ih < 0
                                        || iw < 0
                                        || ih >= h_in as isize
                                        || iw >= w_in as isize
                                    {
                                        continue;
                                    }
                                    let in_idx = ni * in_chw
                                        + ic * h_in * w_in
                                        + ih as usize * w_in
                                        + iw as usize;
                                    let w_idx_local =
                                        oc * weight_per_oc + ic_local * kh * kw + ky * kw + kx;
                                    if want_dx {
                                        let w_val = storage[call.weight.offset + w_idx_local];
                                        storage[call.dx.offset + in_idx] += dy * w_val;
                                    }
                                    if want_dw {
                                        let in_val = storage[call.input.offset + in_idx];
                                        storage[call.dw.offset + w_idx_local] += dy * in_val;
                                    }
                                }
                            }
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

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    #[test]
    fn conv2d_identity_kernel_passes_through_input() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let call = Conv2dCall {
            input: span(0, 9),
            weight: span(9, 1),
            bias: span(10, 1),
            output: span(11, 9),
            n: 1,
            c_in: 1,
            c_out: 1,
            h_in: 3,
            w_in: 3,
            h_out: 3,
            w_out: 3,
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
        conv2d(&mut s, &call);
        assert_eq!(&s[11..20], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn conv2d_3x3_average_kernel_with_padding() {
        let mut s = vec![0.0_f32; 9 + 9 + 1 + 9];
        for (i, slot) in s.iter_mut().take(9).enumerate() {
            *slot = (i + 1) as f32;
        }
        for slot in s.iter_mut().skip(9).take(9) {
            *slot = 1.0 / 9.0;
        }
        let call = Conv2dCall {
            input: span(0, 9),
            weight: span(9, 9),
            bias: span(18, 1),
            output: span(19, 9),
            n: 1,
            c_in: 1,
            c_out: 1,
            h_in: 3,
            w_in: 3,
            h_out: 3,
            w_out: 3,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        };
        conv2d(&mut s, &call);
        assert!((s[19 + 4] - 5.0).abs() < 1e-5);
        for v in &s[19..28] {
            assert!(*v > 0.0);
        }
    }

    fn run_conv2d(x: &[f32], w: &[f32], b: &[f32], geom: ConvGeom) -> Vec<f32> {
        let xn = x.len();
        let wn = w.len();
        let bn = b.len();
        let out_n = (geom.n * geom.c_out * geom.h_out * geom.w_out) as usize;
        let mut s = vec![0.0_f32; xn + wn + bn + out_n];
        s[..xn].copy_from_slice(x);
        s[xn..xn + wn].copy_from_slice(w);
        s[xn + wn..xn + wn + bn].copy_from_slice(b);
        let call = Conv2dCall {
            input: span(0, xn),
            weight: span(xn, wn),
            bias: span(xn + wn, bn),
            output: span(xn + wn + bn, out_n),
            n: geom.n,
            c_in: geom.c_in,
            c_out: geom.c_out,
            h_in: geom.h_in,
            w_in: geom.w_in,
            h_out: geom.h_out,
            w_out: geom.w_out,
            kernel_h: geom.kernel_h,
            kernel_w: geom.kernel_w,
            stride_h: geom.stride_h,
            stride_w: geom.stride_w,
            pad_h: geom.pad_h,
            pad_w: geom.pad_w,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        };
        conv2d(&mut s, &call);
        s[xn + wn + bn..].to_vec()
    }

    #[derive(Clone, Copy)]
    struct ConvGeom {
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

    #[test]
    fn conv2d_grad_matches_finite_difference() {
        let geom = ConvGeom {
            n: 1,
            c_in: 2,
            c_out: 2,
            h_in: 4,
            w_in: 4,
            h_out: 4,
            w_out: 4,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
        };
        let xn = (geom.n * geom.c_in * geom.h_in * geom.w_in) as usize;
        let wn = (geom.c_out * geom.c_in * geom.kernel_h * geom.kernel_w) as usize;
        let bn = geom.c_out as usize;
        let yn = (geom.n * geom.c_out * geom.h_out * geom.w_out) as usize;

        // Deterministic-ish small inputs.
        let x: Vec<f32> = (0..xn).map(|i| ((i as f32) * 0.13 - 1.5).sin()).collect();
        let w: Vec<f32> = (0..wn).map(|i| ((i as f32) * 0.07).cos()).collect();
        let b: Vec<f32> = (0..bn).map(|i| 0.1 * i as f32).collect();
        let dy: Vec<f32> = (0..yn).map(|i| ((i as f32) * 0.21).sin()).collect();

        // Analytic.
        let mut s = vec![0.0_f32; xn + wn + bn + yn + xn + wn + bn];
        s[..xn].copy_from_slice(&x);
        s[xn..xn + wn].copy_from_slice(&w);
        s[xn + wn..xn + wn + bn].copy_from_slice(&b);
        s[xn + wn + bn..xn + wn + bn + yn].copy_from_slice(&dy);
        let dx_off = xn + wn + bn + yn;
        let dw_off = dx_off + xn;
        let db_off = dw_off + wn;
        let call = Conv2dGradCall {
            input: span(0, xn),
            weight: span(xn, wn),
            dy: span(xn + wn + bn, yn),
            dx: span(dx_off, xn),
            dw: span(dw_off, wn),
            db: span(db_off, bn),
            n: geom.n,
            c_in: geom.c_in,
            c_out: geom.c_out,
            h_in: geom.h_in,
            w_in: geom.w_in,
            h_out: geom.h_out,
            w_out: geom.w_out,
            kernel_h: geom.kernel_h,
            kernel_w: geom.kernel_w,
            stride_h: geom.stride_h,
            stride_w: geom.stride_w,
            pad_h: geom.pad_h,
            pad_w: geom.pad_w,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        };
        conv2d_grad(&mut s, &call);
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
            let fd = (dot(&run_conv2d(&xp, &w, &b, geom)) - dot(&run_conv2d(&xm, &w, &b, geom)))
                / (2.0 * h);
            assert!(
                (dx[i] - fd).abs() < 5e-2,
                "dx[{}]: got {}, fd {}",
                i,
                dx[i],
                fd
            );
        }
        for i in 0..wn {
            let mut wp = w.clone();
            wp[i] += h;
            let mut wm = w.clone();
            wm[i] -= h;
            let fd = (dot(&run_conv2d(&x, &wp, &b, geom)) - dot(&run_conv2d(&x, &wm, &b, geom)))
                / (2.0 * h);
            assert!(
                (dw[i] - fd).abs() < 5e-2,
                "dw[{}]: got {}, fd {}",
                i,
                dw[i],
                fd
            );
        }
        for i in 0..bn {
            let mut bp = b.clone();
            bp[i] += h;
            let mut bm = b.clone();
            bm[i] -= h;
            let fd = (dot(&run_conv2d(&x, &w, &bp, geom)) - dot(&run_conv2d(&x, &w, &bm, geom)))
                / (2.0 * h);
            assert!(
                (db[i] - fd).abs() < 5e-2,
                "db[{}]: got {}, fd {}",
                i,
                db[i],
                fd
            );
        }
    }

    #[test]
    fn conv2d_with_bias_adds_per_output_channel() {
        let mut s = [1.0_f32, 3.0, 5.0, 10.0, 20.0, 0.0, 0.0];
        let call = Conv2dCall {
            input: span(0, 1),
            weight: span(1, 2),
            bias: span(3, 2),
            output: span(5, 2),
            n: 1,
            c_in: 1,
            c_out: 2,
            h_in: 1,
            w_in: 1,
            h_out: 1,
            w_out: 1,
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
        conv2d(&mut s, &call);
        assert_eq!(s[5], 1.0 * 3.0 + 10.0);
        assert_eq!(s[6], 1.0 * 5.0 + 20.0);
    }
}
