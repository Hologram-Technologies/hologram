//! Canonical `Resize` op (nearest / bilinear modes) — semantic
//! identity, executable form, and CPU reference kernels.
//!
//! Reference behaviour: NCHW resize from `[h_in, w_in]` to
//! `[h_out, w_out]`. Two interpolation modes ship in canonical:
//!
//! - **0 — nearest**: nearest-neighbor sampling (round half-away
//!   from zero, ONNX `half_pixel` coordinate transform).
//! - **1 — linear**: bilinear interpolation between the four
//!   surrounding input pixels.
//!
//! Cubic interpolation is deferred to a focused ADR (more attribute
//! surface — coefficient `a`, exclude-outside flag, etc.).

use crate::attrs::ResizeAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `resize` (nearest mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Channel count.
    pub c: u32,
    /// Input height.
    pub h_in: u32,
    /// Input width.
    pub w_in: u32,
    /// Output height.
    pub h_out: u32,
    /// Output width.
    pub w_out: u32,
}

/// Marker struct for the canonical `resize` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Resize(pub ResizeAttrs);

impl Op for Resize {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "resize"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Shape
    }
}

/// Forward: nearest-neighbor resize per-(N, C) plane.
pub fn resize_nearest(storage: &mut [f32], call: &ResizeCall) {
    let n = call.n as usize;
    let c = call.c as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let h_out = call.h_out as usize;
    let w_out = call.w_out as usize;
    debug_assert!(h_in > 0 && w_in > 0 && h_out > 0 && w_out > 0);
    let scale_h = h_in as f32 / h_out as f32;
    let scale_w = w_in as f32 / w_out as f32;
    let chw_in = c * h_in * w_in;
    let chw_out = c * h_out * w_out;

    for ni in 0..n {
        for ci in 0..c {
            let plane_in = call.input.offset + ni * chw_in + ci * h_in * w_in;
            let plane_out = call.output.offset + ni * chw_out + ci * h_out * w_out;
            for oh in 0..h_out {
                let ih = nearest_src(oh, scale_h, h_in);
                for ow in 0..w_out {
                    let iw = nearest_src(ow, scale_w, w_in);
                    storage[plane_out + oh * w_out + ow] = storage[plane_in + ih * w_in + iw];
                }
            }
        }
    }
}

/// Forward: bilinear-interpolation resize per-(N, C) plane.
pub fn resize_linear(storage: &mut [f32], call: &ResizeCall) {
    let n = call.n as usize;
    let c = call.c as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let h_out = call.h_out as usize;
    let w_out = call.w_out as usize;
    debug_assert!(h_in > 0 && w_in > 0 && h_out > 0 && w_out > 0);
    let scale_h = h_in as f32 / h_out as f32;
    let scale_w = w_in as f32 / w_out as f32;
    let chw_in = c * h_in * w_in;
    let chw_out = c * h_out * w_out;

    for ni in 0..n {
        for ci in 0..c {
            let plane_in = call.input.offset + ni * chw_in + ci * h_in * w_in;
            let plane_out = call.output.offset + ni * chw_out + ci * h_out * w_out;
            for oh in 0..h_out {
                let sy = src_coord(oh, scale_h, h_in);
                let (y0, y1, fy) = neighbours(sy, h_in);
                for ow in 0..w_out {
                    let sx = src_coord(ow, scale_w, w_in);
                    let (x0, x1, fx) = neighbours(sx, w_in);
                    let v00 = storage[plane_in + y0 * w_in + x0];
                    let v01 = storage[plane_in + y0 * w_in + x1];
                    let v10 = storage[plane_in + y1 * w_in + x0];
                    let v11 = storage[plane_in + y1 * w_in + x1];
                    let top = v00 + (v01 - v00) * fx;
                    let bot = v10 + (v11 - v10) * fx;
                    storage[plane_out + oh * w_out + ow] = top + (bot - top) * fy;
                }
            }
        }
    }
}

#[inline]
fn src_coord(out_idx: usize, scale: f32, in_size: usize) -> f32 {
    let v = (out_idx as f32 + 0.5) * scale - 0.5;
    if v < 0.0 {
        0.0
    } else if v > (in_size - 1) as f32 {
        (in_size - 1) as f32
    } else {
        v
    }
}

#[inline]
fn neighbours(src: f32, in_size: usize) -> (usize, usize, f32) {
    let lo = libm::floorf(src) as usize;
    let lo = lo.min(in_size - 1);
    let hi = (lo + 1).min(in_size - 1);
    let frac = src - lo as f32;
    (lo, hi, frac.clamp(0.0, 1.0))
}

#[inline]
fn nearest_src(out_idx: usize, scale: f32, in_size: usize) -> usize {
    let src = (out_idx as f32 + 0.5) * scale - 0.5;
    if src < 0.0 {
        return 0;
    }
    let rounded = libm::roundf(src) as usize;
    if rounded >= in_size {
        in_size - 1
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_nearest_doubles_each_dim() {
        // 1×1×2×2 input [[1,2],[3,4]] → 1×1×4×4 output via nearest.
        let mut s = vec![0.0_f32; 4 + 16];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = ResizeCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 16 },
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 4,
            w_out: 4,
        };
        resize_nearest(&mut s, &call);
        // Expected: each input pixel covers a 2×2 output region.
        assert_eq!(
            &s[4..20],
            &[
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn resize_linear_blends_between_neighbours() {
        // 1×1×2×2 input [[0, 10], [0, 10]] → 1×1×2×4 output. Linear
        // interpolation along the W axis should give half-step blends.
        let mut s = vec![0.0_f32; 4 + 8];
        s[..4].copy_from_slice(&[0.0, 10.0, 0.0, 10.0]);
        let call = ResizeCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 8 },
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 2,
            w_out: 4,
        };
        resize_linear(&mut s, &call);
        // Each row's four output pixels sample the [0, 10] gradient at
        // 0, 0.25, 0.75, 1.0 (with clamp at borders) — values are
        // monotone non-decreasing and bracketed by [0, 10].
        for r in 0..2 {
            let row = &s[4 + r * 4..4 + r * 4 + 4];
            assert!(row[0] >= -1e-5 && row[3] <= 10.0 + 1e-5);
            for w in 1..4 {
                assert!(row[w] >= row[w - 1] - 1e-5);
            }
        }
    }

    #[test]
    fn resize_nearest_identity_dims_is_copy() {
        let mut s = vec![0.0_f32; 4 + 4];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = ResizeCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 4 },
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 2,
            w_out: 2,
        };
        resize_nearest(&mut s, &call);
        assert_eq!(&s[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }
}
