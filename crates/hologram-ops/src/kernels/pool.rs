//! Canonical 2-D pooling ops (`MaxPool2d`, `AvgPool2d`,
//! `GlobalAvgPool`) — semantic identity, executable form, and CPU
//! reference kernels.
//!
//! NCHW layout. `Pool2dCall` covers max + avg (they share window
//! params); `GlobalAvgPoolCall` is its own shape since the output
//! collapses spatial dims to `1×1` and only needs `[N, C, H, W]`
//! information.

use crate::attrs::{GlobalAvgPoolAttrs, Pool2dAttrs};
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Pre-resolved arguments for a 2-D max/avg pool kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool2dCall {
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
}

/// Identity tag for the canonical 2-D pools sharing `Pool2dCall`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pool2dKind {
    /// Maximum-of-window pool.
    Max,
    /// Average-of-window pool (counts only in-bounds elements).
    Avg,
}

/// Pre-resolved arguments for `GlobalAvgPool` (spatial → 1×1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAvgPoolCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (length = `n * c`).
    pub output: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Channel count.
    pub c: u32,
    /// Spatial height.
    pub h: u32,
    /// Spatial width.
    pub w: u32,
}

/// Marker struct for the canonical `max_pool_2d` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaxPool2d(pub Pool2dAttrs);

impl Op for MaxPool2d {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "max_pool_2d"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Convolution
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::MaxPool2dBackward)
    }
}

/// Marker struct for the canonical `avg_pool_2d` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AvgPool2d(pub Pool2dAttrs);

impl Op for AvgPool2d {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "avg_pool_2d"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Convolution
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::AvgPool2dBackward)
    }
}

/// Marker struct for the canonical `global_avg_pool` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalAvgPool(pub GlobalAvgPoolAttrs);

impl Op for GlobalAvgPool {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "global_avg_pool"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Convolution
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::GlobalAvgPoolBackward)
    }
}

/// Apply max or avg pooling per `kind`.
pub fn dispatch_pool2d(storage: &mut [f32], call: &Pool2dCall, kind: Pool2dKind) {
    let n = call.n as usize;
    let c = call.c as usize;
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
    let chw_in = c * h_in * w_in;
    let chw_out = c * h_out * w_out;

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let (acc, count) = window_fold(
                        storage,
                        call.input.offset + ni * chw_in + ci * h_in * w_in,
                        h_in,
                        w_in,
                        oh as isize * sh as isize - ph,
                        ow as isize * sw as isize - pw,
                        kh,
                        kw,
                        kind,
                    );
                    let idx =
                        call.output.offset + ni * chw_out + ci * h_out * w_out + oh * w_out + ow;
                    storage[idx] = match kind {
                        Pool2dKind::Max => acc,
                        Pool2dKind::Avg => {
                            if count == 0 {
                                0.0
                            } else {
                                acc / count as f32
                            }
                        }
                    };
                }
            }
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn window_fold(
    storage: &[f32],
    plane_off: usize,
    h_in: usize,
    w_in: usize,
    h_start: isize,
    w_start: isize,
    kh: usize,
    kw: usize,
    kind: Pool2dKind,
) -> (f32, usize) {
    let mut acc = match kind {
        Pool2dKind::Max => f32::NEG_INFINITY,
        Pool2dKind::Avg => 0.0,
    };
    let mut count = 0_usize;
    for ky in 0..kh {
        let ih = h_start + ky as isize;
        if ih < 0 || ih >= h_in as isize {
            continue;
        }
        for kx in 0..kw {
            let iw = w_start + kx as isize;
            if iw < 0 || iw >= w_in as isize {
                continue;
            }
            let v = storage[plane_off + ih as usize * w_in + iw as usize];
            acc = match kind {
                Pool2dKind::Max => f32::max(acc, v),
                Pool2dKind::Avg => acc + v,
            };
            count += 1;
        }
    }
    (acc, count)
}

/// Forward: `out[n, c] = mean(input[n, c, :, :])`.
pub fn global_avg_pool(storage: &mut [f32], call: &GlobalAvgPoolCall) {
    let n = call.n as usize;
    let c = call.c as usize;
    let h = call.h as usize;
    let w = call.w as usize;
    let plane = h * w;
    debug_assert!(plane > 0);
    let denom = plane as f32;
    for ni in 0..n {
        for ci in 0..c {
            let off = call.input.offset + (ni * c + ci) * plane;
            let mut sum = 0.0_f32;
            for i in 0..plane {
                sum += storage[off + i];
            }
            storage[call.output.offset + ni * c + ci] = sum / denom;
        }
    }
}

/// Pre-resolved arguments for `MaxPool2d` / `AvgPool2d` backward.
///
/// `input` is only consulted by the max variant (to recompute the
/// argmax position per window). For both variants, accumulates into
/// `dx` — overlapping windows pile up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool2dGradCall {
    /// Forward input span (used by Max only).
    pub input: SlotSpan,
    /// Upstream gradient `dy` (length = `output.len`).
    pub dy: SlotSpan,
    /// Gradient slot for `x`.
    pub dx: SlotSpan,
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
}

/// Pre-resolved arguments for `GlobalAvgPool` backward.
///
/// Broadcasts each `dy[n,c]` uniformly over the spatial plane:
/// `dx[n,c,i,j] += dy[n,c] / (h*w)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAvgPoolGradCall {
    /// Upstream gradient `dy` (length = `n * c`).
    pub dy: SlotSpan,
    /// Gradient slot for `x` (length = `n * c * h * w`).
    pub dx: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Channel count.
    pub c: u32,
    /// Spatial height.
    pub h: u32,
    /// Spatial width.
    pub w: u32,
}

/// Backward of 2-D max/avg pool. Accumulates into `dx`. No-op if
/// `dx.len == 0`.
pub fn dispatch_pool2d_grad(storage: &mut [f32], call: &Pool2dGradCall, kind: Pool2dKind) {
    if call.dx.len == 0 {
        return;
    }
    let n = call.n as usize;
    let c = call.c as usize;
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
    let chw_in = c * h_in * w_in;
    let chw_out = c * h_out * w_out;

    for ni in 0..n {
        for ci in 0..c {
            let plane_in = call.input.offset + ni * chw_in + ci * h_in * w_in;
            let plane_dx = call.dx.offset + ni * chw_in + ci * h_in * w_in;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let h_start = oh as isize * sh as isize - ph;
                    let w_start = ow as isize * sw as isize - pw;
                    let dy = storage
                        [call.dy.offset + ni * chw_out + ci * h_out * w_out + oh * w_out + ow];
                    match kind {
                        Pool2dKind::Avg => {
                            let mut count = 0_usize;
                            for ky in 0..kh {
                                let ih = h_start + ky as isize;
                                if ih < 0 || ih >= h_in as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let iw = w_start + kx as isize;
                                    if iw < 0 || iw >= w_in as isize {
                                        continue;
                                    }
                                    count += 1;
                                }
                            }
                            if count == 0 {
                                continue;
                            }
                            let share = dy / count as f32;
                            for ky in 0..kh {
                                let ih = h_start + ky as isize;
                                if ih < 0 || ih >= h_in as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let iw = w_start + kx as isize;
                                    if iw < 0 || iw >= w_in as isize {
                                        continue;
                                    }
                                    storage[plane_dx + ih as usize * w_in + iw as usize] += share;
                                }
                            }
                        }
                        Pool2dKind::Max => {
                            let mut best = f32::NEG_INFINITY;
                            let mut arg = None::<(usize, usize)>;
                            for ky in 0..kh {
                                let ih = h_start + ky as isize;
                                if ih < 0 || ih >= h_in as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let iw = w_start + kx as isize;
                                    if iw < 0 || iw >= w_in as isize {
                                        continue;
                                    }
                                    let v = storage[plane_in + ih as usize * w_in + iw as usize];
                                    if v > best {
                                        best = v;
                                        arg = Some((ih as usize, iw as usize));
                                    }
                                }
                            }
                            if let Some((ih, iw)) = arg {
                                storage[plane_dx + ih * w_in + iw] += dy;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Backward of `global_avg_pool`. Broadcasts `dy / (h*w)` over the
/// spatial plane.
pub fn global_avg_pool_grad(storage: &mut [f32], call: &GlobalAvgPoolGradCall) {
    if call.dx.len == 0 {
        return;
    }
    let n = call.n as usize;
    let c = call.c as usize;
    let h = call.h as usize;
    let w = call.w as usize;
    let plane = h * w;
    debug_assert!(plane > 0);
    let inv_plane = 1.0 / plane as f32;
    for ni in 0..n {
        for ci in 0..c {
            let dy = storage[call.dy.offset + ni * c + ci] * inv_plane;
            let off = call.dx.offset + (ni * c + ci) * plane;
            for i in 0..plane {
                storage[off + i] += dy;
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
    fn max_pool_2x2_stride_2_picks_max_of_each_window() {
        // 1×1×2×2 input [[1,2],[3,4]] → 2×2 kernel, stride 2 → 1×1 output [4].
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 0.0];
        let call = Pool2dCall {
            input: span(0, 4),
            output: span(4, 1),
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 1,
            w_out: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        dispatch_pool2d(&mut s, &call, Pool2dKind::Max);
        assert_eq!(s[4], 4.0);
    }

    #[test]
    fn avg_pool_averages_only_in_bounds_elements() {
        // 1×1×2×2 input [[1,2],[3,4]] with pad=1, kernel=2, stride=2 → 2×2
        // output where corners see only one in-bounds element.
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let call = Pool2dCall {
            input: span(0, 4),
            output: span(4, 4),
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 2,
            w_out: 2,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
        };
        dispatch_pool2d(&mut s, &call, Pool2dKind::Avg);
        // Each output cell sees exactly one input element with the given
        // padding/stride configuration.
        assert_eq!(&s[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn global_avg_pool_collapses_spatial_dims() {
        // 1×2×2×2 input: channel 0 = [[1,2],[3,4]] (mean 2.5),
        //                channel 1 = [[5,6],[7,8]] (mean 6.5).
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0];
        let call = GlobalAvgPoolCall {
            input: span(0, 8),
            output: span(8, 2),
            n: 1,
            c: 2,
            h: 2,
            w: 2,
        };
        global_avg_pool(&mut s, &call);
        assert!((s[8] - 2.5).abs() < 1e-5);
        assert!((s[9] - 6.5).abs() < 1e-5);
    }

    #[test]
    fn max_pool_grad_routes_dy_to_argmax_position() {
        // 1×1×2×2 input [[1,2],[3,4]], 2×2 kernel stride 2, dy = 7.0.
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 7.0, 0.0, 0.0, 0.0, 0.0];
        let call = Pool2dGradCall {
            input: span(0, 4),
            dy: span(4, 1),
            dx: span(5, 4),
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 1,
            w_out: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        dispatch_pool2d_grad(&mut s, &call, Pool2dKind::Max);
        // Argmax is at position 3 (value 4).
        assert_eq!(&s[5..9], &[0.0, 0.0, 0.0, 7.0]);
    }

    #[test]
    fn avg_pool_grad_distributes_dy_uniformly_in_window() {
        // 1×1×2×2 input → 1×1 output, kernel 2x2 stride 2; dy = 8 → each
        // input position gets 8/4 = 2.
        let mut s = [1.0_f32, 2.0, 3.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0];
        let call = Pool2dGradCall {
            input: span(0, 4),
            dy: span(4, 1),
            dx: span(5, 4),
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            h_out: 1,
            w_out: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        dispatch_pool2d_grad(&mut s, &call, Pool2dKind::Avg);
        assert_eq!(&s[5..9], &[2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn global_avg_pool_grad_broadcasts_uniformly() {
        // 1×1×2×2; dy = 4 → each spatial position gets 4/4 = 1.
        let mut s = [4.0_f32, 0.0, 0.0, 0.0, 0.0];
        let call = GlobalAvgPoolGradCall {
            dy: span(0, 1),
            dx: span(1, 4),
            n: 1,
            c: 1,
            h: 2,
            w: 2,
        };
        global_avg_pool_grad(&mut s, &call);
        assert_eq!(&s[1..5], &[1.0, 1.0, 1.0, 1.0]);
    }
}
