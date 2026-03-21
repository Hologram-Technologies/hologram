use super::helpers::*;
use crate::error::{ExecError, ExecResult};

/// Generic 2D pooling with flattened outer loop.
///
/// Replaces 6-level nested loops (batch × channels × oh × ow × kh × kw)
/// with a flat outer loop over output elements + a flat inner kernel loop.
/// The `fold` closure receives `(accumulator, data_value)` and returns the
/// new accumulator; `finalize` converts the accumulator to the output value.
#[allow(clippy::too_many_arguments)]
#[inline]
fn pool_2d<A: Copy>(
    data: &[f32],
    n: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    init: A,
    fold: impl Fn(A, f32) -> A,
    finalize: impl Fn(A) -> f32,
) -> Vec<f32> {
    let out_len = n * channels * h_out * w_out;
    let mut out = vec![0.0f32; out_len];
    let c_h_w = channels * h_out * w_out;
    let h_w = h_out * w_out;

    for (flat_out, out_val) in out.iter_mut().enumerate() {
        // Decompose flat output index → (batch, c, oh, ow)
        let batch = flat_out / c_h_w;
        let rem = flat_out % c_h_w;
        let c = rem / h_w;
        let rem2 = rem % h_w;
        let oh = rem2 / w_out;
        let ow = rem2 % w_out;

        let mut acc = init;
        // Flat inner kernel loop
        for k_flat in 0..(kh * kw) {
            let ky = k_flat / kw;
            let kx = k_flat % kw;
            let iy = (oh * sh + ky) as isize - ph as isize;
            let ix = (ow * sw + kx) as isize - pw as isize;
            if iy >= 0 && iy < h_in as isize && ix >= 0 && ix < w_in as isize {
                let idx = ((batch * channels + c) * h_in + iy as usize) * w_in + ix as usize;
                if idx < data.len() {
                    acc = fold(acc, data[idx]);
                }
            }
        }
        *out_val = finalize(acc);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_max_pool_2d(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let total = data.len();
    let (channels, h_in, w_in) = infer_nchw(total, 1);
    let n = 1;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let out = pool_2d(
        &data,
        n,
        channels,
        h_in,
        w_in,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        f32::NEG_INFINITY,
        |acc, val| acc.max(val),
        |acc| acc,
    );
    Ok(f32_vec_to_bytes(out))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_avg_pool_2d(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let total = data.len();
    let (channels, h_in, w_in) = infer_nchw(total, 1);
    let n = 1;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    // Accumulator: (sum, count)
    let out = pool_2d(
        &data,
        n,
        channels,
        h_in,
        w_in,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        (0.0f32, 0usize),
        |(sum, count), val| (sum + val, count + 1),
        |(sum, count)| {
            if count > 0 {
                sum / count as f32
            } else {
                0.0
            }
        },
    );
    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_global_avg_pool(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // GlobalAvgPool: [N,C,H,W] → [N,C,1,1]
    let total = data.len();
    let (channels, h, w) = infer_nchw(total, 1);
    let spatial = h * w;

    let mut out = Vec::with_capacity(channels);
    for c in 0..channels {
        let start = c * spatial;
        let end = (start + spatial).min(data.len());
        if start < data.len() {
            let sum: f32 = data[start..end].iter().sum();
            out.push(sum / spatial as f32);
        }
    }
    Ok(f32_vec_to_bytes(out))
}

/// GlobalAvgPool with explicit input shapes (no heuristic guessing).
pub(crate) fn dispatch_global_avg_pool_with_shapes(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let (n, c, h, w) = match input_shapes.first().map(|s| s.as_slice()) {
        Some(&[n, c, h, w]) => (n, c, h, w),
        _ => {
            return Err(ExecError::UnsupportedOp(format!(
                "GlobalAvgPool: expected 4D input shape, got {:?}",
                input_shapes.first()
            )))
        }
    };
    let spatial = h * w;
    let mut out = Vec::with_capacity(n * c);
    for batch in 0..n {
        for ch in 0..c {
            let start = (batch * c + ch) * spatial;
            let end = start + spatial;
            if end <= data.len() {
                let sum: f32 = data[start..end].iter().sum();
                out.push(sum / spatial as f32);
            }
        }
    }
    Ok(f32_vec_to_bytes(out))
}
