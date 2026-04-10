use super::helpers::*;
use crate::error::ExecResult;

pub(crate) fn dispatch_resize(inputs: &[&[u8]], mode: u8) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // inputs[1] = scales or sizes (f32 or i64)
    let scales_bytes = inputs.get(1).copied().unwrap_or(&[][..]);

    if data.is_empty() {
        return Ok(vec![]);
    }

    // Parse scales as f32
    let scales: Vec<f32> = if !scales_bytes.is_empty() && scales_bytes.len() % 4 == 0 {
        cast_f32(scales_bytes)?.to_vec()
    } else {
        vec![1.0; 4]
    };

    // If all scales are 1.0, pass through
    if scales.iter().all(|&s| (s - 1.0).abs() < 1e-6) {
        return Ok(inputs[0].to_vec());
    }

    // 4D NCHW spatial resize: only scale H (dim 2) and W (dim 3).
    // Infer N×C from total elements / (H×W). Scales[0]=N, [1]=C, [2]=H, [3]=W.
    if scales.len() >= 4 {
        let sh = scales[2];
        let sw = scales[3];
        if sh > 0.0 && sw > 0.0 && (sh != 1.0 || sw != 1.0) {
            return dispatch_resize_nchw(&data, sh, sw, mode);
        }
    }

    // Fallback: 1-D flat resize for non-spatial scales.
    let total_scale: f32 = scales.iter().product();
    let out_len = ((data.len() as f32) * total_scale) as usize;
    if out_len == 0 || out_len > data.len() * 64 {
        return Ok(inputs[0].to_vec());
    }

    let out: Vec<f32> = match mode {
        1 => {
            // Linear interpolation
            (0..out_len)
                .map(|i| {
                    let src_f = (i as f32) / total_scale;
                    let lo = src_f.floor() as usize;
                    let hi = (lo + 1).min(data.len() - 1);
                    let frac = src_f - lo as f32;
                    let lo = lo.min(data.len() - 1);
                    data[lo] * (1.0 - frac) + data[hi] * frac
                })
                .collect()
        }
        _ => {
            // Nearest neighbor (mode 0) or cubic/unknown fallback
            (0..out_len)
                .map(|i| {
                    let src = ((i as f32) / total_scale) as usize;
                    data[src.min(data.len() - 1)]
                })
                .collect()
        }
    };

    Ok(f32_vec_to_bytes(out))
}

/// Resize with optional input shape metadata for proper NCHW spatial dims.
pub(crate) fn dispatch_resize_with_shape(
    inputs: &[&[u8]],
    mode: u8,
    input_shape: Option<&[usize]>,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let scales_bytes = inputs.get(1).copied().unwrap_or(&[][..]);

    if data.is_empty() {
        return Ok(vec![]);
    }

    let scales: Vec<f32> = if !scales_bytes.is_empty() && scales_bytes.len() % 4 == 0 {
        cast_f32(scales_bytes)?.to_vec()
    } else {
        vec![1.0; 4]
    };

    if scales.iter().all(|&s| (s - 1.0).abs() < 1e-6) {
        return Ok(inputs[0].to_vec());
    }

    // Use shape metadata to determine NCHW dims.
    if scales.len() >= 4 && (scales[2] != 1.0 || scales[3] != 1.0) {
        if let Some(shape) = input_shape {
            if shape.len() == 4 {
                let n = shape[0];
                let c = shape[1];
                let h_in = shape[2];
                let w_in = shape[3];
                let h_out = (h_in as f32 * scales[2]).round() as usize;
                let w_out = (w_in as f32 * scales[3]).round() as usize;
                return dispatch_resize_nchw_known(
                    &data,
                    ResizeNchwParams::new(n, c, h_in, w_in, h_out, w_out).with_mode(mode),
                );
            }
        }
        // No shape metadata — fall back to flat 1D resize.
        // The heuristic NCHW path misidentifies spatial dims for large tensors.
    }

    // 1D fallback
    dispatch_resize(inputs, mode)
}

/// Shape + mode for [`dispatch_resize_nchw_known`]. Build with
/// [`ResizeNchwParams::new`] (required shape) and chain
/// [`Self::with_mode`] (default `0`, nearest-neighbor).
#[derive(Debug, Clone, Copy)]
struct ResizeNchwParams {
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    mode: u8,
}

impl ResizeNchwParams {
    #[inline]
    fn new(n: usize, c: usize, h_in: usize, w_in: usize, h_out: usize, w_out: usize) -> Self {
        Self {
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            mode: 0,
        }
    }

    #[inline]
    fn with_mode(mut self, mode: u8) -> Self {
        self.mode = mode;
        self
    }
}

/// NCHW resize with known spatial dimensions — no guessing.
fn dispatch_resize_nchw_known(data: &[f32], params: ResizeNchwParams) -> ExecResult<Vec<u8>> {
    let ResizeNchwParams {
        n,
        c,
        h_in,
        w_in,
        h_out,
        w_out,
        mode,
    } = params;
    let nc = n * c;
    let out_total = nc * h_out * w_out;
    let mut out = vec![0.0f32; out_total];

    for ch in 0..nc {
        let in_base = ch * h_in * w_in;
        let out_base = ch * h_out * w_out;
        for oh in 0..h_out {
            for ow in 0..w_out {
                let val = match mode {
                    1 => {
                        let src_h = (oh as f32) * (h_in as f32) / (h_out as f32);
                        let src_w = (ow as f32) * (w_in as f32) / (w_out as f32);
                        let h0 = src_h.floor() as usize;
                        let w0 = src_w.floor() as usize;
                        let h1 = (h0 + 1).min(h_in.saturating_sub(1));
                        let w1 = (w0 + 1).min(w_in.saturating_sub(1));
                        let h0 = h0.min(h_in.saturating_sub(1));
                        let w0 = w0.min(w_in.saturating_sub(1));
                        let fh = src_h - h0 as f32;
                        let fw = src_w - w0 as f32;
                        let idx = |h: usize, w: usize| {
                            (in_base + h * w_in + w).min(data.len().saturating_sub(1))
                        };
                        data[idx(h0, w0)] * (1.0 - fh) * (1.0 - fw)
                            + data[idx(h0, w1)] * (1.0 - fh) * fw
                            + data[idx(h1, w0)] * fh * (1.0 - fw)
                            + data[idx(h1, w1)] * fh * fw
                    }
                    _ => {
                        let sh = (oh as f32 * h_in as f32 / h_out as f32) as usize;
                        let sw = (ow as f32 * w_in as f32 / w_out as f32) as usize;
                        let sh = sh.min(h_in.saturating_sub(1));
                        let sw = sw.min(w_in.saturating_sub(1));
                        let idx = (in_base + sh * w_in + sw).min(data.len().saturating_sub(1));
                        data[idx]
                    }
                };
                out[out_base + oh * w_out + ow] = val;
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

/// NCHW spatial resize — proper 2D nearest-neighbor or bilinear.
///
/// Scales only H and W dims. N and C are inferred from total elements.
/// Much more memory-efficient than flat 1D resize for vision models.
fn dispatch_resize_nchw(data: &[f32], scale_h: f32, scale_w: f32, mode: u8) -> ExecResult<Vec<u8>> {
    // Infer spatial dims. Without explicit shape metadata, we estimate
    // H=W (square) from total elements / nc, then try common aspect ratios.
    // For models compiled with known shapes, the ratio is exact.
    let total = data.len();
    if total == 0 {
        return Ok(vec![]);
    }

    // Try to find H×W such that total = NC × H × W and H_out/W_out are integer.
    // Start with square assumption, then validate.
    let h_out_f = |h: usize| (h as f32 * scale_h).round() as usize;
    let w_out_f = |w: usize| (w as f32 * scale_w).round() as usize;

    // Try common spatial sizes (descending) that divide total evenly.
    let (nc, h_in, w_in) = {
        let mut found = (total, 1, 1);
        // Square spatial: total = nc * s * s
        for s in (1..=1024).rev() {
            if total.is_multiple_of(s * s) {
                let nc = total / (s * s);
                if nc > 0 {
                    found = (nc, s, s);
                    break;
                }
            }
        }
        found
    };

    let h_out = h_out_f(h_in);
    let w_out = w_out_f(w_in);
    let out_total = nc * h_out * w_out;
    let mut out = vec![0.0f32; out_total];

    for c in 0..nc {
        let in_base = c * h_in * w_in;
        let out_base = c * h_out * w_out;
        for oh in 0..h_out {
            for ow in 0..w_out {
                let val = match mode {
                    1 => {
                        // Bilinear
                        let src_h = (oh as f32) / scale_h;
                        let src_w = (ow as f32) / scale_w;
                        let h0 = src_h.floor() as usize;
                        let w0 = src_w.floor() as usize;
                        let h1 = (h0 + 1).min(h_in.saturating_sub(1));
                        let w1 = (w0 + 1).min(w_in.saturating_sub(1));
                        let h0 = h0.min(h_in.saturating_sub(1));
                        let w0 = w0.min(w_in.saturating_sub(1));
                        let fh = src_h - h0 as f32;
                        let fw = src_w - w0 as f32;
                        let v00 = data[in_base + h0 * w_in + w0];
                        let v01 = data[in_base + h0 * w_in + w1];
                        let v10 = data[in_base + h1 * w_in + w0];
                        let v11 = data[in_base + h1 * w_in + w1];
                        v00 * (1.0 - fh) * (1.0 - fw)
                            + v01 * (1.0 - fh) * fw
                            + v10 * fh * (1.0 - fw)
                            + v11 * fh * fw
                    }
                    _ => {
                        // Nearest neighbor
                        let src_h = ((oh as f32) / scale_h) as usize;
                        let src_w = ((ow as f32) / scale_w) as usize;
                        let sh = src_h.min(h_in.saturating_sub(1));
                        let sw = src_w.min(w_in.saturating_sub(1));
                        data[in_base + sh * w_in + sw]
                    }
                };
                out[out_base + oh * w_out + ow] = val;
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

pub(crate) fn dispatch_pad(inputs: &[&[u8]], mode: u8) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let pads_bytes = inputs.get(1).copied().unwrap_or(&[][..]);

    if pads_bytes.is_empty() {
        return Ok(inputs[0].to_vec());
    }

    // Pads are i64: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    let pads: Vec<i64> = if pads_bytes.len() % 8 == 0 {
        iter_i64(pads_bytes).collect()
    } else {
        // Try as f32 (some models pass pads as float)
        cast_f32(pads_bytes)?.iter().map(|&v| v as i64).collect()
    };

    if pads.iter().all(|&p| p == 0) {
        return Ok(inputs[0].to_vec());
    }

    // Simple 1-D padding: sum all begin pads and end pads
    let ndim = pads.len() / 2;
    let total_begin: usize = pads[..ndim].iter().map(|&p| p.max(0) as usize).sum();
    let total_end: usize = pads[ndim..].iter().map(|&p| p.max(0) as usize).sum();

    let pad_val = match mode {
        0 => 0.0f32, // constant
        _ => 0.0f32, // reflect/edge simplified to constant
    };

    let out_len = total_begin + data.len() + total_end;
    let mut out = vec![pad_val; out_len];
    out[total_begin..total_begin + data.len()].copy_from_slice(&data);

    if mode == 1 && data.len() > 1 {
        // Reflect: mirror edges
        for (i, v) in out[..total_begin].iter_mut().enumerate() {
            let src = total_begin - i;
            *v = if src < data.len() { data[src] } else { data[0] };
        }
        let tail_start = total_begin + data.len();
        for (i, v) in out[tail_start..tail_start + total_end]
            .iter_mut()
            .enumerate()
        {
            let src = data.len().saturating_sub(2).saturating_sub(i);
            *v = data[src];
        }
    } else if mode == 2 {
        // Edge: replicate border
        let first = data[0];
        let last = *data.last().unwrap_or(&0.0);
        out[..total_begin].fill(first);
        let tail_start = total_begin + data.len();
        out[tail_start..tail_start + total_end].fill(last);
    }

    Ok(f32_vec_to_bytes(out))
}
