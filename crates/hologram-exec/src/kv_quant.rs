//! Per-channel affine quantization for KV cache storage.
//!
//! Provides q8 (8-bit) and q4 (4-bit) quantization/dequantization routines
//! used by `KvCacheState` to compress key/value tensors. Each head at each
//! position is treated as an independent channel with its own scale and
//! zero-point, giving affine (min-max) quantization.

// ── Per-channel affine quantization ──────────────────────────────────

/// Per-channel scale and zero-point for affine quantization.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChannelParams {
    pub(crate) scale: f32,
    pub(crate) zero_point: f32,
}

/// Quantize a single channel (one head at one position) to q8.
/// Returns params. `out` must have len == `data.len()`.
#[inline]
pub(crate) fn quantize_channel_q8(data: &[f32], out: &mut [u8]) -> ChannelParams {
    // Fused min/max in a single pass (4-wide manual unroll for autovec).
    let n = data.len();
    let (mut min, mut max) = (data[0], data[0]);
    let chunks = n / 4;
    let base = data.as_ptr();
    for c in 0..chunks {
        let off = c * 4;
        unsafe {
            let a = *base.add(off);
            let b = *base.add(off + 1);
            let c = *base.add(off + 2);
            let d = *base.add(off + 3);
            let lo = a.min(b).min(c.min(d));
            let hi = a.max(b).max(c.max(d));
            min = min.min(lo);
            max = max.max(hi);
        }
    }
    for &v in &data[chunks * 4..n] {
        min = min.min(v);
        max = max.max(v);
    }

    // Degenerate: all values identical.
    if (max - min).abs() < f32::EPSILON {
        out.iter_mut().for_each(|b| *b = 0);
        return ChannelParams {
            scale: 1.0,
            zero_point: -min,
        };
    }
    let scale = (max - min) / 255.0;
    let inv_scale = 255.0 / (max - min);
    let bias = -min * inv_scale + 0.5; // +0.5 replaces .round() with truncation

    // Quantize: fused multiply-add + truncate (avoids round() per element).
    for i in 0..n {
        let q = (data[i] * inv_scale + bias) as i32;
        out[i] = q.clamp(0, 255) as u8;
    }
    ChannelParams {
        scale,
        zero_point: -min * (1.0 / scale),
    }
}

/// Quantize a single channel to q4 (16 levels).
/// `out` must have len == `data.len().div_ceil(2)`.
#[inline]
pub(crate) fn quantize_channel_q4(data: &[f32], out: &mut [u8]) -> ChannelParams {
    let n = data.len();
    let (mut min, mut max) = (data[0], data[0]);
    for &v in &data[1..] {
        min = min.min(v);
        max = max.max(v);
    }
    if (max - min).abs() < f32::EPSILON {
        out.iter_mut().for_each(|b| *b = 0);
        return ChannelParams {
            scale: 1.0,
            zero_point: -min,
        };
    }
    let scale = (max - min) / 15.0;
    let inv_scale = 15.0 / (max - min);
    let bias = -min * inv_scale + 0.5;

    // Pack two 4-bit indices per byte.
    let pairs = n / 2;
    for p in 0..pairs {
        let hi = (data[p * 2] * inv_scale + bias) as u8;
        let lo = (data[p * 2 + 1] * inv_scale + bias) as u8;
        out[p] = (hi.min(15) << 4) | lo.min(15);
    }
    if n & 1 != 0 {
        let hi = (data[n - 1] * inv_scale + bias) as u8;
        out[pairs] = hi.min(15) << 4;
    }
    ChannelParams {
        scale,
        zero_point: -min * (1.0 / scale),
    }
}

/// Dequantize q8 indices back to f32.
/// The compiler autovectorizes this to NEON/SSE at opt-level >= 2.
#[inline]
pub(crate) fn dequantize_q8(indices: &[u8], params: &ChannelParams, out: &mut [f32]) {
    let scale = params.scale;
    let zp = params.zero_point;
    let n = indices.len();
    for i in 0..n {
        out[i] = (indices[i] as f32 - zp) * scale;
    }
}

/// Dequantize q8 with fused sign-flip: `out[i] = ((idx - zp) * scale) * signs[i]`.
/// Eliminates a separate `vec_mul_inplace` pass on the WHT read path.
#[inline]
pub(crate) fn dequantize_q8_signed(
    indices: &[u8],
    params: &ChannelParams,
    signs: &[f32],
    out: &mut [f32],
) {
    let scale = params.scale;
    let zp = params.zero_point;
    let n = indices.len();
    for i in 0..n {
        out[i] = (indices[i] as f32 - zp) * scale * signs[i];
    }
}

/// Dequantize q4 packed indices back to f32.
#[inline]
pub(crate) fn dequantize_q4(
    packed: &[u8],
    n_elems: usize,
    params: &ChannelParams,
    out: &mut [f32],
) {
    let scale = params.scale;
    let zp = params.zero_point;
    let pairs = n_elems / 2;
    for p in 0..pairs {
        let byte = packed[p];
        out[p * 2] = ((byte >> 4) as f32 - zp) * scale;
        out[p * 2 + 1] = ((byte & 0x0F) as f32 - zp) * scale;
    }
    if n_elems & 1 != 0 {
        out[n_elems - 1] = ((packed[pairs] >> 4) as f32 - zp) * scale;
    }
}

/// Dequantize q4 with fused sign-flip.
#[inline]
pub(crate) fn dequantize_q4_signed(
    packed: &[u8],
    n_elems: usize,
    params: &ChannelParams,
    signs: &[f32],
    out: &mut [f32],
) {
    let scale = params.scale;
    let zp = params.zero_point;
    let pairs = n_elems / 2;
    for p in 0..pairs {
        let byte = packed[p];
        out[p * 2] = ((byte >> 4) as f32 - zp) * scale * signs[p * 2];
        out[p * 2 + 1] = ((byte & 0x0F) as f32 - zp) * scale * signs[p * 2 + 1];
    }
    if n_elems & 1 != 0 {
        out[n_elems - 1] = ((packed[pairs] >> 4) as f32 - zp) * scale * signs[n_elems - 1];
    }
}
