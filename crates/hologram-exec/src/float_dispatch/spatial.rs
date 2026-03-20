use super::helpers::*;
use crate::error::ExecResult;

pub(super) fn dispatch_resize(inputs: &[&[u8]], mode: u8) -> ExecResult<Vec<u8>> {
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

    // Simple 1-D resize using the product of all scales
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

pub(super) fn dispatch_pad(inputs: &[&[u8]], mode: u8) -> ExecResult<Vec<u8>> {
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
