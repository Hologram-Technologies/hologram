use super::helpers::*;
use crate::error::ExecResult;

pub(super) fn unary_map(inputs: &[&[u8]], f: impl Fn(f32) -> f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<f32> = x.iter().map(|&v| f(v)).collect();
    Ok(f32_vec_to_bytes(out))
}

pub(super) fn binary_elementwise(
    inputs: &[&[u8]],
    f: impl Fn(f32, f32) -> f32,
) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let out_len = a.len().max(b.len());
    let out: Vec<f32> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]))
        .collect();
    Ok(f32_vec_to_bytes(out))
}

/// Binary elementwise with proper N-D broadcasting using input shapes.
///
/// Follows numpy broadcasting rules: dimensions are compared right-to-left,
/// and each dimension must be either equal or 1. A dimension of 1 is broadcast
/// (repeated) to match the other operand's dimension.
pub(super) fn binary_elementwise_broadcast(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    f: impl Fn(f32, f32) -> f32,
) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let sa = &input_shapes[0];
    let sb = &input_shapes[1];

    // Fast path: same shape or one is scalar — cycling is correct.
    if sa == sb || a.len() == 1 || b.len() == 1 {
        let out_len = a.len().max(b.len());
        let out: Vec<f32> = (0..out_len)
            .map(|i| f(a[i % a.len()], b[i % b.len()]))
            .collect();
        return Ok(f32_vec_to_bytes(out));
    }

    // Validate shapes match data sizes.
    let a_prod: usize = sa.iter().product();
    let b_prod: usize = sb.iter().product();
    if a_prod != a.len() || b_prod != b.len() {
        // Shape doesn't match data — fall back to cycling.
        let out_len = a.len().max(b.len());
        let out: Vec<f32> = (0..out_len)
            .map(|i| f(a[i % a.len()], b[i % b.len()]))
            .collect();
        return Ok(f32_vec_to_bytes(out));
    }

    // Compute broadcast output shape. Fall back to cycling if shapes
    // are not broadcast-compatible (e.g. non-f32 dtype mismatch).
    let out_shape = match broadcast_shapes(sa, sb) {
        Some(s) => s,
        None => {
            let out_len = a.len().max(b.len());
            let out: Vec<f32> = (0..out_len)
                .map(|i| f(a[i % a.len()], b[i % b.len()]))
                .collect();
            return Ok(f32_vec_to_bytes(out));
        }
    };
    let out_len: usize = out_shape.iter().product();

    // If broadcast would inflate output beyond both input sizes, the compiled
    // input_shapes are stale (0-sentinels resolved to wrong values creating
    // orthogonal broadcast dimensions at runtime). Fall back to cycling.
    if out_len > a.len().max(b.len()) {
        let safe_len = a.len().max(b.len());
        let out: Vec<f32> = (0..safe_len)
            .map(|i| f(a[i % a.len()], b[i % b.len()]))
            .collect();
        return Ok(f32_vec_to_bytes(out));
    }

    // Compute strides for index mapping.
    let a_strides = compute_broadcast_strides(sa, &out_shape);
    let b_strides = compute_broadcast_strides(sb, &out_shape);
    let out_strides = compute_strides_small(&out_shape);

    let out: Vec<f32> = (0..out_len)
        .map(|flat_idx| {
            let a_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &a_strides);
            let b_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &b_strides);
            f(a[a_idx], b[b_idx])
        })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

/// Binary compare with proper N-D broadcasting.
pub(super) fn binary_compare_broadcast(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    f: impl Fn(f32, f32) -> bool,
) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let sa = &input_shapes[0];
    let sb = &input_shapes[1];

    if sa == sb || a.len() == 1 || b.len() == 1 {
        let out_len = a.len().max(b.len());
        let out: Vec<u8> = (0..out_len)
            .map(|i| {
                if f(a[i % a.len()], b[i % b.len()]) {
                    1u8
                } else {
                    0u8
                }
            })
            .collect();
        return Ok(out);
    }

    let a_prod: usize = sa.iter().product();
    let b_prod: usize = sb.iter().product();
    if a_prod != a.len() || b_prod != b.len() {
        let out_len = a.len().max(b.len());
        let out: Vec<u8> = (0..out_len)
            .map(|i| {
                if f(a[i % a.len()], b[i % b.len()]) {
                    1u8
                } else {
                    0u8
                }
            })
            .collect();
        return Ok(out);
    }

    let out_shape = match broadcast_shapes(sa, sb) {
        Some(s) => s,
        None => {
            let out_len = a.len().max(b.len());
            let out: Vec<u8> = (0..out_len)
                .map(|i| {
                    if f(a[i % a.len()], b[i % b.len()]) {
                        1u8
                    } else {
                        0u8
                    }
                })
                .collect();
            return Ok(out);
        }
    };
    let out_len: usize = out_shape.iter().product();

    let a_strides = compute_broadcast_strides(sa, &out_shape);
    let b_strides = compute_broadcast_strides(sb, &out_shape);
    let out_strides = compute_strides_small(&out_shape);

    let out: Vec<u8> = (0..out_len)
        .map(|flat_idx| {
            let a_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &a_strides);
            let b_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &b_strides);
            if f(a[a_idx], b[b_idx]) {
                1u8
            } else {
                0u8
            }
        })
        .collect();
    Ok(out)
}

/// Convert raw bytes to per-element booleans (0 or 1).
///
/// If the buffer is f32-aligned, each 4-byte f32 becomes one bool (nonzero → 1).
/// Otherwise, each byte is a boolean directly.
pub(super) fn to_bools(data: &[u8]) -> Vec<u8> {
    // If all bytes are 0 or 1 the data is packed boolean (1 byte per element),
    // which is what binary_compare / IsNaN / Cast-to-bool ops emit.
    // Checking this FIRST avoids misinterpreting packed bools as f32 when the
    // length happens to be a multiple of 4 — that would collapse 4 booleans
    // into one f32 value, corrupting attention masks.
    if data.iter().all(|&b| b <= 1) {
        return data.iter().map(|&v| (v != 0) as u8).collect();
    }
    // Otherwise, data is f32-encoded (e.g., a float attention mask of 0.0/1.0
    // from a Cast or Mul — bytes will include values > 1 like 0x3F, 0x80).
    if data.len().is_multiple_of(4) && data.len() >= 4 {
        if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(data) {
            return floats.iter().map(|&v| (v != 0.0) as u8).collect();
        }
    }
    // Fallback: byte-level booleans.
    data.iter().map(|&v| (v != 0) as u8).collect()
}

pub(super) fn binary_byte_bool(inputs: &[&[u8]], f: impl Fn(u8, u8) -> u8) -> ExecResult<Vec<u8>> {
    let a = to_bools(inputs[0]);
    let b = to_bools(inputs[1]);
    let out_len = a.len().max(b.len());
    let out: Vec<u8> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]))
        .collect();
    Ok(out)
}

pub(super) fn unary_byte_bool(inputs: &[&[u8]], f: impl Fn(u8) -> u8) -> ExecResult<Vec<u8>> {
    let bools = to_bools(inputs[0]);
    let out: Vec<u8> = bools.iter().map(|&x| f(x)).collect();
    Ok(out)
}

pub(super) fn binary_compare(
    inputs: &[&[u8]],
    f: impl Fn(f32, f32) -> bool,
) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let out_len = a.len().max(b.len());
    let out: Vec<u8> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]) as u8)
        .collect();
    Ok(out)
}

pub(super) fn dispatch_isnan(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<u8> = x.iter().map(|v| v.is_nan() as u8).collect();
    Ok(out)
}
