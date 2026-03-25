use super::helpers::*;
use crate::error::ExecResult;

/// Reshape: data passes through, shape is read from the shape tensor (inputs[1]).
/// Returns `(data_bytes, new_shape)`.
///
/// Avoids copying the data buffer when the reshape is a pure metadata change
/// (shape product matches element count). Only copies when broadcast expansion
/// is needed.
pub fn dispatch_reshape_with_shape(inputs: &[&[u8]]) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let n_elems = inputs[0].len() / 4; // assume f32

    if inputs.len() >= 2 && !inputs[1].is_empty() {
        let shape = parse_shape_values(inputs[1], n_elems).unwrap_or_else(|| vec![n_elems]);

        let shape_product: usize = shape.iter().product();
        if shape_product == n_elems {
            // Pure reshape — data passes through unchanged. Copy only the bytes.
            Ok((inputs[0].to_vec(), shape))
        } else if shape_product > n_elems && n_elems > 0 && shape_product <= n_elems * 1024 {
            // Broadcast expansion (e.g. GQA key repeat): replicate data.
            let src = cast_f32(inputs[0])?;
            let expanded = broadcast_to(&src, n_elems, &shape);
            Ok((f32_vec_to_bytes(expanded), shape))
        } else {
            // Can't match — fall back to 1-D.
            Ok((inputs[0].to_vec(), vec![n_elems]))
        }
    } else {
        // No shape tensor — return 1D.
        Ok((inputs[0].to_vec(), vec![n_elems]))
    }
}

/// Transpose: physically reorder f32 data according to `perm`.
/// Returns `(permuted_bytes, output_shape)`.
pub fn dispatch_transpose(
    input: &[u8],
    perm: &[u8],
    input_shape: &[usize],
) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let ndim = perm.len();

    // Early returns before any allocation or cast.
    if ndim == 0 || input_shape.is_empty() {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }
    if perm.iter().any(|&p| (p as usize) >= input_shape.len()) {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }

    // Check if perm is identity (no-op transpose).
    let is_identity = perm.iter().enumerate().all(|(i, &p)| p as usize == i);
    if is_identity {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }

    let src = cast_f32(input)?;
    let strides = compute_strides_small(input_shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();
    let out_strides = compute_strides_small(&out_shape);

    let total = src.len();
    // Verify shape matches data length.
    let shape_elems: usize = input_shape
        .iter()
        .copied()
        .fold(1usize, usize::saturating_mul);
    if shape_elems != total {
        // Shape doesn't match data — fall back to pass-through.
        return Ok((input.to_vec(), input_shape.to_vec()));
    }
    let mut dst = vec![0.0f32; total];
    for (flat_idx, dst_val) in dst.iter_mut().enumerate().take(total) {
        let mut remaining = flat_idx;
        let mut src_flat = 0usize;
        for (d, &p) in perm.iter().enumerate() {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            src_flat += coord * strides[p as usize];
        }
        // Guard against stride overflow from shape mismatches.
        if src_flat >= total {
            continue;
        }
        *dst_val = src[src_flat];
    }
    Ok((f32_vec_to_bytes(dst), out_shape))
}

/// Broadcast `src` to fill `target_shape` using numpy-style broadcasting.
///
/// Infers the source shape from `n_src` and `target_shape`: each source dim
/// is either 1 (broadcast) or equals the target dim.
pub(crate) fn broadcast_to(src: &[f32], n_src: usize, target_shape: &[usize]) -> Vec<f32> {
    let target_elems: usize = target_shape
        .iter()
        .copied()
        .fold(1usize, usize::saturating_mul);
    if src.is_empty() || target_elems == 0 {
        return vec![0.0f32; target_elems];
    }

    // Infer source shape: for each target dim, source dim is target dim
    // if it divides evenly into remaining elements, else 1 (broadcast).
    let ndim = target_shape.len();
    let mut src_shape = vec![1usize; ndim];
    let mut remaining = n_src;
    for i in (0..ndim).rev() {
        if remaining > 1 && target_shape[i] > 0 && remaining.is_multiple_of(target_shape[i]) {
            src_shape[i] = target_shape[i];
            remaining /= target_shape[i];
        }
    }

    let src_strides = compute_strides_small(&src_shape);
    let tgt_strides = compute_strides_small(target_shape);

    let mut out = Vec::with_capacity(target_elems);
    for flat_idx in 0..target_elems {
        let mut src_flat = 0usize;
        let mut rem = flat_idx;
        for d in 0..ndim {
            let coord = rem / tgt_strides[d];
            rem %= tgt_strides[d];
            // Wrap coordinate if source dim is 1 (broadcast).
            let src_coord = if src_shape[d] == 1 { 0 } else { coord };
            src_flat += src_coord * src_strides[d];
        }
        out.push(src[src_flat]);
    }
    out
}

/// Parse shape values from a shape tensor's raw bytes.
///
/// Handles both i64 and i32 encoded shapes with alignment-safe reads.
/// Resolves a single -1/0 dimension from the element count.
fn parse_shape_values(shape_bytes: &[u8], n_elems: usize) -> Option<Vec<usize>> {
    if shape_bytes.is_empty() {
        return None;
    }

    let shape_vals: Vec<i64> = if shape_bytes.len().is_multiple_of(8) {
        let vals: Vec<i64> = shape_bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let reasonable = vals.iter().all(|&v| v >= -1 && v <= n_elems as i64 + 1);
        if reasonable {
            vals
        } else if shape_bytes.len().is_multiple_of(4) {
            shape_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect()
        } else {
            vals
        }
    } else if shape_bytes.len().is_multiple_of(4) {
        shape_bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
            .collect()
    } else {
        return None;
    };

    let shape: Vec<usize> = shape_vals
        .iter()
        .map(|&v| {
            if v == -1 || v == 0 {
                0
            } else if v < 0 {
                1
            } else {
                v as usize
            }
        })
        .collect();

    let zero_count = shape.iter().filter(|&&d| d == 0).count();
    if zero_count == 1 {
        let known: usize = shape.iter().filter(|&&d| d > 0).product::<usize>().max(1);
        let unknown = if known > 0 { n_elems / known } else { n_elems };
        Some(
            shape
                .iter()
                .map(|&d| if d == 0 { unknown } else { d })
                .collect(),
        )
    } else {
        Some(shape)
    }
}
