use super::helpers::*;
use crate::error::ExecResult;

/// Reshape: data passes through, shape is read from the shape tensor (inputs[1]).
/// Returns `(data_bytes, new_shape)`.
pub fn dispatch_reshape_with_shape(inputs: &[&[u8]]) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let data = inputs[0].to_vec();
    if inputs.len() >= 2 && !inputs[1].is_empty() {
        let n_elems = data.len() / 4; // assume f32

        let shape = crate::eval::shape_resolve::parse_shape_values(inputs[1], n_elems)
            .unwrap_or_else(|| vec![n_elems]);

        let shape_product: usize = shape.iter().product();
        if shape_product == n_elems {
            Ok((data, shape))
        } else if shape_product > n_elems && n_elems > 0 && shape_product <= n_elems * 1024 {
            // Broadcast expansion (e.g. GQA key repeat): replicate data.
            let src = cast_f32(&data)?;
            let expanded = broadcast_to(&src, n_elems, &shape);
            Ok((f32_vec_to_bytes(expanded), shape))
        } else {
            // Can't match — fall back to 1-D.
            Ok((data, vec![n_elems]))
        }
    } else {
        // No shape tensor — return 1D.
        let n = data.len() / 4;
        Ok((data, vec![n]))
    }
}

/// Transpose: physically reorder f32 data according to `perm`.
/// Returns `(permuted_bytes, output_shape)`.
pub fn dispatch_transpose(
    input: &[u8],
    perm: &[u8],
    input_shape: &[usize],
) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let src = cast_f32(input)?;
    let ndim = perm.len();

    if ndim == 0 || input_shape.is_empty() {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }

    // Guard: perm must not reference dims beyond the input shape.
    if perm.iter().any(|&p| (p as usize) >= input_shape.len()) {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }

    let strides = compute_strides(input_shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();
    let out_strides = compute_strides(&out_shape);

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
pub(super) fn broadcast_to(src: &[f32], n_src: usize, target_shape: &[usize]) -> Vec<f32> {
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

    let src_strides = compute_strides(&src_shape);
    let tgt_strides = compute_strides(target_shape);

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
