use super::elementwise::to_bools;
use super::helpers::*;
use crate::error::ExecResult;

pub(super) fn dispatch_top_k(inputs: &[&[u8]], _axis: usize, largest: bool) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // K from inputs[1] (i64 scalar)
    let k = if inputs.len() >= 2 && inputs[1].len() >= 8 {
        iter_i64(inputs[1]).next().unwrap_or(1) as usize
    } else if inputs.len() >= 2 && inputs[1].len() >= 4 {
        cast_f32(inputs[1])?.first().copied().unwrap_or(1.0) as usize
    } else {
        1
    };

    let k = k.min(data.len());
    // Simple: sort all elements, take top/bottom K
    // Returns values only (indices would need separate output handling)
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    if largest {
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    indexed.truncate(k);

    // Output: interleave values and indices as f32
    // Standard TopK produces two outputs, but since we have single output dispatch,
    // output values as f32 (the primary output)
    let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
    Ok(f32_vec_to_bytes(values))
}

pub(super) fn dispatch_scatter_nd(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [data, indices, updates]
    let data = cast_f32(inputs[0])?;
    let indices_bytes = inputs[1];
    let updates = cast_f32(inputs[2])?;

    let mut out = data.into_owned();

    // Simple 1-D scatter: indices are i64, each indexing into the flat output
    let indices: Vec<usize> = iter_i64(indices_bytes).map(|v| v as usize).collect();

    for (i, &idx) in indices.iter().enumerate() {
        if idx < out.len() && i < updates.len() {
            out[idx] = updates[i];
        }
    }

    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_cumsum(inputs: &[&[u8]], _axis: usize) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let mut out = data.into_owned();

    // Simple 1-D cumulative sum
    for i in 1..out.len() {
        out[i] += out[i - 1];
    }

    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_nonzero(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // NonZero returns indices of non-zero elements as i64
    let indices: Vec<i64> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(i, _)| i as i64)
        .collect();

    Ok(bytemuck::cast_slice(&indices).to_vec())
}

pub(super) fn dispatch_compress(inputs: &[&[u8]], _axis: usize) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // condition from inputs[1]: boolean mask
    let cond = to_bools(inputs[1]);

    let out: Vec<f32> = data
        .iter()
        .zip(cond.iter().chain(std::iter::repeat(&0u8)))
        .filter(|(_, &c)| c != 0)
        .map(|(&v, _)| v)
        .collect();

    Ok(f32_vec_to_bytes(out))
}

pub(super) fn dispatch_reverse_sequence(
    inputs: &[&[u8]],
    _batch_axis: usize,
    _time_axis: usize,
) -> ExecResult<Vec<u8>> {
    // Simple: reverse the entire f32 sequence
    let data = cast_f32(inputs[0])?;
    let mut out = data.into_owned();
    out.reverse();
    Ok(f32_vec_to_bytes(out))
}
