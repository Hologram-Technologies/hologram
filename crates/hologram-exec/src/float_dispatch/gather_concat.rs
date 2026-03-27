use super::elementwise::to_bools;
use super::helpers::*;
use crate::error::{ExecError, ExecResult};
use hologram_core::op::FloatDType;

pub(crate) fn dispatch_gather(
    inputs: &[&[u8]],
    dim: usize,
    dtype: FloatDType,
) -> ExecResult<Vec<u8>> {
    let index_bytes = inputs[0];
    let n_indices = index_bytes.len() / 8;
    let table_bytes = inputs[1];

    match dtype {
        FloatDType::I64 => {
            // i64 gather (shape subgraph): indices select individual i64 values.
            let n_table = table_bytes.len() / 8;
            let mut out = Vec::with_capacity(n_indices * 8);
            for idx in iter_i64(index_bytes).map(|v| v as usize) {
                if idx >= n_table {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("i64 index < {n_table}"),
                        actual: format!("index = {idx}"),
                    });
                }
                let val = read_i64_at(table_bytes, idx).unwrap();
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        FloatDType::I32 => {
            // i32 gather: indices select individual i32 values.
            let n_table = table_bytes.len() / 4;
            let mut out = Vec::with_capacity(n_indices * 4);
            for idx in iter_i64(index_bytes).map(|v| v as usize) {
                if idx >= n_table {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("i32 index < {n_table}"),
                        actual: format!("index = {idx}"),
                    });
                }
                let val = read_i32_at(table_bytes, idx).unwrap();
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        _ => {
            // f32 embedding gather (default for F32, F16, BF16, etc.).
            let table = cast_f32(table_bytes)?;
            let dim = if dim > 0 { dim } else { 1 };
            let vocab = table.len() / dim;
            let mut out = Vec::with_capacity(n_indices * dim * 4);
            for idx in iter_i64(index_bytes).map(|v| v as usize) {
                if idx >= vocab {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("index < {vocab}"),
                        actual: format!("index = {idx}"),
                    });
                }
                out.extend_from_slice(bytemuck::cast_slice(&table[idx * dim..(idx + 1) * dim]));
            }
            Ok(out)
        }
    }
}

pub(crate) fn dispatch_concat(
    inputs: &[&[u8]],
    size_a: usize,
    size_b: usize,
    dtype: FloatDType,
) -> ExecResult<Vec<u8>> {
    let a_bytes = inputs[0];
    let b_bytes = inputs[1];

    let elem_size = dtype.byte_size();

    // For non-f32 types (I64, I32, etc.), use byte-level operations.
    // This prevents i64 data from being split at 4-byte f32 boundaries.
    if !matches!(dtype, FloatDType::F32) {
        if size_a <= 1 && size_b <= 1 {
            // axis=0 concat: simple byte append.
            let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
            out.extend_from_slice(a_bytes);
            out.extend_from_slice(b_bytes);
            return Ok(out);
        }
        // Interleave at element granularity (not f32 granularity).
        let row_bytes_a = size_a * elem_size;
        let row_bytes_b = size_b * elem_size;
        if row_bytes_a > 0 && row_bytes_b > 0 {
            let rows_a = a_bytes.len() / row_bytes_a;
            let rows_b = b_bytes.len() / row_bytes_b;
            if rows_a == rows_b && rows_a > 0 {
                let mut out = Vec::with_capacity(rows_a * (row_bytes_a + row_bytes_b));
                for i in 0..rows_a {
                    out.extend_from_slice(&a_bytes[i * row_bytes_a..(i + 1) * row_bytes_a]);
                    out.extend_from_slice(&b_bytes[i * row_bytes_b..(i + 1) * row_bytes_b]);
                }
                return Ok(out);
            }
        }
        // Fallback: simple append.
        let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
        out.extend_from_slice(a_bytes);
        out.extend_from_slice(b_bytes);
        return Ok(out);
    }

    // F32 path (original behavior).
    if size_a <= 1 && size_b <= 1 {
        // axis=0 concat: simple byte append.
        let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
        out.extend_from_slice(a_bytes);
        out.extend_from_slice(b_bytes);
        return Ok(out);
    }

    if size_a > 0 && a_bytes.len().is_multiple_of(4) && b_bytes.len().is_multiple_of(4) {
        let a = cast_f32(a_bytes)?;
        let b = cast_f32(b_bytes)?;
        let rows_a = a.len() / size_a;
        let rows_b = b.len() / size_b;
        if rows_a == rows_b && rows_a > 0 {
            // Last-axis concat: interleave rows.
            let mut out = Vec::with_capacity(rows_a * (size_a + size_b));
            for i in 0..rows_a {
                out.extend_from_slice(&a[i * size_a..(i + 1) * size_a]);
                out.extend_from_slice(&b[i * size_b..(i + 1) * size_b]);
            }
            Ok(f32_vec_to_bytes(out))
        } else {
            // Fallback: simple append (axis=0 or shape mismatch).
            let mut out = Vec::with_capacity(a.len() + b.len());
            out.extend_from_slice(&a);
            out.extend_from_slice(&b);
            Ok(f32_vec_to_bytes(out))
        }
    } else {
        // Data doesn't cleanly partition into f32 rows — raw byte concat.
        let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
        out.extend_from_slice(a_bytes);
        out.extend_from_slice(b_bytes);
        Ok(out)
    }
}

pub(crate) fn dispatch_where(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [cond (u8 or f32), x (f32), y (f32)]
    // Condition is normalized to per-element booleans via to_bools(),
    // which handles both u8 masks and f32-encoded booleans uniformly.
    let cond = to_bools(inputs[0]);
    let x = cast_f32(inputs[1])?;
    let y = cast_f32(inputs[2])?;

    let n = cond.len().max(x.len()).max(y.len());

    let out: Vec<f32> = (0..n)
        .map(|i| {
            if cond[i % cond.len()] != 0 {
                x[i % x.len()]
            } else {
                y[i % y.len()]
            }
        })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

pub(crate) fn dispatch_range(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [start, limit, delta] — each is a scalar, dtype is either f32 or i64.
    let read_scalar = |b: &[u8]| -> f32 {
        match b.len() {
            8 => i64::from_le_bytes(b.try_into().unwrap_or([0; 8])) as f32,
            4 => f32::from_le_bytes(b.try_into().unwrap_or([0; 4])),
            _ => cast_f32(b)
                .map(|v| v.first().copied().unwrap_or(0.0))
                .unwrap_or(0.0),
        }
    };
    let start = read_scalar(inputs[0]);
    let limit = read_scalar(inputs[1]);
    let delta = read_scalar(inputs[2]);
    let n = if delta != 0.0 {
        ((limit - start) / delta).ceil().max(0.0) as usize
    } else {
        0
    };
    let out: Vec<f32> = (0..n).map(|i| start + i as f32 * delta).collect();
    Ok(f32_vec_to_bytes(out))
}

pub(crate) fn dispatch_shape(
    inputs: &[&[u8]],
    dtype: FloatDType,
    _start: i64,
    _end: i64,
) -> ExecResult<Vec<u8>> {
    // float_dispatch is a kernel-level path with no shape metadata, so it can
    // only infer total element count — not the per-axis dims. Return the element
    // count as a single i64. (The executor path has access to tracked shapes and
    // performs proper per-axis shape extraction with start/end slicing.)
    let elem_bytes = dtype.byte_size();
    let n_elements = if elem_bytes > 0 {
        inputs[0].len() as i64 / elem_bytes as i64
    } else {
        inputs[0].len() as i64
    };
    Ok(bytemuck::cast_slice(&[n_elements]).to_vec())
}

/// N-D gather: `output[k] = data[indices[k]]` for flat-index access.
///
/// `inputs[0]` = data (f32 flat buffer)
/// `inputs[1]` = indices (i64 or i32 flat buffer — detected by buffer alignment)
///
/// Each index value is a flat offset into the data f32 array.
/// Out-of-bounds indices clamp to 0.0 (matches ONNX GatherND behavior for
/// empty-or-zero outputs on invalid indices).
pub(super) fn dispatch_gather_nd(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    if inputs.len() < 2 || inputs[1].is_empty() {
        return Ok(vec![]);
    }
    let data = cast_f32(inputs[0])?;
    let idx_bytes = inputs[1];

    // Detect index dtype from buffer alignment: i64=8B per element, i32=4B.
    let indices: Vec<usize> = if idx_bytes.len().is_multiple_of(8) {
        bytemuck::cast_slice::<u8, i64>(idx_bytes)
            .iter()
            .map(|&i| i.max(0) as usize)
            .collect()
    } else {
        bytemuck::cast_slice::<u8, i32>(idx_bytes)
            .iter()
            .map(|&i| i.max(0) as usize)
            .collect()
    };

    let out: Vec<f32> = indices
        .iter()
        .map(|&i| *data.get(i).unwrap_or(&0.0))
        .collect();
    Ok(f32_vec_to_bytes(out))
}

/// Slice a tensor's shape according to ONNX Shape opset-15 `start`/`end` attributes.
///
/// Returns an i64 buffer containing `in_shape[s..e]` where `s` and `e` are
/// clamped/normalised from `start`/`end` exactly as the ONNX spec requires:
/// - `start = i64::MAX` is treated as "end of dims" (only meaningful for end).
/// - Negative indices count from the rank end.
/// - Indices are clamped to `[0, rank]`.
///
/// Used by the executor's `FloatOp::Shape` handler. Exposed `pub` so that
/// unit tests can exercise start/end slicing without requiring a full compiled
/// graph (the AiGraph pipeline constant-folds Shape when input dims are concrete).
pub fn dispatch_shape_sliced(
    in_shape: &[usize],
    _dtype: FloatDType,
    start: i64,
    end: i64,
) -> ExecResult<Vec<u8>> {
    let rank = in_shape.len() as i64;
    let s = if start < 0 {
        (rank + start).max(0) as usize
    } else {
        (start as usize).min(in_shape.len())
    };
    let e = if end == i64::MAX {
        in_shape.len()
    } else if end < 0 {
        (rank + end).max(0) as usize
    } else {
        (end as usize).min(in_shape.len())
    };
    if s >= e {
        return Ok(vec![]);
    }
    let sliced: Vec<i64> = in_shape[s..e].iter().map(|&d| d as i64).collect();
    Ok(bytemuck::cast_slice(&sliced).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
        f32_vec_to_bytes(v.to_vec())
    }

    fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice::<u8, f32>(b).to_vec()
    }

    #[test]
    fn gather_nd_basic_1d() {
        // data = [10.0, 20.0, 30.0], indices = [2, 0, 1] → [30.0, 10.0, 20.0]
        let data = f32_to_bytes(&[10.0f32, 20.0, 30.0]);
        let idx: Vec<i32> = vec![2, 0, 1];
        let idx_bytes: Vec<u8> = bytemuck::cast_slice::<i32, u8>(&idx).to_vec();
        let result = dispatch_gather_nd(&[&data, &idx_bytes]).unwrap();
        let out = bytes_to_f32(&result);
        assert_eq!(out, &[30.0f32, 10.0, 20.0]);
    }

    #[test]
    fn gather_nd_out_of_bounds_clamps() {
        // index 5 is OOB for data of length 3 → clamp to 0.0
        // Use 3 i32 indices (12 bytes) to avoid ambiguity with the i64 detection
        // heuristic (which triggers when buffer length is divisible by 8).
        let data = f32_to_bytes(&[1.0f32, 2.0, 3.0]);
        let idx: Vec<i32> = vec![0, 5, 2];
        let idx_bytes: Vec<u8> = bytemuck::cast_slice::<i32, u8>(&idx).to_vec();
        let result = dispatch_gather_nd(&[&data, &idx_bytes]).unwrap();
        let out = bytes_to_f32(&result);
        assert_eq!(out[0], 1.0); // data[0]
        assert_eq!(out[1], 0.0); // OOB → clamp to 0.0
        assert_eq!(out[2], 3.0); // data[2]
    }

    #[test]
    fn gather_nd_i64_indices() {
        let data = f32_to_bytes(&[5.0f32, 6.0, 7.0]);
        let idx: Vec<i64> = vec![1, 2];
        let idx_bytes: Vec<u8> = bytemuck::cast_slice::<i64, u8>(&idx).to_vec();
        let result = dispatch_gather_nd(&[&data, &idx_bytes]).unwrap();
        let out = bytes_to_f32(&result);
        assert_eq!(out, &[6.0f32, 7.0]);
    }

    #[test]
    fn gather_nd_empty_indices_returns_empty() {
        let data = f32_to_bytes(&[1.0f32, 2.0]);
        let result = dispatch_gather_nd(&[&data, &[]]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn gather_nd_single_input_returns_empty() {
        let data = f32_to_bytes(&[1.0f32, 2.0]);
        let result = dispatch_gather_nd(&[&data]).unwrap();
        assert!(result.is_empty());
    }
}
