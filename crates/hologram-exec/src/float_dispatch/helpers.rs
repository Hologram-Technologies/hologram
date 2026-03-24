use crate::error::ExecResult;

/// Allocate space for `n` f32s in `out_buf` and return a mutable f32 slice.
///
/// Writes directly into `out_buf` — no intermediate Vec allocation.
#[inline]
pub(super) fn alloc_f32_in(out_buf: &mut Vec<u8>, n: usize) -> &mut [f32] {
    let start = out_buf.len();
    out_buf.resize(start + n * 4, 0);
    bytemuck::cast_slice_mut(&mut out_buf[start..])
}

/// Transpose a row-major matrix [rows × cols] → [cols × rows].
#[cfg_attr(all(feature = "accelerate", target_os = "macos"), allow(dead_code))]
pub(super) fn transpose_f32(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

pub(super) fn cast_f32(bytes: &[u8]) -> ExecResult<std::borrow::Cow<'_, [f32]>> {
    match bytemuck::try_cast_slice(bytes) {
        Ok(s) => Ok(std::borrow::Cow::Borrowed(s)),
        Err(_) => Ok(std::borrow::Cow::Owned(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )),
    }
}

/// Iterator over i64 values read from potentially-misaligned bytes.
pub(super) fn iter_i64(bytes: &[u8]) -> impl Iterator<Item = i64> + '_ {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
}

/// Read a single i64 at element index `idx` from potentially-misaligned bytes.
pub(super) fn read_i64_at(bytes: &[u8], idx: usize) -> Option<i64> {
    let off = idx * 8;
    bytes
        .get(off..off + 8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
}

/// Iterator over i32 values read from potentially-misaligned bytes.
pub(super) fn iter_i32(bytes: &[u8]) -> impl Iterator<Item = i32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
}

/// Read a single i32 at element index `idx` from potentially-misaligned bytes.
pub(super) fn read_i32_at(bytes: &[u8], idx: usize) -> Option<i32> {
    let off = idx * 4;
    bytes
        .get(off..off + 4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
}

/// Zero-copy conversion from `Vec<f32>` to `Vec<u8>`.
///
/// Takes ownership and reinterprets the backing allocation in-place —
/// no memcpy, no extra allocation.
pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let len = data.len() * 4;
    let cap = data.capacity() * 4;
    let ptr = data.as_ptr() as *mut u8;
    std::mem::forget(data);
    // SAFETY: f32 has alignment >= u8; len/cap are scaled correctly.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub(super) fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
pub(super) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

use smallvec::SmallVec;

/// Most tensors have ≤6 dimensions; SmallVec avoids heap allocation for common cases.
pub(super) type StrideVec = SmallVec<[usize; 6]>;

/// Compute strides for a shape (row-major). Stack-allocated for ≤6 dims.
pub(super) fn compute_strides_small(shape: &[usize]) -> StrideVec {
    let mut strides = SmallVec::from_elem(1usize, shape.len());
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute strides for a shape (row-major). Public API (returns Vec).
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    compute_strides_small(shape).to_vec()
}

/// Compute broadcast strides: for dimensions where `src` has size 1 (broadcast),
/// the stride is 0 (same element repeated). Otherwise, uses normal strides.
pub(super) fn compute_broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> StrideVec {
    let src_strides = compute_strides_small(src_shape);
    let offset = out_shape.len() - src_shape.len();
    let mut strides = SmallVec::from_elem(0usize, out_shape.len());
    for i in 0..src_shape.len() {
        if src_shape[i] != 1 {
            strides[i + offset] = src_strides[i];
        }
        // else: stride stays 0 (broadcast dimension)
    }
    strides
}

/// Convert a flat output index to a flat source index using broadcast strides.
#[inline]
pub(super) fn broadcast_flat_index(
    flat_idx: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    src_strides: &[usize],
) -> usize {
    let mut src_idx = 0;
    let mut remaining = flat_idx;
    for i in 0..out_shape.len() {
        let coord = remaining / out_strides[i];
        remaining %= out_strides[i];
        src_idx += coord * src_strides[i];
    }
    src_idx
}

/// Compute numpy-style broadcast output shape.
/// Returns `None` if shapes are not broadcast-compatible (dimensions must be
/// equal or one of them must be 1).
pub(super) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if da != db && da != 1 && db != 1 {
            return None; // Not broadcast-compatible
        }
        result.push(da.max(db));
    }
    Some(result)
}

/// Heuristic to infer (C, H, W) from total element count and batch size.
pub(super) fn infer_nchw(total: usize, n: usize) -> (usize, usize, usize) {
    let per_batch = total / n.max(1);
    // Try common channel counts: 1, 3, then factors
    for &c in &[3, 1, 64, 128, 256, 512, 32, 16] {
        if per_batch.is_multiple_of(c) {
            let spatial = per_batch / c;
            let h = (spatial as f32).sqrt() as usize;
            if h > 0 && spatial.is_multiple_of(h) {
                return (c, h, spatial / h);
            }
        }
    }
    // Fallback: single channel, try square
    let h = (per_batch as f32).sqrt() as usize;
    if h > 0 && per_batch.is_multiple_of(h) {
        (1, h, per_batch / h)
    } else {
        (1, 1, per_batch)
    }
}
