//! Runtime shape resolution for TapeKernel dispatch.
//!
//! Ops bake dimension parameters (size, k, m, n, input_h, input_w) at compile
//! time. When runtime tensor sizes differ from compiled sizes (variable-length
//! sequences, different spatial resolutions), these baked values are stale.
//!
//! This module provides pure functions that resolve baked dimensions using
//! runtime `TensorMeta` from the `BufferArena`. Each function falls back to
//! the baked value when runtime metadata is unavailable, preserving backward
//! compatibility with archives that lack shape metadata.

use hologram_core::op::TensorMeta;
use smallvec::SmallVec;

/// Runtime metadata for dispatch inputs, parallel to `&[&[u8]]` input slices.
pub type InputMetas = SmallVec<[Option<TensorMeta>; 4]>;

// ── Last-dimension resolution (norms, softmax, reduce) ──────────────────────

/// Resolve the last-axis dimension for norm/softmax/reduce ops.
///
/// Priority:
/// 1. Runtime input meta's last dim (if available and > 0)
/// 2. Weight tensor length (for LayerNorm: input[1] is always the norm dim)
/// 3. Baked compiled size (if > 0 and divides input element count)
/// 4. Input element count (1-D fallback)
pub fn resolve_last_dim(
    compiled_size: u32,
    input_meta: Option<&TensorMeta>,
    input_byte_len: usize,
) -> usize {
    let n_floats = input_byte_len / 4;

    // Best: use N-D shape from runtime metadata.
    // Validate that meta's total elements matches buffer (catches dtype mismatches
    // where meta was computed with wrong elem_size, e.g., I64 vs F32).
    if let Some(meta) = input_meta {
        if let Some(d) = meta.last_dim() {
            let d = d as usize;
            let meta_total = meta.n_elems();
            if d > 0 && meta_total == n_floats && n_floats.is_multiple_of(d) {
                return d;
            }
        }
    }
    // Fallback: use compiled size if it cleanly divides the input.
    if compiled_size > 0 && n_floats > 0 && n_floats.is_multiple_of(compiled_size as usize) {
        return compiled_size as usize;
    }
    // Last resort: treat entire input as one row.
    n_floats
}

/// Resolve last dim for LayerNorm specifically, using weight tensor as
/// authoritative source (weight length == norm dimension, always).
pub fn resolve_last_dim_with_weight(
    compiled_size: u32,
    input_meta: Option<&TensorMeta>,
    input_byte_len: usize,
    weight_byte_len: usize,
) -> usize {
    // Weight tensor length is always the norm dimension for LayerNorm.
    let weight_size = weight_byte_len / 4;
    if weight_size > 0 {
        return weight_size;
    }
    resolve_last_dim(compiled_size, input_meta, input_byte_len)
}

// ── MatMul dimension resolution ─────────────────────────────────────────────

/// Resolve M, K, N for MatMul from runtime metadata.
///
/// For A [batch..., M, K] x B [batch..., K, N] → [batch..., M, N]:
/// - K = A's last dim = B's second-to-last dim
/// - M = A's second-to-last dim
/// - N = B's last dim
///
/// Falls back to compiled values, then to buffer-length inference.
pub fn resolve_matmul_dims(
    compiled_m: u32,
    compiled_k: u32,
    compiled_n: u32,
    a_meta: Option<&TensorMeta>,
    b_meta: Option<&TensorMeta>,
    a_byte_len: usize,
    b_byte_len: usize,
) -> (usize, usize, usize) {
    let a_floats = a_byte_len / 4;
    let b_floats = b_byte_len / 4;

    // Try N-D metadata first.
    if let (Some(a), Some(b)) = (a_meta, b_meta) {
        // Debug: uncomment to trace matmul resolution
        // eprintln!("matmul metas: A={:?} B={:?}", a.shape(), b.shape());
        if a.ndim >= 2 && b.ndim >= 2 {
            let k = a.last_dim().unwrap_or(compiled_k) as usize;
            let m = a.second_last_dim().unwrap_or(compiled_m) as usize;
            let n = b.last_dim().unwrap_or(compiled_n) as usize;
            if k > 0 && m > 0 && n > 0 {
                return (m, k, n);
            }
        }
    }

    // Try individual metas.
    let k: usize = a_meta
        .and_then(|a: &TensorMeta| a.last_dim())
        .or_else(|| b_meta.and_then(|b: &TensorMeta| b.second_last_dim()))
        .map(|d: u32| d as usize)
        .unwrap_or(compiled_k as usize);

    let m: usize = a_meta
        .and_then(|a: &TensorMeta| a.second_last_dim())
        .map(|d: u32| d as usize)
        .unwrap_or(compiled_m as usize);

    let n: usize = b_meta
        .and_then(|b: &TensorMeta| b.last_dim())
        .map(|d: u32| d as usize)
        .unwrap_or(compiled_n as usize);

    if k > 0 && m > 0 && n > 0 {
        return (m, k, n);
    }

    // Buffer-length inference (existing infer_matmul_dims pattern).
    let ck = if compiled_k > 0 {
        compiled_k as usize
    } else if compiled_m > 0 && a_floats > 0 {
        a_floats / compiled_m as usize
    } else {
        // Can't infer — return compiled values as-is.
        return (
            compiled_m as usize,
            compiled_k as usize,
            compiled_n as usize,
        );
    };
    let cm = if a_floats > 0 && ck > 0 {
        a_floats / ck
    } else {
        compiled_m as usize
    };
    let cn = if b_floats > 0 && ck > 0 {
        b_floats / ck
    } else {
        compiled_n as usize
    };
    (cm, ck, cn)
}

// ── Spatial dimension resolution (Conv2d, pooling, etc.) ────────────────────

/// Resolve spatial (H, W) for NCHW vision ops from runtime metadata.
///
/// For input [N, C, H, W]:
/// - H = dims[ndim-2]
/// - W = dims[ndim-1]
pub fn resolve_spatial_dims(
    compiled_h: u32,
    compiled_w: u32,
    input_meta: Option<&TensorMeta>,
) -> (usize, usize) {
    if let Some(meta) = input_meta {
        if let Some((h, w)) = meta.spatial_hw() {
            if h > 0 && w > 0 {
                return (h as usize, w as usize);
            }
        }
    }
    (compiled_h as usize, compiled_w as usize)
}

/// Resolve (channels, spatial_h, spatial_w) for GlobalAvgPool from NCHW meta.
pub fn resolve_global_avg_pool_dims(
    compiled_c: u32,
    compiled_h: u32,
    compiled_w: u32,
    input_meta: Option<&TensorMeta>,
) -> (usize, usize, usize) {
    if let Some(meta) = input_meta {
        if meta.ndim >= 3 {
            let c = meta.dims[meta.ndim as usize - 3] as usize;
            let h = meta.dims[meta.ndim as usize - 2] as usize;
            let w = meta.dims[meta.ndim as usize - 1] as usize;
            if c > 0 && h > 0 && w > 0 {
                return (c, h, w);
            }
        }
    }
    (
        compiled_c as usize,
        compiled_h as usize,
        compiled_w as usize,
    )
}

// ── Transpose shape resolution ──────────────────────────────────────────────

/// Resolve input shape for Transpose from runtime metadata.
pub fn resolve_transpose_shape(
    compiled_shape: &[u32; 8],
    ndim: u8,
    input_meta: Option<&TensorMeta>,
) -> Vec<usize> {
    let n = ndim as usize;
    if let Some(meta) = input_meta {
        if meta.ndim as usize == n {
            return meta.shape().iter().map(|&d| d as usize).collect();
        }
    }
    compiled_shape[..n].iter().map(|&d| d as usize).collect()
}

// ── Output meta computation ─────────────────────────────────────────────────

/// Compute runtime output TensorMeta from input metas and actual output byte count.
///
/// This is the "TCP header" approach: instead of relying on compiled shapes,
/// derive the output shape from the operation semantics + actual inputs.
/// Returns the compiled meta when input metas are insufficient.
pub fn compute_output_meta(
    input_metas: &InputMetas,
    compiled_meta: Option<TensorMeta>,
    out_bytes: usize,
    elem_size: usize,
) -> Option<TensorMeta> {
    let out_elems = if elem_size > 0 {
        out_bytes / elem_size
    } else {
        out_bytes / 4
    };
    let dtype = input_metas
        .first()
        .and_then(|m| m.as_ref())
        .map(|m| m.dtype)
        .unwrap_or(hologram_core::op::FloatDType::F32);

    // If we have a compiled meta whose element count matches actual output,
    // it's correct — use it (preserves N-D shape).
    if let Some(ref cm) = compiled_meta {
        if cm.n_elems() == out_elems {
            return compiled_meta;
        }
    }

    // Try to derive from first input meta (covers unary, norm, softmax,
    // activation, and other element-preserving ops).
    if let Some(Some(input_meta)) = input_metas.first() {
        if input_meta.n_elems() == out_elems {
            // Element-preserving: output shape = input shape.
            return Some(*input_meta);
        }
    }

    // For binary ops: if output elems match the larger input, use its shape.
    if input_metas.len() >= 2 {
        for m in input_metas.iter().flatten() {
            if m.n_elems() == out_elems {
                return Some(*m);
            }
        }
    }

    // If compiled meta exists but element count differs, try to adjust
    // one dimension to match actual output (variable-length scaling).
    // Use simple first-match here (not input-disambiguated) because this
    // handles ALL ops, not just Reshape. Reshape has its own handler with
    // input-based disambiguation in the execution loop.
    if let Some(cm) = compiled_meta {
        if let Some(adjusted) = scale_meta_to_fit(cm, out_elems, None) {
            return Some(adjusted);
        }
    }

    // Last resort: 1-D meta from actual output.
    if out_elems > 0 {
        Some(TensorMeta::new(dtype, &[out_elems]))
    } else {
        None
    }
}

/// Scale one dimension of `compiled` to make its total elements equal `target_elems`.
///
/// When multiple dimensions could be scaled (e.g., seq=32 and heads=32 both
/// produce valid results), uses `input_meta` to disambiguate: the scaled
/// dimension should match a dimension in the input that actually changed
/// from its compiled value.
pub fn scale_meta_to_fit(
    compiled: TensorMeta,
    target_elems: usize,
    input_meta: Option<&TensorMeta>,
) -> Option<TensorMeta> {
    let compiled_elems = compiled.n_elems();
    if compiled.ndim == 0 || compiled_elems == 0 || target_elems == 0 {
        return None;
    }
    if compiled_elems == target_elems {
        return Some(compiled);
    }

    let ratio = target_elems as f64 / compiled_elems as f64;
    let mut best: Option<(usize, u32)> = None;

    for i in 0..compiled.ndim as usize {
        let old_dim = compiled.dims[i] as usize;
        if old_dim == 0 {
            continue;
        }
        let scaled = (old_dim as f64 * ratio).round() as u32;
        let mut check = compiled;
        check.dims[i] = scaled;
        if check.n_elems() != target_elems {
            continue;
        }
        // This dim produces a valid scaling. Check if the input meta
        // confirms it's the right one (the scaled value matches an
        // input dimension that differs from the compiled dim).
        let confirmed = input_meta.is_some_and(|im| {
            (0..im.ndim as usize).any(|j| im.dims[j] == scaled && scaled != compiled.dims[i])
        });
        match best {
            None => best = Some((i, scaled)),
            Some(_) if confirmed => {
                best = Some((i, scaled));
                break; // Input-confirmed match — highest confidence.
            }
            _ => {} // Keep the first unconfirmed match if no better option.
        }
    }

    best.map(|(i, scaled)| {
        let mut adjusted = compiled;
        adjusted.dims[i] = scaled;
        adjusted
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::FloatDType;

    fn meta(shape: &[usize]) -> TensorMeta {
        TensorMeta::new(FloatDType::F32, shape)
    }

    #[test]
    fn resolve_last_dim_from_meta() {
        let m = meta(&[1, 320, 64, 64]);
        // Total elements = 1*320*64*64 = 1310720. Buffer = 1310720*4 bytes.
        let byte_len = 1310720 * 4;
        assert_eq!(resolve_last_dim(0, Some(&m), byte_len), 64);
    }

    #[test]
    fn resolve_last_dim_fallback_compiled() {
        assert_eq!(resolve_last_dim(320, None, 320 * 4 * 4), 320);
    }

    #[test]
    fn resolve_last_dim_fallback_buffer() {
        assert_eq!(resolve_last_dim(0, None, 100 * 4), 100);
    }

    #[test]
    fn resolve_last_dim_with_weight_overrides() {
        let m = meta(&[1, 320, 64, 64]);
        // Weight says 320, meta says 64 — weight wins for LayerNorm.
        assert_eq!(resolve_last_dim_with_weight(0, Some(&m), 0, 320 * 4), 320);
    }

    #[test]
    fn resolve_matmul_from_meta() {
        let a = meta(&[1, 32, 64]); // [batch, M, K]
        let b = meta(&[1, 64, 128]); // [batch, K, N]
        let (m, k, n) = resolve_matmul_dims(0, 0, 0, Some(&a), Some(&b), 0, 0);
        assert_eq!((m, k, n), (32, 64, 128));
    }

    #[test]
    fn resolve_spatial_from_meta() {
        let m = meta(&[1, 64, 128, 256]);
        let (h, w) = resolve_spatial_dims(0, 0, Some(&m));
        assert_eq!((h, w), (128, 256));
    }

    #[test]
    fn resolve_spatial_fallback() {
        let (h, w) = resolve_spatial_dims(64, 64, None);
        assert_eq!((h, w), (64, 64));
    }

    #[test]
    fn resolve_transpose_from_meta() {
        let compiled = [1, 4, 64, 64, 0, 0, 0, 0];
        let m = meta(&[1, 4, 128, 128]);
        let shape = resolve_transpose_shape(&compiled, 4, Some(&m));
        assert_eq!(shape, vec![1, 4, 128, 128]);
    }
}
