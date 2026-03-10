//! Float-domain kernel dispatch for `FloatOp` graph operations.
//!
//! All kernels operate on `&[u8]` inputs interpreted as `&[f32]` via bytemuck,
//! matching the pattern used by `MatMulLut4`/`MatMulLut8`.

use hologram_core::op::{bits_to_f32, FloatDType, FloatOp, OpCategory};

use crate::error::{ExecError, ExecResult};

/// Parameters for a GEMM (General Matrix Multiply) operation:
/// `C = alpha * op(A) * op(B) + beta * C`
#[derive(Debug, Clone, Copy)]
pub struct GemmParams {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

/// Dispatch a `FloatOp` with the given byte-buffer inputs.
///
/// Category-based dispatch: generic kernel patterns (unary, binary, compare,
/// byte-bool) are handled by `OpCategory`, while ops needing dedicated logic
/// are dispatched individually via `dispatch_custom`.
pub fn dispatch_float(op: &FloatOp, inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    match op.category() {
        OpCategory::UnaryElementwise => unary_map(inputs, |v| op.apply_unary(v)),
        OpCategory::BinaryElementwise => binary_elementwise(inputs, |a, b| op.apply_binary(a, b)),
        OpCategory::BinaryCompare => binary_compare(inputs, |a, b| op.apply_compare(a, b)),
        OpCategory::BinaryByteBool => binary_byte_bool(inputs, |a, b| op.apply_byte_bool(a, b)),
        OpCategory::UnaryByteBool => unary_byte_bool(inputs, |a| if a != 0 { 0 } else { 1 }),
        OpCategory::UnaryToU8 => dispatch_isnan(inputs),
        OpCategory::Custom => dispatch_custom(op, inputs),
    }
}

/// Dispatch ops that need dedicated kernel logic.
fn dispatch_custom(op: &FloatOp, inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    match op {
        FloatOp::MatMul { m, k, n } => {
            dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize)
        }
        FloatOp::Gemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
        } => dispatch_gemm(
            inputs,
            GemmParams {
                m: *m as usize,
                n: *n as usize,
                k: *k as usize,
                alpha: bits_to_f32(*alpha),
                beta: bits_to_f32(*beta),
                trans_a: *trans_a,
                trans_b: *trans_b,
            },
        ),
        FloatOp::Softmax { size } => dispatch_softmax(inputs, *size as usize),
        FloatOp::LogSoftmax { size } => dispatch_log_softmax(inputs, *size as usize),
        FloatOp::RmsNorm { size, epsilon } => {
            dispatch_rms_norm(inputs, *size as usize, bits_to_f32(*epsilon))
        }
        FloatOp::LayerNorm { size, epsilon } => {
            dispatch_layer_norm(inputs, *size as usize, bits_to_f32(*epsilon))
        }
        FloatOp::ReduceSum { size } => dispatch_reduce(inputs, *size as usize, reduce_sum),
        FloatOp::ReduceMean { size } => dispatch_reduce(inputs, *size as usize, reduce_mean),
        FloatOp::ReduceMax { size } => dispatch_reduce(inputs, *size as usize, reduce_max),
        FloatOp::ReduceMin { size } => dispatch_reduce(inputs, *size as usize, reduce_min),
        FloatOp::Gather { dim, dtype } => dispatch_gather(inputs, *dim as usize, *dtype),
        FloatOp::Concat {
            size_a,
            size_b,
            dtype,
        } => dispatch_concat(inputs, *size_a as usize, *size_b as usize, *dtype),
        FloatOp::Reshape | FloatOp::Transpose { .. } | FloatOp::GatherND => Ok(inputs[0].to_vec()),
        FloatOp::Cast { from, to } => dispatch_cast(inputs, *from, *to),
        FloatOp::Embed { dim } => dispatch_embed(inputs, *dim as usize),
        FloatOp::Where => dispatch_where(inputs),
        FloatOp::Range => dispatch_range(inputs),
        FloatOp::Shape { dtype } => dispatch_shape(inputs, *dtype),
        FloatOp::RotaryEmbedding { dim, base } => {
            dispatch_rope(inputs, *dim as usize, bits_to_f32(*base))
        }
        FloatOp::Attention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
        } => dispatch_attention(
            inputs,
            *head_dim as usize,
            *num_q_heads as usize,
            *num_kv_heads as usize,
            bits_to_f32(*scale),
            *causal,
        ),
        FloatOp::Dequantize => dispatch_dequantize(inputs),
        _ => unreachable!("non-custom op {:?} routed to dispatch_custom", op),
    }
}

/// Dispatch a fused chain of unary element-wise f32 ops.
///
/// Applies each op in sequence to every element, avoiding intermediate buffers.
pub fn dispatch_fused_chain(chain: &[FloatOp], inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<f32> = x
        .iter()
        .map(|&v| {
            let mut val = v;
            for op in chain {
                val = op.apply_unary(val);
            }
            val
        })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn cast_f32(bytes: &[u8]) -> ExecResult<&[f32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| ExecError::ShapeMismatch {
        expected: "f32-aligned bytes".into(),
        actual: e.to_string(),
    })
}

fn cast_i64(bytes: &[u8]) -> ExecResult<&[i64]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| ExecError::ShapeMismatch {
        expected: "i64-aligned bytes".into(),
        actual: e.to_string(),
    })
}

fn cast_i32(bytes: &[u8]) -> ExecResult<&[i32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| ExecError::ShapeMismatch {
        expected: "i32-aligned bytes".into(),
        actual: e.to_string(),
    })
}

/// Zero-copy conversion from `Vec<f32>` to `Vec<u8>`.
///
/// Takes ownership and reinterprets the backing allocation in-place —
/// no memcpy, no extra allocation.
pub(crate) fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let len = data.len() * 4;
    let cap = data.capacity() * 4;
    let ptr = data.as_ptr() as *mut u8;
    std::mem::forget(data);
    // SAFETY: f32 has alignment >= u8; len/cap are scaled correctly.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ── Unary ────────────────────────────────────────────────────────────────────

fn unary_map(inputs: &[&[u8]], f: impl Fn(f32) -> f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<f32> = x.iter().map(|&v| f(v)).collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Binary ───────────────────────────────────────────────────────────────────

fn binary_elementwise(inputs: &[&[u8]], f: impl Fn(f32, f32) -> f32) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let out_len = a.len().max(b.len());
    let out: Vec<f32> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]))
        .collect();
    Ok(f32_vec_to_bytes(out))
}

// ── MatMul ───────────────────────────────────────────────────────────────────

#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod blas {
    use super::GemmParams;

    #[allow(non_camel_case_types)]
    type cblas_int = i32;

    const CBLAS_ROW_MAJOR: cblas_int = 101;
    const CBLAS_NO_TRANS: cblas_int = 111;
    const CBLAS_TRANS: cblas_int = 112;

    extern "C" {
        fn cblas_sgemm(
            order: cblas_int,
            trans_a: cblas_int,
            trans_b: cblas_int,
            m: cblas_int,
            n: cblas_int,
            k: cblas_int,
            alpha: f32,
            a: *const f32,
            lda: cblas_int,
            b: *const f32,
            ldb: cblas_int,
            beta: f32,
            c: *mut f32,
            ldc: cblas_int,
        );
    }

    /// BLAS sgemm: C = A * B (row-major, no transpose).
    pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        sgemm_full(
            GemmParams {
                m,
                n,
                k,
                alpha: 1.0,
                beta: 0.0,
                trans_a: false,
                trans_b: false,
            },
            a,
            b,
            c,
        );
    }

    /// BLAS sgemm: C = alpha * op(A) * op(B) + beta * C (row-major).
    pub fn sgemm_full(p: GemmParams, a: &[f32], b: &[f32], c: &mut [f32]) {
        let ta = if p.trans_a {
            CBLAS_TRANS
        } else {
            CBLAS_NO_TRANS
        };
        let tb = if p.trans_b {
            CBLAS_TRANS
        } else {
            CBLAS_NO_TRANS
        };
        let lda = if p.trans_a {
            p.m as cblas_int
        } else {
            p.k as cblas_int
        };
        let ldb = if p.trans_b {
            p.k as cblas_int
        } else {
            p.n as cblas_int
        };
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                ta,
                tb,
                p.m as cblas_int,
                p.n as cblas_int,
                p.k as cblas_int,
                p.alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                p.beta,
                c.as_mut_ptr(),
                p.n as cblas_int,
            );
        }
    }
}

/// Dispatch a MatMul using runtime-aware shape inference.
///
/// The compiled `k` (inner dimension) is used as a hint. When it cannot
/// cleanly divide both inputs, we attempt to infer k from the compiled `n`
/// and `m` hints, or from common factors between the two buffer lengths.
pub fn dispatch_matmul(inputs: &[&[u8]], m: usize, k: usize, n: usize) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;

    let actual_k = infer_matmul_k(k, m, n, a.len(), b.len())?;
    let actual_m = a.len() / actual_k;
    let actual_n = b.len() / actual_k;

    // Cap output to prevent OOM from shape inference errors.
    let out_size = actual_m.saturating_mul(actual_n);
    if out_size > 256 * 1024 * 1024 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("matmul output < 1GB (compiled k={k})"),
            actual: format!(
                "[{actual_m},{actual_k}]x[{actual_k},{actual_n}] = {} floats",
                out_size
            ),
        });
    }

    let mut out = vec![0.0f32; out_size];

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        blas::sgemm(actual_m, actual_n, actual_k, a, b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        // ikj loop order: inner j loop is stride-1 on both out[] and b[],
        // enabling auto-vectorization and cache-friendly access.
        for i in 0..actual_m {
            for p in 0..actual_k {
                let a_val = a[i * actual_k + p];
                let b_row = &b[p * actual_n..(p + 1) * actual_n];
                let o_row = &mut out[i * actual_n..(i + 1) * actual_n];
                for j in 0..actual_n {
                    o_row[j] += a_val * b_row[j];
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

/// Batched matmul: A[batch, M, K] × B[batch, K, N] → C[batch, M, N].
///
/// Each batch independently computes a 2D matrix multiply. This is required
/// for multi-head attention where Q@K^T operates per-head.
///
/// Returns `(output_bytes, output_shape)`.
pub fn dispatch_batched_matmul(
    inputs: &[&[u8]],
    a_shape: &[usize],
    b_shape: &[usize],
) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;

    // Last 2 dims are the matrix dims; everything before is batch.
    let mat_m = a_shape[a_shape.len() - 2];
    let mat_k = a_shape[a_shape.len() - 1];
    let mat_n = b_shape[b_shape.len() - 1];

    let batch: usize = a_shape[..a_shape.len() - 2]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);

    let a_stride = mat_m * mat_k;
    let b_stride = mat_k * mat_n;
    let c_stride = mat_m * mat_n;

    // Validate sizes.
    if batch * a_stride > a.len() || batch * b_stride > b.len() {
        return Err(ExecError::ShapeMismatch {
            expected: format!(
                "batched matmul: batch={batch} A=[{mat_m},{mat_k}] B=[{mat_k},{mat_n}]"
            ),
            actual: format!("a_len={}, b_len={}", a.len(), b.len()),
        });
    }

    let out_size = batch * c_stride;
    if out_size > 256 * 1024 * 1024 {
        return Err(ExecError::ShapeMismatch {
            expected: "batched matmul output < 1GB".to_string(),
            actual: format!("{out_size} floats"),
        });
    }

    let mut out = vec![0.0f32; out_size];

    for bat in 0..batch {
        let a_off = bat * a_stride;
        let b_off = bat * b_stride;
        let c_off = bat * c_stride;
        let a_slice = &a[a_off..a_off + a_stride];
        let b_slice = &b[b_off..b_off + b_stride];
        let c_slice = &mut out[c_off..c_off + c_stride];

        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(mat_m, mat_n, mat_k, a_slice, b_slice, c_slice);
        }

        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            for i in 0..mat_m {
                for p in 0..mat_k {
                    let a_val = a_slice[i * mat_k + p];
                    let b_row = &b_slice[p * mat_n..(p + 1) * mat_n];
                    let o_row = &mut c_slice[i * mat_n..(i + 1) * mat_n];
                    for j in 0..mat_n {
                        o_row[j] += a_val * b_row[j];
                    }
                }
            }
        }
    }

    // Output shape: A's batch dims + [M, N]
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(mat_n);

    Ok((f32_vec_to_bytes(out), out_shape))
}

/// Infer the shared inner dimension `k` for MatMul A[M,K] × B[K,N].
///
/// Uses compiled k/m/n as hints. When compiled k is wrong (doesn't divide
/// both inputs), tries to infer k from compiled n (B's last dim, typically
/// concrete for weight matrices) or from common factors.
fn infer_matmul_k(
    compiled_k: usize,
    compiled_m: usize,
    compiled_n: usize,
    a_len: usize,
    b_len: usize,
) -> ExecResult<usize> {
    // Primary: compiled k divides both inputs cleanly.
    if compiled_k > 1 && a_len.is_multiple_of(compiled_k) && b_len.is_multiple_of(compiled_k) {
        return Ok(compiled_k);
    }
    // Fallback 1: if compiled n is known, infer k = b_len / n.
    if compiled_n > 1 && b_len.is_multiple_of(compiled_n) {
        let k = b_len / compiled_n;
        if k > 0 && a_len.is_multiple_of(k) {
            return Ok(k);
        }
    }
    // Fallback 2: if compiled m is known, infer k = a_len / m.
    if compiled_m > 1 && a_len.is_multiple_of(compiled_m) {
        let k = a_len / compiled_m;
        if k > 0 && b_len.is_multiple_of(k) {
            return Ok(k);
        }
    }
    // Fallback 3: when compiled_n is known, try k = b_len / compiled_n even
    // if a_len % k != 0 — this handles batched MatMul where A is [batch, m, k].
    // Also try using compiled_n as k directly (common for square weight matrices).
    if compiled_n > 1 {
        // Try compiled_n as k (e.g. weight is [2048, 2048], both k and n are 2048)
        if a_len.is_multiple_of(compiled_n) && b_len.is_multiple_of(compiled_n) {
            return Ok(compiled_n);
        }
    }

    // Fallback 4: use GCD of a_len and b_len as k (finds the largest shared dim).
    let g = gcd(a_len, b_len);
    if g > 1 {
        // Verify output won't be absurdly large (m * n < 256M floats)
        let m_cand = a_len / g;
        let n_cand = b_len / g;
        if m_cand.saturating_mul(n_cand) < 256 * 1024 * 1024 {
            return Ok(g);
        }
        // GCD is too large, try smaller common factors by dividing GCD
        // Use compiled_n hint if available
        if compiled_n > 1 && g.is_multiple_of(compiled_n) {
            let k = g / compiled_n * compiled_n; // round down to compiled_n multiple
            if k > 1 && a_len.is_multiple_of(k) && b_len.is_multiple_of(k) {
                return Ok(k);
            }
        }
    }

    // Last resort: compiled_k works if it divides both (even k=1 for scalar matmul).
    if compiled_k > 0 && a_len.is_multiple_of(compiled_k) && b_len.is_multiple_of(compiled_k) {
        // Guard against k=1 producing impossibly large outputs.
        let m_cand = a_len / compiled_k;
        let n_cand = b_len / compiled_k;
        if m_cand.saturating_mul(n_cand) < 256 * 1024 * 1024 {
            return Ok(compiled_k);
        }
    }
    Err(ExecError::ShapeMismatch {
        expected: format!("matmul k dividing both inputs (compiled k={compiled_k}, m={compiled_m}, n={compiled_n})"),
        actual: format!("a={a_len}, b={b_len}"),
    })
}

// ── Softmax ──────────────────────────────────────────────────────────────────

fn dispatch_softmax(inputs: &[&[u8]], size: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── RmsNorm ──────────────────────────────────────────────────────────────────

fn dispatch_rms_norm(inputs: &[&[u8]], size: usize, epsilon: f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    if weight.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight: [{size}]"),
            actual: format!("{} floats", weight.len()),
        });
    }
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
        let rms = (ms + epsilon).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = (*v / rms) * w;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

fn dispatch_layer_norm(inputs: &[&[u8]], size: usize, epsilon: f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias = cast_f32(inputs[2])?;
    if weight.len() != size || bias.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight/bias: [{size}]"),
            actual: format!("weight={}, bias={}", weight.len(), bias.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let mean: f32 = row.iter().sum::<f32>() / size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / size as f32;
        let std = (var + epsilon).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v - mean) / std) * weight[i] + bias[i];
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Reductions ───────────────────────────────────────────────────────────────

fn dispatch_reduce(
    inputs: &[&[u8]],
    size: usize,
    f: impl Fn(&[f32]) -> f32,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let out: Vec<f32> = x.chunks(size).map(f).collect();
    Ok(f32_vec_to_bytes(out))
}

fn reduce_sum(row: &[f32]) -> f32 {
    row.iter().sum()
}

fn reduce_mean(row: &[f32]) -> f32 {
    row.iter().sum::<f32>() / row.len() as f32
}

fn reduce_max(row: &[f32]) -> f32 {
    row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn reduce_min(row: &[f32]) -> f32 {
    row.iter().cloned().fold(f32::INFINITY, f32::min)
}

// ── Gather ───────────────────────────────────────────────────────────────────

fn dispatch_gather(inputs: &[&[u8]], dim: usize, dtype: FloatDType) -> ExecResult<Vec<u8>> {
    let indices = cast_i64(inputs[0])?;
    let table_bytes = inputs[1];

    match dtype {
        FloatDType::I64 => {
            // i64 gather (shape subgraph): indices select individual i64 values.
            let table_i64 = cast_i64(table_bytes)?;
            let mut out = Vec::with_capacity(indices.len() * 8);
            for &idx in indices {
                let idx = idx as usize;
                if idx >= table_i64.len() {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("i64 index < {}", table_i64.len()),
                        actual: format!("index = {idx}"),
                    });
                }
                out.extend_from_slice(&table_i64[idx].to_le_bytes());
            }
            Ok(out)
        }
        FloatDType::I32 => {
            // i32 gather: indices select individual i32 values.
            let table_i32 = cast_i32(table_bytes)?;
            let mut out = Vec::with_capacity(indices.len() * 4);
            for &idx in indices {
                let idx = idx as usize;
                if idx >= table_i32.len() {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("i32 index < {}", table_i32.len()),
                        actual: format!("index = {idx}"),
                    });
                }
                out.extend_from_slice(&table_i32[idx].to_le_bytes());
            }
            Ok(out)
        }
        _ => {
            // f32 embedding gather (default for F32, F16, BF16, etc.).
            let table = cast_f32(table_bytes)?;
            let dim = if dim > 0 { dim } else { 1 };
            let vocab = table.len() / dim;
            let mut out = Vec::with_capacity(indices.len() * dim);
            for &idx in indices {
                let idx = idx as usize;
                if idx >= vocab {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("index < {vocab}"),
                        actual: format!("index = {idx}"),
                    });
                }
                out.extend_from_slice(&table[idx * dim..(idx + 1) * dim]);
            }
            Ok(f32_vec_to_bytes(out))
        }
    }
}

// ── Concat ───────────────────────────────────────────────────────────────────

/// Check if raw bytes look like valid f32 data vs i64 byte reinterpretation.
///
/// On little-endian, an i64 value < 2^31 (all shape dimensions) has its high
/// 4 bytes as 0x00000000. When reinterpreted as two f32 values, the second
/// (high word) is always 0.0. Real f32 shape data (after Cast) contains
fn dispatch_concat(
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
            out.extend_from_slice(a);
            out.extend_from_slice(b);
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

// ── Erf approximation (moved to FloatOp::apply_unary) ───────────────────

// ── Boolean / byte-wise ops ─────────────────────────────────────────────

fn binary_byte_bool(inputs: &[&[u8]], f: impl Fn(u8, u8) -> u8) -> ExecResult<Vec<u8>> {
    let a = inputs[0];
    let b = inputs[1];
    let out_len = a.len().max(b.len());
    let out: Vec<u8> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]))
        .collect();
    Ok(out)
}

fn unary_byte_bool(inputs: &[&[u8]], f: impl Fn(u8) -> u8) -> ExecResult<Vec<u8>> {
    let out: Vec<u8> = inputs[0].iter().map(|&x| f(x)).collect();
    Ok(out)
}

// ── Comparison ops (f32 → u8) ───────────────────────────────────────────

fn binary_compare(inputs: &[&[u8]], f: impl Fn(f32, f32) -> bool) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let out_len = a.len().max(b.len());
    let out: Vec<u8> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]) as u8)
        .collect();
    Ok(out)
}

// ── IsNaN (f32 → u8) ───────────────────────────────────────────────────

fn dispatch_isnan(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<u8> = x.iter().map(|v| v.is_nan() as u8).collect();
    Ok(out)
}

// ── Gemm ────────────────────────────────────────────────────────────────

fn dispatch_gemm(inputs: &[&[u8]], p: GemmParams) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let c = cast_f32(inputs[2])?;
    // Derive m and n from actual inputs — compile-time values may be wrong.
    let k = p.k;
    let n = if k > 0 { b.len() / k } else { 0 };
    let m = if k > 0 { a.len() / k } else { 0 };
    let mut out = vec![0.0f32; m * n];

    // Copy bias (C) into output — BLAS computes C := alpha*A*B + beta*C in-place.
    if p.beta != 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                out[idx] = if idx < c.len() { c[idx] } else { 0.0 };
            }
        }
    }

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        blas::sgemm_full(GemmParams { m, n, k, ..p }, a, b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for q in 0..k {
                    let a_val = if p.trans_a {
                        a[q * m + i]
                    } else {
                        a[i * k + q]
                    };
                    let b_val = if p.trans_b {
                        b[j * k + q]
                    } else {
                        b[q * n + j]
                    };
                    sum += a_val * b_val;
                }
                let c_val = if i * n + j < c.len() {
                    c[i * n + j]
                } else {
                    0.0
                };
                out[i * n + j] = p.alpha * sum + p.beta * c_val;
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── LogSoftmax ──────────────────────────────────────────────────────────

fn dispatch_log_softmax(inputs: &[&[u8]], size: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
        for v in row.iter_mut() {
            *v -= log_sum_exp;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Embed ───────────────────────────────────────────────────────────────

fn dispatch_embed(inputs: &[&[u8]], dim: usize) -> ExecResult<Vec<u8>> {
    // inputs[0] = token_ids (i64 or u32), inputs[1] = table (f32) [vocab, dim]
    let raw = inputs[0];
    let table = cast_f32(inputs[1])?;
    let vocab = table.len() / dim;

    // Detect token ID dtype: i64 (8 bytes each) or u32 (4 bytes each).
    let token_ids: Vec<usize> = if raw.len().is_multiple_of(8) {
        // Prefer i64 — matches the typical INT64 graph input dtype.
        let i64s: &[i64] = bytemuck::try_cast_slice(raw).map_err(|e| ExecError::ShapeMismatch {
            expected: "i64-aligned bytes".into(),
            actual: e.to_string(),
        })?;
        i64s.iter().map(|&v| v as usize).collect()
    } else {
        let u32s: &[u32] = bytemuck::try_cast_slice(raw).map_err(|e| ExecError::ShapeMismatch {
            expected: "u32-aligned bytes".into(),
            actual: e.to_string(),
        })?;
        u32s.iter().map(|&v| v as usize).collect()
    };

    let mut out = Vec::with_capacity(token_ids.len() * dim);
    for idx in token_ids {
        if idx >= vocab {
            return Err(ExecError::ShapeMismatch {
                expected: format!("token id < {vocab}"),
                actual: format!("token id = {idx}"),
            });
        }
        out.extend_from_slice(&table[idx * dim..(idx + 1) * dim]);
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Where ───────────────────────────────────────────────────────────────

fn dispatch_where(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [cond (u8), x (f32), y (f32)]
    let cond = inputs[0];
    let x = cast_f32(inputs[1])?;
    let y = cast_f32(inputs[2])?;
    let out: Vec<f32> = cond
        .iter()
        .zip(x.iter().zip(y.iter()))
        .map(|(&c, (&xv, &yv))| if c != 0 { xv } else { yv })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Range ───────────────────────────────────────────────────────────────

fn dispatch_range(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [start (f32), limit (f32), delta (f32)] — each is a scalar
    let start = cast_f32(inputs[0])?[0];
    let limit = cast_f32(inputs[1])?[0];
    let delta = cast_f32(inputs[2])?[0];
    let n = ((limit - start) / delta).ceil() as usize;
    let out: Vec<f32> = (0..n).map(|i| start + i as f32 * delta).collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Cast ────────────────────────────────────────────────────────────────

fn dispatch_cast(inputs: &[&[u8]], from: FloatDType, to: FloatDType) -> ExecResult<Vec<u8>> {
    if from == to {
        return Ok(inputs[0].to_vec());
    }
    let data = inputs[0];
    let from_size = from.byte_size();

    // If the data doesn't divide evenly by the declared `from` dtype but
    // DOES divide evenly by the `to` dtype (or by 4 for f32), the upstream
    // already converted and this Cast is a no-op.  This handles chains of
    // Casts where dtype metadata wasn't fully propagated.
    if from_size > 0 && !data.len().is_multiple_of(from_size) {
        return Ok(data.to_vec());
    }

    match (from, to) {
        (FloatDType::I64, FloatDType::F32) => {
            let src = cast_i64(data)?;
            let out: Vec<f32> = src.iter().map(|&v| v as f32).collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::I32, FloatDType::F32) => {
            let src: &[i32] =
                bytemuck::try_cast_slice(data).map_err(|e| ExecError::ShapeMismatch {
                    expected: "i32-aligned bytes".into(),
                    actual: e.to_string(),
                })?;
            let out: Vec<f32> = src.iter().map(|&v| v as f32).collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::F32, FloatDType::I64) => {
            let src = cast_f32(data)?;
            let out: Vec<i64> = src.iter().map(|&v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::I32) => {
            let src = cast_f32(data)?;
            let out: Vec<i32> = src.iter().map(|&v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::Bool, FloatDType::F32) => {
            let out: Vec<f32> = data
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::F32, FloatDType::Bool) => {
            let src = cast_f32(data)?;
            Ok(src
                .iter()
                .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
                .collect())
        }
        // Fallback: pass-through (same bytes, different interpretation).
        _ => Ok(data.to_vec()),
    }
}

// ── Shape ───────────────────────────────────────────────────────────────

fn dispatch_shape(inputs: &[&[u8]], dtype: FloatDType) -> ExecResult<Vec<u8>> {
    let elem_bytes = dtype.byte_size();
    let n_elements = if elem_bytes > 0 {
        inputs[0].len() as i64 / elem_bytes as i64
    } else {
        inputs[0].len() as i64
    };
    Ok(bytemuck::cast_slice(&[n_elements]).to_vec())
}

// ── Attention ───────────────────────────────────────────────────────────

fn dispatch_attention(
    inputs: &[&[u8]],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    causal: bool,
) -> ExecResult<Vec<u8>> {
    // Q: [num_q_heads, seq, head_dim], K/V: [num_kv_heads, seq, head_dim]
    let q = cast_f32(inputs[0])?;
    let k = cast_f32(inputs[1])?;
    let v = cast_f32(inputs[2])?;

    let seq_q = q.len() / (num_q_heads * head_dim);
    let seq_k = k.len() / (num_kv_heads * head_dim);
    let group_size = num_q_heads / num_kv_heads.max(1);

    let mut out = vec![0.0f32; num_q_heads * seq_q * head_dim];
    // Allocate scores buffer once, reuse across all heads.
    let mut scores = vec![0.0f32; seq_q * seq_k];

    // Per-head attention: iterate over Q heads, map to KV head via group_size.
    for qh in 0..num_q_heads {
        let kh = qh / group_size;
        let q_off = qh * seq_q * head_dim;
        let k_off = kh * seq_k * head_dim;
        let o_off = qh * seq_q * head_dim;

        let q_head = &q[q_off..q_off + seq_q * head_dim];
        let k_head = &k[k_off..k_off + seq_k * head_dim];
        let v_head = &v[k_off..k_off + seq_k * head_dim];

        // scores = Q_head × K_head^T * scale → [seq_q, seq_k]
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm_full(
                GemmParams {
                    m: seq_q,
                    n: seq_k,
                    k: head_dim,
                    alpha: scale,
                    beta: 0.0,
                    trans_a: false,
                    trans_b: true,
                },
                q_head,
                k_head,
                &mut scores,
            );
            // Apply causal mask after BLAS.
            if causal {
                for i in 0..seq_q {
                    for j in (i + 1)..seq_k {
                        scores[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            // Fused QK^T with causal mask: skip upper triangle entirely.
            for i in 0..seq_q {
                let limit = if causal { (i + 1).min(seq_k) } else { seq_k };
                for j in 0..limit {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                    }
                    scores[i * seq_k + j] = dot * scale;
                }
                // Fill masked positions with -inf.
                if causal {
                    for j in limit..seq_k {
                        scores[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Softmax each row (2-pass: max+exp+sum, then divide).
        for row in scores.chunks_mut(seq_k) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                *val = (*val - max).exp();
                sum += *val;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for val in row.iter_mut() {
                    *val *= inv;
                }
            }
        }

        // out_head = scores × V_head → [seq_q, head_dim]
        // ikj loop order for cache-friendly access to V.
        let out_head = &mut out[o_off..o_off + seq_q * head_dim];
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(seq_q, head_dim, seq_k, &scores, v_head, out_head);
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            out_head.fill(0.0);
            for i in 0..seq_q {
                for j in 0..seq_k {
                    let s = scores[i * seq_k + j];
                    if s == 0.0 {
                        continue;
                    }
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    let o_row = &mut out_head[i * head_dim..(i + 1) * head_dim];
                    for d in 0..head_dim {
                        o_row[d] += s * v_row[d];
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── Dequantize ──────────────────────────────────────────────────────────

fn dispatch_dequantize(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // Q4_0 dequantization: blocks of 18 bytes (2 byte scale + 16 nibbles = 32 values)
    let data = inputs[0];
    let block_size = 18;
    if !data.len().is_multiple_of(block_size) {
        // Not Q4_0 format — just pass through
        return Ok(data.to_vec());
    }
    let n_blocks = data.len() / block_size;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for block in data.chunks(block_size) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for byte_idx in 0..16 {
            let byte = block[2 + byte_idx];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = (byte >> 4) as i8 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }
    Ok(f32_vec_to_bytes(out))
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // subnormal
        let val = (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

fn dispatch_rope(inputs: &[&[u8]], dim: usize, base: f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;

    // Position input: either a single u32 start offset, or absent (sequential from 0).
    let start_pos: usize = if inputs.len() >= 2 && inputs[1].len() == 4 {
        u32::from_le_bytes([inputs[1][0], inputs[1][1], inputs[1][2], inputs[1][3]]) as usize
    } else {
        0
    };

    let half = dim / 2;
    let mut out = x.to_vec();
    // Apply RoPE to each chunk (token position). For full-sequence inference,
    // each chunk gets its sequential position: start_pos, start_pos+1, ...
    for (token_idx, chunk) in out.chunks_mut(dim).enumerate() {
        let pos = (start_pos + token_idx) as f32;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x0 = chunk[i];
            let x1 = chunk[i + half];
            chunk[i] = x0 * cos_a - x1 * sin_a;
            chunk[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Shape-aware ops (called from executor with ShapeMap) ─────────────────────

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
            let expanded = broadcast_to(src, n_elems, &shape);
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
fn broadcast_to(src: &[f32], n_src: usize, target_shape: &[usize]) -> Vec<f32> {
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

/// Compute row-major strides from a shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Shape inference functions (infer_output_shape, infer_custom_output_shape)
// have been consolidated into eval::shape_resolve::resolve_float_shape().

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(data).to_vec()
    }

    #[test]
    fn test_float_add() {
        let a = f32_bytes(&[1.0, 2.0, 3.0]);
        let b = f32_bytes(&[4.0, 5.0, 6.0]);
        let result = dispatch_float(&FloatOp::Add, &[&a, &b]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_float_add_broadcast() {
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let b = f32_bytes(&[10.0]);
        let result = dispatch_float(&FloatOp::Add, &[&a, &b]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_float_relu() {
        let x = f32_bytes(&[-1.0, 0.0, 1.0, 2.0]);
        let result = dispatch_float(&FloatOp::Relu, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_float_sigmoid() {
        let x = f32_bytes(&[0.0]);
        let result = dispatch_float(&FloatOp::Sigmoid, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_float_matmul() {
        // [2,3] × [3,2] → [2,2]
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f32_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = dispatch_float(&FloatOp::MatMul { m: 2, k: 3, n: 2 }, &[&a, &b]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(out, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_float_softmax() {
        let x = f32_bytes(&[1.0, 2.0, 3.0]);
        let result = dispatch_float(&FloatOp::Softmax { size: 3 }, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn test_float_rms_norm() {
        use hologram_core::op::f32_to_bits;
        let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let w = f32_bytes(&[1.0, 1.0, 1.0, 1.0]);
        let result = dispatch_float(
            &FloatOp::RmsNorm {
                size: 4,
                epsilon: f32_to_bits(1e-5),
            },
            &[&x, &w],
        )
        .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        // rms = sqrt((1+4+9+16)/4 + 1e-5) ≈ sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-5).sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-4);
        assert!((out[3] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_float_gather() {
        // vocab=3, dim=2
        let indices = bytemuck::cast_slice::<i64, u8>(&[0i64, 2]).to_vec();
        let table = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dispatch_float(
            &FloatOp::Gather {
                dim: 2,
                dtype: FloatDType::F32,
            },
            &[&indices, &table],
        )
        .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_float_fused_swiglu() {
        let gate = f32_bytes(&[0.0, 1.0]);
        let up = f32_bytes(&[2.0, 3.0]);
        let result = dispatch_float(&FloatOp::FusedSwiGLU, &[&gate, &up]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        // silu(0)*2 = 0, silu(1)*3 = 0.7310...*3 ≈ 2.1932
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - silu(1.0) * 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_float_reduce_sum() {
        let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dispatch_float(&FloatOp::ReduceSum { size: 3 }, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[6.0, 15.0]);
    }

    #[test]
    fn test_float_concat() {
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0]); // 2 rows of 2
        let b = f32_bytes(&[5.0, 6.0]); // 2 rows of 1
        let result = dispatch_float(
            &FloatOp::Concat {
                size_a: 2,
                size_b: 1,
                dtype: FloatDType::F32,
            },
            &[&a, &b],
        )
        .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }
}
