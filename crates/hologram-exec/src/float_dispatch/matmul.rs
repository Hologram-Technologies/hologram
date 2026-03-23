use super::helpers::*;
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

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(super) mod blas {
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
        blas::sgemm(actual_m, actual_n, actual_k, &a, &b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
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

/// MatMul writing directly into a pre-allocated output buffer.
pub fn dispatch_matmul_into(
    inputs: &[&[u8]],
    m: usize,
    k: usize,
    n: usize,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;

    let actual_k = infer_matmul_k(k, m, n, a.len(), b.len())?;
    let actual_m = a.len() / actual_k;
    let actual_n = b.len() / actual_k;

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
        blas::sgemm(actual_m, actual_n, actual_k, &a, &b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
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

    out_buf.extend_from_slice(bytemuck::cast_slice(&out));
    Ok(())
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

    // Support broadcast: 2-D B (shared weight) reuses the same matrix for
    // every batch slice. b_batch_count=1 means b_off stays at 0 each iteration.
    let b_batch_count = if b_stride > 0 {
        (b.len() / b_stride).max(1)
    } else {
        1
    };

    // Validate sizes.
    if batch * a_stride > a.len() || b_stride > b.len() {
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
        let b_off = (bat % b_batch_count) * b_stride;
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

pub(super) fn dispatch_gemm(inputs: &[&[u8]], p: GemmParams, quant_b: u8) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = super::cast::decode_weights(inputs[1], quant_b)?;
    let c: std::borrow::Cow<'_, [f32]> = if inputs.len() > 2 {
        cast_f32(inputs[2])?
    } else {
        std::borrow::Cow::Owned(vec![])
    };
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
        blas::sgemm_full(GemmParams { m, n, k, ..p }, &a, &b, &mut out);
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
