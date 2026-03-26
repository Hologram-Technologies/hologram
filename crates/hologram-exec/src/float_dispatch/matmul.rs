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

    // Detect batched matmul: when compiled m and n are non-zero and the total
    // elements exceed m*k (for A) or k*n (for B), there are batch dimensions.
    let mk = m.max(1) * actual_k;
    let kn = actual_k * n.max(1);

    let (batch, actual_m, actual_n) = if m > 0
        && n > 0
        && mk > 0
        && kn > 0
        && a.len() > mk
        && a.len().is_multiple_of(mk)
        && (b.len().is_multiple_of(kn) || b.len() == kn)
    {
        // Batched: A has batch leading dims, B may be batched or broadcast.
        let batch_a = a.len() / mk;
        let batch_b = if b.len() > kn && b.len().is_multiple_of(kn) {
            b.len() / kn
        } else {
            1
        };
        if batch_a == batch_b || batch_b == 1 {
            (batch_a, m, n)
        } else {
            // Batch mismatch — fall back to flat 2D.
            (1, a.len() / actual_k, b.len() / actual_k)
        }
    } else {
        // Flat 2D matmul (no batch dims or m/n unknown).
        (1, a.len() / actual_k, b.len() / actual_k)
    };

    let out_size = batch * actual_m * actual_n;
    if out_size > 256 * 1024 * 1024 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("matmul output < 1GB (compiled m={m} k={k} n={n})"),
            actual: format!(
                "batch={batch} [{actual_m},{actual_k}]x[{actual_k},{actual_n}] = {out_size} floats",
            ),
        });
    }

    let mut out = vec![0.0f32; out_size];

    if batch == 1 {
        // Single (possibly flattened) 2D matmul.
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(actual_m, actual_n, actual_k, &a, &b, &mut out);
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            matmul_k_outer(&a, &b, &mut out, actual_m, actual_k, actual_n);
        }
    } else {
        // Batched matmul: compute one [m, k] × [k, n] per batch.
        let a_stride = actual_m * actual_k;
        let b_stride = if b.len() == kn {
            0
        } else {
            actual_k * actual_n
        };
        let o_stride = actual_m * actual_n;
        for i in 0..batch {
            let a_slice = &a[i * a_stride..(i + 1) * a_stride];
            let b_slice = if b_stride > 0 {
                &b[i * b_stride..(i + 1) * b_stride]
            } else {
                &b[..kn] // broadcast: same B for all batches
            };
            let o_slice = &mut out[i * o_stride..(i + 1) * o_stride];
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            {
                blas::sgemm(actual_m, actual_n, actual_k, a_slice, b_slice, o_slice);
            }
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            {
                matmul_k_outer(a_slice, b_slice, o_slice, actual_m, actual_k, actual_n);
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

/// MatMul writing directly into a pre-allocated output buffer (zero intermediate Vec).
/// Infer actual (m, k, n) dimensions from compiled values and runtime buffer sizes.
///
/// When the runtime buffer has fewer elements than compiled m*k (variable-length
/// execution like decode with 1 token instead of 2048), adapts m to match the
/// actual buffer size. For batched matmul, preserves m and detects batch count.
#[allow(dead_code)] // Replaced by shape_resolve::resolve_matmul_dims; kept for reference
pub(crate) fn infer_matmul_dims(
    compiled_m: usize,
    compiled_k: usize,
    compiled_n: usize,
    a_elems: usize,
    b_elems: usize,
) -> (usize, usize, usize) {
    let actual_k =
        infer_matmul_k(compiled_k, compiled_m, compiled_n, a_elems, b_elems).unwrap_or(compiled_k);

    let mk = compiled_m.max(1) * actual_k;
    let kn = actual_k * compiled_n.max(1);

    if compiled_m > 0
        && compiled_n > 0
        && mk > 0
        && kn > 0
        && a_elems > mk
        && a_elems.is_multiple_of(mk)
        && (b_elems.is_multiple_of(kn) || b_elems == kn)
    {
        // Batched case: keep compiled m and n, batch is implicit.
        (compiled_m, actual_k, compiled_n)
    } else if actual_k > 0 {
        // Non-batched: infer m from buffer size.
        let actual_m = a_elems / actual_k;
        let actual_n = if b_elems >= actual_k {
            b_elems / actual_k
        } else {
            compiled_n
        };
        (actual_m, actual_k, actual_n)
    } else {
        (compiled_m, compiled_k, compiled_n)
    }
}

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

    // Detect batched matmul (same logic as dispatch_matmul).
    let mk = m.max(1) * actual_k;
    let kn = actual_k * n.max(1);

    let (batch, actual_m, actual_n) = if m > 0
        && n > 0
        && mk > 0
        && kn > 0
        && a.len() > mk
        && a.len().is_multiple_of(mk)
        && (b.len().is_multiple_of(kn) || b.len() == kn)
    {
        let batch_a = a.len() / mk;
        let batch_b = if b.len() > kn && b.len().is_multiple_of(kn) {
            b.len() / kn
        } else {
            1
        };
        if batch_a == batch_b || batch_b == 1 {
            (batch_a, m, n)
        } else {
            (1, a.len() / actual_k, b.len() / actual_k)
        }
    } else {
        (1, a.len() / actual_k, b.len() / actual_k)
    };

    let out_size = batch * actual_m * actual_n;
    if out_size > 256 * 1024 * 1024 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("matmul output < 1GB (compiled m={m} k={k} n={n})"),
            actual: format!(
                "batch={batch} [{actual_m},{actual_k}]x[{actual_k},{actual_n}] = {out_size} floats",
            ),
        });
    }

    let out = alloc_f32_in(out_buf, out_size);

    if batch == 1 {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(actual_m, actual_n, actual_k, &a, &b, out);
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            matmul_k_outer(&a, &b, out, actual_m, actual_k, actual_n);
        }
    } else {
        let a_stride = actual_m * actual_k;
        let b_stride = if b.len() == kn {
            0
        } else {
            actual_k * actual_n
        };
        let o_stride = actual_m * actual_n;
        for i in 0..batch {
            let a_slice = &a[i * a_stride..(i + 1) * a_stride];
            let b_slice = if b_stride > 0 {
                &b[i * b_stride..(i + 1) * b_stride]
            } else {
                &b[..kn]
            };
            let o_slice = &mut out[i * o_stride..(i + 1) * o_stride];
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            {
                blas::sgemm(actual_m, actual_n, actual_k, a_slice, b_slice, o_slice);
            }
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            {
                matmul_k_outer(a_slice, b_slice, o_slice, actual_m, actual_k, actual_n);
            }
        }
    }

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
            matmul_k_outer(a_slice, b_slice, c_slice, mat_m, mat_k, mat_n);
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
pub(crate) fn infer_matmul_k(
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

pub(crate) fn dispatch_gemm(inputs: &[&[u8]], p: GemmParams, quant_b: u8) -> ExecResult<Vec<u8>> {
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
        for (idx, o) in out.iter_mut().enumerate() {
            *o = if idx < c.len() { c[idx] } else { 0.0 };
        }
    }

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        blas::sgemm_full(GemmParams { m, n, k, ..p }, &a, &b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        // Pre-transpose to row-major if needed (one-time cost), then use
        // the cache-friendly k-outer loop. This is 3-5x faster than the
        // previous i,j-outer loop with runtime transpose conditionals.
        let a_rm = if p.trans_a {
            std::borrow::Cow::Owned(transpose_f32(&a, k, m))
        } else {
            std::borrow::Cow::Borrowed(&*a)
        };
        let b_rm = if p.trans_b {
            std::borrow::Cow::Owned(transpose_f32(&b, n, k))
        } else {
            std::borrow::Cow::Borrowed(&*b)
        };

        matmul_k_outer(&a_rm, &b_rm, &mut out, m, k, n);

        // Apply alpha/beta scaling if needed.
        if p.alpha != 1.0 || p.beta != 0.0 {
            for (idx, o) in out.iter_mut().enumerate() {
                let c_val = if idx < c.len() { c[idx] } else { 0.0 };
                *o = p.alpha * *o + p.beta * c_val;
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── Shared matmul kernel ────────────────────────────────────────────────

/// Cache-friendly register-blocked matmul: C[m,n] += A[m,k] × B[k,n].
///
/// Processes MR×NR output tiles (4×8) in registers, accumulating across the
/// full K dimension before writing back. This gives ~2-3x over the naive
/// k-outer loop on non-BLAS platforms by maximizing register reuse and
/// enabling autovectorization of the NR-wide inner accumulation.
///
/// Falls back to scalar k-outer for remainder rows/columns that don't
/// fill a complete tile.
#[inline]
#[cfg_attr(all(feature = "accelerate", target_os = "macos"), allow(dead_code))]
fn matmul_k_outer(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    const MR: usize = 4;
    const NR: usize = 8;

    let m_tiles = m / MR;
    let n_tiles = n / NR;
    let m_rem = m % MR;
    let n_rem = n % NR;

    // Tiled body: MR×NR register blocks.
    for it in 0..m_tiles {
        let i = it * MR;
        for jt in 0..n_tiles {
            let j = jt * NR;
            let mut acc = [[0.0f32; NR]; MR];
            for p in 0..k {
                let b_off = p * n + j;
                for ii in 0..MR {
                    let a_val = a[(i + ii) * k + p];
                    for jj in 0..NR {
                        acc[ii][jj] += a_val * b[b_off + jj];
                    }
                }
            }
            for (ii, acc_row) in acc.iter().enumerate() {
                out[(i + ii) * n + j..(i + ii) * n + j + NR].copy_from_slice(acc_row);
            }
        }
        // Remainder columns for tiled rows.
        if n_rem > 0 {
            let j = n_tiles * NR;
            for ii in 0..MR {
                let row = i + ii;
                for p in 0..k {
                    let a_val = a[row * k + p];
                    for jj in 0..n_rem {
                        out[row * n + j + jj] += a_val * b[p * n + j + jj];
                    }
                }
            }
        }
    }

    // Remainder rows: scalar k-outer for the bottom strip.
    if m_rem > 0 {
        let i = m_tiles * MR;
        for ii in 0..m_rem {
            let row = i + ii;
            for p in 0..k {
                let a_val = a[row * k + p];
                let b_row = &b[p * n..(p + 1) * n];
                let o_row = &mut out[row * n..(row + 1) * n];
                for j in 0..n {
                    o_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}
