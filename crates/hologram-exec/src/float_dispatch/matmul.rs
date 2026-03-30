use super::helpers::*;
use crate::error::{ExecError, ExecResult};
use hologram_core::op::FloatOp;

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

        let do_batch = |i: usize, o_slice: &mut [f32]| {
            let a_slice = &a[i * a_stride..(i + 1) * a_stride];
            let b_slice = if b_stride > 0 {
                &b[i * b_stride..(i + 1) * b_stride]
            } else {
                &b[..kn]
            };
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            {
                blas::sgemm(actual_m, actual_n, actual_k, a_slice, b_slice, o_slice);
            }
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            {
                matmul_k_outer(a_slice, b_slice, o_slice, actual_m, actual_k, actual_n);
            }
        };

        #[cfg(feature = "parallel")]
        if batch >= 2 {
            use rayon::prelude::*;
            out.par_chunks_mut(o_stride)
                .enumerate()
                .for_each(|(i, o_slice)| do_batch(i, o_slice));
        } else {
            do_batch(0, &mut out);
        }

        #[cfg(not(feature = "parallel"))]
        for i in 0..batch {
            do_batch(i, &mut out[i * o_stride..(i + 1) * o_stride]);
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

        let do_batch = |i: usize, o_slice: &mut [f32]| {
            let a_slice = &a[i * a_stride..(i + 1) * a_stride];
            let b_slice = if b_stride > 0 {
                &b[i * b_stride..(i + 1) * b_stride]
            } else {
                &b[..kn]
            };
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            {
                blas::sgemm(actual_m, actual_n, actual_k, a_slice, b_slice, o_slice);
            }
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            {
                matmul_k_outer(a_slice, b_slice, o_slice, actual_m, actual_k, actual_n);
            }
        };

        #[cfg(feature = "parallel")]
        if batch >= 2 {
            use rayon::prelude::*;
            out.par_chunks_mut(o_stride)
                .enumerate()
                .for_each(|(i, o_slice)| do_batch(i, o_slice));
        } else {
            do_batch(0, out);
        }

        #[cfg(not(feature = "parallel"))]
        for i in 0..batch {
            do_batch(i, &mut out[i * o_stride..(i + 1) * o_stride]);
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

    let mut out = vec![0.0f32; out_size];

    let do_batch = |bat: usize, c_slice: &mut [f32]| {
        let a_off = bat * a_stride;
        let b_off = (bat % b_batch_count) * b_stride;
        let a_slice = &a[a_off..a_off + a_stride];
        let b_slice = &b[b_off..b_off + b_stride];

        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(mat_m, mat_n, mat_k, a_slice, b_slice, c_slice);
        }

        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            matmul_k_outer(a_slice, b_slice, c_slice, mat_m, mat_k, mat_n);
        }
    };

    #[cfg(feature = "parallel")]
    if batch >= 2 {
        use rayon::prelude::*;
        out.par_chunks_mut(c_stride)
            .enumerate()
            .for_each(|(bat, c_slice)| do_batch(bat, c_slice));
    } else {
        do_batch(0, &mut out);
    }

    #[cfg(not(feature = "parallel"))]
    for bat in 0..batch {
        do_batch(bat, &mut out[bat * c_stride..(bat + 1) * c_stride]);
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
/// Validate a k candidate: must divide both inputs and not produce an absurdly
/// large output (guards against k=1 or erroneous small values).
#[inline]
fn try_k(k: usize, a_len: usize, b_len: usize) -> Option<usize> {
    if k == 0 || !a_len.is_multiple_of(k) || !b_len.is_multiple_of(k) {
        return None;
    }
    let m_cand = a_len / k;
    let n_cand = b_len / k;
    if m_cand.saturating_mul(n_cand) < 256 * 1024 * 1024 {
        Some(k)
    } else {
        None
    }
}

pub(crate) fn infer_matmul_k(
    compiled_k: usize,
    compiled_m: usize,
    compiled_n: usize,
    a_len: usize,
    b_len: usize,
) -> ExecResult<usize> {
    // Primary: compiled k is high-confidence — no output-size guard needed.
    if compiled_k > 1 && a_len.is_multiple_of(compiled_k) && b_len.is_multiple_of(compiled_k) {
        return Ok(compiled_k);
    }

    // Build candidate list in priority order; validate each with try_k.
    let g = gcd(a_len, b_len);
    let candidates = [
        // k = b_len / n: weight's last dim is usually concrete.
        if compiled_n > 1 && b_len.is_multiple_of(compiled_n) {
            b_len / compiled_n
        } else {
            0
        },
        // k = a_len / m: activation's last dim.
        if compiled_m > 1 && a_len.is_multiple_of(compiled_m) {
            a_len / compiled_m
        } else {
            0
        },
        // compiled_n as k: square weight matrix case.
        compiled_n,
        // GCD: largest shared dimension.
        g,
        // GCD sub-divisor when GCD is too large: round down to compiled_n multiple.
        if compiled_n > 1 && g.is_multiple_of(compiled_n) {
            g / compiled_n * compiled_n
        } else {
            0
        },
        // Last resort: compiled_k including k=1 (guarded against huge output).
        compiled_k,
    ];
    for k in candidates {
        if let Some(k) = try_k(k, a_len, b_len) {
            return Ok(k);
        }
    }

    Err(ExecError::ShapeMismatch {
        expected: format!(
            "matmul k dividing both inputs (compiled k={compiled_k}, m={compiled_m}, n={compiled_n})"
        ),
        actual: format!("a={a_len}, b={b_len}"),
    })
}

pub(crate) fn dispatch_gemm(inputs: &[&[u8]], p: GemmParams, quant_b: u8) -> ExecResult<Vec<u8>> {
    // ── Fast path: fused Q4_0 dequant-matmul ──────────────────────────
    // Skip the full dequantization when B is Q4_0, not transposed, and
    // dimensions align to block boundaries.  This avoids materializing
    // the entire K×N f32 weight matrix.
    if quant_b == 1 && !p.trans_b && !p.trans_a && p.alpha == 1.0 && p.beta == 0.0 {
        let b_q4 = inputs[1];
        let expected_f32_count = b_q4.len() / Q4_0_BLOCK_BYTES * Q4_0_BLOCK_VALUES;
        let k = p.k;
        if k > 0 && expected_f32_count > 0 {
            let n = expected_f32_count / k;
            let a = cast_f32(inputs[0])?;
            let m = if k > 0 { a.len() / k } else { 0 };
            if n.is_multiple_of(Q4_0_BLOCK_VALUES) && m > 0 && n > 0 {
                let mut out = vec![0.0f32; m * n];
                matmul_dequant_q4_0(&a, b_q4, &mut out, m, k, n);
                return Ok(f32_vec_to_bytes(out));
            }
        }
    }

    // ── General path: dequantize B, then standard matmul ──────────────
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

// Minimum M-tile rows to justify rayon threads (thread overhead threshold).
#[allow(dead_code)]
const PAR_M_TILE_THRESHOLD: usize = 8;

/// Wrapper to send a raw `*mut f32` across rayon threads.
/// SAFETY: callers must guarantee non-overlapping writes per thread.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// K-dimension block size for L2 cache blocking. A KC×NR panel (256×8 = 8 KB)
/// fits in L1; a KC×N panel for N=2048 (2 MB) fits in L2.
const KC: usize = 256;

/// Micro-kernel: accumulate A[i..i+MR, k_start..k_end] × B[k_start..k_end, j..j+NR]
/// into `acc`. The accumulator is NOT zeroed — caller manages initialization.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn micro_kernel<const MR: usize, const NR: usize>(
    a: &[f32],
    b: &[f32],
    acc: &mut [[f32; NR]; MR],
    i: usize,
    j: usize,
    k_start: usize,
    k_end: usize,
    k_stride: usize,
    n: usize,
) {
    for p in k_start..k_end {
        let b_off = p * n + j;
        for ii in 0..MR {
            let a_val = a[(i + ii) * k_stride + p];
            for jj in 0..NR {
                acc[ii][jj] += a_val * b[b_off + jj];
            }
        }
    }
}

/// Micro-kernel operating on a packed (contiguous) B panel. The panel is
/// laid out as `packed_b[p * NR + jj]` with stride NR, eliminating strided
/// access to the original B matrix (stride N, which wastes cache lines when
/// N is large).
#[inline(always)]
fn micro_kernel_packed<const MR: usize, const NR: usize>(
    a: &[f32],
    packed_b: &[f32],
    acc: &mut [[f32; NR]; MR],
    i: usize,
    k_start: usize,
    k_end: usize,
    k_stride: usize,
) {
    let kc_len = k_end - k_start;
    for p in 0..kc_len {
        let b_off = p * NR;
        for ii in 0..MR {
            let a_val = a[(i + ii) * k_stride + k_start + p];
            for jj in 0..NR {
                acc[ii][jj] += a_val * packed_b[b_off + jj];
            }
        }
    }
}

/// Pack B[k_start..k_end, j..j+NR] into a contiguous NR-strided buffer.
/// Cost: one sequential copy of KC×NR floats (~8 KB for KC=256, NR=8).
#[inline]
fn pack_b_panel<const NR: usize>(
    b: &[f32],
    packed: &mut [f32],
    k_start: usize,
    k_end: usize,
    j: usize,
    n: usize,
) {
    for p in 0..(k_end - k_start) {
        let src_off = (k_start + p) * n + j;
        let dst_off = p * NR;
        packed[dst_off..dst_off + NR].copy_from_slice(&b[src_off..src_off + NR]);
    }
}

/// Cache-friendly register-blocked matmul with L2 cache blocking:
/// C[m,n] += A[m,k] × B[k,n].
///
/// Uses Goto/BLIS-style loop ordering: each thread owns a strip of M-tile
/// rows, then iterates over K-blocks (KC=256) and N-tiles internally. This
/// keeps B panels resident in L2 across all N-tile iterations within a
/// K-block, giving ~2x improvement for K≥512 vs the unblocked kernel.
///
/// For K < KC the K-loop executes a single iteration with no overhead.
///
/// Processes MR×NR output tiles (4×8) in registers. Falls back to scalar
/// k-outer for remainder rows/columns that don't fill a complete tile.
#[inline]
#[cfg_attr(all(feature = "accelerate", target_os = "macos"), allow(dead_code))]
pub(crate) fn matmul_k_outer(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    const MR: usize = 4;
    const NR: usize = 8;

    let m_tiles = m / MR;
    let n_tiles = n / NR;
    let m_rem = m % MR;
    let n_rem = n % NR;

    // Whether B-panel packing is worthwhile: only when multiple M-tile rows
    // reuse the same packed panel (amortizes the copy cost).
    let use_packing = m_tiles > 1;

    // Core logic for one strip of M-tile rows (used by both parallel and sequential paths).
    // Each strip processes all N-tiles × K-blocks, keeping B panels L2-resident.
    let process_m_tile = |it: usize, out_ptr: SendPtr| {
        let out_ptr = out_ptr.0;
        let i = it * MR;
        // Stack-allocated packed B panel: KC × NR = 256 × 8 = 8 KB.
        let mut packed_b = [0.0f32; KC * NR];

        // Tiled body: MR×NR output tiles with KC blocking.
        for jt in 0..n_tiles {
            let j = jt * NR;
            let mut acc = [[0.0f32; NR]; MR];
            for kc_start in (0..k).step_by(KC) {
                let kc_end = (kc_start + KC).min(k);
                if use_packing {
                    let kc_len = kc_end - kc_start;
                    pack_b_panel::<NR>(b, &mut packed_b[..kc_len * NR], kc_start, kc_end, j, n);
                    micro_kernel_packed::<MR, NR>(
                        a,
                        &packed_b[..kc_len * NR],
                        &mut acc,
                        i,
                        kc_start,
                        kc_end,
                        k,
                    );
                } else {
                    micro_kernel::<MR, NR>(a, b, &mut acc, i, j, kc_start, kc_end, k, n);
                }
            }
            for (ii, acc_row) in acc.iter().enumerate() {
                let off = (i + ii) * n + j;
                unsafe {
                    std::ptr::copy_nonoverlapping(acc_row.as_ptr(), out_ptr.add(off), NR);
                }
            }
        }
        // Remainder columns: use MR×4 tile when possible (4-wide autovectorizable),
        // then scalar for the last 0-3 columns.
        if n_rem > 0 {
            let j = n_tiles * NR;
            let mut j_off = 0;
            // MR×4 tile for first 4 remainder columns.
            if n_rem >= 4 {
                let mut acc = [[0.0f32; 4]; MR];
                for kc_start in (0..k).step_by(KC) {
                    let kc_end = (kc_start + KC).min(k);
                    micro_kernel::<MR, 4>(a, b, &mut acc, i, j, kc_start, kc_end, k, n);
                }
                for (ii, acc_row) in acc.iter().enumerate() {
                    for (jj, &v) in acc_row.iter().enumerate() {
                        unsafe { *out_ptr.add((i + ii) * n + j + jj) = v };
                    }
                }
                j_off = 4;
            }
            // Scalar for remaining 0-3 columns.
            for jj in j_off..n_rem {
                let mut acc = [0.0f32; MR];
                for kc_start in (0..k).step_by(KC) {
                    let kc_end = (kc_start + KC).min(k);
                    for p in kc_start..kc_end {
                        for (ii, a_acc) in acc.iter_mut().enumerate() {
                            *a_acc += a[(i + ii) * k + p] * b[p * n + j + jj];
                        }
                    }
                }
                for (ii, &a_acc) in acc.iter().enumerate() {
                    unsafe { *out_ptr.add((i + ii) * n + j + jj) = a_acc };
                }
            }
        }
    };

    // Parallel path: split M-tile rows across rayon threads.
    // Each thread writes to non-overlapping output rows, so no synchronization needed.
    #[cfg(feature = "parallel")]
    if m_tiles >= PAR_M_TILE_THRESHOLD {
        use rayon::prelude::*;
        let out_ptr = SendPtr(out.as_mut_ptr());
        // SAFETY: each iteration writes exclusively to out[i*n..(i+MR)*n],
        // which is non-overlapping across different `it` values.
        (0..m_tiles)
            .into_par_iter()
            .for_each(|it| process_m_tile(it, out_ptr));

        // Remainder rows (sequential — typically ≤3 rows).
        if m_rem > 0 {
            let i = m_tiles * MR;
            let raw = out_ptr.0;
            for ii in 0..m_rem {
                let row = i + ii;
                for kc_start in (0..k).step_by(KC) {
                    let kc_end = (kc_start + KC).min(k);
                    for p in kc_start..kc_end {
                        let a_val = a[row * k + p];
                        let b_row = &b[p * n..p * n + n];
                        let o_row = unsafe { std::slice::from_raw_parts_mut(raw.add(row * n), n) };
                        for j in 0..n {
                            o_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
        return;
    }

    // Sequential path: same KC-blocked tiling without rayon.
    let out_ptr = SendPtr(out.as_mut_ptr());
    for it in 0..m_tiles {
        process_m_tile(it, out_ptr);
    }

    // Remainder rows: scalar k-outer with KC blocking for the bottom strip.
    if m_rem > 0 {
        let i = m_tiles * MR;
        for ii in 0..m_rem {
            let row = i + ii;
            for kc_start in (0..k).step_by(KC) {
                let kc_end = (kc_start + KC).min(k);
                for p in kc_start..kc_end {
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
}

// ── Fused Q4_0 dequant-matmul ─────────────────────────────────────────
//
// Instead of dequantizing the entire K×N weight matrix to f32 (which doubles
// memory bandwidth and allocates K×N×4 bytes), dequantize one KC×NR panel at
// a time directly into the stack-allocated packed_b buffer.  The micro-kernel
// is unchanged — it operates on the same packed f32 panel.

/// Q4_0 block size: 18 bytes → 32 f32 values.
const Q4_0_BLOCK_BYTES: usize = 18;
/// Number of f32 values produced by one Q4_0 block.
const Q4_0_BLOCK_VALUES: usize = 32;

/// Dequantize a KC×NR panel of Q4_0 weights directly into a packed f32 buffer.
///
/// Reads Q4_0 blocks from `b_q4` (row-major K×N layout, where each row of N
/// elements is stored as N/32 blocks of 18 bytes) and writes dequantized f32s
/// into `packed` with NR stride (same layout as `pack_b_panel`).
///
/// Requires: `n` is a multiple of `Q4_0_BLOCK_VALUES` (32).
#[inline]
fn dequant_pack_q4_0_panel<const NR: usize>(
    b_q4: &[u8],
    packed: &mut [f32],
    k_start: usize,
    k_end: usize,
    j: usize,
    n: usize,
) {
    let blocks_per_row = n / Q4_0_BLOCK_VALUES;
    let block_col = j / Q4_0_BLOCK_VALUES;
    let pos_in_block = j % Q4_0_BLOCK_VALUES;

    for p_idx in 0..(k_end - k_start) {
        let p = k_start + p_idx;
        let block_offset = (p * blocks_per_row + block_col) * Q4_0_BLOCK_BYTES;
        let block = &b_q4[block_offset..block_offset + Q4_0_BLOCK_BYTES];
        let scale = super::cast::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        let dst = &mut packed[p_idx * NR..(p_idx + 1) * NR];
        for (jj, d) in dst.iter_mut().enumerate() {
            let pos = pos_in_block + jj;
            let val = if pos < 16 {
                (block[2 + pos] & 0x0F) as i8 - 8
            } else {
                (block[2 + pos - 16] >> 4) as i8 - 8
            };
            *d = val as f32 * scale;
        }
    }
}

/// Dequantize a row segment of Q4_0 weights into a contiguous f32 buffer.
///
/// Used for remainder paths where packing is not worthwhile.  Dequantizes
/// `b_q4[row, col_start..col_end]` into `out`.
#[inline]
fn dequant_q4_0_row_segment(
    b_q4: &[u8],
    out: &mut [f32],
    row: usize,
    col_start: usize,
    n_cols: usize,
    n: usize,
) {
    let blocks_per_row = n / Q4_0_BLOCK_VALUES;
    for (jj, o) in out.iter_mut().enumerate().take(n_cols) {
        let col = col_start + jj;
        let block_col = col / Q4_0_BLOCK_VALUES;
        let pos = col % Q4_0_BLOCK_VALUES;
        let block_offset = (row * blocks_per_row + block_col) * Q4_0_BLOCK_BYTES;
        let block = &b_q4[block_offset..block_offset + Q4_0_BLOCK_BYTES];
        let scale = super::cast::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let val = if pos < 16 {
            (block[2 + pos] & 0x0F) as i8 - 8
        } else {
            (block[2 + pos - 16] >> 4) as i8 - 8
        };
        *o = val as f32 * scale;
    }
}

/// Process one MR-row strip: dequant-pack B panels and run micro-kernel.
#[inline]
#[allow(clippy::too_many_arguments)]
fn dequant_q4_0_m_strip(
    a: &[f32],
    b_q4: &[u8],
    out_ptr: *mut f32,
    i: usize,
    k: usize,
    n: usize,
    n_tiles: usize,
    n_rem: usize,
) {
    const MR: usize = 4;
    const NR: usize = 8;
    let mut packed_b = [0.0f32; KC * NR];

    // Tiled body: MR×NR output tiles with KC blocking + on-the-fly dequant.
    for jt in 0..n_tiles {
        let j = jt * NR;
        let mut acc = [[0.0f32; NR]; MR];
        for kc_start in (0..k).step_by(KC) {
            let kc_end = (kc_start + KC).min(k);
            let kc_len = kc_end - kc_start;
            dequant_pack_q4_0_panel::<NR>(
                b_q4,
                &mut packed_b[..kc_len * NR],
                kc_start,
                kc_end,
                j,
                n,
            );
            micro_kernel_packed::<MR, NR>(
                a,
                &packed_b[..kc_len * NR],
                &mut acc,
                i,
                kc_start,
                kc_end,
                k,
            );
        }
        for (ii, acc_row) in acc.iter().enumerate() {
            let off = (i + ii) * n + j;
            unsafe { std::ptr::copy_nonoverlapping(acc_row.as_ptr(), out_ptr.add(off), NR) };
        }
    }

    // Remainder columns — dequant one element at a time, accumulate scalar.
    if n_rem > 0 {
        let j = n_tiles * NR;
        let mut b_val = 0.0f32;
        for jj in 0..n_rem {
            let mut acc = [0.0f32; MR];
            for kc_start in (0..k).step_by(KC) {
                let kc_end = (kc_start + KC).min(k);
                for p in kc_start..kc_end {
                    dequant_q4_0_row_segment(
                        b_q4,
                        std::slice::from_mut(&mut b_val),
                        p,
                        j + jj,
                        1,
                        n,
                    );
                    for (ii, a_acc) in acc.iter_mut().enumerate() {
                        *a_acc += a[(i + ii) * k + p] * b_val;
                    }
                }
            }
            for (ii, &a_acc) in acc.iter().enumerate() {
                unsafe { *out_ptr.add((i + ii) * n + j + jj) = a_acc };
            }
        }
    }
}

/// Process remainder rows (< MR): dequant one B row at a time, scalar accumulate.
fn dequant_q4_0_remainder_rows(
    a: &[f32],
    b_q4: &[u8],
    out: &mut [f32],
    m_start: usize,
    m_rem: usize,
    k: usize,
    n: usize,
) {
    let mut b_row = vec![0.0f32; n];
    for ii in 0..m_rem {
        let row = m_start + ii;
        for kc_start in (0..k).step_by(KC) {
            let kc_end = (kc_start + KC).min(k);
            for p in kc_start..kc_end {
                let a_val = a[row * k + p];
                dequant_q4_0_row_segment(b_q4, &mut b_row, p, 0, n, n);
                let o_row = &mut out[row * n..(row + 1) * n];
                for j in 0..n {
                    o_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}

/// Fused Q4_0 dequantize-matmul: C[m,n] += A[m,k] × dequant(B_q4[k,n]).
///
/// Same tiling structure as `matmul_k_outer` (KC=256, MR=4, NR=8) but replaces
/// B-panel packing with on-the-fly Q4_0 dequantization.  Never materializes
/// the full K×N f32 weight matrix — only a KC×NR panel (8 KB) lives on stack.
///
/// Requires: `n` is a multiple of 32, `k * n / 32 * 18 == b_q4.len()`.
pub(crate) fn matmul_dequant_q4_0(
    a: &[f32],
    b_q4: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const MR: usize = 4;
    const NR: usize = 8;

    let m_tiles = m / MR;
    let n_tiles = n / NR;
    let m_rem = m % MR;
    let n_rem = n % NR;

    #[cfg(feature = "parallel")]
    if m_tiles >= PAR_M_TILE_THRESHOLD {
        use rayon::prelude::*;
        let out_ptr = SendPtr(out.as_mut_ptr());
        (0..m_tiles).into_par_iter().for_each(|it| {
            let ptr = out_ptr;
            dequant_q4_0_m_strip(a, b_q4, ptr.0, it * MR, k, n, n_tiles, n_rem);
        });
        if m_rem > 0 {
            dequant_q4_0_remainder_rows(a, b_q4, out, m_tiles * MR, m_rem, k, n);
        }
        return;
    }

    let out_ptr = out.as_mut_ptr();
    for it in 0..m_tiles {
        dequant_q4_0_m_strip(a, b_q4, out_ptr, it * MR, k, n, n_tiles, n_rem);
    }
    if m_rem > 0 {
        dequant_q4_0_remainder_rows(a, b_q4, out, m_tiles * MR, m_rem, k, n);
    }
}

// ── Epilogue fusion: matmul + activation ─────────────────────────────

/// Fused matmul + activation dispatch. Runs the standard vectorized matmul
/// kernel, then applies activation as a tight post-pass on the output buffer.
///
/// This preserves autovectorization of both the matmul inner loop and the
/// activation loop, while eliminating one arena slot allocation + one tape
/// instruction dispatch vs the unfused path.
pub fn dispatch_matmul_activation_into(
    inputs: &[&[u8]],
    m: usize,
    k: usize,
    n: usize,
    activation: &FloatOp,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    // Run the standard matmul (fully vectorized, cache-friendly).
    dispatch_matmul_into(inputs, m, k, n, out_buf)?;

    // Epilogue: apply activation in-place on the just-written output.
    // Data is cache-hot. Tight scalar loop — no arena overhead.
    if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>(out_buf) {
        for v in floats.iter_mut() {
            *v = activation.apply_unary(*v);
        }
    }

    Ok(())
}

/// Fused matmul + bias + activation dispatch. Runs the standard matmul,
/// then applies bias addition and activation in a single pass over the
/// cache-hot output. Eliminates both intermediate buffers that the
/// unfused MatMul → Add(bias) → Activation path requires.
pub fn dispatch_matmul_bias_activation_into(
    inputs: &[&[u8]],
    m: usize,
    k: usize,
    n: usize,
    bias: &[f32],
    activation: &FloatOp,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    // Standard matmul (fully vectorized, cache-friendly).
    dispatch_matmul_into(inputs, m, k, n, out_buf)?;

    // Fused epilogue: bias + activation in one pass.
    // Data is cache-hot from the matmul write.
    if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>(out_buf) {
        let bias_len = bias.len();
        if bias_len > 0 {
            for row in floats.chunks_mut(bias_len) {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = activation.apply_unary(*v + bias[j]);
                }
            }
        }
    }

    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal f32→f16 conversion for test data encoding.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3FF;
        if exp <= 0 {
            sign as u16 // flush to zero
        } else if exp >= 31 {
            (sign | 0x7C00) as u16 // infinity
        } else {
            (sign | ((exp as u32) << 10) | mant) as u16
        }
    }

    /// Encode f32 weights into Q4_0 format (18-byte blocks of 32 values each).
    fn encode_q4_0(weights: &[f32], k: usize, n: usize) -> Vec<u8> {
        assert_eq!(weights.len(), k * n);
        assert_eq!(n % Q4_0_BLOCK_VALUES, 0, "n must be a multiple of 32");
        let blocks_per_row = n / Q4_0_BLOCK_VALUES;
        let mut out = vec![0u8; k * blocks_per_row * Q4_0_BLOCK_BYTES];

        for row in 0..k {
            for bc in 0..blocks_per_row {
                let start = row * n + bc * Q4_0_BLOCK_VALUES;
                let vals = &weights[start..start + Q4_0_BLOCK_VALUES];

                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };

                let block_off = (row * blocks_per_row + bc) * Q4_0_BLOCK_BYTES;
                let scale_bits = f32_to_f16_bits(scale);
                out[block_off] = scale_bits as u8;
                out[block_off + 1] = (scale_bits >> 8) as u8;

                for i in 0..16 {
                    let lo_q = ((vals[i] / scale).round() as i8).clamp(-8, 7) + 8;
                    let hi_q = ((vals[16 + i] / scale).round() as i8).clamp(-8, 7) + 8;
                    out[block_off + 2 + i] = (lo_q as u8) | ((hi_q as u8) << 4);
                }
            }
        }
        out
    }

    /// Fused Q4_0 dequant-matmul must match dequant-then-matmul (bit-exact).
    #[test]
    fn dequant_matmul_q4_0_matches_reference() {
        let m = 5; // m_tiles=1, m_rem=1
        let k = 64;
        let n = 64;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 5) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let b_q4 = encode_q4_0(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q4_0(&b_q4);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q4_0(&a, &b_q4, &mut fused_out, m, k, n);

        for (idx, (&r, &f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
            assert_eq!(
                r.to_bits(),
                f.to_bits(),
                "mismatch at [{idx}]: ref={r}, fused={f}"
            );
        }
    }

    /// m=1 decode path — only remainder rows, no tiled body.
    #[test]
    fn dequant_matmul_q4_0_m1_decode() {
        let m = 1;
        let k = 128;
        let n = 64;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 11 + 2) % 100) as f32 / 100.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let b_q4 = encode_q4_0(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q4_0(&b_q4);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q4_0(&a, &b_q4, &mut fused_out, m, k, n);

        for (idx, (&r, &f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
            assert_eq!(
                r.to_bits(),
                f.to_bits(),
                "m=1 mismatch at [{idx}]: ref={r}, fused={f}"
            );
        }
    }

    /// Large prefill — exercises parallel path (m_tiles >= 8).
    #[test]
    fn dequant_matmul_q4_0_large_prefill() {
        let m = 32;
        let k = 256;
        let n = 256;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 1) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 7) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let b_q4 = encode_q4_0(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q4_0(&b_q4);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q4_0(&a, &b_q4, &mut fused_out, m, k, n);

        let max_err = ref_out
            .iter()
            .zip(fused_out.iter())
            .map(|(&r, &f)| (r - f).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err == 0.0,
            "large prefill max absolute error: {max_err}"
        );
    }
}
