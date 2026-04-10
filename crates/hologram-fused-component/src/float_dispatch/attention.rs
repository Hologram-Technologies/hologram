use std::borrow::Cow;

use super::helpers::*;
#[cfg(all(feature = "accelerate", target_os = "macos"))]
use super::matmul::GemmParams;
use crate::error::{ExecError, ExecResult};

// ── SIMD-accelerated attention primitives ────────────────────────────────
// Functions are conditionally compiled per target — suppress dead_code warnings
// from cfg branches that don't apply to the current compilation target.

/// Dot product of two f32 slices, using SIMD where available.
///
/// Falls back to scalar on unsupported platforms. The slices must have
/// the same length (typically `head_dim`, 64-128 for most models).
#[allow(dead_code)] // Used in non-BLAS path and tests
#[inline(always)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        dot_f32_neon(a, b)
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        dot_f32_avx2_fma(a, b)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    {
        dot_f32_scalar(a, b)
    }
}

/// Fused `out[i] += w * v[i]` using SIMD where available.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn accumulate_weighted(out: &mut [f32], v: &[f32], w: f32) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "aarch64")]
    {
        accumulate_weighted_neon(out, v, w);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        accumulate_weighted_avx2_fma(out, v, w);
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    {
        accumulate_weighted_scalar(out, v, w);
    }
}

/// Scale all elements: `out[i] *= factor`.
#[allow(dead_code)]
#[inline(always)]
fn scale_slice(out: &mut [f32], factor: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        scale_slice_neon(out, factor);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        scale_slice_avx2(out, factor);
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    {
        for val in out.iter_mut() {
            *val *= factor;
        }
    }
}

// ── Scalar fallbacks (used on platforms without SIMD) ────────────────────

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn accumulate_weighted_scalar(out: &mut [f32], v: &[f32], w: f32) {
    for (o, &val) in out.iter_mut().zip(v.iter()) {
        *o += w * val;
    }
}

// ── NEON (aarch64) ──────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // Used in non-BLAS path
#[inline(always)]
fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = vld1q_f32(a_ptr.add(i * 4));
            let vb = vld1q_f32(b_ptr.add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let mut sum = vaddvq_f32(acc);
        for i in 0..remainder {
            sum += a[chunks * 4 + i] * b[chunks * 4 + i];
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[inline(always)]
fn accumulate_weighted_neon(out: &mut [f32], v: &[f32], w: f32) {
    use std::arch::aarch64::*;
    let n = out.len();
    let chunks = n / 4;
    let remainder = n % 4;
    unsafe {
        let vw = vdupq_n_f32(w);
        let o_ptr = out.as_mut_ptr();
        let v_ptr = v.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let vo = vld1q_f32(o_ptr.add(off));
            let vv = vld1q_f32(v_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vfmaq_f32(vo, vw, vv));
        }
        for i in 0..remainder {
            let idx = chunks * 4 + i;
            out[idx] += w * v[idx];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[inline(always)]
fn scale_slice_neon(out: &mut [f32], factor: f32) {
    use std::arch::aarch64::*;
    let n = out.len();
    let chunks = n / 4;
    let remainder = n % 4;
    unsafe {
        let vf = vdupq_n_f32(factor);
        let ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            vst1q_f32(ptr.add(off), vmulq_f32(v, vf));
        }
        for i in 0..remainder {
            out[chunks * 4 + i] *= factor;
        }
    }
}

// ── AVX2 + FMA (x86_64) ────────────────────────────────────────────────

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        // Horizontal sum: 8 → 4 → 2 → 1
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum4 = _mm_add_ps(lo, hi);
        let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        let mut sum = _mm_cvtss_f32(sum1);
        for i in 0..remainder {
            sum += a[chunks * 8 + i] * b[chunks * 8 + i];
        }
        sum
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn accumulate_weighted_avx2_fma(out: &mut [f32], v: &[f32], w: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 8;
    let remainder = n % 8;
    unsafe {
        let vw = _mm256_set1_ps(w);
        let o_ptr = out.as_mut_ptr();
        let v_ptr = v.as_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let vo = _mm256_loadu_ps(o_ptr.add(off));
            let vv = _mm256_loadu_ps(v_ptr.add(off));
            _mm256_storeu_ps(o_ptr.add(off), _mm256_fmadd_ps(vw, vv, vo));
        }
        for i in 0..remainder {
            let idx = chunks * 8 + i;
            out[idx] += w * v[idx];
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
fn scale_slice_avx2(out: &mut [f32], factor: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 8;
    let remainder = n % 8;
    unsafe {
        let vf = _mm256_set1_ps(factor);
        let ptr = out.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(ptr.add(off));
            _mm256_storeu_ps(ptr.add(off), _mm256_mul_ps(v, vf));
        }
        for i in 0..remainder {
            out[chunks * 8 + i] *= factor;
        }
    }
}

/// Transpose from [seq, n_heads, head_dim] to [n_heads, seq, head_dim].
///
/// Single flat loop with index decomposition — avoids triple-nested loops
/// that defeat autovectorization and harm cache locality.
fn transpose_heads(data: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    let nh_hd = n_heads * head_dim;
    let s_hd = seq * head_dim;
    for (flat, &val) in data.iter().enumerate() {
        // Decompose flat index in source layout [seq, n_heads, head_dim]
        let t = flat / nh_hd;
        let h = (flat % nh_hd) / head_dim;
        let d = flat % head_dim;
        // Write to dest layout [n_heads, seq, head_dim]
        out[h * s_hd + t * head_dim + d] = val;
    }
    out
}

/// Threshold below which a normalized softmax weight is considered negligible.
/// At long context, 90%+ of positions fall below this — skipping their V accumulation
/// yields significant decode speedup with zero measurable quality loss.
/// Override at runtime via `HOLOGRAM_sparse_v_threshold()` env var (e.g. `1e-4`
/// for more aggressive pruning at very long contexts).
fn sparse_v_threshold() -> f32 {
    static THRESHOLD: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("HOLOGRAM_sparse_v_threshold()")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-6)
    })
}

/// Parameters for [`dispatch_attention`]. Required knobs are the three head
/// counts + head_dim; the optional `scale` defaults to `1.0 / sqrt(head_dim)`
/// and `causal` defaults to `true`. Use [`Self::with_heads_first`] /
/// [`Self::with_sparse_v`] / [`Self::with_scale`] / [`Self::with_causal`] to
/// override.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionParams {
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub scale: f32,
    pub causal: bool,
    pub heads_first: bool,
    pub sparse_v: bool,
}

impl AttentionParams {
    #[inline]
    pub fn new(head_dim: usize, num_q_heads: usize, num_kv_heads: usize) -> Self {
        Self {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale: if head_dim > 0 {
                1.0 / (head_dim as f32).sqrt()
            } else {
                1.0
            },
            causal: true,
            heads_first: false,
            sparse_v: false,
        }
    }

    #[inline]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    #[inline]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    #[inline]
    pub fn with_heads_first(mut self, heads_first: bool) -> Self {
        self.heads_first = heads_first;
        self
    }

    #[inline]
    pub fn with_sparse_v(mut self, sparse_v: bool) -> Self {
        self.sparse_v = sparse_v;
        self
    }
}

pub(crate) fn dispatch_attention(inputs: &[&[u8]], params: AttentionParams) -> ExecResult<Vec<u8>> {
    let AttentionParams {
        head_dim,
        num_q_heads,
        num_kv_heads,
        scale,
        causal,
        heads_first,
        sparse_v,
    } = params;
    let q_raw = cast_f32(inputs[0])?;
    let k_raw = cast_f32(inputs[1])?;
    let v_raw = cast_f32(inputs[2])?;

    // Validate buffer sizes before inferring sequence lengths.
    let q_stride = num_q_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;
    if q_stride == 0 || kv_stride == 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!(
                "non-zero head config (q_heads={num_q_heads}, kv_heads={num_kv_heads}, head_dim={head_dim})"
            ),
            actual: "zero stride".into(),
        });
    }
    if q_raw.len() % q_stride != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("Q length divisible by num_q_heads*head_dim={q_stride}"),
            actual: format!("Q has {} f32 elements", q_raw.len()),
        });
    }
    if k_raw.len() % kv_stride != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("K length divisible by num_kv_heads*head_dim={kv_stride}"),
            actual: format!("K has {} f32 elements", k_raw.len()),
        });
    }
    if v_raw.len() != k_raw.len() {
        return Err(ExecError::ShapeMismatch {
            expected: format!(
                "V length == K length ({}) [Q={}, K={}, V={} f32; q_heads={}, kv_heads={}, head_dim={}]",
                k_raw.len(), q_raw.len(), k_raw.len(), v_raw.len(),
                num_q_heads, num_kv_heads, head_dim,
            ),
            actual: format!("V has {} f32 elements", v_raw.len()),
        });
    }

    let seq_q = q_raw.len() / q_stride;
    let seq_k = k_raw.len() / kv_stride;
    if seq_q == 0 || seq_k == 0 {
        return Err(ExecError::ShapeMismatch {
            expected: "non-zero sequence lengths".into(),
            actual: format!(
                "seq_q={seq_q}, seq_k={seq_k} (Q={}, K={} f32 elems)",
                q_raw.len(),
                k_raw.len()
            ),
        });
    }

    // Ensure all inputs are in heads-first [heads, seq, head_dim] layout.
    #[allow(clippy::type_complexity)]
    let (q, k, v): (Cow<[f32]>, Cow<[f32]>, Cow<[f32]>) = if heads_first {
        (
            Cow::Borrowed(&*q_raw),
            Cow::Borrowed(&*k_raw),
            Cow::Borrowed(&*v_raw),
        )
    } else {
        (
            Cow::Owned(transpose_heads(&q_raw, seq_q, num_q_heads, head_dim)),
            Cow::Owned(transpose_heads(&k_raw, seq_k, num_kv_heads, head_dim)),
            Cow::Owned(transpose_heads(&v_raw, seq_k, num_kv_heads, head_dim)),
        )
    };
    // Optional additive mask from input[3] (ONNX attention mask).
    // Shape: broadcastable to [num_q_heads, seq_q, seq_k].
    let mask: Option<Cow<[f32]>> = if inputs.len() >= 4 && !inputs[3].is_empty() {
        Some(cast_f32(inputs[3])?)
    } else {
        None
    };

    let group_size = num_q_heads / num_kv_heads.max(1);

    let mut out = vec![0.0f32; num_q_heads * seq_q * head_dim];
    // Scores buffer only needed for BLAS path (non-BLAS uses online softmax).
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    let mut scores = vec![0.0f32; seq_q * seq_k];

    // Pre-compute per-head offsets: avoids repeated division/multiplication
    // in the per-head loop. For GQA, multiple Q heads map to the same KV head.
    let q_stride = seq_q * head_dim;
    let k_stride = seq_k * head_dim;
    let head_offsets: Vec<(usize, usize, usize)> = (0..num_q_heads)
        .map(|qh| {
            let kh = qh / group_size;
            (qh * q_stride, kh * k_stride, qh * q_stride)
        })
        .collect();

    for &(q_off, k_off, o_off) in &head_offsets {
        let q_head = &q[q_off..q_off + q_stride];
        let k_head = &k[k_off..k_off + k_stride];
        let v_head = &v[k_off..k_off + k_stride];

        // scores = Q_head × K_head^T * scale → [seq_q, seq_k]
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            super::matmul::blas::sgemm_full(
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
            // Apply mask after BLAS.
            if let Some(ref m) = mask {
                // Additive mask: scores[i,j] += mask[broadcast(i,j)].
                // Mask shape is typically [1, 1, seq_q, seq_k] or [seq_q, seq_k].
                // The last seq_q*seq_k elements are the per-position mask.
                let mask_2d_size = seq_q * seq_k;
                let mask_offset = if m.len() > mask_2d_size {
                    m.len() - mask_2d_size // Take the last [seq_q, seq_k] slice.
                } else {
                    0
                };
                for idx in 0..mask_2d_size {
                    if mask_offset + idx < m.len() {
                        scores[idx] += m[mask_offset + idx];
                    }
                }
            } else if causal {
                // When seq_q < seq_k (KV cache decode), use absolute positions.
                for i in 0..seq_q {
                    let abs_pos = seq_k - seq_q + i;
                    for j in (abs_pos + 1)..seq_k {
                        scores[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            // Online softmax attention (Flash Attention-style).
            // Fuses QK^T, softmax, and score×V into a single pass per query row.
            // Avoids materializing the full [seq_q, seq_k] scores matrix.
            let out_head = &mut out[o_off..o_off + seq_q * head_dim];
            out_head.fill(0.0);

            for i in 0..seq_q {
                let abs_pos = seq_k - seq_q + i;
                let limit = if causal && mask.is_none() {
                    (abs_pos + 1).min(seq_k)
                } else {
                    seq_k
                };
                let q_row = &q_head[i * head_dim..(i + 1) * head_dim];
                let o_row = &mut out_head[i * head_dim..(i + 1) * head_dim];

                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0f32;

                for j in 0..limit {
                    let k_row = &k_head[j * head_dim..(j + 1) * head_dim];
                    let dot = dot_f32(q_row, k_row);
                    let mut score = dot * scale;
                    if let Some(ref m) = mask {
                        let mask_idx = i * seq_k + j;
                        score += m[mask_idx % m.len()];
                    }

                    // Online softmax update: if new score exceeds running max,
                    // rescale the accumulated output and sum.
                    if score > row_max {
                        let correction = (row_max - score).exp();
                        row_sum *= correction;
                        scale_slice(o_row, correction);
                        row_max = score;
                    }

                    let w = (score - row_max).exp();
                    row_sum += w;

                    // Sparse V: skip V accumulation for negligible weights.
                    // At long context 90%+ of positions have near-zero weight;
                    // skipping them avoids head_dim multiply-adds per position.
                    if sparse_v && w < sparse_v_threshold() {
                        continue;
                    }

                    // Accumulate weighted V into output (SIMD-accelerated).
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    accumulate_weighted(o_row, v_row, w);
                }

                // Normalize by sum.
                if row_sum > 0.0 {
                    scale_slice(o_row, 1.0 / row_sum);
                }
            }
        }

        // BLAS path: use the existing 3-phase approach with score buffer.
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            // scores = Q_head × K_head^T * scale → [seq_q, seq_k]
            // (BLAS sgemm already computed scores above.)

            // Softmax each row (2-pass: max+exp+sum, then divide).
            // Sparse V: zero out negligible weights after normalization.
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
                        // Sparse V: zero out negligible normalized weights.
                        if sparse_v && *val < sparse_v_threshold() {
                            *val = 0.0;
                        }
                    }
                }
            }

            // out_head = scores × V_head → [seq_q, head_dim]
            let out_head = &mut out[o_off..o_off + seq_q * head_dim];
            super::matmul::blas::sgemm(seq_q, head_dim, seq_k, &scores, v_head, out_head);
        }
    }

    if heads_first {
        // ONNX: output stays in [n_heads, seq, head_dim] layout.
        Ok(f32_vec_to_bytes(out))
    } else {
        // GGUF: transpose output from [n_heads, seq, head_dim] to [seq, n_heads, head_dim].
        // Single flat loop with index decomposition.
        let mut final_out = vec![0.0f32; out.len()];
        let s_hd = seq_q * head_dim;
        let nh_hd = num_q_heads * head_dim;
        for (flat, &val) in out.iter().enumerate() {
            // Decompose flat index in source layout [n_heads, seq, head_dim]
            let h = flat / s_hd;
            let t = (flat % s_hd) / head_dim;
            let d = flat % head_dim;
            // Write to dest layout [seq, n_heads, head_dim]
            final_out[t * nh_hd + h * head_dim + d] = val;
        }
        Ok(f32_vec_to_bytes(final_out))
    }
}

pub(crate) fn dispatch_rope(
    inputs: &[&[u8]],
    dim: usize,
    base: f32,
    n_heads: usize,
    start_pos: usize,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;

    let half = dim / 2;
    let n_heads = n_heads.max(1);
    let mut out = x.into_owned();

    // Pre-compute frequency table: freq[i] = 1.0 / base^(2i/dim).
    // Avoids calling powf() per element in the inner loop.
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();

    // Apply RoPE to each chunk of `dim` elements. Multiple heads per token
    // share the same position: pos = chunk_index / n_heads.
    // Uses interleaved convention (ggml): pairs (0,1), (2,3), (4,5), ...
    for (chunk_idx, chunk) in out.chunks_mut(dim).enumerate() {
        let token_pos = chunk_idx / n_heads;
        let pos = (start_pos + token_pos) as f32;
        for (i, &freq) in freqs.iter().enumerate() {
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x0 = chunk[2 * i];
            let x1 = chunk[2 * i + 1];
            chunk[2 * i] = x0 * cos_a - x1 * sin_a;
            chunk[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
    Ok(f32_vec_to_bytes(out))
}
