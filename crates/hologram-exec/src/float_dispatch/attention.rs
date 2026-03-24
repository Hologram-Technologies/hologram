use std::borrow::Cow;

use super::helpers::*;
#[cfg(all(feature = "accelerate", target_os = "macos"))]
use super::matmul::GemmParams;
use crate::error::{ExecError, ExecResult};

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

pub(crate) fn dispatch_attention(
    inputs: &[&[u8]],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    causal: bool,
    heads_first: bool,
) -> ExecResult<Vec<u8>> {
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

    // GGUF inputs arrive as [seq, n_heads, head_dim] — transpose to heads-first.
    // ONNX inputs arrive as [n_heads, seq, head_dim] — already heads-first (zero-copy).
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
                    let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                    let mut score = dot * scale;
                    if let Some(ref m) = mask {
                        let mask_idx = i * seq_k + j;
                        score += m[mask_idx % m.len()];
                    }

                    // Online softmax update: if new score exceeds running max,
                    // rescale the accumulated output and sum.
                    if score > row_max {
                        let correction = (row_max - score).exp();
                        // Rescale running sum and accumulated output.
                        row_sum *= correction;
                        for val in o_row.iter_mut().take(head_dim) {
                            *val *= correction;
                        }
                        row_max = score;
                    }

                    let w = (score - row_max).exp();
                    row_sum += w;

                    // Accumulate weighted V into output.
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    for (o, &v) in o_row.iter_mut().zip(v_row.iter()) {
                        *o += w * v;
                    }
                }

                // Normalize by sum.
                if row_sum > 0.0 {
                    let inv = 1.0 / row_sum;
                    for val in o_row.iter_mut().take(head_dim) {
                        *val *= inv;
                    }
                }
            }
        }

        // BLAS path: use the existing 3-phase approach with score buffer.
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            // scores = Q_head × K_head^T * scale → [seq_q, seq_k]
            // (BLAS sgemm already computed scores above.)

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
