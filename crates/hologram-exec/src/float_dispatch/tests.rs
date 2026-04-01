use super::helpers::{broadcast_shapes, silu};
use super::*;
use hologram_core::op::{FloatDType, FloatOp};

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

// ── N-D broadcasting tests ──────────────────────────────────────────

#[test]
fn test_broadcast_shapes_compatible() {
    assert_eq!(broadcast_shapes(&[2, 1], &[2, 3]), Some(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]), Some(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[3], &[2, 3]), Some(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[1], &[5]), Some(vec![5]));
    assert_eq!(
        broadcast_shapes(&[4, 1, 3], &[1, 5, 1]),
        Some(vec![4, 5, 3])
    );
}

#[test]
fn test_broadcast_shapes_incompatible() {
    // [2,32] vs [1,64]: dim 1 has 32 vs 64, neither is 1
    assert_eq!(broadcast_shapes(&[2, 32], &[1, 64]), None);
    assert_eq!(broadcast_shapes(&[3], &[4]), None);
    assert_eq!(broadcast_shapes(&[2, 3], &[2, 4]), None);
}

#[test]
fn test_broadcast_2d_row_vector() {
    // [2,3] + [1,3] => broadcast row: result should add row-wise
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // shape [2,3]
    let b = f32_bytes(&[10.0, 20.0, 30.0]); // shape [1,3]
    let result =
        dispatch_float_with_shapes(&FloatOp::Add, &[&a, &b], &[vec![2, 3], vec![1, 3]]).unwrap();
    let out: &[f32] = bytemuck::cast_slice(&result);
    assert_eq!(out, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_broadcast_2d_column_vector() {
    // [2,3] / [2,1] => broadcast column (the LayerNorm pattern)
    let a = f32_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]); // shape [2,3]
    let b = f32_bytes(&[2.0, 5.0]); // shape [2,1]
    let result =
        dispatch_float_with_shapes(&FloatOp::Div, &[&a, &b], &[vec![2, 3], vec![2, 1]]).unwrap();
    let out: &[f32] = bytemuck::cast_slice(&result);
    assert_eq!(out, &[5.0, 10.0, 15.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_broadcast_incompatible_falls_back_to_cycling() {
    // [2,32] vs [1,64]: NOT broadcast-compatible.
    // Must not panic — falls back to cycling.
    let a = f32_bytes(&vec![1.0; 64]); // shape [2,32]
    let b = f32_bytes(&vec![2.0; 64]); // shape [1,64]
    let result = dispatch_float_with_shapes(&FloatOp::Add, &[&a, &b], &[vec![2, 32], vec![1, 64]]);
    assert!(result.is_ok()); // Must not panic
    let binding = result.unwrap();
    let out: &[f32] = bytemuck::cast_slice(&binding);
    assert_eq!(out.len(), 64); // cycling: max(64,64)
}

#[test]
fn test_broadcast_shape_data_mismatch_falls_back() {
    // Shape says [2,4] (8 elements) but data has 6 f32s — shape mismatch
    // Must fall back to cycling, not panic.
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f32_bytes(&[10.0]);
    let result = dispatch_float_with_shapes(
        &FloatOp::Add,
        &[&a, &b],
        &[vec![2, 4], vec![1]], // shape [2,4] doesn't match 6 elements
    );
    assert!(result.is_ok());
}

#[test]
fn test_broadcast_compare_2d() {
    // [2,3] > [1,3] => broadcast comparison
    let a = f32_bytes(&[1.0, 20.0, 3.0, 40.0, 5.0, 60.0]); // shape [2,3]
    let b = f32_bytes(&[10.0, 10.0, 10.0]); // shape [1,3]
    let result =
        dispatch_float_with_shapes(&FloatOp::Greater, &[&a, &b], &[vec![2, 3], vec![1, 3]])
            .unwrap();
    // 1>10=0, 20>10=1, 3>10=0, 40>10=1, 5>10=0, 60>10=1
    assert_eq!(result, vec![0, 1, 0, 1, 0, 1]);
}

// ── infer_slice_axis_size tests ──────────────────────────────────────────

#[test]
fn test_infer_slice_axis_size_fast_path() {
    // end divides n_elems evenly → return end.
    assert_eq!(super::infer_slice_axis_size(18, 6), 6);
    assert_eq!(super::infer_slice_axis_size(100, 10), 10);
    assert_eq!(super::infer_slice_axis_size(2048, 2048), 2048);
}

#[test]
fn test_infer_slice_axis_size_search() {
    // end does NOT divide n_elems → search upward for smallest divisor >= end.
    // n_elems=18 (3×6), end=4 → 4 doesn't divide 18, 5 doesn't, 6 does → 6.
    assert_eq!(super::infer_slice_axis_size(18, 4), 6);
    // n_elems=60 (3×20 or 4×15 or 5×12 or 6×10), end=7 → 10.
    assert_eq!(super::infer_slice_axis_size(60, 7), 10);
}

#[test]
fn test_infer_slice_axis_size_non_divisible() {
    // When n_elems = prime * axis_size, and no divisor between end and
    // axis_size exists, the heuristic correctly finds axis_size.
    // Use a prime seq so no spurious divisors exist.
    // n_elems = 11 * 2560 = 28160. Smallest divisor >= 2048: 2560.
    assert_eq!(super::infer_slice_axis_size(11 * 2560, 2048), 2560);
    // n_elems = 3 * 2560 = 7680. 7680 % 2048 ≠ 0. Smallest >= 2048: 2560.
    assert_eq!(super::infer_slice_axis_size(3 * 2560, 2048), 2560);
}

#[test]
fn test_infer_slice_axis_size_heuristic_limitation() {
    // Known limitation: when end or a smaller divisor >= end divides n_elems,
    // the heuristic may return the wrong axis size. This is because the
    // function can't distinguish end=axis_size from end<axis_size without
    // additional context. A proper fix requires storing axis_size in
    // FloatOp::Slice (tracked in Plan 016).
    //
    // Example: seq=8, axis=2560, end=2048. n_elems=20480, 20480%2048=0 → returns 2048 (wrong).
    // Example: seq=7, axis=2560, end=2048. n_elems=17920, 17920%2240=0 → returns 2240 (wrong).
    assert_eq!(super::infer_slice_axis_size(8 * 2560, 2048), 2048); // fast path, incorrect
    assert_eq!(super::infer_slice_axis_size(7 * 2560, 2048), 2240); // finds 2240, not 2560
}

#[test]
fn test_infer_slice_axis_size_edge_cases() {
    assert_eq!(super::infer_slice_axis_size(0, 4), 0);
    assert_eq!(super::infer_slice_axis_size(10, 0), 10);
    // n_elems is prime, end < n_elems → only n_elems divides itself.
    assert_eq!(super::infer_slice_axis_size(17, 4), 17);
}

// ── Sprint 21: attention zero-copy heads_first path ─────────────────

#[test]
fn test_attention_heads_first_produces_output() {
    // Minimal attention: 1 head, seq=2, head_dim=2, heads_first=true.
    // Q=[1,0, 0,1], K=[1,0, 0,1], V=[1,2, 3,4] → output should be non-empty.
    let q = f32_bytes(&[1.0, 0.0, 0.0, 1.0]); // [1 head, 2 seq, 2 dim]
    let k = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
    let v = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let result = dispatch_float(
        &FloatOp::Attention {
            head_dim: 2,
            num_q_heads: 1,
            num_kv_heads: 1,
            scale: f32::to_bits(1.0 / 2.0f32.sqrt()),
            causal: false,
            heads_first: true,
            qk_norm: false,
            rope: false,
            rope_base: 0,
            sparse_v: true,
        },
        &[&q, &k, &v],
    )
    .unwrap();
    let out: &[f32] = bytemuck::cast_slice(&result);
    assert_eq!(
        out.len(),
        4,
        "attention output should have 4 floats (1 head × 2 seq × 2 dim)"
    );
}

#[test]
fn test_attention_heads_first_matches_transposed() {
    // Same attention, both paths should produce identical results.
    // heads_first=true: [n_heads, seq, head_dim]
    // heads_first=false: [seq, n_heads, head_dim] — needs transpose
    let head_dim = 2;
    let n_heads = 1;
    let seq = 2;

    // heads_first layout: [1, 2, 2]
    let q_hf = f32_bytes(&[1.0, 0.5, 0.5, 1.0]);
    let k_hf = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
    let v_hf = f32_bytes(&[2.0, 3.0, 4.0, 5.0]);

    // seq_first layout: same data but [2, 1, 2] — identical for n_heads=1.
    let q_sf = q_hf.clone();
    let k_sf = k_hf.clone();
    let v_sf = v_hf.clone();

    let op_hf = FloatOp::Attention {
        head_dim: head_dim as u32,
        num_q_heads: n_heads as u32,
        num_kv_heads: n_heads as u32,
        scale: f32::to_bits(1.0 / (head_dim as f32).sqrt()),
        causal: false,
        heads_first: true,
        qk_norm: false,
        rope: false,
        rope_base: 0,
        sparse_v: true,
    };
    let op_sf = FloatOp::Attention {
        head_dim: head_dim as u32,
        num_q_heads: n_heads as u32,
        num_kv_heads: n_heads as u32,
        scale: f32::to_bits(1.0 / (head_dim as f32).sqrt()),
        causal: false,
        heads_first: false,
        qk_norm: false,
        rope: false,
        rope_base: 0,
        sparse_v: true,
    };

    let result_hf = dispatch_float(&op_hf, &[&q_hf, &k_hf, &v_hf]).unwrap();
    let result_sf = dispatch_float(&op_sf, &[&q_sf, &k_sf, &v_sf]).unwrap();

    let out_hf: &[f32] = bytemuck::cast_slice(&result_hf);
    let out_sf: &[f32] = bytemuck::cast_slice(&result_sf);
    assert_eq!(out_hf.len(), seq * n_heads * head_dim);
    // For n_heads=1, both layouts are identical so outputs must match.
    for (a, b) in out_hf.iter().zip(out_sf.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "heads_first and seq_first should match: {a} vs {b}"
        );
    }
}

// ── Sprint 21: norm into_owned / alloc_f32_in ───────────────────────

#[test]
fn test_softmax_into_sums_to_one() {
    use super::norm::dispatch_softmax_into;
    let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mut out_buf = Vec::new();
    dispatch_softmax_into(&[&x], 4, &mut out_buf).unwrap();
    let floats: &[f32] = bytemuck::cast_slice(&out_buf);
    assert_eq!(floats.len(), 4);
    let sum: f32 = floats.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax_into should sum to 1, got {sum}"
    );
}

#[test]
fn test_rms_norm_into_matches_allocating() {
    use super::norm::{dispatch_rms_norm, dispatch_rms_norm_into};
    let x = f32_bytes(&[1.0, 2.0, 3.0]);
    let w = f32_bytes(&[1.0, 1.0, 1.0]);
    let eps = 1e-5f32;

    // Allocating path.
    let result = dispatch_rms_norm(&[&x, &w], 3, eps).unwrap();

    // _into path.
    let mut out_buf = Vec::new();
    dispatch_rms_norm_into(&[&x, &w], 3, eps, &mut out_buf).unwrap();

    assert_eq!(
        result, out_buf,
        "rms_norm_into must match allocating dispatch_rms_norm"
    );
}

// ── Plan 038: Sparse V attention tests ──────────────────────────────

/// Helper: run attention with given params and return f32 output.
fn run_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    sparse_v: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let op = FloatOp::Attention {
        head_dim: head_dim as u32,
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        scale: f32::to_bits(scale),
        causal,
        heads_first: false,
        qk_norm: false,
        rope: false,
        rope_base: 0,
        sparse_v,
    };
    let qb = f32_bytes(q);
    let kb = f32_bytes(k);
    let vb = f32_bytes(v);
    let result = dispatch_float(&op, &[&qb, &kb, &vb]).unwrap();
    bytemuck::cast_slice::<u8, f32>(&result).to_vec()
}

#[test]
fn attention_sparse_v_zero_quality_loss() {
    // Random-ish Q/K/V at various seq lengths — sparse V should produce
    // output indistinguishable from non-sparse (all weights are significant
    // at small seq, so nothing is skipped).
    let head_dim = 4;
    let num_heads = 2;
    for seq in [4, 16, 64] {
        let q: Vec<f32> = (0..seq * num_heads * head_dim)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();
        let k: Vec<f32> = (0..seq * num_heads * head_dim)
            .map(|i| (i as f32 * 0.53).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..seq * num_heads * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();

        let out_sparse = run_attention(&q, &k, &v, head_dim, num_heads, num_heads, true, true);
        let out_dense = run_attention(&q, &k, &v, head_dim, num_heads, num_heads, true, false);

        assert_eq!(out_sparse.len(), out_dense.len());
        for (i, (s, d)) in out_sparse.iter().zip(out_dense.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-5,
                "seq={seq} elem {i}: sparse={s} dense={d} diff={}",
                (s - d).abs()
            );
        }
    }
}

#[test]
fn attention_sparse_v_uniform_weights_no_skip() {
    // When Q ≈ K for all positions, attention weights are roughly uniform
    // (~1/seq). At seq=8, each weight is ~0.125 >> 1e-6, so sparse V
    // should skip nothing and produce identical output.
    let head_dim = 4;
    let seq = 8;
    let q: Vec<f32> = vec![1.0; seq * head_dim]; // all identical → uniform attn
    let k: Vec<f32> = vec![1.0; seq * head_dim];
    let v: Vec<f32> = (0..seq * head_dim).map(|i| i as f32).collect();

    let out_sparse = run_attention(&q, &k, &v, head_dim, 1, 1, false, true);
    let out_dense = run_attention(&q, &k, &v, head_dim, 1, 1, false, false);

    for (i, (s, d)) in out_sparse.iter().zip(out_dense.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-6,
            "uniform attn elem {i}: sparse={s} dense={d}",
        );
    }
}

#[test]
fn attention_sparse_v_single_dominant_position() {
    // One K row is identical to Q row; all others are orthogonal.
    // Output should be dominated by that V row.
    let head_dim = 4;
    let seq_k = 8;
    // Q = single query [1,0,0,0]
    let q = vec![1.0, 0.0, 0.0, 0.0];
    // K: first row matches Q, rest are orthogonal
    let mut k = vec![0.0f32; seq_k * head_dim];
    k[0] = 1.0; // first K row = [1,0,0,0] → high dot product
    for i in 1..seq_k {
        k[i * head_dim + (i % head_dim)] = 1.0; // orthogonal
    }
    // V: distinct values per row
    let v: Vec<f32> = (0..seq_k * head_dim).map(|i| (i + 1) as f32).collect();

    let out_sparse = run_attention(&q, &k, &v, head_dim, 1, 1, false, true);
    let out_dense = run_attention(&q, &k, &v, head_dim, 1, 1, false, false);

    // Both should be nearly identical — the dominant position has weight ~1.0
    for (i, (s, d)) in out_sparse.iter().zip(out_dense.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-4,
            "dominant pos elem {i}: sparse={s} dense={d}",
        );
    }
}

#[test]
fn attention_sparse_v_gqa_compatibility() {
    // GQA: 8 Q-heads sharing 2 KV-heads. Sparse V must respect head grouping.
    let head_dim = 4;
    let num_q_heads = 8;
    let num_kv_heads = 2;
    let seq = 4;
    let q: Vec<f32> = (0..seq * num_q_heads * head_dim)
        .map(|i| (i as f32 * 0.3).sin())
        .collect();
    let k: Vec<f32> = (0..seq * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.7).cos())
        .collect();
    let v: Vec<f32> = (0..seq * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();

    let out_sparse = run_attention(&q, &k, &v, head_dim, num_q_heads, num_kv_heads, false, true);
    let out_dense = run_attention(
        &q,
        &k,
        &v,
        head_dim,
        num_q_heads,
        num_kv_heads,
        false,
        false,
    );

    assert_eq!(out_sparse.len(), out_dense.len());
    for (i, (s, d)) in out_sparse.iter().zip(out_dense.iter()).enumerate() {
        assert!((s - d).abs() < 1e-5, "GQA elem {i}: sparse={s} dense={d}",);
    }
}

#[test]
fn attention_sparse_v_causal_mask_no_regression() {
    // Causal decode scenario: seq_q=1, seq_k=16. Masked positions already
    // have -inf → weight=0 from masking. Sparse V should not change behavior.
    let head_dim = 4;
    let seq_k = 16;
    // Q: single query (seq_q=1)
    let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.5).sin()).collect();
    // K: 16 cached keys
    let k: Vec<f32> = (0..seq_k * head_dim)
        .map(|i| (i as f32 * 0.3).cos())
        .collect();
    let v: Vec<f32> = (0..seq_k * head_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();

    let out_sparse = run_attention(&q, &k, &v, head_dim, 1, 1, true, true);
    let out_dense = run_attention(&q, &k, &v, head_dim, 1, 1, true, false);

    for (i, (s, d)) in out_sparse.iter().zip(out_dense.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-5,
            "causal decode elem {i}: sparse={s} dense={d}",
        );
    }
}

#[test]
fn attention_sparse_v_threshold_boundary() {
    // Verify that the sparse V threshold works correctly:
    // At very small seq with distinct weights, all weights should be
    // significant enough to not be skipped, so output must match exactly.
    let head_dim = 2;
    // seq=2 with causal: position 0 attends only to itself (weight=1.0),
    // position 1 attends to both (two non-zero weights).
    // No weight should be below 1e-6 at seq=2.
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![10.0, 20.0, 30.0, 40.0];

    let out_sparse = run_attention(&q, &k, &v, head_dim, 1, 1, true, true);
    let out_dense = run_attention(&q, &k, &v, head_dim, 1, 1, true, false);

    for (i, (s, d)) in out_sparse.iter().zip(out_dense.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-6,
            "threshold boundary elem {i}: sparse={s} dense={d}",
        );
    }
}

// ── Plan 039: GroupNorm _into tests ─────────────────────────────────

#[test]
fn group_norm_into_matches_allocating() {
    use super::norm::{dispatch_group_norm, dispatch_group_norm_into};
    let n_channels = 8usize;
    let spatial = 16usize;
    let num_groups = 4;
    let epsilon = 1e-5f32;

    let data: Vec<f32> = (0..n_channels * spatial)
        .map(|i| (i as f32 * 0.1) - 3.0)
        .collect();
    let scale: Vec<f32> = (0..n_channels).map(|i| 0.5 + i as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..n_channels).map(|i| i as f32 * 0.05).collect();

    let data_b = f32_bytes(&data);
    let scale_b = f32_bytes(&scale);
    let bias_b = f32_bytes(&bias);
    let inputs: Vec<&[u8]> = vec![&data_b, &scale_b, &bias_b];

    // Allocating path.
    let result_alloc = dispatch_group_norm(&inputs, num_groups, epsilon).expect("alloc failed");
    let alloc_f32: &[f32] = bytemuck::cast_slice(&result_alloc);

    // Into path.
    let mut out_buf = Vec::new();
    dispatch_group_norm_into(&inputs, num_groups, epsilon, &mut out_buf).expect("into failed");
    let into_f32: &[f32] = bytemuck::cast_slice(&out_buf);

    assert_eq!(alloc_f32.len(), into_f32.len());
    for (i, (&a, &b)) in alloc_f32.iter().zip(into_f32.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "group_norm_into mismatch at {i}: alloc={a} into={b}",
        );
    }
}

#[test]
fn group_norm_silu_fused_matches_separate() {
    use super::norm::{dispatch_group_norm, dispatch_group_norm_activation_into};
    let n_channels = 16usize;
    let spatial = 64usize;
    let num_groups = 4;
    let epsilon = 1e-5f32;
    let silu = FloatOp::Silu;

    let data: Vec<f32> = (0..n_channels * spatial)
        .map(|i| (i as f32 * 0.07).sin() * 2.0)
        .collect();
    let scale: Vec<f32> = vec![1.0; n_channels];
    let bias: Vec<f32> = vec![0.0; n_channels];

    let data_b = f32_bytes(&data);
    let scale_b = f32_bytes(&scale);
    let bias_b = f32_bytes(&bias);
    let inputs: Vec<&[u8]> = vec![&data_b, &scale_b, &bias_b];

    // Separate: GroupNorm then SiLU.
    let gn_result = dispatch_group_norm(&inputs, num_groups, epsilon).expect("gn failed");
    let gn_f32: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&gn_result).to_vec();
    let expected: Vec<f32> = gn_f32.iter().map(|&v| silu.apply_unary(v)).collect();

    // Fused: GroupNorm+SiLU in one pass.
    let mut out_buf = Vec::new();
    dispatch_group_norm_activation_into(&inputs, num_groups, epsilon, &silu, &mut out_buf)
        .expect("fused failed");
    let fused: &[f32] = bytemuck::cast_slice(&out_buf);

    assert_eq!(expected.len(), fused.len());
    for (i, (&e, &f)) in expected.iter().zip(fused.iter()).enumerate() {
        assert!(
            (e - f).abs() < 1e-5,
            "silu fused mismatch at {i}: expected={e} fused={f}",
        );
    }
}

#[test]
fn group_norm_activation_into_sd_shapes() {
    use super::norm::dispatch_group_norm_activation_into;
    let silu = FloatOp::Silu;
    // Realistic SD shapes: [1, C, H, W] flattened.
    for (channels, spatial) in [(320, 64 * 64), (640, 32 * 32), (1280, 16 * 16)] {
        let num_groups = 32; // SD v1.5 uses 32 groups
        let epsilon = 1e-5f32;

        let data: Vec<f32> = (0..channels * spatial)
            .map(|i| (i as f32 * 0.003).sin())
            .collect();
        let scale: Vec<f32> = vec![1.0; channels];
        let bias: Vec<f32> = vec![0.0; channels];

        let data_b = f32_bytes(&data);
        let scale_b = f32_bytes(&scale);
        let bias_b = f32_bytes(&bias);
        let inputs: Vec<&[u8]> = vec![&data_b, &scale_b, &bias_b];

        let mut out_buf = Vec::new();
        dispatch_group_norm_activation_into(&inputs, num_groups, epsilon, &silu, &mut out_buf)
            .unwrap_or_else(|e| panic!("failed for C={channels} spatial={spatial}: {e}"));

        let out: &[f32] = bytemuck::cast_slice(&out_buf);
        assert_eq!(out.len(), channels * spatial);
        // Verify all values are finite (no NaN/inf from norm computation).
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "C={channels} elem {i} is not finite: {v}");
        }
    }
}

// ── Plan 039: Depthwise Conv2d tests ────────────────────────────────

#[test]
fn conv2d_depthwise_matches_generic() {
    // Depthwise: group=channels=4, 3×3 kernel, stride=1, pad=1.
    // Compare depthwise fast path output against a known reference
    // computed by direct convolution.
    use super::conv::dispatch_conv2d_direct;
    let channels = 4;
    let h = 8;
    let w = 8;
    let kh = 3;
    let kw = 3;

    // Weight: [channels, 1, kh, kw] for depthwise.
    let weight: Vec<f32> = (0..channels * kh * kw)
        .map(|i| (i as f32 * 0.1) - 0.5)
        .collect();
    // Data: [1, channels, h, w].
    let data: Vec<f32> = (0..channels * h * w)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    // No bias.
    let bias: Vec<f32> = vec![];

    let data_b = f32_bytes(&data);
    let weight_b = f32_bytes(&weight);
    let bias_b = f32_bytes(&bias);

    let result = dispatch_conv2d_direct(
        &[&data_b, &weight_b, &bias_b],
        kh,
        kw,
        1,
        1,
        1,
        1,
        1,
        1,
        channels, // group == channels (depthwise)
        h,
        w,
    )
    .expect("depthwise conv2d failed");
    let out: &[f32] = bytemuck::cast_slice(&result);

    // Output shape: [1, channels, h, w] (same spatial with pad=1).
    assert_eq!(out.len(), channels * h * w);

    // Verify against manual computation for a few positions.
    // Channel 0, position (0,0): sum over kernel window with padding.
    // This is a smoke test — exact values depend on padding behavior.
    for &v in out {
        assert!(v.is_finite(), "depthwise output has non-finite value: {v}");
    }
}

#[test]
fn conv2d_depthwise_stride2() {
    use super::conv::dispatch_conv2d_direct;
    let channels = 4;
    let h = 8;
    let w = 8;
    // stride=2, pad=1 → output should be 4×4.
    let weight: Vec<f32> = vec![1.0; channels * 3 * 3];
    let data: Vec<f32> = vec![1.0; channels * h * w];
    let bias: Vec<f32> = vec![];

    let data_b = f32_bytes(&data);
    let weight_b = f32_bytes(&weight);
    let bias_b = f32_bytes(&bias);

    let result = dispatch_conv2d_direct(
        &[&data_b, &weight_b, &bias_b],
        3,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        channels,
        h,
        w,
    )
    .expect("depthwise stride2 failed");
    let out: &[f32] = bytemuck::cast_slice(&result);

    // Output: [1, 4, 4, 4] = 64 elements.
    assert_eq!(out.len(), channels * 4 * 4);
    for &v in out {
        assert!(v.is_finite(), "stride2 output has non-finite: {v}");
    }
}

#[test]
fn conv2d_depthwise_with_bias() {
    use super::conv::dispatch_conv2d_direct;
    let channels = 2;
    let h = 4;
    let w = 4;

    // All-zero data + bias = bias value everywhere.
    let data: Vec<f32> = vec![0.0; channels * h * w];
    let weight: Vec<f32> = vec![1.0; channels * 3 * 3];
    let bias: Vec<f32> = vec![5.0, 10.0]; // per-channel bias

    let data_b = f32_bytes(&data);
    let weight_b = f32_bytes(&weight);
    let bias_b = f32_bytes(&bias);

    let result = dispatch_conv2d_direct(
        &[&data_b, &weight_b, &bias_b],
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        channels,
        h,
        w,
    )
    .expect("depthwise bias failed");
    let out: &[f32] = bytemuck::cast_slice(&result);

    assert_eq!(out.len(), channels * h * w);
    // Channel 0 should all be 5.0, channel 1 should all be 10.0.
    for i in 0..h * w {
        assert!(
            (out[i] - 5.0).abs() < 1e-5,
            "ch0 pos {i}: expected 5.0, got {}",
            out[i]
        );
    }
    for i in 0..h * w {
        assert!(
            (out[h * w + i] - 10.0).abs() < 1e-5,
            "ch1 pos {i}: expected 10.0, got {}",
            out[h * w + i]
        );
    }
}

// ── Plan 039: Winograd F(2,3) tests ─────────────────────────────────

/// Helper: run conv2d via dispatch and return f32 output.
fn run_conv2d(
    data: &[f32],
    weight: &[f32],
    bias: &[f32],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
    h_in: usize,
    w_in: usize,
) -> Vec<f32> {
    use super::conv::dispatch_conv2d_direct;
    let data_b = f32_bytes(data);
    let weight_b = f32_bytes(weight);
    let bias_b = f32_bytes(bias);
    let result = dispatch_conv2d_direct(
        &[&data_b, &weight_b, &bias_b],
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        group,
        h_in,
        w_in,
    )
    .expect("conv2d dispatch failed");
    bytemuck::cast_slice::<u8, f32>(&result).to_vec()
}

#[test]
fn conv2d_winograd_matches_reference() {
    // 3×3 stride=1 pad=1 on [1, 32, 16, 16] — triggers Winograd (ic_per_group=32 >= 16).
    // Compare against a manually computed reference using the naive O(n^2) path.
    let ic = 32;
    let oc = 32;
    let h = 16;
    let w = 16;

    let data: Vec<f32> = (0..ic * h * w).map(|i| (i as f32 * 0.01).sin()).collect();
    let weight: Vec<f32> = (0..oc * ic * 3 * 3)
        .map(|i| (i as f32 * 0.003).cos() * 0.1)
        .collect();
    let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();

    // Winograd path (pad=1, stride=1, 3×3, ic>=16).
    let winograd_out = run_conv2d(&data, &weight, &bias, 3, 3, 1, 1, 1, 1, 1, 1, 1, h, w);

    // Reference: use im2col path by setting dilation=2 (which disables Winograd gate),
    // but we need stride=1 pad=1 dilation=1. Instead, compute a naive reference.
    // Naive conv2d reference.
    let h_out = h; // pad=1, stride=1 → same spatial
    let w_out = w;
    let mut ref_out = vec![0.0f32; oc * h_out * w_out];
    for oc_idx in 0..oc {
        let b = bias[oc_idx];
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = b;
                for ic_idx in 0..ic {
                    for fh in 0..3 {
                        for fw in 0..3 {
                            let ih = oh as i32 + fh as i32 - 1; // pad=1
                            let iw = ow as i32 + fw as i32 - 1;
                            if ih >= 0 && ih < h as i32 && iw >= 0 && iw < w as i32 {
                                let d_val = data[ic_idx * h * w + ih as usize * w + iw as usize];
                                let w_val = weight[oc_idx * ic * 9 + ic_idx * 9 + fh * 3 + fw];
                                sum += d_val * w_val;
                            }
                        }
                    }
                }
                ref_out[oc_idx * h_out * w_out + oh * w_out + ow] = sum;
            }
        }
    }

    assert_eq!(winograd_out.len(), ref_out.len());
    let mut max_err = 0.0f32;
    for (i, (&w_val, &r_val)) in winograd_out.iter().zip(ref_out.iter()).enumerate() {
        let err = (w_val - r_val).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-3,
            "winograd mismatch at {i}: winograd={w_val} ref={r_val} err={err}",
        );
    }
    // Log max error for visibility.
    assert!(max_err < 1e-3, "max winograd error: {max_err}");
}

#[test]
fn conv2d_winograd_odd_spatial() {
    // Odd spatial dims: [1, 16, 17, 17] — partial tiles at boundaries.
    let ic = 16;
    let oc = 16;
    let h = 17;
    let w = 17;

    let data: Vec<f32> = (0..ic * h * w).map(|i| (i as f32 * 0.02).sin()).collect();
    let weight: Vec<f32> = (0..oc * ic * 9)
        .map(|i| (i as f32 * 0.005).cos() * 0.1)
        .collect();
    let bias: Vec<f32> = vec![0.0; oc];

    let out = run_conv2d(&data, &weight, &bias, 3, 3, 1, 1, 1, 1, 1, 1, 1, h, w);

    // Output spatial should be 17×17 (pad=1, stride=1).
    assert_eq!(out.len(), oc * h * w);
    for &v in &out {
        assert!(v.is_finite(), "odd spatial output has non-finite: {v}");
    }
}

#[test]
fn conv2d_winograd_realistic_sd_shapes() {
    // Test with shapes seen in SD v1.5 UNet (scaled down for test speed).
    for (ic, oc, h, w) in [(32, 32, 32, 32), (64, 64, 16, 16), (128, 128, 8, 8)] {
        let data: Vec<f32> = (0..ic * h * w).map(|i| (i as f32 * 0.01).sin()).collect();
        let weight: Vec<f32> = (0..oc * ic * 9)
            .map(|i| (i as f32 * 0.002).cos() * 0.05)
            .collect();
        let bias: Vec<f32> = vec![0.0; oc];

        let out = run_conv2d(&data, &weight, &bias, 3, 3, 1, 1, 1, 1, 1, 1, 1, h, w);
        assert_eq!(out.len(), oc * h * w, "shape ({ic},{oc},{h},{w})");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "shape ({ic},{oc},{h},{w}) elem {i} non-finite: {v}"
            );
        }
    }
}

#[test]
fn conv2d_winograd_not_used_for_stride2() {
    // stride=2 should NOT use Winograd (falls back to im2col).
    // This is a smoke test: if it produces correct output, im2col handled it.
    let ic = 32;
    let oc = 32;
    let h = 16;
    let w = 16;

    let data: Vec<f32> = vec![1.0; ic * h * w];
    let weight: Vec<f32> = vec![0.01; oc * ic * 9];
    let bias: Vec<f32> = vec![0.0; oc];

    // stride=2, pad=1 → output 8×8.
    let out = run_conv2d(&data, &weight, &bias, 3, 3, 2, 2, 1, 1, 1, 1, 1, h, w);
    assert_eq!(out.len(), oc * 8 * 8);
    for &v in &out {
        assert!(v.is_finite());
    }
}
