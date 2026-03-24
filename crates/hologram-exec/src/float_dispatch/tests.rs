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
