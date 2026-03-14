//! Op-chain tests for shape correctness.
//!
//! These tests target connected-op patterns where bugs compound silently.
//! Each test models a real pattern from TinyLlama's computation graph.
//!
//! Note on API scope:
//! - `dispatch_float` handles 2-input Concat, Neg, Mul, Add directly.
//! - `Slice` is dispatched inside the executor (executor.rs), not via
//!   `dispatch_float`. Slice chain tests are in exec_conformance.rs (ONNX pipeline).
//! - N-input Concat (>2) is also executor-level only.

use bytemuck::cast_slice;
use hologram_core::op::{FloatDType, FloatOp};
use hologram_exec::float_dispatch::dispatch_float;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(b).to_vec()
}

fn i64_to_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ── 2-input Concat size-preservation (rotate_half building block) ─────────────

/// Concat of two equal-size f32 tensors must return 2× the input size.
/// This is the core building block for rotate_half in TinyLlama RoPE.
#[test]
fn concat_two_halves_doubles_size() {
    let half_elems: usize = 4 * 2 * 4; // n_heads=4, seq=2, half_head_dim=4
    let a: Vec<f32> = (0..half_elems).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..half_elems).map(|i| (i + 100) as f32).collect();

    let a_bytes = f32_to_bytes(&a);
    let b_bytes = f32_to_bytes(&b);

    let result = dispatch_float(
        &FloatOp::Concat {
            size_a: 1,
            size_b: 1,
            dtype: FloatDType::F32,
        },
        &[&a_bytes, &b_bytes],
    )
    .expect("concat");

    let out = bytes_to_f32(&result);
    assert_eq!(
        out.len(),
        2 * half_elems,
        "concat must double element count"
    );

    // First half = a, second half = b.
    assert_eq!(&out[..half_elems], a.as_slice());
    assert_eq!(&out[half_elems..], b.as_slice());
}

/// Neg → 2-input Concat: concat(neg(b), a) must have same size as concat(a, b).
/// This is the rotate_half pattern: output = concat(neg(x_second), x_first).
#[test]
fn neg_then_concat_preserves_total_size() {
    let n: usize = 32; // arbitrary element count per half
    let x_first: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let x_second: Vec<f32> = (0..n).map(|i| (i + n) as f32 * 0.1).collect();

    let x_first_bytes = f32_to_bytes(&x_first);
    let x_second_bytes = f32_to_bytes(&x_second);

    let neg_second = dispatch_float(&FloatOp::Neg, &[&x_second_bytes]).expect("neg");

    let rotated = dispatch_float(
        &FloatOp::Concat {
            size_a: 1,
            size_b: 1,
            dtype: FloatDType::F32,
        },
        &[&neg_second, &x_first_bytes],
    )
    .expect("concat rotated");

    let out = bytes_to_f32(&rotated);
    assert_eq!(
        out.len(),
        2 * n,
        "rotate_half concat must have size 2*n: expected {}, got {}",
        2 * n,
        out.len()
    );

    // Values: [-x_second..., x_first...]
    for (i, &v) in out[..n].iter().enumerate() {
        let expected = -x_second[i];
        assert!(
            (v - expected).abs() < 1e-6,
            "neg values mismatch at {i}: {v} != {expected}"
        );
    }
    for (i, &v) in out[n..].iter().enumerate() {
        let expected = x_first[i];
        assert!(
            (v - expected).abs() < 1e-6,
            "pass-through values mismatch at {i}: {v} != {expected}"
        );
    }
}

/// RoPE elementwise chain: x*cos + rotate_half(x)*sin must not change element count.
///
/// With rotate_half(x) same size as x, and Mul/Add being elementwise,
/// the output must be same size as input.
#[test]
fn rope_elementwise_preserves_size() {
    let n: usize = 4 * 5 * 8; // n_heads * seq * head_dim = 160

    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let cos_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).cos()).collect();
    let sin_v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();

    // Build rotate_half(x) using 2-input concat on pre-split halves.
    let half = n / 2;
    let x_first: Vec<f32> = (0..half).map(|i| x[i]).collect();
    let x_second: Vec<f32> = (half..n).map(|i| x[i]).collect();

    let x_bytes = f32_to_bytes(&x);
    let cos_bytes = f32_to_bytes(&cos_v);
    let sin_bytes = f32_to_bytes(&sin_v);
    let x_first_bytes = f32_to_bytes(&x_first);
    let neg_second = dispatch_float(&FloatOp::Neg, &[&f32_to_bytes(&x_second)]).expect("neg");

    let rotated = dispatch_float(
        &FloatOp::Concat {
            size_a: 1,
            size_b: 1,
            dtype: FloatDType::F32,
        },
        &[&neg_second, &x_first_bytes],
    )
    .expect("concat rotated");

    assert_eq!(
        rotated.len(),
        x_bytes.len(),
        "rotate_half must match x size"
    );

    // x*cos + rotate_half(x)*sin
    let x_cos = dispatch_float(&FloatOp::Mul, &[&x_bytes, &cos_bytes]).expect("x*cos");
    let rot_sin = dispatch_float(&FloatOp::Mul, &[&rotated, &sin_bytes]).expect("rot*sin");
    let out = dispatch_float(&FloatOp::Add, &[&x_cos, &rot_sin]).expect("add");

    let result = bytes_to_f32(&out);
    assert_eq!(
        result.len(),
        n,
        "RoPE output must have same size as input: expected {n}, got {}",
        result.len()
    );
}

// ── Reshape -1 inference ──────────────────────────────────────────────────────

/// Regression: Reshape with -1 must infer n_heads correctly (NOT head_dim).
///
/// For Q = [1, seq, n_heads*head_dim] = [1, 5, 2048], reshaping to
/// [1, 5, -1, 64] must produce [1, 5, 32, 64] — not [1, 5, 64, 64].
///
/// The [1, 5, 64, 64] wrong result would give hidden_size=4096 per position,
/// which is the root cause of the TinyLlama ONNX batched MatMul shape mismatch.
#[test]
fn reshape_neg1_infers_n_heads_not_head_dim() {
    use hologram_exec::float_dispatch::dispatch_reshape_with_shape;

    let n_heads: usize = 32;
    let seq: usize = 5;
    let head_dim: usize = 64;
    let hidden = n_heads * head_dim; // 2048
    let total = seq * hidden; // 10240

    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let data_bytes = f32_to_bytes(&data);

    // Shape tensor: [1, 5, -1, 64] as i64.
    // Note: the actual ONNX shape tensor from TinyLlama is [1, seq, -1, head_dim].
    let shape_i64: Vec<i64> = vec![1, seq as i64, -1, head_dim as i64];
    let shape_bytes = i64_to_bytes(&shape_i64);

    let (out_bytes, out_shape) =
        dispatch_reshape_with_shape(&[&data_bytes, &shape_bytes]).expect("reshape");

    let out_floats: Vec<f32> = bytemuck::cast_slice(&out_bytes).to_vec();
    assert_eq!(
        out_floats.len(),
        total,
        "reshape must preserve element count"
    );

    let inferred_heads = out_shape.get(2).copied().unwrap_or(0);
    assert_eq!(
        inferred_heads, n_heads,
        "Reshape -1 must infer n_heads={n_heads} (not head_dim={head_dim}); got {inferred_heads}. \
        If wrong, this causes A=[seq, n_heads*head_dim*2] in the projection MatMul."
    );
    assert_eq!(
        out_shape,
        vec![1, seq, n_heads, head_dim],
        "Reshape output shape must be [1, seq, n_heads, head_dim]"
    );
}

/// Regression: Reshape -1 for KV heads.
/// [1, seq, 256] → shape_tensor [1, seq, -1, 64] → [1, seq, 4, 64]
#[test]
fn reshape_neg1_infers_kv_heads() {
    use hologram_exec::float_dispatch::dispatch_reshape_with_shape;

    let n_kv_heads: usize = 4;
    let seq: usize = 5;
    let head_dim: usize = 64;
    let kv_hidden = n_kv_heads * head_dim; // 256
    let total = seq * kv_hidden; // 1280

    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let data_bytes = f32_to_bytes(&data);

    let shape_i64: Vec<i64> = vec![1, seq as i64, -1, head_dim as i64];
    let shape_bytes = i64_to_bytes(&shape_i64);

    let (out_bytes, out_shape) =
        dispatch_reshape_with_shape(&[&data_bytes, &shape_bytes]).expect("reshape kv");

    let out_floats: Vec<f32> = bytemuck::cast_slice(&out_bytes).to_vec();
    assert_eq!(out_floats.len(), total, "element count preserved");
    assert_eq!(
        out_shape.get(2).copied().unwrap_or(0),
        n_kv_heads,
        "must infer n_kv_heads={n_kv_heads}"
    );
    assert_eq!(out_shape, vec![1, seq, n_kv_heads, head_dim]);
}

// ── NaN detector i64 values ───────────────────────────────────────────────────

/// Regression: 2-input i64 Concat must produce correct values, not garbled bytes.
/// An i64 value of -1 (0xFFFFFFFFFFFFFFFF) looks like two f32 NaN values.
/// The NaN detector in executor.rs was fixed to skip non-f32 ops.
/// This test validates the 2-input i64 Concat output bytes are correct.
#[test]
fn i64_concat_two_inputs_correct_values() {
    let a_i64: Vec<i64> = vec![1i64, 40i64]; // [batch, seq]
    let b_i64: Vec<i64> = vec![-1i64, 64i64]; // [-1, head_dim] for shape tensor

    let a_bytes = i64_to_bytes(&a_i64);
    let b_bytes = i64_to_bytes(&b_i64);

    let concat_result = dispatch_float(
        &FloatOp::Concat {
            size_a: 1,
            size_b: 1,
            dtype: FloatDType::I64,
        },
        &[&a_bytes, &b_bytes],
    )
    .expect("i64 concat");

    // 4 i64 values = 32 bytes.
    assert_eq!(concat_result.len(), 32, "4 i64 values = 32 bytes");

    let result_i64: Vec<i64> = concat_result
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        result_i64,
        vec![1i64, 40, -1, 64],
        "i64 Concat values must be exact"
    );
}

// ── Batched MatMul shape guard ────────────────────────────────────────────────

/// Regression: batched MatMul must error when K-dim doesn't match.
/// TinyLlama bug: A=[40, 4096] but weight=[2048, 2048] → must return error.
#[test]
fn batched_matmul_rejects_k_dim_mismatch() {
    use hologram_exec::float_dispatch::dispatch_batched_matmul;

    let a: Vec<f32> = vec![0.0f32; 1 * 40 * 4096]; // wrong hidden (should be 2048)
    let b: Vec<f32> = vec![0.0f32; 2048 * 2048]; // actual output-projection weight

    let result = dispatch_batched_matmul(
        &[cast_slice(&a), cast_slice(&b)],
        &[1, 40, 4096],
        &[2048, 2048],
    );

    assert!(
        result.is_err(),
        "batched MatMul with mismatched K-dim must return Err, not silently produce wrong output"
    );
}

/// Correctness: batched MatMul with matching K-dim must succeed.
#[test]
fn batched_matmul_correct_shapes_succeed() {
    use hologram_exec::float_dispatch::dispatch_batched_matmul;

    let a: Vec<f32> = vec![1.0f32; 1 * 40 * 2048];
    let b: Vec<f32> = vec![0.5f32; 2048 * 2048];

    let (out_bytes, out_shape) = dispatch_batched_matmul(
        &[cast_slice(&a), cast_slice(&b)],
        &[1, 40, 2048],
        &[2048, 2048],
    )
    .expect("batched matmul should succeed with correct shapes");

    let out: Vec<f32> = bytemuck::cast_slice(&out_bytes).to_vec();
    assert_eq!(out_shape, vec![1, 40, 2048]);
    assert_eq!(out.len(), 40 * 2048);
    // Each output = sum of 2048 ones × 0.5 = 1024.0
    for &v in &out {
        assert!((v - 1024.0f32).abs() < 0.5, "expected ~1024.0, got {v}");
    }
}

// ── FloatOp::Shape start/end slicing ─────────────────────────────────────────

/// Regression: Shape(K, start=2, end=4) of a [1,2,6,8] tensor must return [6,8].
///
/// TinyLlama uses this pattern for GQA attention: K has shape
/// [batch, n_kv_heads, seq_len, head_dim], and `Shape(K, start=2, end=4)`
/// extracts [seq_len, head_dim] for use in downstream Reshape/Expand ops.
///
/// The original bug: `FloatOp::Shape` ignored `start`/`end`, returning ALL dims.
/// This test directly exercises `dispatch_shape_sliced` (the kernel backing the
/// executor's Shape handler) to verify correct slicing without requiring a full
/// compiled graph (the AiGraph pipeline constant-folds Shape ops).
#[test]
fn shape_start_end_extracts_seq_and_head_dim() {
    use hologram_core::op::FloatDType;
    use hologram_exec::float_dispatch::dispatch_shape_sliced;

    // K: [batch=1, n_kv_heads=2, seq=6, head_dim=8]
    let in_shape = vec![1usize, 2, 6, 8];

    // Shape(K, start=2, end=4) → [seq, head_dim] = [6, 8]
    let out = dispatch_shape_sliced(&in_shape, FloatDType::I64, 2, 4).expect("shape sliced 2:4");
    let dims: Vec<i64> = bytemuck::cast_slice(&out).to_vec();
    assert_eq!(
        dims,
        vec![6i64, 8],
        "Shape(start=2,end=4) of [1,2,6,8] must return [6,8], got {dims:?}"
    );
}

/// Shape(K, start=0, end=2) must return the batch and head-count dims.
#[test]
fn shape_start_end_extracts_batch_and_heads() {
    use hologram_core::op::FloatDType;
    use hologram_exec::float_dispatch::dispatch_shape_sliced;

    let in_shape = vec![1usize, 2, 6, 8];
    let out = dispatch_shape_sliced(&in_shape, FloatDType::I64, 0, 2).expect("shape sliced 0:2");
    let dims: Vec<i64> = bytemuck::cast_slice(&out).to_vec();
    assert_eq!(
        dims,
        vec![1i64, 2],
        "Shape(start=0,end=2) of [1,2,6,8] must return [1,2]"
    );
}

/// Shape with no start/end (start=0, end=i64::MAX) must return all dims.
#[test]
fn shape_no_bounds_returns_all_dims() {
    use hologram_core::op::FloatDType;
    use hologram_exec::float_dispatch::dispatch_shape_sliced;

    let in_shape = vec![1usize, 2, 6, 8];
    let out =
        dispatch_shape_sliced(&in_shape, FloatDType::I64, 0, i64::MAX).expect("shape all dims");
    let dims: Vec<i64> = bytemuck::cast_slice(&out).to_vec();
    assert_eq!(
        dims,
        vec![1i64, 2, 6, 8],
        "Shape with full range must return all dims"
    );
}

/// Shape with negative start/end must count from rank end (ONNX spec).
#[test]
fn shape_negative_indices_count_from_end() {
    use hologram_core::op::FloatDType;
    use hologram_exec::float_dispatch::dispatch_shape_sliced;

    // [1,2,6,8]: start=-2, end=i64::MAX → dims[-2:] = [6, 8]
    let in_shape = vec![1usize, 2, 6, 8];
    let out =
        dispatch_shape_sliced(&in_shape, FloatDType::I64, -2, i64::MAX).expect("negative start");
    let dims: Vec<i64> = bytemuck::cast_slice(&out).to_vec();
    assert_eq!(
        dims,
        vec![6i64, 8],
        "Shape(start=-2) of [1,2,6,8] must return [6,8]"
    );
}
