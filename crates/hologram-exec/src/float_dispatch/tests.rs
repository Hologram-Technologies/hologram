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
