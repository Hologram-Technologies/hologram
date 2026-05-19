//! Hologram's prism-canonical axis impls smoke + equivalence tests.
//!
//! Verifies that:
//! - `prism::tensor::TensorAxis::matmul` resolves on hologram markers
//!   and produces correct f32 matmul outputs.
//! - `prism::tensor::ActivationAxis::{relu, sigmoid_q}` resolves and
//!   produces correct outputs.
//! - The `AxisExtension::dispatch_kernel` routing (companion-macro-
//!   emitted) reaches both kernels by their numeric id.

use prism::tensor::{ActivationAxis, TensorAxis};
use prism::pipeline::AxisExtension;
use hologram_backend::{
    HologramF32Tensor4x4Matmul,
    HologramF32VectorActivation16,
};

fn write_f32_row(bytes: &mut [u8], row: &[f32]) {
    for (i, v) in row.iter().enumerate() {
        bytes[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn read_f32_row(bytes: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn tensor_axis_matmul_resolves_on_hologram_marker() {
    // 4×4 identity × identity → identity. Bytes are little-endian f32.
    const N: usize = 4;
    let mat_bytes = 4 * N * N;
    let mut input = vec![0u8; 2 * mat_bytes];
    let mut a = vec![0f32; N * N];
    let mut b = vec![0f32; N * N];
    for i in 0..N {
        a[i * N + i] = 1.0;
        b[i * N + i] = 1.0;
    }
    write_f32_row(&mut input[..mat_bytes], &a);
    write_f32_row(&mut input[mat_bytes..], &b);

    let mut out = vec![0u8; mat_bytes];
    let n = <HologramF32Tensor4x4Matmul as TensorAxis>::matmul(&input, &mut out).unwrap();
    assert_eq!(n, mat_bytes);
    let c = read_f32_row(&out, N * N);
    let expected: Vec<f32> = (0..N * N).map(|i| if i / N == i % N { 1.0 } else { 0.0 }).collect();
    assert_eq!(c, expected);
}

#[test]
fn activation_axis_relu_clamps_negatives() {
    const N: usize = 16;
    let mut input = vec![0u8; 4 * N];
    let values: Vec<f32> = (0..N).map(|i| (i as f32) - 8.0).collect();
    write_f32_row(&mut input, &values);

    let mut out = vec![0u8; 4 * N];
    let n = <HologramF32VectorActivation16 as ActivationAxis>::relu(&input, &mut out).unwrap();
    assert_eq!(n, 4 * N);
    let result = read_f32_row(&out, N);
    for (i, r) in result.iter().enumerate().take(N) {
        let v = (i as f32) - 8.0;
        assert_eq!(*r, if v > 0.0 { v } else { 0.0 });
    }
}

#[test]
fn activation_axis_sigmoid_q_is_monotonic() {
    const N: usize = 16;
    let mut input = vec![0u8; 4 * N];
    let values: Vec<f32> = (0..N).map(|i| (i as f32) - 8.0).collect();
    write_f32_row(&mut input, &values);

    let mut out = vec![0u8; 4 * N];
    <HologramF32VectorActivation16 as ActivationAxis>::sigmoid_q(&input, &mut out).unwrap();
    let result = read_f32_row(&out, N);
    // Monotone non-decreasing in the input.
    let mut prev = result[0];
    for &v in &result[1..] {
        assert!(v >= prev, "sigmoid not monotonic: prev={prev}, v={v}");
        prev = v;
    }
    // Sigmoid(0) ≈ 0.5 (index 8: input = 0.0).
    assert!((result[8] - 0.5).abs() < 1e-3);
}

#[test]
fn axis_extension_dispatch_routes_through_kernel_ids() {
    // `axis!` companion-macro emits `AxisExtension::dispatch_kernel`
    // matching each method against its compile-time module-level
    // `KERNEL_*: u32` id (emitted alongside the trait declaration).
    // The resolver routes `kernel_id = 0` (the matmul slot) to `matmul`.
    const N: usize = 4;
    let mat_bytes = 4 * N * N;
    let mut input = vec![0u8; 2 * mat_bytes];
    let mut a = vec![0f32; N * N];
    let mut b = vec![0f32; N * N];
    for i in 0..N {
        a[i * N + i] = 2.0;
        b[i * N + i] = 3.0;
    }
    write_f32_row(&mut input[..mat_bytes], &a);
    write_f32_row(&mut input[mat_bytes..], &b);

    let mut out = vec![0u8; mat_bytes];
    // TensorAxis has one method (matmul) → KERNEL_MATMUL = 0.
    let n = <HologramF32Tensor4x4Matmul as AxisExtension>::dispatch_kernel(
        0, &input, &mut out,
    ).unwrap();
    assert_eq!(n, mat_bytes);
    let c = read_f32_row(&out, N * N);
    assert!((c[0] - 6.0).abs() < 1e-6);
    assert!((c[5] - 6.0).abs() < 1e-6);

    // An out-of-range kernel id returns ShapeViolation rather than
    // panicking — verifies the companion macro's default catch-all.
    let oor = <HologramF32Tensor4x4Matmul as AxisExtension>::dispatch_kernel(
        99, &input, &mut out,
    );
    assert!(oor.is_err());
}
